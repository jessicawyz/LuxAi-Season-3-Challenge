import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import deque
import random
import jax

from envs.vec_env import make_vec_env
from envs.action_head import HierarchicalActionHead
from selfplay import OpponentPool, create_selfplay_pool, PFSPConfig
from eval import evaluate_and_update_pool, EvalConfig
from baseline_agent import Agent as BaselineAgent
from dataclasses import asdict

# IL reward imports
try:
    from envs.il_rewards import create_il_reward_shaper
    IL_AVAILABLE = True
except ImportError:
    IL_AVAILABLE = False
    print("[MAPPO] IL reward module not available")

class ConvLSTMCell(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(self, x, hidden_state):
        h, c = hidden_state
        
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class SpatialEncoder(nn.Module):
    
    def __init__(self, input_channels: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.convlstm = ConvLSTMCell(128, hidden_dim, kernel_size=3)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, spatial_features, hidden_state=None):
        batch_size = spatial_features.shape[0]
        
        x = F.relu(self.conv1(spatial_features))
        x = F.relu(self.conv2(x))
        
        if hidden_state is None:
            h = torch.zeros(
                batch_size, self.hidden_dim,
                spatial_features.shape[2], spatial_features.shape[3],
                device=spatial_features.device
            )
            c = torch.zeros_like(h)
            hidden_state = (h, c)
        
        h, new_hidden_state = self.convlstm(x, hidden_state)
        
        pooled = self.pool(h).flatten(1)
        encoded = self.fc(pooled)
        
        return encoded, new_hidden_state


class TransformerUnitEncoder(nn.Module):
        
    def __init__(
        self,
        unit_feature_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(unit_feature_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, unit_features, unit_mask):
        batch_size = unit_features.shape[0]
        num_units = unit_features.shape[1]
        
        has_valid_units = unit_mask.any(dim=1)
        
        x = self.input_proj(unit_features)
        
        encoded = torch.zeros_like(x)
        aggregated = torch.zeros(batch_size, x.shape[-1], device=x.device)
        
        if has_valid_units.any():
            attn_mask = ~unit_mask
            
            valid_indices = torch.where(has_valid_units)[0]
            
            if len(valid_indices) > 0:
                x_valid = x[valid_indices]
                attn_mask_valid = attn_mask[valid_indices]
                
                encoded_valid = self.transformer(x_valid, src_key_padding_mask=attn_mask_valid)
                
                encoded_valid = self.output_proj(encoded_valid)
                
                encoded[valid_indices] = encoded_valid
                
                mask_expanded = unit_mask[valid_indices].unsqueeze(-1).float()
                aggregated_valid = (encoded_valid * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
                aggregated[valid_indices] = aggregated_valid
        
        return encoded, aggregated


@dataclass
class MAPPOConfig:
    
    num_envs: int = 8
    reward_mode: str = "dense"
    
    # Training parameters
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps per update
    n_epochs: int = 10  # PPO epochs per update
    batch_size: int = 256  # Mini-batch size
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # If None, no clipping
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5
    
    # Exploration
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500_000
    
    # Architecture
    spatial_channels: int = 23
    unit_feature_dim: int = 10
    global_feature_dim: int = 12
    hidden_dim: int = 256
    convlstm_hidden_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    
    # Environment
    max_units: int = 16
    map_width: int = 24
    map_height: int = 24
    unit_sap_range: int = 4
    
    # Checkpointing
    snapshot_freq: int = 100_000
    eval_freq: int = 50_000
    checkpoint_dir: str = "checkpoint"
    
    # Self-play parameters
    use_selfplay: bool = True
    selfplay_ratio: float = 0.5
    num_eval_opponents: int = 5
    use_baseline_opponent: bool = False
    max_pool_size: int = 20
    elo_k_factor: float = 32.0
    
    # IL reward parameters
    use_il_reward: bool = False
    il_model_path: str = "checkpoint/unit_unet.pth"
    il_reward_weight: float = 0.1  # lambda
    il_reward_bonus: float = 0.1   # bonus per matching action
    il_reward_penalty: float = 0.1  # penalty per mismatched action
    il_reward_anneal: bool = True
    il_reward_anneal_start: float = 0.5
    il_reward_anneal_end: float = 0.05
    il_reward_anneal_steps: int = 500_000
    il_compare_sap_targets: bool = False  # Compare SAP targets (needs SAP-UNet)
    target_il_agreement_rate: float = 0.6
    
    log_freq: int = 1000
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MAPPOActor(nn.Module):
    """Actor network that outputs action distributions"""
    
    def __init__(self, config: MAPPOConfig):
        super().__init__()
        
        self.config = config
        
        self.spatial_encoder = SpatialEncoder(
            input_channels=config.spatial_channels,
            hidden_dim=config.convlstm_hidden_dim,
            output_dim=config.hidden_dim
        )
        
        self.unit_encoder = TransformerUnitEncoder(
            unit_feature_dim=config.unit_feature_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers
        )
        
        self.global_proj = nn.Linear(config.global_feature_dim, config.hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        
        self.action_head = HierarchicalActionHead(
            feature_dim=config.hidden_dim,
            max_units=config.max_units,
            unit_sap_range=config.unit_sap_range,
            hidden_dim=config.hidden_dim,
        )
    
    def forward(
        self,
        spatial_features,
        unit_features,
        unit_mask,
        global_features,
        unit_energies,
        unit_positions,
        tile_types,
        spatial_hidden_state=None,
        epsilon=0.0,
        deterministic=False
    ):
        batch_size = spatial_features.shape[0]
        
        # Encode features
        spatial_encoded, new_spatial_hidden_state = self.spatial_encoder(
            spatial_features, spatial_hidden_state
        )
        unit_encoded, unit_aggregated = self.unit_encoder(unit_features, unit_mask)
        global_encoded = F.relu(self.global_proj(global_features))
        
        # Fuse features
        fused = self.fusion(torch.cat([
            spatial_encoded, unit_aggregated, global_encoded
        ], dim=-1))
        fused_per_unit = fused.unsqueeze(1).expand(-1, self.config.max_units, -1)
        unit_features_combined = fused_per_unit + unit_encoded
        
        # Get action logits
        action_type_logits, sap_target_logits, mask_info = self.action_head(
            unit_features_combined,
            unit_mask,
            unit_energies,
            unit_positions,
            self.config.map_width,
            self.config.map_height,
            tile_types,
            unit_move_cost=2,
            unit_sap_cost=40,
        )
        
        # Sample or select actions
        if deterministic:
            actions, action_info = self.action_head.sample_actions(
                action_type_logits,
                sap_target_logits,
                mask_info["action_type_mask"],
                mask_info["sap_target_mask"],
                unit_mask,
                deterministic=True
            )
        else:
            # Sample from distribution
            actions, action_info = self.action_head.sample_actions(
                action_type_logits,
                sap_target_logits,
                mask_info["action_type_mask"],
                mask_info["sap_target_mask"],
                unit_mask,
                deterministic=False
            )
        
        # Compute log probabilities
        log_probs = self.compute_log_probs(
            actions,
            action_type_logits,
            sap_target_logits,
            mask_info["action_type_mask"],
            mask_info["sap_target_mask"],
            unit_mask
        )
        
        # Compute entropy
        entropy = self.compute_entropy(
            action_type_logits,
            sap_target_logits,
            mask_info["action_type_mask"],
            mask_info["sap_target_mask"],
            unit_mask
        )
        
        return actions, log_probs, entropy, new_spatial_hidden_state, action_info
    
    def compute_log_probs(
        self,
        actions,
        action_type_logits,
        sap_target_logits,
        action_type_mask,
        sap_target_mask,
        unit_mask
    ):
        """Compute log probabilities of actions"""
        batch_size = actions.shape[0]
        
        # Apply masks to logits
        action_type_logits_masked = action_type_logits.clone()
        action_type_logits_masked[~action_type_mask] = -1e10
        
        sap_target_logits_masked = sap_target_logits.clone()
        sap_target_logits_masked[~sap_target_mask] = -1e10
        
        # Action type log probs
        action_type_log_probs = F.log_softmax(action_type_logits_masked, dim=-1)
        selected_action_type_log_probs = action_type_log_probs.gather(
            -1, actions[:, :, 0].unsqueeze(-1)
        ).squeeze(-1)
        
        # SAP target log probs (only for SAP actions)
        sap_target_log_probs = F.log_softmax(sap_target_logits_masked, dim=-1)
        selected_sap_target_log_probs = sap_target_log_probs.gather(
            -1, actions[:, :, 1].unsqueeze(-1)
        ).squeeze(-1)
        
        # Combine log probs (only count SAP target for SAP actions, action type 5)
        is_sap = (actions[:, :, 0] == 5).float()
        log_probs = selected_action_type_log_probs + is_sap * selected_sap_target_log_probs
        
        # Apply unit mask
        log_probs = log_probs * unit_mask.float()
        
        # Sum over units
        log_probs = log_probs.sum(dim=1)
        
        return log_probs
    
    def compute_entropy(
        self,
        action_type_logits,
        sap_target_logits,
        action_type_mask,
        sap_target_mask,
        unit_mask
    ):
        """Compute entropy of action distributions"""
        # Apply masks
        action_type_logits_masked = action_type_logits.clone()
        action_type_logits_masked[~action_type_mask] = -1e10
        
        sap_target_logits_masked = sap_target_logits.clone()
        sap_target_logits_masked[~sap_target_mask] = -1e10
        
        # Compute distributions
        action_type_probs = F.softmax(action_type_logits_masked, dim=-1)
        sap_target_probs = F.softmax(sap_target_logits_masked, dim=-1)
        
        # Entropy for action types
        action_type_entropy = -(action_type_probs * F.log_softmax(action_type_logits_masked, dim=-1)).sum(dim=-1)
        
        # Entropy for SAP targets
        sap_target_entropy = -(sap_target_probs * F.log_softmax(sap_target_logits_masked, dim=-1)).sum(dim=-1)
        
        # Average entropy (weighted by unit mask)
        entropy = (action_type_entropy + sap_target_entropy) * unit_mask.float()
        entropy = entropy.sum(dim=1) / (unit_mask.float().sum(dim=1) + 1e-8)
        
        return entropy.mean()


class MAPPOCritic(nn.Module):
    """Centralized critic that outputs state values"""
    
    def __init__(self, config: MAPPOConfig):
        super().__init__()
        
        self.config = config
        
        # Global spatial encoder (sees both teams)
        self.spatial_encoder = SpatialEncoder(
            input_channels=config.spatial_channels * 2,
            hidden_dim=config.convlstm_hidden_dim,
            output_dim=config.hidden_dim
        )
        
        # Unit encoders for both teams
        self.unit_encoder_0 = TransformerUnitEncoder(
            unit_feature_dim=config.unit_feature_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers
        )
        
        self.unit_encoder_1 = TransformerUnitEncoder(
            unit_feature_dim=config.unit_feature_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers
        )
        
        # Global features projection
        self.global_proj = nn.Linear(config.global_feature_dim * 2, config.hidden_dim)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
    
    def forward(
        self,
        global_spatial_features,
        team_0_unit_features,
        team_0_unit_mask,
        team_1_unit_features,
        team_1_unit_mask,
        global_features,
        spatial_hidden_state=None
    ):
        # Encode spatial features
        spatial_encoded, new_spatial_hidden_state = self.spatial_encoder(
            global_spatial_features, spatial_hidden_state
        )
        
        # Encode units
        _, team_0_aggregated = self.unit_encoder_0(team_0_unit_features, team_0_unit_mask)
        _, team_1_aggregated = self.unit_encoder_1(team_1_unit_features, team_1_unit_mask)
        
        # Encode global features
        global_encoded = F.relu(self.global_proj(global_features))
        
        # Combine all features
        combined = torch.cat([
            spatial_encoded,
            team_0_aggregated,
            team_1_aggregated,
            global_encoded
        ], dim=-1)
        
        # Compute value
        value = self.value_head(combined).squeeze(-1)
        
        return value, new_spatial_hidden_state


class RolloutBuffer:
    """Buffer for storing trajectories for PPO"""
    
    def __init__(self, n_steps: int, num_envs: int, device: str):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: Dict,
        actions: Dict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor
    ):
        self.observations.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)
        self.log_probs.append(log_probs)
        
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True
    
    def get(self, last_values: torch.Tensor):
        """Get all data and compute advantages using GAE"""
        assert self.full, "Buffer not full"
        
        # Stack everything
        rewards = torch.stack(self.rewards)  # (n_steps, num_envs, 2)
        values = torch.stack(self.values)  # (n_steps, num_envs, 2)
        dones = torch.stack(self.dones)  # (n_steps, num_envs)
        
        # Compute advantages using GAE for each team
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        for team_id in range(2):
            team_rewards = rewards[:, :, team_id]
            team_values = values[:, :, team_id]
            team_last_values = last_values[:, team_id]
            
            team_advantages, team_returns = self._compute_gae(
                team_rewards,
                team_values,
                team_last_values,
                dones
            )
            
            advantages[:, :, team_id] = team_advantages
            returns[:, :, team_id] = team_returns
        
        return advantages, returns
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """Compute Generalized Advantage Estimation"""
        n_steps = rewards.shape[0]
        num_envs = rewards.shape[1]
        
        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.zeros(num_envs, device=rewards.device)
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        
        return advantages, returns


def soft_update(target_net, source_net, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class BaselineAgentWrapper:
    """Wrapper to use baseline agent in training loop"""
    
    def __init__(self, env_cfg: dict, device: str = "cpu"):
        self.agent = BaselineAgent("player_1", env_cfg)
        self.device = device
        self.env_cfg = env_cfg
    
    def get_actions(self, obs_raw_batch, step: int) -> torch.Tensor:
        """Convert observations and get actions from baseline agent"""
        batch_size = len(obs_raw_batch["player_1"])
        max_units = self.env_cfg["max_units"]
        
        actions_batch = []
        for i in range(batch_size):
            obs_single = obs_raw_batch["player_1"][i]
            
            # Convert JAX arrays to numpy if needed
            obs_dict = {}
            for key, value in obs_single.items():
                if hasattr(value, "__array__"):
                    obs_dict[key] = np.array(value)
                else:
                    obs_dict[key] = value
            
            # Get actions from baseline agent
            actions = self.agent.act(step, obs_dict, remainingOverageTime=60)
            actions_batch.append(actions)
        
        # Convert to tensor
        actions_tensor = torch.tensor(
            np.stack(actions_batch), 
            dtype=torch.long, 
            device=self.device
        )
        
        return actions_tensor


def train_mappo(config: MAPPOConfig):
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize opponent pool for self-play
    if config.use_selfplay:
        pfsp_config = PFSPConfig(
            pfsp_ratio=config.selfplay_ratio,
        )
        opponent_pool = create_selfplay_pool(
            checkpoint_dir=config.checkpoint_dir,
            model_type="mappo",
            max_pool_size=config.max_pool_size,
            elo_k_factor=config.elo_k_factor,
            pfsp_config=pfsp_config,
        )
        print(f"Initialized opponent pool with {len(opponent_pool.opponents)} opponents")
    else:
        opponent_pool = None
    
    # Initialize environment
    env = make_vec_env(
        num_envs=config.num_envs,
        reward_mode=config.reward_mode,
        device=config.device
    )
    
    # Initialize IL reward shaper if enabled
    il_reward_shaper = None
    if config.use_il_reward:
        print(f"Initializing IL reward shaper...")
        try:
            il_reward_shaper = create_il_reward_shaper(
                il_model_path=config.il_model_path,
                num_envs=config.num_envs,
                device=config.device,
                bonus_per_match=config.il_reward_bonus,
                penalty_per_mismatch=config.il_reward_penalty,
                weight=config.il_reward_weight,
                anneal=config.il_reward_anneal,
                anneal_start=config.il_reward_anneal_start,
                anneal_end=config.il_reward_anneal_end,
                anneal_steps=config.il_reward_anneal_steps,
                compare_sap_targets=config.il_compare_sap_targets,
                map_size=config.map_width,
                max_units=config.max_units,
            )
            print(f"IL reward shaper initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize IL reward shaper: {e}")
            print(f"Continuing without IL rewards")
            config.use_il_reward = False
    
    # Initialize baseline agent if using it as opponent
    baseline_wrapper = None
    if config.use_baseline_opponent:
        env_cfg = {
            "max_units": config.max_units,
            "map_width": config.map_width,
            "map_height": config.map_height,
        }
        baseline_wrapper = BaselineAgentWrapper(env_cfg, device=config.device)
        print("Initialized baseline agent as opponent")
    
    # Initialize networks
    actor_0 = MAPPOActor(config).to(config.device)
    actor_1 = MAPPOActor(config).to(config.device)
    critic_0 = MAPPOCritic(config).to(config.device)
    critic_1 = MAPPOCritic(config).to(config.device)
    
    # Optimizers
    actor_0_optimizer = torch.optim.Adam(actor_0.parameters(), lr=config.learning_rate)
    actor_1_optimizer = torch.optim.Adam(actor_1.parameters(), lr=config.learning_rate)
    critic_0_optimizer = torch.optim.Adam(critic_0.parameters(), lr=config.learning_rate)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    
    # Rollout buffers
    rollout_buffer_0 = RolloutBuffer(config.n_steps, config.num_envs, config.device)
    rollout_buffer_1 = RolloutBuffer(config.n_steps, config.num_envs, config.device)
    
    # Initialize environment
    obs = env.reset()
    
    # Tracking
    global_step = 0
    episode_rewards = np.zeros(config.num_envs)
    best_elo = -np.inf
    current_opponent = None
    opponent_actor_0 = None
    opponent_actor_1 = None
    
    start_time = time.time()
    
    print("Starting MAPPO training...")
    
    while global_step < config.total_timesteps:
        
        # Epsilon decay
        epsilon = max(
            config.epsilon_end,
            config.epsilon_start - (config.epsilon_start - config.epsilon_end) * global_step / config.epsilon_decay_steps
        )
        
        # Decide whether to use opponent from pool
        use_opponent = (
            config.use_selfplay and 
            opponent_pool is not None and 
            len(opponent_pool.opponents) > 0 and
            np.random.random() < config.selfplay_ratio
        )
        
        # Sample new opponent if needed
        if use_opponent and (current_opponent is None or np.random.random() < 0.1):
            try:
                current_opponent = opponent_pool.sample_opponent(
                    mode="pfsp",
                    current_step=global_step
                )
                opponent_checkpoint = torch.load(current_opponent.path, map_location=config.device)
                opponent_actor_0 = MAPPOActor(config).to(config.device)
                opponent_actor_1 = MAPPOActor(config).to(config.device)
                opponent_actor_0.load_state_dict(opponent_checkpoint["actor_0"])
                opponent_actor_1.load_state_dict(opponent_checkpoint["actor_1"])
                opponent_actor_0.eval()
                opponent_actor_1.eval()
                print(f"Sampled opponent: {os.path.basename(current_opponent.path)} (ELO: {current_opponent.elo_rating:.1f})")
            except Exception as e:
                print(f"Warning: Could not load opponent: {e}")
                use_opponent = False
                current_opponent = None
        
        # === Collect rollouts ===
        rollout_buffer_0.reset()
        rollout_buffer_1.reset()
        
        # Initialize IL info dict
        il_info = {}
        
        for step in range(config.n_steps):
            with torch.no_grad():
                # Get actions from actors
                actions_0, log_probs_0, _, _, _ = actor_0(
                    spatial_features=obs["team_0"]["spatial_features"],
                    unit_features=obs["team_0"]["unit_features"],
                    unit_mask=obs["team_0"]["unit_mask"],
                    global_features=obs["team_0"]["global_features"],
                    unit_energies=obs["team_0"]["unit_features"][:, :, 2] * 400,
                    unit_positions=(obs["team_0"]["unit_features"][:, :, :2] *
                                  torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                    tile_types=obs["team_0"]["spatial_features"][:, 1] * 2,
                    epsilon=epsilon,
                    deterministic=False
                )
                
                # Player 1: baseline, opponent, or current policy
                if config.use_baseline_opponent and baseline_wrapper is not None:
                    obs_raw = []
                    for i in range(config.num_envs):
                        obs_single = jax.tree.map(lambda x: x[i], env.prev_obs_raw)
                        obs_single_np = jax.tree.map(lambda x: np.array(x), obs_single)
                        obs_raw.append(obs_single_np)
                    
                    obs_raw_batch = {"player_1": [asdict(obs["player_1"]) for obs in obs_raw]}
                    actions_1 = baseline_wrapper.get_actions(obs_raw_batch, step=global_step + step)
                    log_probs_1 = torch.zeros(config.num_envs, device=config.device)  # Dummy log probs
                elif use_opponent and opponent_actor_1 is not None:
                    actions_1, log_probs_1, _, _, _ = opponent_actor_1(
                        spatial_features=obs["team_1"]["spatial_features"],
                        unit_features=obs["team_1"]["unit_features"],
                        unit_mask=obs["team_1"]["unit_mask"],
                        global_features=obs["team_1"]["global_features"],
                        unit_energies=obs["team_1"]["unit_features"][:, :, 2] * 400,
                        unit_positions=(obs["team_1"]["unit_features"][:, :, :2] *
                                      torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                        tile_types=obs["team_1"]["spatial_features"][:, 1] * 2,
                        epsilon=0.0,
                        deterministic=False
                    )
                else:
                    actions_1, log_probs_1, _, _, _ = actor_1(
                        spatial_features=obs["team_1"]["spatial_features"],
                        unit_features=obs["team_1"]["unit_features"],
                        unit_mask=obs["team_1"]["unit_mask"],
                        global_features=obs["team_1"]["global_features"],
                        unit_energies=obs["team_1"]["unit_features"][:, :, 2] * 400,
                        unit_positions=(obs["team_1"]["unit_features"][:, :, :2] *
                                      torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                        tile_types=obs["team_1"]["spatial_features"][:, 1] * 2,
                        epsilon=epsilon,
                        deterministic=False
                    )
                
                # Get values from critics
                values_0, _ = critic_0(
                    global_spatial_features=obs["global"]["spatial_features"],
                    team_0_unit_features=obs["global"]["team_0_units"],
                    team_0_unit_mask=obs["global"]["team_0_unit_mask"],
                    team_1_unit_features=obs["global"]["team_1_units"],
                    team_1_unit_mask=obs["global"]["team_1_unit_mask"],
                    global_features=obs["global"]["global_features"]
                )
                
                values_1, _ = critic_1(
                    global_spatial_features=obs["global"]["spatial_features"],
                    team_0_unit_features=obs["global"]["team_0_units"],
                    team_0_unit_mask=obs["global"]["team_0_unit_mask"],
                    team_1_unit_features=obs["global"]["team_1_units"],
                    team_1_unit_mask=obs["global"]["team_1_unit_mask"],
                    global_features=obs["global"]["global_features"]
                )
            
            # Step environment
            actions_dict = {
                "player_0": actions_0,
                "player_1": actions_1
            }
            
            next_obs, rewards, dones, truncated, infos = env.step(actions_dict, seed=global_step + step)
            
            # Compute IL rewards if enabled
            il_rewards_0 = torch.zeros(config.num_envs, device=config.device)
            il_rewards_1 = torch.zeros(config.num_envs, device=config.device)
            
            if config.use_il_reward and il_reward_shaper is not None:
                try:
                    # Extract raw observations from vectorized environment
                    if hasattr(env, 'prev_obs_raw') and env.prev_obs_raw is not None:
                        # Convert JAX observations to list of dicts per environment
                        raw_obs_batch = []
                        for i in range(config.num_envs):
                            # Extract single environment observation
                            obs_single = jax.tree.map(lambda x: x[i], env.prev_obs_raw)
                            # Convert to numpy
                            obs_single = jax.tree.map(lambda x: np.array(x), obs_single)
                            raw_obs_batch.append(obs_single)
                        
                        # Compute IL rewards for team 0
                        obs_team_0 = [asdict(o["player_0"]) for o in raw_obs_batch]
                        actions_team_0 = actions_0.cpu().numpy()
                        unit_masks_0 = obs["team_0"]["unit_mask"].cpu().numpy()
                        
                        game_params = {
                            "max_units": config.max_units,
                            "unit_move_cost": env.env_params.unit_move_cost,
                            "unit_sap_cost": env.env_params.unit_sap_cost,
                            "unit_sap_range": env.env_params.unit_sap_range,
                            "unit_sensor_range": env.env_params.unit_sensor_range,
                            "nebula_tile_energy_reduction": env.env_params.nebula_tile_energy_reduction,
                            "unit_sap_dropoff_factor": env.env_params.unit_sap_dropoff_factor,
                            "unit_energy_void_factor": env.env_params.unit_energy_void_factor,
                        }
                        
                        il_r0, il_i0 = il_reward_shaper.compute_rewards(
                            obs_batch=obs_team_0,
                            team_ids=[0] * config.num_envs,
                            rl_actions=actions_team_0,
                            unit_masks=unit_masks_0,
                            game_params=game_params,
                            global_step=global_step + step,
                        )
                        il_rewards_0 = torch.from_numpy(il_r0).float().to(config.device)
                        
                        # Compute IL rewards for team 1
                        obs_team_1 = [asdict(o["player_1"]) for o in raw_obs_batch]
                        actions_team_1 = actions_1.cpu().numpy()
                        unit_masks_1 = obs["team_1"]["unit_mask"].cpu().numpy()
                        
                        il_r1, il_i1 = il_reward_shaper.compute_rewards(
                            obs_batch=obs_team_1,
                            team_ids=[1] * config.num_envs,
                            rl_actions=actions_team_1,
                            unit_masks=unit_masks_1,
                            game_params=game_params,
                            global_step=global_step + step,
                        )
                        il_rewards_1 = torch.from_numpy(il_r1).float().to(config.device)
                        
                        # Store IL info for logging (only once per rollout)
                        if step == 0:
                            il_info = {}
                            il_info.update({f"team_0_{k}": v for k, v in il_i0.items()})
                            il_info.update({f"team_1_{k}": v for k, v in il_i1.items()})
                
                except Exception as e:
                    if step == 0:  # Only print once
                        print(f"Warning: IL reward computation failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Add IL rewards to environment rewards per-environment
            if config.use_il_reward and il_reward_shaper is not None:
                for i in range(config.num_envs):
                    # rewards[i] is a tensor [player_0_reward, player_1_reward]
                    il_reward_add = torch.tensor(
                        [il_rewards_0[i].item(), il_rewards_1[i].item()],
                        dtype=rewards[i].dtype,
                        device=rewards[i].device
                    )
                    rewards[i] = rewards[i] + il_reward_add
            
            # Store in buffers
            values = torch.stack([values_0, values_1], dim=1)  # (num_envs, 2)
            log_probs = torch.stack([log_probs_0, log_probs_1], dim=1)  # (num_envs, 2)
            
            rollout_buffer_0.add(obs, actions_dict, rewards, dones, values, log_probs)
            rollout_buffer_1.add(obs, actions_dict, rewards, dones, values, log_probs)
            
            # Track rewards
            for i in range(config.num_envs):
                episode_rewards[i] += rewards[i].sum().item()
                if dones[i]:
                    episode_rewards[i] = 0.0
                    
                    # Reset IL state for this environment
                    if config.use_il_reward and il_reward_shaper is not None:
                        il_reward_shaper.reset(env_idx=i)
            
            obs = next_obs
            global_step += config.num_envs
        
        # === Compute advantages ===
        with torch.no_grad():
            last_values_0, _ = critic_0(
                global_spatial_features=obs["global"]["spatial_features"],
                team_0_unit_features=obs["global"]["team_0_units"],
                team_0_unit_mask=obs["global"]["team_0_unit_mask"],
                team_1_unit_features=obs["global"]["team_1_units"],
                team_1_unit_mask=obs["global"]["team_1_unit_mask"],
                global_features=obs["global"]["global_features"]
            )
            
            last_values_1, _ = critic_1(
                global_spatial_features=obs["global"]["spatial_features"],
                team_0_unit_features=obs["global"]["team_0_units"],
                team_0_unit_mask=obs["global"]["team_0_unit_mask"],
                team_1_unit_features=obs["global"]["team_1_units"],
                team_1_unit_mask=obs["global"]["team_1_unit_mask"],
                global_features=obs["global"]["global_features"]
            )
            
            last_values = torch.stack([last_values_0, last_values_1], dim=1)
        
        advantages_0, returns_0 = rollout_buffer_0.get(last_values)
        advantages_1, returns_1 = rollout_buffer_1.get(last_values)
        
        # === PPO updates ===
        for epoch in range(config.n_epochs):
            # Update actor 0 and critic 0
            update_ppo(
                actor_0, critic_0,
                actor_0_optimizer, critic_0_optimizer,
                rollout_buffer_0,
                advantages_0[:, :, 0],
                returns_0[:, :, 0],
                config,
                team_id=0
            )
            
            # Update actor 1 and critic 1 (skip if using baseline)
            if not config.use_baseline_opponent:
                update_ppo(
                    actor_1, critic_1,
                    actor_1_optimizer, critic_1_optimizer,
                    rollout_buffer_1,
                    advantages_1[:, :, 1],
                    returns_1[:, :, 1],
                    config,
                    team_id=1
                )
        
        # Logging
        if global_step % config.log_freq == 0:
            mean_reward = np.mean(episode_rewards)
            elapsed = (time.time() - start_time) / 60
            log_msg = f"[{elapsed:.2f} min] Step {global_step} | Epsilon {epsilon:.3f} | Mean Reward {mean_reward:.2f}"
            if config.use_baseline_opponent:
                log_msg += " | Opponent: Baseline"
            elif use_opponent and current_opponent:
                log_msg += f" | Opponent ELO {current_opponent.elo_rating:.1f}"
            
            # Add IL statistics if available
            if config.use_il_reward and il_info:
                # Average across teams
                il_agreement = (il_info.get("team_0_il_agreement_rate", 0) + 
                               il_info.get("team_1_il_agreement_rate", 0)) / 2
                il_weight = il_info.get("team_0_il_weight", 0)
                log_msg += f" | IL Agree {il_agreement:.2%} | IL λ {il_weight:.3f}"
            
            print(log_msg)
        
        # Evaluation and checkpoint management
        if global_step % config.eval_freq == 0 and global_step > 0:
            print(f"\n{'='*60}")
            print(f"EVALUATION AT STEP {global_step}")
            print(f"{'='*60}")
            
            # Save latest checkpoint
            latest_path = os.path.join(config.checkpoint_dir, "mappo_latest.pt")
            checkpoint = {
                "actor_0": actor_0.state_dict(),
                "actor_1": actor_1.state_dict(),
                "critic_0": critic_0.state_dict(),
                "critic_1": critic_1.state_dict(),
                "global_step": global_step,
                "config": config.__dict__,
            }
            torch.save(checkpoint, latest_path)
            print(f"Saved latest checkpoint")
            
            # Evaluate against opponent pool
            if config.use_selfplay and opponent_pool is not None and len(opponent_pool.opponents) > 0:
                eval_config = EvalConfig(
                    num_eval_episodes=10,
                    num_envs=config.num_envs,
                    device=config.device,
                )
                
                try:
                    eval_results, estimated_elo = evaluate_and_update_pool(
                        latest_path,
                        "mappo",
                        opponent_pool,
                        eval_config,
                        num_opponents=min(config.num_eval_opponents, len(opponent_pool.opponents)),
                        update_pool=True,
                    )
                    
                    print(f"Evaluation Results:")
                    print(f"  Win Rate: {eval_results['overall_win_rate']:.2%}")
                    print(f"  Estimated ELO: {estimated_elo:.1f}")
                    print(f"  Opponents Evaluated: {eval_results['num_opponents']}")
                    
                    # Save best checkpoint if ELO improved
                    if estimated_elo > best_elo:
                        best_elo = estimated_elo
                        best_path = os.path.join(config.checkpoint_dir, "mappo_best.pt")
                        torch.save(checkpoint, best_path)
                        print(f"Best checkpoint saved to mappo_best.pt (ELO: {best_elo:.1f})")
                    
                    # Add current checkpoint to opponent pool
                    opponent_pool.add_checkpoint(
                        checkpoint_path=latest_path,
                        model_type="mappo",
                        initial_rating=estimated_elo,
                        checkpoint_step=global_step,
                    )
                    
                    # Save opponent pool state
                    pool_state_path = os.path.join(config.checkpoint_dir, "mappo_population.json")
                    opponent_pool.save_pool_state(pool_state_path)
                    
                except Exception as e:
                    print(f"Warning: Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Skipping evaluation (no opponents in pool)")
            
            print(f"{'='*60}\n")
        
        # Save snapshot checkpoints
        if global_step % config.snapshot_freq == 0 and global_step > 0:
            snapshot_path = os.path.join(config.checkpoint_dir, f"mappo_{global_step}.pt")
            checkpoint = {
                "actor_0": actor_0.state_dict(),
                "actor_1": actor_1.state_dict(),
                "critic_0": critic_0.state_dict(),
                "critic_1": critic_1.state_dict(),
                "global_step": global_step,
                "config": config.__dict__,
            }
            torch.save(checkpoint, snapshot_path)
            print(f"Saved snapshot checkpoint at step {global_step}")
    
    # Save final checkpoint
    final_checkpoint = {
        "actor_0": actor_0.state_dict(),
        "actor_1": actor_1.state_dict(),
        "critic_0": critic_0.state_dict(),
        "critic_1": critic_1.state_dict(),
        "config": config.__dict__,
        "global_step": global_step,
    }
    torch.save(final_checkpoint, os.path.join(config.checkpoint_dir, "mappo_final.pt"))
    torch.save(final_checkpoint, os.path.join(config.checkpoint_dir, "mappo_latest.pt"))
    
    # Save pool state
    if opponent_pool is not None:
        pool_state_path = os.path.join(config.checkpoint_dir, "mappo_population.json")
        opponent_pool.save_pool_state(pool_state_path)
    
    print("Training complete! Final checkpoints saved.")
    
    env.close()


def update_ppo(
    actor, critic,
    actor_optimizer, critic_optimizer,
    rollout_buffer,
    advantages,
    returns,
    config,
    team_id
):
    """Perform one PPO update"""
    
    # Flatten data
    n_steps = config.n_steps
    num_envs = config.num_envs
    total_samples = n_steps * num_envs
    
    # Normalize advantages
    advantages_flat = advantages.flatten()
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
    
    returns_flat = returns.flatten()
    
    # Create mini-batches
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    n_batches = total_samples // config.batch_size
    
    for batch_idx in range(n_batches):
        batch_indices = indices[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size]
        
        # Get batch data
        batch_step_indices = batch_indices // num_envs
        batch_env_indices = batch_indices % num_envs
        
        # Reconstruct observations for batch
        batch_obs = {}
        for key in rollout_buffer.observations[0].keys():
            batch_obs[key] = {}
            for feat_key in rollout_buffer.observations[0][key].keys():
                feat_list = [rollout_buffer.observations[step][key][feat_key][env_idx]
                           for step, env_idx in zip(batch_step_indices, batch_env_indices)]
                batch_obs[key][feat_key] = torch.stack(feat_list)
        
        # Get batch actions
        if team_id == 0:
            player_key = "player_0"
            team_key = "team_0"
        else:
            player_key = "player_1"
            team_key = "team_1"
        
        batch_actions = torch.stack([
            rollout_buffer.actions[step][player_key][env_idx]
            for step, env_idx in zip(batch_step_indices, batch_env_indices)
        ])
        
        batch_old_log_probs = torch.stack([
            rollout_buffer.log_probs[step][env_idx, team_id]
            for step, env_idx in zip(batch_step_indices, batch_env_indices)
        ])
        
        batch_advantages = advantages_flat[batch_indices]
        batch_returns = returns_flat[batch_indices]
        
        # Forward pass through actor
        _, new_log_probs, entropy, _, _ = actor(
            spatial_features=batch_obs[team_key]["spatial_features"],
            unit_features=batch_obs[team_key]["unit_features"],
            unit_mask=batch_obs[team_key]["unit_mask"],
            global_features=batch_obs[team_key]["global_features"],
            unit_energies=batch_obs[team_key]["unit_features"][:, :, 2] * 400,
            unit_positions=(batch_obs[team_key]["unit_features"][:, :, :2] *
                          torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
            tile_types=batch_obs[team_key]["spatial_features"][:, 1] * 2,
            epsilon=0.0,
            deterministic=False
        )
        
        # PPO loss
        ratio = torch.exp(new_log_probs - batch_old_log_probs)
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy loss
        entropy_loss = -entropy
        
        # Total actor loss
        total_actor_loss = actor_loss + config.ent_coef * entropy_loss
        
        # Update actor
        actor_optimizer.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        actor_optimizer.step()
        
        # Forward pass through critic
        new_values, _ = critic(
            global_spatial_features=batch_obs["global"]["spatial_features"],
            team_0_unit_features=batch_obs["global"]["team_0_units"],
            team_0_unit_mask=batch_obs["global"]["team_0_unit_mask"],
            team_1_unit_features=batch_obs["global"]["team_1_units"],
            team_1_unit_mask=batch_obs["global"]["team_1_unit_mask"],
            global_features=batch_obs["global"]["global_features"]
        )
        
        # Value loss
        if config.clip_range_vf is not None:
            batch_old_values = torch.stack([
                rollout_buffer.values[step][env_idx, team_id]
                for step, env_idx in zip(batch_step_indices, batch_env_indices)
            ])
            values_clipped = batch_old_values + torch.clamp(
                new_values - batch_old_values,
                -config.clip_range_vf,
                config.clip_range_vf
            )
            value_loss1 = F.mse_loss(new_values, batch_returns)
            value_loss2 = F.mse_loss(values_clipped, batch_returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(new_values, batch_returns)
        
        # Update critic
        critic_optimizer.zero_grad()
        (config.vf_coef * value_loss).backward()
        nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        critic_optimizer.step()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MAPPO agent for Lux AI S3")
    
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--reward-mode", type=str, default="dense", choices=["sparse", "dense"], help="Reward mode")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--epsilon-start", type=float, default=0.3, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=500_000, help="Epsilon decay steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--use-selfplay", action="store_true", default=True, help="Use self-play training")
    parser.add_argument("--no-selfplay", action="store_false", dest="use_selfplay", help="Disable self-play")
    parser.add_argument("--use-baseline-opponent", action="store_true", default=False, help="Use baseline agent as opponent")
    parser.add_argument("--selfplay-ratio", type=float, default=0.5, help="Ratio of training vs opponent pool")
    parser.add_argument("--num-eval-opponents", type=int, default=5, help="Number of opponents for evaluation")
    parser.add_argument("--max-pool-size", type=int, default=20, help="Max opponent pool size")
    parser.add_argument("--elo-k-factor", type=float, default=32.0, help="ELO K-factor")
    parser.add_argument("--snapshot-freq", type=int, default=100_000, help="Snapshot save frequency (steps)")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency (steps)")
    parser.add_argument("--log-freq", type=int, default=1000, help="Log frequency (steps)")
    
    # IL reward arguments
    parser.add_argument("--use-il-reward", action="store_true", help="Enable IL reward shaping")
    parser.add_argument("--il-model-path", type=str, default="checkpoint/unit_unet.pth", help="Path to IL model")
    parser.add_argument("--il-reward-weight", type=float, default=0.1, help="IL reward weight (lambda)")
    parser.add_argument("--il-reward-bonus", type=float, default=0.1, help="Bonus per matching action")
    parser.add_argument("--il-reward-penalty", type=float, default=0.1, help="Penalty per mismatched action")
    parser.add_argument("--il-reward-anneal", action="store_true", default=True, help="Anneal IL weight")
    parser.add_argument("--il-reward-anneal-start", type=float, default=0.5, help="IL anneal start weight")
    parser.add_argument("--il-reward-anneal-end", type=float, default=0.05, help="IL anneal end weight")
    parser.add_argument("--il-reward-anneal-steps", type=int, default=500_000, help="IL anneal steps")
    parser.add_argument("--il-compare-sap-targets", action="store_true", help="Compare SAP targets")
    
    args = parser.parse_args()
    
    config = MAPPOConfig(
        num_envs=args.num_envs,
        reward_mode=args.reward_mode,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        checkpoint_dir=args.checkpoint_dir,
        use_selfplay=args.use_selfplay,
        use_baseline_opponent=args.use_baseline_opponent,
        selfplay_ratio=args.selfplay_ratio,
        num_eval_opponents=args.num_eval_opponents,
        max_pool_size=args.max_pool_size,
        elo_k_factor=args.elo_k_factor,
        snapshot_freq=args.snapshot_freq,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        # IL reward parameters
        use_il_reward=args.use_il_reward,
        il_model_path=args.il_model_path,
        il_reward_weight=args.il_reward_weight,
        il_reward_bonus=args.il_reward_bonus,
        il_reward_penalty=args.il_reward_penalty,
        il_reward_anneal=args.il_reward_anneal,
        il_reward_anneal_start=args.il_reward_anneal_start,
        il_reward_anneal_end=args.il_reward_anneal_end,
        il_reward_anneal_steps=args.il_reward_anneal_steps,
        il_compare_sap_targets=args.il_compare_sap_targets,
    )
    
    train_mappo(config)