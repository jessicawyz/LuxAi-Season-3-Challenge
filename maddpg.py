import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import deque
import random
import jax

from envs.vec_env import make_vec_env
from envs.action_head import HierarchicalActionHead
from selfplay import create_selfplay_pool, PFSPConfig
from eval import evaluate_and_update_pool, EvalConfig
from baseline_agent import Agent as BaselineAgent
from dataclasses import asdict

import matplotlib.pyplot as plt
plt.close('all')

# IL reward imports
try:
    from envs.il_rewards import create_il_reward_shaper
    IL_AVAILABLE = True
except ImportError:
    IL_AVAILABLE = False
    print("[MADDPG] IL reward module not available")

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
class MADDPGConfig:
    
    num_envs: int = 8
    reward_mode: str = "dense"
    
    
    total_timesteps: int = 10_000_000
    learning_rate_actor: float = 1e-4
    learning_rate_critic: float = 3e-4
    buffer_size: int = 1000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  
    learning_starts: int = 10_000
    train_frequency: int = 4
    gradient_steps: int = 1
    target_update_frequency: int = 1
    
    
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500_000
    
    
    spatial_channels: int = 23 
    unit_feature_dim: int = 10
    global_feature_dim: int = 12
    hidden_dim: int = 64
    convlstm_hidden_dim: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 2
    
    
    max_units: int = 16
    map_width: int = 24
    map_height: int = 24
    unit_sap_range: int = 4
    
    
    snapshot_freq: int = 100_000
    eval_freq: int = 50_000
    checkpoint_dir: str = "checkpoint"
    
    # Self-play parameters
    use_selfplay: bool = True
    selfplay_ratio: float = 0.5 
    num_eval_opponents: int = 5
    use_baseline_opponent: bool = False  # Use baseline agent instead of selfplay 
    max_pool_size: int = 20
    elo_k_factor: float = 32.0
    
    # IL reward parameters
    use_il_reward: bool = False
    il_model_path: str = "IL/imitation_learning/weights/model.pth"
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


class MADDPGActor(nn.Module):
    
    def __init__(self, config: MADDPGConfig):
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
        epsilon=0.0
    ):
        
        batch_size = spatial_features.shape[0]
        
        
        spatial_encoded, new_spatial_hidden_state = self.spatial_encoder(
            spatial_features, spatial_hidden_state
        )
        unit_encoded, unit_aggregated = self.unit_encoder(unit_features, unit_mask)
        global_encoded = F.relu(self.global_proj(global_features))
        
        
        fused = self.fusion(torch.cat([
            spatial_encoded, unit_aggregated, global_encoded
        ], dim=-1))
        fused_per_unit = fused.unsqueeze(1).expand(-1, self.config.max_units, -1)
        unit_features_combined = fused_per_unit + unit_encoded
        
        
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
        
        
        if epsilon > 0 and self.training:
            
            explore_mask = torch.rand(batch_size, self.config.max_units, device=action_type_logits.device) < epsilon
            
            
            random_action_types = torch.randint(
                0, 6,
                (batch_size, self.config.max_units),
                device=action_type_logits.device
            )
            
            
            action_type_mask = mask_info["action_type_mask"]
            valid_random_actions = torch.where(
                action_type_mask.gather(-1, random_action_types.unsqueeze(-1)).squeeze(-1),
                random_action_types,
                torch.argmax(action_type_logits, dim=-1)
            )
            
            
            deterministic_actions = torch.argmax(action_type_logits, dim=-1)
            action_types = torch.where(
                explore_mask & unit_mask,
                valid_random_actions,
                deterministic_actions
            )
        else:
            
            action_types = torch.argmax(action_type_logits, dim=-1)
        
        
        sap_targets = torch.argmax(sap_target_logits, dim=-1)
        
        
        actions, action_info = self.action_head.sample_actions(
            action_type_logits,
            sap_target_logits,
            mask_info["action_type_mask"],
            mask_info["sap_target_mask"],
            unit_mask,
            deterministic=True
        )
        
        return actions, new_spatial_hidden_state, action_info


class MADDPGCritic(nn.Module):
    
    def __init__(self, config: MADDPGConfig):
        super().__init__()
        
        self.config = config
        
        
        # Global spatial features are: team_0 spatial (21) + team_1 spatial (21) = 42 channels
        # Each team spatial already includes memory (19 base + 2 memory)
        self.spatial_encoder = SpatialEncoder(
            input_channels=config.spatial_channels * 2,  # 21 * 2 = 42 channels
            hidden_dim=config.convlstm_hidden_dim,
            output_dim=config.hidden_dim
        )
        
        
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
        
        
        self.global_proj = nn.Linear(config.global_feature_dim * 2, config.hidden_dim)
        
        
        self.action_encoder_0 = nn.Sequential(
            nn.Linear(config.max_units * 3, config.hidden_dim),
            nn.ReLU()
        )
        
        self.action_encoder_1 = nn.Sequential(
            nn.Linear(config.max_units * 3, config.hidden_dim),
            nn.ReLU()
        )
        
        
        self.q_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 6, config.hidden_dim),
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
        actions_0,
        actions_1,
        spatial_hidden_state=None
    ):
        
        
        spatial_encoded, new_spatial_hidden_state = self.spatial_encoder(
            global_spatial_features, spatial_hidden_state
        )
        
        
        _, team_0_aggregated = self.unit_encoder_0(team_0_unit_features, team_0_unit_mask)
        _, team_1_aggregated = self.unit_encoder_1(team_1_unit_features, team_1_unit_mask)
        
        
        global_encoded = F.relu(self.global_proj(global_features))
        
        
        actions_0_flat = actions_0.flatten(1).float()  
        actions_1_flat = actions_1.flatten(1).float()
        
        actions_0_encoded = self.action_encoder_0(actions_0_flat)
        actions_1_encoded = self.action_encoder_1(actions_1_flat)
        
        
        combined = torch.cat([
            spatial_encoded,
            team_0_aggregated,
            team_1_aggregated,
            global_encoded,
            actions_0_encoded,
            actions_1_encoded
        ], dim=-1)
        
        
        q_value = self.q_head(combined).squeeze(-1)
        
        return q_value, new_spatial_hidden_state


class ReplayBuffer:
    
    
    def __init__(self, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, obs, actions, rewards, next_obs, dones):

        self.buffer.append((obs, actions, rewards, next_obs, dones))
    
    def sample(self, batch_size: int):

        samples = random.sample(self.buffer, batch_size)
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = zip(*samples)       
        obs_batch = self._stack_obs(obs_list)
        next_obs_batch = self._stack_obs(next_obs_list)
        
        actions_batch = {
            "player_0": torch.stack([a["player_0"] for a in actions_list]),
            "player_1": torch.stack([a["player_1"] for a in actions_list])
        }
        
        rewards_batch = torch.stack(rewards_list)       
        dones_batch = torch.stack(dones_list).float()
        
        return obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch
    
    def _stack_obs(self, obs_list):
        
        obs_batch = {}
        
        for agent_key in ["team_0", "team_1", "global"]:
            obs_batch[agent_key] = {}
            for feat_key in obs_list[0][agent_key].keys():
                obs_batch[agent_key][feat_key] = torch.stack([
                    obs[agent_key][feat_key] for obs in obs_list
                ])
        
        return obs_batch
    
    def __len__(self):
        return len(self.buffer)


def soft_update(target_net, source_net, tau):
    
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class BaselineAgentWrapper:
    
    def __init__(self, env_cfg: dict, device: str = "cpu"):
        self.agent = BaselineAgent("player_1", env_cfg)
        self.device = device
        self.env_cfg = env_cfg
    
    def get_actions(self, obs_raw_batch, step: int) -> torch.Tensor:
        
        batch_size = len(obs_raw_batch["player_1"])
        max_units = self.env_cfg["max_units"]
        
        # Get actions for each environment
        actions_batch = []
        for i in range(batch_size):
            # Extract single environment observation (already a dict)
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

def exponential_smooth(data, alpha=0.1):
    """Apply exponential smoothing to data"""
    if len(data) == 0:
        return []
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
    return smoothed

def train_maddpg(config: MADDPGConfig):
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize opponent pool for self-play
    if config.use_selfplay:
        pfsp_config = PFSPConfig(
            pfsp_ratio=config.selfplay_ratio,
        )
        opponent_pool = create_selfplay_pool(
            checkpoint_dir=config.checkpoint_dir,
            model_type="maddpg",
            max_pool_size=config.max_pool_size,
            elo_k_factor=config.elo_k_factor,
            pfsp_config=pfsp_config,
        )
        print(f"Initialized opponent pool with {len(opponent_pool.opponents)} opponents")
    else:
        opponent_pool = None
    
    
    env = make_vec_env(
        num_envs=config.num_envs,
        reward_mode=config.reward_mode,
        device=config.device
    )
    
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
    
    
    actor_0 = MADDPGActor(config).to(config.device)
    actor_1 = MADDPGActor(config).to(config.device)
    critic_0 = MADDPGCritic(config).to(config.device)
    critic_1 = MADDPGCritic(config).to(config.device)
    
    
    target_actor_0 = MADDPGActor(config).to(config.device)
    target_actor_1 = MADDPGActor(config).to(config.device)
    target_critic_0 = MADDPGCritic(config).to(config.device)
    target_critic_1 = MADDPGCritic(config).to(config.device)
    
    
    target_actor_0.load_state_dict(actor_0.state_dict())
    target_actor_1.load_state_dict(actor_1.state_dict())
    target_critic_0.load_state_dict(critic_0.state_dict())
    target_critic_1.load_state_dict(critic_1.state_dict())
    
    
    actor_0_optimizer = torch.optim.Adam(actor_0.parameters(), lr=config.learning_rate_actor)
    actor_1_optimizer = torch.optim.Adam(actor_1.parameters(), lr=config.learning_rate_actor)
    critic_0_optimizer = torch.optim.Adam(critic_0.parameters(), lr=config.learning_rate_critic)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate_critic)
    
    
    replay_buffer = ReplayBuffer(config.buffer_size, config.device)
    
    
    # State tracking
    global_step = 0
    episode_rewards = np.zeros((config.num_envs, 2))

    # Tracking
    reward_history = []
    actor_loss_history = []
    critic_loss_history = []
    best_elo = -float('inf')
    
    # Opponent actors (for self-play mode)
    best_elo = -np.inf
    current_opponent = None
    opponent_actor_0 = None
    opponent_actor_1 = None
    
    
    obs = env.reset(seed=0)
    
    print(f"Starting MADDPG training...")
    print(f"Device: {config.device}")
    print(f"Num envs: {config.num_envs}")
    
    start_time = time.time()

    while global_step < config.total_timesteps:
        
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
                # Load opponent actors
                opponent_checkpoint = torch.load(current_opponent.path, map_location=config.device)
                opponent_actor_0 = MADDPGActor(config).to(config.device)
                opponent_actor_1 = MADDPGActor(config).to(config.device)
                opponent_actor_0.load_state_dict(opponent_checkpoint["actor_0"])
                opponent_actor_1.load_state_dict(opponent_checkpoint["actor_1"])
                opponent_actor_0.eval()
                opponent_actor_1.eval()
                print(f"Sampled opponent: {os.path.basename(current_opponent.path)} (ELO: {current_opponent.elo_rating:.1f})")
            except Exception as e:
                print(f"Warning: Could not load opponent: {e}")
                use_opponent = False
                current_opponent = None
        
        
        with torch.no_grad():
            # Player 0 always uses current policy
            actions_0, _, _ = actor_0(
                spatial_features=obs["team_0"]["spatial_features"],
                unit_features=obs["team_0"]["unit_features"],
                unit_mask=obs["team_0"]["unit_mask"],
                global_features=obs["team_0"]["global_features"],
                unit_energies=obs["team_0"]["unit_features"][:, :, 2] * 400,
                unit_positions=(obs["team_0"]["unit_features"][:, :, :2] *
                              torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                tile_types=obs["team_0"]["spatial_features"][:, 1] * 2,
                epsilon=epsilon
            )
            
            # Player 1: use baseline agent, opponent, or current policy
            if config.use_baseline_opponent and baseline_wrapper is not None:
                # Get raw observations from environment
                obs_raw = []
                for i in range(config.num_envs):
                    obs_single = jax.tree.map(lambda x: x[i], env.prev_obs_raw)
                    obs_single_np = jax.tree.map(lambda x: np.array(x), obs_single)
                    obs_raw.append(obs_single_np)
                
                # Convert to dict format for baseline agent
                obs_raw_batch = {"player_1": [asdict(obs["player_1"]) for obs in obs_raw]}
                actions_1 = baseline_wrapper.get_actions(obs_raw_batch, step=global_step)
            elif use_opponent and opponent_actor_1 is not None:
                actions_1, _, _ = opponent_actor_1(
                    spatial_features=obs["team_1"]["spatial_features"],
                    unit_features=obs["team_1"]["unit_features"],
                    unit_mask=obs["team_1"]["unit_mask"],
                    global_features=obs["team_1"]["global_features"],
                    unit_energies=obs["team_1"]["unit_features"][:, :, 2] * 400,
                    unit_positions=(obs["team_1"]["unit_features"][:, :, :2] *
                                  torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                    tile_types=obs["team_1"]["spatial_features"][:, 1] * 2,
                    epsilon=0.0 
                )
            else:
                actions_1, _, _ = actor_1(
                    spatial_features=obs["team_1"]["spatial_features"],
                    unit_features=obs["team_1"]["unit_features"],
                    unit_mask=obs["team_1"]["unit_mask"],
                    global_features=obs["team_1"]["global_features"],
                    unit_energies=obs["team_1"]["unit_features"][:, :, 2] * 400,
                    unit_positions=(obs["team_1"]["unit_features"][:, :, :2] *
                                  torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                    tile_types=obs["team_1"]["spatial_features"][:, 1] * 2,
                    epsilon=epsilon
                )
        
        
        actions_dict = {
            "player_0": actions_0,
            "player_1": actions_1
        }
        
        next_obs, rewards, dones, truncated, infos = env.step(actions_dict, seed=global_step)
        
        # Compute IL rewards if enabled
        il_rewards_0 = torch.zeros(config.num_envs, device=config.device)
        il_rewards_1 = torch.zeros(config.num_envs, device=config.device)
        il_info = {}
        
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
                        global_step=global_step,
                    )
                    il_rewards_0 = torch.from_numpy(il_r0).float().to(config.device)
                    il_info.update({f"team_0_{k}": v for k, v in il_i0.items()})
                    
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
                        global_step=global_step,
                    )
                    il_rewards_1 = torch.from_numpy(il_r1).float().to(config.device)
                    il_info.update({f"team_1_{k}": v for k, v in il_i1.items()})
            
            except Exception as e:
                print(f"Warning: IL reward computation failed: {e}")
                import traceback
                traceback.print_exc()
        
        global_step += config.num_envs
        
        
        for i in range(config.num_envs):
            obs_single = {k: {kk: v[i] for kk, v in obs[k].items()} for k in obs.keys()}
            next_obs_single = {k: {kk: v[i] for kk, v in next_obs[k].items()} for k in next_obs.keys()}
            actions_single = {k: v[i] for k, v in actions_dict.items()}
            rewards_single = rewards[i]
            
            # Add IL rewards to this environment's rewards
            if config.use_il_reward and il_reward_shaper is not None:
                # rewards_single is a tensor [player_0_reward, player_1_reward]
                il_reward_add = torch.tensor(
                    [il_rewards_0[i].item(), il_rewards_1[i].item()],
                    dtype=rewards_single.dtype,
                    device=rewards_single.device
                )
                rewards_single = rewards_single + il_reward_add
            
            done_single = dones[i]
            
            replay_buffer.add(obs_single, actions_single, rewards_single, next_obs_single, done_single)
            
            episode_rewards[i, 0] += rewards_single[0].item()  # Team 0
            episode_rewards[i, 1] += rewards_single[1].item()  # Team 1

            if done_single:
                episode_rewards[i, 0] = 0.0
                episode_rewards[i, 1] = 0.0
                
                # Reset IL state for this environment
                if config.use_il_reward and il_reward_shaper is not None:
                    il_reward_shaper.reset(env_idx=i)
        
        obs = next_obs
        
        
        if global_step > config.learning_starts and global_step % config.train_frequency == 0:
            for _ in range(config.gradient_steps):
                
                obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = replay_buffer.sample(
                    config.batch_size
                )
                
                
                for team_id, actor, target_actor, critic, target_critic, actor_opt, critic_opt in [
                    (0, actor_0, target_actor_0, critic_0, target_critic_0, actor_0_optimizer, critic_0_optimizer),
                    (1, actor_1, target_actor_1, critic_1, target_critic_1, actor_1_optimizer, critic_1_optimizer)
                ]:
                    
                    with torch.no_grad():
                        
                        next_actions_0, _, _ = target_actor_0(
                            spatial_features=next_obs_batch["team_0"]["spatial_features"],
                            unit_features=next_obs_batch["team_0"]["unit_features"],
                            unit_mask=next_obs_batch["team_0"]["unit_mask"],
                            global_features=next_obs_batch["team_0"]["global_features"],
                            unit_energies=next_obs_batch["team_0"]["unit_features"][:, :, 2] * 400,
                            unit_positions=(next_obs_batch["team_0"]["unit_features"][:, :, :2] *
                                          torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                            tile_types=next_obs_batch["team_0"]["spatial_features"][:, 1] * 2,
                            epsilon=0.0
                        )
                        
                        next_actions_1, _, _ = target_actor_1(
                            spatial_features=next_obs_batch["team_1"]["spatial_features"],
                            unit_features=next_obs_batch["team_1"]["unit_features"],
                            unit_mask=next_obs_batch["team_1"]["unit_mask"],
                            global_features=next_obs_batch["team_1"]["global_features"],
                            unit_energies=next_obs_batch["team_1"]["unit_features"][:, :, 2] * 400,
                            unit_positions=(next_obs_batch["team_1"]["unit_features"][:, :, :2] *
                                          torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                            tile_types=next_obs_batch["team_1"]["spatial_features"][:, 1] * 2,
                            epsilon=0.0
                        )
                        
                        
                        target_q, _ = target_critic(
                            global_spatial_features=next_obs_batch["global"]["spatial_features"],
                            team_0_unit_features=next_obs_batch["global"]["team_0_units"],
                            team_0_unit_mask=next_obs_batch["global"]["team_0_unit_mask"],
                            team_1_unit_features=next_obs_batch["global"]["team_1_units"],
                            team_1_unit_mask=next_obs_batch["global"]["team_1_unit_mask"],
                            global_features=next_obs_batch["global"]["global_features"],
                            actions_0=next_actions_0,
                            actions_1=next_actions_1,
                        )
                        
                        
                        target = rewards_batch[:, team_id] + config.gamma * (1 - dones_batch) * target_q
                    
                    
                    current_q, _ = critic(
                        global_spatial_features=obs_batch["global"]["spatial_features"],
                        team_0_unit_features=obs_batch["global"]["team_0_units"],
                        team_0_unit_mask=obs_batch["global"]["team_0_unit_mask"],
                        team_1_unit_features=obs_batch["global"]["team_1_units"],
                        team_1_unit_mask=obs_batch["global"]["team_1_unit_mask"],
                        global_features=obs_batch["global"]["global_features"],
                        actions_0=actions_batch["player_0"],
                        actions_1=actions_batch["player_1"],
                    )
                    
                    
                    critic_loss = F.mse_loss(current_q, target)
                    
                    
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_opt.step()
                    
                    if team_id == 0:
                        critic_loss_history.append(critic_loss.item()) # Tracking
                    
                    curr_actions_team, _, _ = actor(
                        spatial_features=obs_batch[f"team_{team_id}"]["spatial_features"],
                        unit_features=obs_batch[f"team_{team_id}"]["unit_features"],
                        unit_mask=obs_batch[f"team_{team_id}"]["unit_mask"],
                        global_features=obs_batch[f"team_{team_id}"]["global_features"],
                        unit_energies=obs_batch[f"team_{team_id}"]["unit_features"][:, :, 2] * 400,
                        unit_positions=(obs_batch[f"team_{team_id}"]["unit_features"][:, :, :2] *
                                      torch.tensor([config.map_width, config.map_height], device=config.device)).long(),
                        tile_types=obs_batch[f"team_{team_id}"]["spatial_features"][:, 1] * 2,
                        epsilon=0.0
                    )
                    
                    
                    if team_id == 0:
                        action_0_for_q = curr_actions_team
                        action_1_for_q = actions_batch["player_1"].detach()
                    else:
                        action_0_for_q = actions_batch["player_0"].detach()
                        action_1_for_q = curr_actions_team
                    
                    
                    q_for_actor, _ = critic(
                        global_spatial_features=obs_batch["global"]["spatial_features"],
                        team_0_unit_features=obs_batch["global"]["team_0_units"],
                        team_0_unit_mask=obs_batch["global"]["team_0_unit_mask"],
                        team_1_unit_features=obs_batch["global"]["team_1_units"],
                        team_1_unit_mask=obs_batch["global"]["team_1_unit_mask"],
                        global_features=obs_batch["global"]["global_features"],
                        actions_0=action_0_for_q,
                        actions_1=action_1_for_q,
                    )
                    
                    
                    actor_loss = -q_for_actor.mean()
                    
                    
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_opt.step()

                    if team_id == 0:
                        actor_loss_history.append(actor_loss.item()) # Tracking
                
                
                if global_step % config.target_update_frequency == 0:
                    soft_update(target_actor_0, actor_0, config.tau)
                    soft_update(target_actor_1, actor_1, config.tau)
                    soft_update(target_critic_0, critic_0, config.tau)
                    soft_update(target_critic_1, critic_1, config.tau)
        
        
        # Logging
        if global_step % config.log_freq == 0:
            mean_reward = np.mean(episode_rewards[:, 0])  # Team 0

            reward_history.append(mean_reward) # Tracking

            elapsed = (time.time() - start_time) / 60
            log_msg = f"[{elapsed:.2f} min] Step {global_step} | Epsilon {epsilon:.3f} | Mean Reward {mean_reward:.2f} | Buffer {len(replay_buffer)}"
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
            latest_path = os.path.join(config.checkpoint_dir, "maddpg_latest.pt")
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
                        "maddpg",
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
                        best_path = os.path.join(config.checkpoint_dir, "maddpg_best.pt")
                        torch.save(checkpoint, best_path)
                        print(f"Best checkpoint saved to maddpg_best.pt (ELO: {best_elo:.1f})")
                    
                    # Add current checkpoint to opponent pool
                    opponent_pool.add_checkpoint(
                        checkpoint_path=latest_path,
                        model_type="maddpg",
                        initial_rating=estimated_elo,
                        checkpoint_step=global_step,
                    )
                    
                    # Save opponent pool state
                    pool_state_path = os.path.join(config.checkpoint_dir, "maddpg_population.json")
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
            snapshot_path = os.path.join(config.checkpoint_dir, f"maddpg_{global_step}.pt")
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
    
    # Plot training metrics
    print("\nGenerating training plots...")
    timestamp = int(time.time())
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Reward history
    if reward_history:
        smoothed_rewards = exponential_smooth(reward_history, alpha=0.1)
        axes[0].plot(reward_history, color='tab:orange', linewidth=1, alpha=0.6, label='Raw')
        axes[0].plot(smoothed_rewards, color='tab:blue', linewidth=3, label='Smoothed')
        axes[0].set_title('Reward History')
        axes[0].set_xlabel('Log Step')
        axes[0].set_ylabel('Mean Reward')
        axes[0].grid(True)
        axes[0].legend(loc='upper left')
    
    # Actor loss history
    if actor_loss_history:
        smoothed_actor_loss = exponential_smooth(actor_loss_history, alpha=0.1)
        axes[1].plot(actor_loss_history, color='tab:orange', linewidth=1, alpha=0.6, label='Raw')
        axes[1].plot(smoothed_actor_loss, color='tab:blue', linewidth=3, label='Smoothed')
        axes[1].set_title('Actor Loss History')
        axes[1].set_xlabel('Gradient Step')
        axes[1].set_ylabel('Actor Loss')
        axes[1].grid(True)
        axes[1].legend(loc='upper left')
    
    # Critic loss history
    if critic_loss_history:
        smoothed_critic_loss = exponential_smooth(critic_loss_history, alpha=0.1)
        axes[2].plot(critic_loss_history, color='tab:orange', linewidth=1, alpha=0.6, label='Raw')
        axes[2].plot(smoothed_critic_loss, color='tab:blue', linewidth=3, label='Smoothed')
        axes[2].set_title('Critic Loss History')
        axes[2].set_xlabel('Gradient Step')
        axes[2].set_ylabel('Critic Loss')
        axes[2].grid(True)
        axes[2].legend(loc='upper left')
    
    plt.tight_layout()
    plot_filename = f'{timestamp}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Training plots saved to {plot_filename}")
    plt.close()
    
    # Save final checkpoint
    final_checkpoint = {
        "actor_0": actor_0.state_dict(),
        "actor_1": actor_1.state_dict(),
        "critic_0": critic_0.state_dict(),
        "critic_1": critic_1.state_dict(),
        "config": config.__dict__,
        "global_step": global_step,
    }
    torch.save(final_checkpoint, os.path.join(config.checkpoint_dir, "maddpg_final.pt"))
    
    # Also save as latest
    torch.save(final_checkpoint, os.path.join(config.checkpoint_dir, "maddpg_latest.pt"))
    
    # Save pool state
    if opponent_pool is not None:
        pool_state_path = os.path.join(config.checkpoint_dir, "maddpg_population.json")
        opponent_pool.save_pool_state(pool_state_path)
    
    print("Training complete! Final checkpoints saved.")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MADDPG agent for Lux AI S3")
    
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--reward-mode", type=str, default="dense", choices=["sparse", "dense"], help="Reward mode")
    parser.add_argument("--learning-rate-actor", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--learning-rate-critic", type=float, default=1e-4, help="Critic learning rate")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
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
    parser.add_argument("--il-model-path", type=str, default="IL/imitation_learning/weights/model.pth", help="Path to IL model")
    parser.add_argument("--il-reward-weight", type=float, default=0.1, help="IL reward weight (lambda)")
    parser.add_argument("--il-reward-bonus", type=float, default=0.1, help="Bonus per matching action")
    parser.add_argument("--il-reward-penalty", type=float, default=0.1, help="Penalty per mismatched action")
    parser.add_argument("--il-reward-anneal", action="store_true", default=True, help="Anneal IL weight")
    parser.add_argument("--il-reward-anneal-start", type=float, default=0.5, help="IL anneal start weight")
    parser.add_argument("--il-reward-anneal-end", type=float, default=0.05, help="IL anneal end weight")
    parser.add_argument("--il-reward-anneal-steps", type=int, default=500_000, help="IL anneal steps")
    parser.add_argument("--il-compare-sap-targets", action="store_true", help="Compare SAP targets")
    
    args = parser.parse_args()
    
    config = MADDPGConfig(
        num_envs=args.num_envs,
        reward_mode=args.reward_mode,
        total_timesteps=args.total_timesteps,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
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
    
    train_maddpg(config)