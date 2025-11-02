import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Any
import torch

from luxai_s3.env import LuxAIS3Env #type: ignore
from luxai_s3.params import EnvParams #type: ignore
from luxai_s3.state import EnvState, EnvObs #type: ignore

from envs.featurize import LuxFeaturizer, create_ctde_observation
from envs.rewards import LuxRewardShaper, RelicMemory

from dataclasses import asdict


class LuxS3Wrapper:
    """
    Gymnasium-style wrapper around LuxAI_S3 environment.
    """
    
    def __init__(
        self,
        env_params: Optional[EnvParams] = None,
        reward_mode: str = "dense",
        auto_reset: bool = True,
        device: str = "cpu",
        
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_matches: int = 3,
        
        # IL reward shaping
        use_il_reward: bool = False,
        il_reward_coef: float = 0.1,
        il_unit_unet_path: str = "IL/imitation_learning/weights/unit_unet.pth",
        il_sap_unet_path: str = "IL/imitation_learning/weights/sap_unet.pth",
    ):
        if env_params is None:
            env_params = EnvParams()
        
        self.env_params = env_params
        self.device = device
        
        # Create base environment
        fixed_params = EnvParams(
            max_units=env_params.max_units,
            num_teams=2,
            map_width=env_params.map_width,
            map_height=env_params.map_height,
            max_energy_nodes=6,
            max_relic_nodes=6,
            relic_config_size=5,
            map_type=env_params.map_type,
        )
        self.env = LuxAIS3Env(auto_reset=auto_reset, fixed_env_params=fixed_params)
        
        # Create featurizer
        self.featurizer = LuxFeaturizer(
            map_width=env_params.map_width,
            map_height=env_params.map_height,
            max_units=env_params.max_units,
            max_relic_nodes=6,
            num_teams=2,
        )
        
        # Create reward shapers
        self.reward_shapers = {
            0: LuxRewardShaper(reward_mode=reward_mode),
            1: LuxRewardShaper(reward_mode=reward_mode),
        }
        
        # Create relic memories
        self.relic_memories = {
            0: RelicMemory(max_relic_nodes=6, relic_config_size=5),
            1: RelicMemory(max_relic_nodes=6, relic_config_size=5),
        }
        
        # Exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_matches = epsilon_decay_matches
        
        # IL reward shaping
        self.use_il_reward = use_il_reward
        self.il_reward_shaper = None
        if use_il_reward:
            try:
                from envs.il_rewards import ILRewardShaper
                self.il_reward_shaper = ILRewardShaper(
                    unit_unet_path=il_unit_unet_path,
                    sap_unet_path=il_sap_unet_path,
                    reward_coef=il_reward_coef,
                    device=device,
                )
                print(f"IL reward shaping enabled (coef={il_reward_coef})")
            except Exception as e:
                print(f"Warning: Could not initialize IL reward shaper: {e}")
                self.use_il_reward = False
        
        # State tracking
        self.state = None
        self.prev_obs = None
        self.current_match = 0
        self.episode_step = 0
        
    def reset(self, key: jax.random.PRNGKey) -> Tuple[Dict, EnvState]:
        # Reset environment
        obs_raw, state = self.env.reset(key, self.env_params)
        
        # Reset relic memories
        for memory in self.relic_memories.values():
            memory.reset()
        
        # Reset IL reward shaper
        if self.use_il_reward and self.il_reward_shaper is not None:
            self.il_reward_shaper.reset()
        
        # Reset state tracking
        self.state = state
        self.prev_obs = None
        self.current_match = 0
        self.episode_step = 0
        
        # Convert to CTDE observations
        obs_dict = self._process_observation(obs_raw, state)
        
        return obs_dict, state
    
    #### Per step
    def step(
        self,
        key: jax.random.PRNGKey,
        state: EnvState,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[Dict, EnvState, Dict, Dict, Dict, Dict]:
        
        if isinstance(actions["player_0"], torch.Tensor):
            actions["player_0"] = actions["player_0"].cpu().numpy()
        if isinstance(actions["player_1"], torch.Tensor):
            actions["player_1"] = actions["player_1"].cpu().numpy()
        
        # Step
        obs_raw, state, env_rewards, env_dones, env_truncated, info = self.env.step(
            key, state, actions, self.env_params
        )
        
        # Track match transitions
        prev_match = self.current_match
        self.current_match = self.episode_step // self.env_params.max_steps_in_match
        match_changed = self.current_match != prev_match
        
        # Process observations
        obs_dict = self._process_observation(obs_raw, state)
        
        # Compute shaped rewards
        rewards = {}
        for team_id in [0, 1]:
            player_key = f"player_{team_id}"
            
            if self.prev_obs is not None:
                prev_obs_team = self.prev_obs[player_key]
                curr_obs_team = obs_raw[player_key]
                
                # Compute reward
                rewards[player_key] = self.reward_shapers[team_id].compute_reward(
                    prev_obs_team,
                    curr_obs_team,
                    env_dones[player_key],
                    info,
                    team_id,
                )
                
                # Add IL reward if enabled
                if self.use_il_reward and self.il_reward_shaper is not None:
                    il_reward = self.il_reward_shaper.compute_reward(
                        rl_actions=actions[player_key],
                        obs_raw=obs_raw,  # Pass full obs with both teams
                        team_id=team_id,
                    )
                    rewards[player_key] += il_reward
                
                # Update relic memory
                prev_points = prev_obs_team["team_points"][team_id]
                curr_points = curr_obs_team["team_points"][team_id]
                points_gained = curr_points - prev_points
                
                # Get active unit positions
                unit_positions = np.array(curr_obs_team["units"]["position"][team_id])
                unit_mask = np.array(curr_obs_team["units_mask"][team_id])
                active_positions = unit_positions[unit_mask]
                
                self.relic_memories[team_id].update(
                    curr_obs_team,
                    points_gained,
                    active_positions
                )
            else:
                rewards[player_key] = 0.0
        
        # Store current obs for next step
        self.prev_obs = obs_raw
        self.episode_step += 1
        
        # Add epsilon to info
        epsilon = self.get_epsilon()
        info["epsilon"] = epsilon
        info["current_match"] = self.current_match
        info["match_changed"] = match_changed
        
        # Add memory info
        for team_id in [0, 1]:
            info[f"team_{team_id}_known_relics"] = len(self.relic_memories[team_id].discovered_relics)
        
        return obs_dict, state, rewards, env_dones, env_truncated, info
    
    def _process_observation(
        self,
        obs_raw: Dict,
        state: EnvState
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        
        # Extract observations for each team
        obs_team_0 = asdict(obs_raw["player_0"])
        obs_team_1 = asdict(obs_raw["player_1"])
        
        # Create base features for each team
        team_0_features = self.featurizer.featurize(obs_team_0, team_id=0, device=self.device)
        team_1_features = self.featurizer.featurize(obs_team_1, team_id=1, device=self.device)
        
        # Add memory features to each team
        for team_id, team_features in [(0, team_0_features), (1, team_1_features)]:
            memory_features = self.relic_memories[team_id].get_memory_features(
                self.env_params.map_width,
                self.env_params.map_height
            )
            memory_tensor = torch.from_numpy(memory_features).float().to(self.device)
            
            team_features["spatial_features"] = torch.cat([
                team_features["spatial_features"],
                memory_tensor
            ], dim=0)
        
        # Global observation from team features
        global_spatial = torch.cat([
            team_0_features["spatial_features"],
            team_1_features["spatial_features"] 
        ], dim=0) 
        
        # Combine global scalars
        global_scalars = torch.cat([
            team_0_features["global_features"],
            team_1_features["global_features"]
        ], dim=0)
        
        # Build CTDE observation
        ctde_obs = {
            "team_0": team_0_features,
            "team_1": team_1_features,
            "global": {
                "spatial_features": global_spatial,
                "global_features": global_scalars,
                "team_0_units": team_0_features["unit_features"],
                "team_1_units": team_1_features["unit_features"],
                "team_0_unit_mask": team_0_features["unit_mask"],
                "team_1_unit_mask": team_1_features["unit_mask"],
            }
        }
        
        return ctde_obs
    
    def get_epsilon(self) -> float:
        """
        Current epsilon for exploration.
        Decays linearly over first N matches
        """
        if self.current_match >= self.epsilon_decay_matches:
            return self.epsilon_end
        
        # Linear decay
        progress = self.current_match / self.epsilon_decay_matches
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        
        return epsilon
    
    def render(self):
        
        if self.state is not None:
            self.env.render(self.state, self.env_params)
    
    @property
    def observation_space(self):
        
        return {
            "spatial_channels": 19,  # 19 base + 2 memory
            "unit_features": 10,
            "global_features": 12,
            "relic_features": 3,
        }
    
    @property
    def action_space(self):
        
        return {
            "action_type": 6,
            "sap_targets": (2 * self.env_params.unit_sap_range + 1) ** 2,
        }


def create_env(
    reward_mode: str = "dense",
    device: str = "cpu",
    **kwargs
) -> LuxS3Wrapper:
    
    env_params = EnvParams()
    
    for key, value in kwargs.items():
        if hasattr(env_params, key):
            setattr(env_params, key, value)
    
    env = LuxS3Wrapper(
        env_params=env_params,
        reward_mode=reward_mode,
        device=device,
        **{k: v for k, v in kwargs.items() if k not in dir(env_params)}
    )
    
    return env