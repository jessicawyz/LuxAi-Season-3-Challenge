import jax
import jax.numpy as jnp
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from functools import partial

from luxai_s3.env import LuxAIS3Env #type: ignore
from luxai_s3.params import EnvParams #type: ignore
from luxai_s3.state import EnvState #type: ignore

from envs.lux_s3_env import LuxS3Wrapper
from dataclasses import asdict

class VecLuxS3Env:
    
    def __init__(
        self,
        num_envs: int,
        env_params: Optional[EnvParams] = None,
        reward_mode: str = "dense",
        device: str = "cpu",
        **kwargs
    ):
        """
        Init parallel vec env
        
        Args:
            num_envs
            env_params
            reward_mode
            device
        """
        self.num_envs = num_envs
        self.device = device
        
        if env_params is None:
            env_params = EnvParams()
        self.env_params = env_params
        
        self.wrapper = LuxS3Wrapper(
            env_params=env_params,
            reward_mode=reward_mode,
            auto_reset=True,
            device=device,
            **kwargs
        )
        
        # Create JAX env
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
        self.base_env = LuxAIS3Env(auto_reset=True, fixed_env_params=fixed_params)
        
        self.vec_reset = jax.vmap(
            lambda key: self.base_env.reset(key, self.env_params),
            in_axes=0
        )
        
        self.vec_step = jax.vmap(
            lambda key, state, action: self.base_env.step(key, state, action, self.env_params),
            in_axes=(0, 0, 0)
        )
        
        self.vec_reset = jax.jit(self.vec_reset)
        self.vec_step = jax.jit(self.vec_step)
        
        # State tracking
        self.states = None
        self.prev_obs_raw = None
        self.episode_steps = np.zeros(num_envs, dtype=np.int32)
        self.current_matches = np.zeros(num_envs, dtype=np.int32)
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        
        # Create separate keys for each environment
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, self.num_envs)
        
        # Reset all environments in parallel
        obs_raw_batch, states_batch = self.vec_reset(keys)
        
        # Store states
        self.states = states_batch
        self.prev_obs_raw = obs_raw_batch
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.current_matches = np.zeros(self.num_envs, dtype=np.int32)
        
        # Process observations for each env
        obs_list = []
        for i in range(self.num_envs):
            obs_single = jax.tree.map(lambda x: x[i], obs_raw_batch)
            # Convert to CPU numpy
            obs_single = jax.tree.map(lambda x: np.array(x), obs_single)
            
            # Get CTDE observation
            obs_dict = self.wrapper._process_observation(
                obs_single,
                jax.tree.map(lambda x: x[i], states_batch)
            )
            obs_list.append(obs_dict)
        
        # Stack observations
        stacked_obs = self._stack_observations(obs_list)
        
        return stacked_obs
    
    def step(
        self,
        actions: Dict[str, torch.Tensor],
        seed: Optional[int] = None
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        
        # Create keys
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, self.num_envs)
        
        # Convert actions to numpy/JAX format
        actions_jax = {
            "player_0": jnp.array(actions["player_0"].cpu().numpy()),
            "player_1": jnp.array(actions["player_1"].cpu().numpy()),
        }
        
        # Step all environments in parallel
        obs_raw_batch, states_batch, rewards_batch, dones_batch, truncated_batch, info_batch = self.vec_step(
            keys, self.states, actions_jax
        )
        
        # Store new states
        self.states = states_batch
        
        # Process observations and compute rewards for each env
        obs_list = []
        rewards_list = []
        infos = []
        
        for i in range(self.num_envs):
            obs_single = jax.tree.map(lambda x: x[i], obs_raw_batch)
            prev_obs_single = jax.tree.map(lambda x: x[i], self.prev_obs_raw)
            
            # Convert to numpy
            obs_single = jax.tree.map(lambda x: np.array(x), obs_single)
            prev_obs_single = jax.tree.map(lambda x: np.array(x), prev_obs_single)
            
            # Track match transitions
            prev_match = self.current_matches[i]
            self.current_matches[i] = self.episode_steps[i] // self.env_params.max_steps_in_match
            match_changed = self.current_matches[i] != prev_match
            
            # Get CTDE observation
            obs_dict = self.wrapper._process_observation(
                obs_single,
                jax.tree.map(lambda x: x[i], states_batch)
            )
            obs_list.append(obs_dict)
            
            # Compute shaped rewards for each team
            env_rewards = {}
            for team_id in [0, 1]:
                player_key = f"player_{team_id}"
                
                prev_obs_team = asdict(prev_obs_single[player_key])
                curr_obs_team = asdict(obs_single[player_key])
                
                done = bool(dones_batch[player_key][i])
                
                # Compute reward
                reward = self.wrapper.reward_shapers[team_id].compute_reward(
                    prev_obs_team,
                    curr_obs_team,
                    done,
                    {},
                    team_id,
                )
                env_rewards[player_key] = reward
                
                # Update relic memory
                prev_points = prev_obs_team["team_points"][team_id]
                curr_points = curr_obs_team["team_points"][team_id]
                points_gained = curr_points - prev_points
                
                # Get active unit positions
                unit_positions = np.array(curr_obs_team["units"]["position"][team_id])
                unit_mask = np.array(curr_obs_team["units_mask"][team_id])
                active_positions = unit_positions[unit_mask]
                
                self.wrapper.relic_memories[team_id].update(
                    curr_obs_team,
                    points_gained,
                    active_positions
                )
            
            rewards_list.append([env_rewards["player_0"][0], env_rewards["player_1"][0]])
            
            # Build info dict
            epsilon = self.wrapper.get_epsilon()
            info = {
                "epsilon": epsilon,
                "current_match": int(self.current_matches[i]),
                "match_changed": match_changed,
                "episode_step": int(self.episode_steps[i]),
            }
            
            # Add memory info
            for team_id in [0, 1]:
                info[f"team_{team_id}_known_relics"] = len(
                    self.wrapper.relic_memories[team_id].discovered_relics
                )
            
            infos.append(info)
            
            self.episode_steps[i] += 1
        
        # Update prev obs
        self.prev_obs_raw = obs_raw_batch
        
        # Stack observations and rewards
        stacked_obs = self._stack_observations(obs_list)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        
        # Convert to tensors
        dones_tensor = torch.tensor(
            [bool(dones_batch["player_0"][i]) for i in range(self.num_envs)],
            dtype=torch.bool,
            device=self.device
        )
        truncated_tensor = torch.tensor(
            [bool(truncated_batch["player_0"][i]) for i in range(self.num_envs)],
            dtype=torch.bool,
            device=self.device
        )
        
        # Reset memories
        for i in range(self.num_envs):
            if dones_tensor[i] or truncated_tensor[i]:
                self.wrapper.relic_memories[0].reset()
                self.wrapper.relic_memories[1].reset()
                self.episode_steps[i] = 0
                self.current_matches[i] = 0
        
        return stacked_obs, rewards_tensor, dones_tensor, truncated_tensor, infos
    
    def _stack_observations(self, obs_list: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        
        # Stack team_0 observations
        team_0_obs = {
            "spatial_features": torch.stack([obs["team_0"]["spatial_features"] for obs in obs_list]),
            "unit_features": torch.stack([obs["team_0"]["unit_features"] for obs in obs_list]),
            "unit_mask": torch.stack([obs["team_0"]["unit_mask"] for obs in obs_list]),
            "global_features": torch.stack([obs["team_0"]["global_features"] for obs in obs_list]),
            "relic_features": torch.stack([obs["team_0"]["relic_features"] for obs in obs_list]),
        }
        
        # Stack team_1 observations
        team_1_obs = {
            "spatial_features": torch.stack([obs["team_1"]["spatial_features"] for obs in obs_list]),
            "unit_features": torch.stack([obs["team_1"]["unit_features"] for obs in obs_list]),
            "unit_mask": torch.stack([obs["team_1"]["unit_mask"] for obs in obs_list]),
            "global_features": torch.stack([obs["team_1"]["global_features"] for obs in obs_list]),
            "relic_features": torch.stack([obs["team_1"]["relic_features"] for obs in obs_list]),
        }
        
        # Stack global observations
        global_obs = {
            "spatial_features": torch.stack([obs["global"]["spatial_features"] for obs in obs_list]),
            "global_features": torch.stack([obs["global"]["global_features"] for obs in obs_list]),
            "team_0_units": torch.stack([obs["global"]["team_0_units"] for obs in obs_list]),
            "team_1_units": torch.stack([obs["global"]["team_1_units"] for obs in obs_list]),
            "team_0_unit_mask": torch.stack([obs["global"]["team_0_unit_mask"] for obs in obs_list]),
            "team_1_unit_mask": torch.stack([obs["global"]["team_1_unit_mask"] for obs in obs_list]),
        }
        
        return {
            "team_0": team_0_obs,
            "team_1": team_1_obs,
            "global": global_obs,
        }
    
    def close(self):
        
        pass


def make_vec_env(
    num_envs: int,
    reward_mode: str = "dense",
    device: str = "cpu",
    **kwargs
) -> VecLuxS3Env:
    
    return VecLuxS3Env(
        num_envs=num_envs,
        reward_mode=reward_mode,
        device=device,
        **kwargs
    )
