import jax
import jax.numpy as jnp
from luxai_s3.env import LuxAIS3Env

class LuxTrainEnv:
    def __init__(self):
        self.env = LuxAIS3Env()
        self.max_units = 16
        
    def reset(self, key):
        obs, state = self.env.reset(key)
        return obs, state
    
    def step(self, key, state, actions_dict):
        """Convert actions dict to proper format and step environment"""
        # Convert actions dict to the format Lux AI S3 expects
        lux_actions = self._convert_actions_to_lux_format(actions_dict)
        
        # Step environment
        obs_st, state_st, rewards, terminated_dict, truncated_dict, info = self.env.step(
            key, state, lux_actions
        )
        
        return obs_st, state_st, rewards, terminated_dict, truncated_dict, info
    
    def _convert_actions_to_lux_format(self, actions_dict):
        """Convert our actions to Lux AI S3 format: dict with player_0 and player_1 keys"""
        lux_actions = {}
        
        for player in ['player_0', 'player_1']:
            if player in actions_dict:
                # actions_dict[player] should be a (max_units, 3) array
                action_array = actions_dict[player]
                
                # Ensure it's the right shape and type
                if hasattr(action_array, 'shape') and action_array.shape == (self.max_units, 3):
                    lux_actions[player] = action_array
                else:
                    # Fallback: create zero actions
                    lux_actions[player] = jnp.zeros((self.max_units, 3), dtype=jnp.int32)
            else:
                # Default to no-op actions
                lux_actions[player] = jnp.zeros((self.max_units, 3), dtype=jnp.int32)
        
        return lux_actions