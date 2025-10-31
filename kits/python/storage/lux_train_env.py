"""
Simple Lux AI S3 Environment Wrapper
"""
import jax
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams

class LuxTrainEnv:
    """Simple wrapper around Lux AI S3 environment"""
    def __init__(self):
        self.env_params = EnvParams()
        self.env = LuxAIS3Env(auto_reset=True)
        self.max_units = 16
    
    def reset(self, key):
        """Reset environment"""
        obs, state = self.env.reset(key, self.env_params)
        return obs, state
    
    def step(self, key, state, actions):
        """
        Step environment
        
        Args:
            key: JAX random key
            state: Current environment state
            actions: Dict with 'player_0' and 'player_1' action arrays
        
        Returns:
            obs, state, rewards, terminated, truncated, info
        """
        return self.env.step(key, state, actions, self.env_params)