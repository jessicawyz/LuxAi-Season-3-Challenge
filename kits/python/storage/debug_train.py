import jax
import jax.numpy as jnp
import torch
import numpy as np
import random
from config import Config
from feature_engineer import SimpleFeatureEngineer
from model import SimpleLuxCNN
from ppo import SimplePPO
from agent import SimpleAgent
from luxai_s3.env import LuxAIS3Env
from reward import create_reward_calculator
from action_masking import create_action_masker, convert_to_env_actions

# Test episode reset sequence
env = LuxAIS3Env()
def extract_observation_data(obs_dict):
    """
    Simple observation extraction without corruption
    """
    processed_obs = {}
    
    for player_key, player_obs in obs_dict.items():
        if player_key.startswith('player'):
            # Convert to simple dict structure without deep processing
            player_data = {}
            
            # Extract units data carefully
            if hasattr(player_obs, 'units'):
                player_data['units'] = {
                    'energy': np.array(player_obs.units.energy),
                    'position': np.array(player_obs.units.position)
                }
            
            # Extract mask
            if hasattr(player_obs, 'units_mask'):
                player_data['units_mask'] = np.array(player_obs.units_mask)
            
            # Extract other important fields
            for attr in ['team_points', 'steps', 'match_steps', 'sensor_mask']:
                if hasattr(player_obs, attr):
                    player_data[attr] = np.array(getattr(player_obs, attr))
            
            # Map features
            if hasattr(player_obs, 'map_features'):
                player_data['map_features'] = {
                    'energy': np.array(player_obs.map_features.energy),
                    'tile_type': np.array(player_obs.map_features.tile_type)
                }
            
            # Relic nodes
            if hasattr(player_obs, 'relic_nodes'):
                player_data['relic_nodes'] = np.array(player_obs.relic_nodes)
                player_data['relic_nodes_mask'] = np.array(player_obs.relic_nodes_mask)
            
            processed_obs[player_key] = player_data
    
    return processed_obs

print("=== EPISODE 1 ===")
key = jax.random.PRNGKey(1000)
# Test the new extraction
obs, state = env.reset(key)
numpy_obs = extract_observation_data(obs)
print("After extraction - units energy:", numpy_obs['player_0']['units']['energy'][0][:5])
print(f"Reset state - match_steps: {state.match_steps}, steps: {state.steps}")

# Play a full episode
for step in range(505):  # Same as your 504 steps + reset
    key, subkey = jax.random.split(key)
    actions = {
        'player_0': jnp.zeros((16, 3), dtype=jnp.int32),
        'player_1': jnp.zeros((16, 3), dtype=jnp.int32)
    }
    obs, state, rewards, terminated, truncated, info = env.step(subkey, state, actions)
    
    # Print step 504 specifically
    if step == 503:  # Since we start from step 0, step 503 is the 504th step
        print(f"\n--- STEP 504 ---")
        print(f"Step: {step + 1}, Match steps: {state.match_steps}")
        print("Units energy at step 504:", obs['player_0'].units.energy[0][:5])
        print("Units mask at step 504:", obs['player_0'].units_mask[0][:5])

print(f"End state - match_steps: {state.match_steps}, steps: {state.steps}")
print("Final units energy:", obs['player_0'].units.energy[0][:5])

print("\n=== EPISODE 2 - AFTER RESET ===")
key, subkey = jax.random.split(key)
obs2, state2 = env.reset(subkey)
print(f"Reset state - match_steps: {state2.match_steps}, steps: {state2.steps}")
print("Reset units energy:", obs2['player_0'].units.energy[0][:5])
print("Reset units mask:", obs2['player_0'].units_mask[0][:5])

# Also step Episode 2 to step 504 and check
print("\n=== EPISODE 2 - STEPPING TO 504 ===")
for step in range(505):
    key, subkey = jax.random.split(key)
    actions = {
        'player_0': jnp.zeros((16, 3), dtype=jnp.int32),
        'player_1': jnp.zeros((16, 3), dtype=jnp.int32)
    }
    obs2, state2, rewards, terminated, truncated, info = env.step(subkey, state2, actions)
    
    # Print step 504 specifically
    if step == 503:  # Since we start from step 0, step 503 is the 504th step
        print(f"\n--- EPISODE 2 STEP 504 ---")
        print(f"Step: {step + 1}, Match steps: {state2.match_steps}")
        print("Units energy at step 504:", obs2['player_0'].units.energy[0][:5])
        print("Units mask at step 504:", obs2['player_0'].units_mask[0][:5])
        break