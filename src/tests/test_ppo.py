import json
import jax
import jax.numpy as jnp
import haiku as hk
import flax
import flax.serialization
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../kits/python')))
from env_wrapper import BatchedLuxEnv
from networks import ActorCritic
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
from luxai_s3.state import serialize_env_actions, serialize_env_states

# -------------------------
# Hyperparameters
# -------------------------
action_dim_per_unit = 6
player = "player_0"
seed = 0

# -------------------------
# Define actor-critic and load trained params
# -------------------------
def actor_critic_fn(obs):
    model = ActorCritic(action_dim_per_unit)
    return model(obs)
actor_critic = hk.transform(actor_critic_fn)

# Initialize dummy params (for structure)
dummy_obs = jnp.zeros((1, 100), dtype=jnp.float32)
key = jax.random.PRNGKey(seed)
params = actor_critic.init(key, dummy_obs)

# Load trained weights
with open("../../checkpoints/ppo_params.pkl", "rb") as f:
    params = flax.serialization.from_bytes(params, f.read())

# -------------------------
# Initialize environment
# -------------------------
env = LuxAIS3Env(auto_reset=False)
env_params = EnvParams(map_type=0, max_steps_in_match=50)
key, reset_key = jax.random.split(key)
obs, state = env.reset(reset_key, params=env_params)

# -------------------------
# Rollout using trained policy
# -------------------------
states = [state]
actions = []

for t in range(env_params.max_steps_in_match):
    logits, value = actor_critic.apply(params, key, jnp.array(obs["player_0"]).reshape(1, -1))
    action = jax.random.categorical(key, logits)
    action = int(action)
    
    env_action = env.action_space(env_params).sample(key)  # replace later with structured action
    actions.append(env_action)
    obs, state, reward, terminated, truncated, info = env.step(key, state, env_action, params=env_params)
    states.append(state)
    if terminated or truncated:
        break

# -------------------------
# Serialize and save episode
# -------------------------
states_serialized = serialize_env_states(states)
episode = dict(
    observations=states_serialized,
    actions=serialize_env_actions(actions),
    params=flax.serialization.to_state_dict(env_params),
    metadata=dict(seed=seed)
)

output_path = "episode.json"
with open(output_path, "w") as f:
    json.dump(episode, f)

print(f"Saved {output_path} for visualization.")
