import jax
import jax.numpy as jnp
import torch
import numpy as np
import random
from config import Config
from feature_engineer import WorkingFeatureEngineer  # Fixed import
from model import WorkingModel  # Fixed import
from ppo import WorkingPPO  # Fixed import
from agent import SimpleAgent
from lux_train_env import LuxTrainEnv  # Use your wrapper
from reward import WorkingRewardCalculator  # Fixed import
from action_masking import get_valid_actions  # Direct import
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from baseline_agent import BaselineAgent


def main():
    config = Config()
    
    # Initialize components
    feature_engineer = WorkingFeatureEngineer()
    model = WorkingModel(
        spatial_channels=24,
        global_dim=20,
        n_main_actions=config.n_main_actions,
        hidden_dim=config.hidden_dim
    )
    ppo = WorkingPPO(model, config)
    agent = SimpleAgent(config, model, ppo)
    
    # Create baseline agent for player_1 (FIXED, no learning)
    baseline_agent = BaselineAgent(player_id=1)
    
    reward_calculator = WorkingRewardCalculator(use_sparse=config.use_sparse_rewards)
    env = LuxTrainEnv()
    
    print("=== BASELINE TRAINING MODE ===")
    print("🤖 Player_0: Learning Agent")
    print("🤖 Player_1: Baseline Agent (FIXED)")
    print(f"Batch size: {config.batch_size}")
    print("=" * 60)
    
    # Training phases
    baseline_training_episodes = 100  # Train against baseline first
    self_play_episodes = config.n_episodes - baseline_training_episodes
    
    for episode in range(config.n_episodes):
        key = jax.random.PRNGKey(episode + 1000)
        key, subkey = jax.random.split(key)
        obs_dict, state = env.reset(subkey)
        feature_engineer.reset()
        reward_calculator.reset_for_game()
        baseline_agent.reset()  # Just reset state, not weights
        
        episode_rewards = {'player_0': 0}
        episode_points = {'player_0': 0, 'player_1': 0}
        done = False
        step_count = 0
        total_experiences = 0
        
        # Determine training mode
        use_baseline = episode < baseline_training_episodes
        mode = "BASELINE" if use_baseline else "SELF-PLAY"
        
        print(f"\n🔄 Episode {episode} [{mode}]")
        
        while not done and step_count < config.max_steps_per_match:
            # Convert observation to proper format
            numpy_obs = extract_observation_data(obs_dict)
            
            # Get actions
            try:
                if use_baseline:
                    # Player_0: Learning agent
                    features_p0 = feature_engineer.process(numpy_obs['player_0'], player_id=0)
                    actions_p0, log_probs_p0, values_p0 = agent.act(features_p0, features_p0['unit_info'], player_id=0)
                    
                    # Player_1: Baseline agent (FIXED - no learning)
                    actions_p1 = baseline_agent.act(numpy_obs['player_1'])
                    
                    training_data_p0 = (actions_p0, log_probs_p0, values_p0, features_p0)
                    
                    # Convert to Lux format
                    lux_actions = {
                        "player_0": agent._create_lux_action_array(actions_p0, features_p0['unit_info'], numpy_obs['player_0'], 0),
                        "player_1": actions_p1
                    }
                else:
                    # Self-play: both use learning agent
                    lux_actions, training_data = agent.act_for_env(numpy_obs, feature_engineer)
                    training_data_p0 = training_data['player_0']
                
            except Exception as e:
                print(f"⚠️  Error getting actions: {e}")
                lux_actions = {
                    'player_0': jnp.zeros((16, 3), dtype=jnp.int32),
                    'player_1': jnp.zeros((16, 3), dtype=jnp.int32)
                }
                training_data_p0 = ({}, {}, {}, {'unit_info': []})
            
            # Step environment
            key, subkey = jax.random.split(key)
            obs_dict_next, state_next, rewards, terminated_dict, truncated_dict, info = env.step(
                subkey, state, lux_actions
            )
            
            # Convert next observation
            numpy_obs_next = extract_observation_data(obs_dict_next)
            
            # Compute rewards ONLY for learning agent (player_0)
            match_ended = terminated_dict['player_0'] or truncated_dict['player_0']
            match_winner = None
            
            if match_ended:
                if 'team_points' in numpy_obs_next['player_0']:
                    p0_points = numpy_obs_next['player_0']['team_points'][0]
                    p1_points = numpy_obs_next['player_0']['team_points'][1]
                    match_winner = 'player_0' if p0_points > p1_points else 'player_1' if p1_points > p0_points else None
            
            reward_p0 = reward_calculator.compute_reward(
                player='player_0',
                obs=numpy_obs['player_0'],
                next_obs=numpy_obs_next['player_0'],
                match_ended=match_ended,
                match_winner=match_winner
            )
            
            episode_rewards['player_0'] += reward_p0
            
            # Store experiences ONLY for learning agent (player_0)
            step_experiences = store_learning_experiences(
                agent, training_data_p0, reward_p0, 
                numpy_obs_next, feature_engineer
            )
            total_experiences += step_experiences
            
            if step_experiences > 0 and step_count <= 3:
                print(f"  ✓ Stored {step_experiences} experiences for learning agent")
                print(f"  Step Reward P0: {reward_p0:.3f}")
            
            # Update for next step
            obs_dict = obs_dict_next
            state = state_next
            
            # Track points
            if 'player_0' in numpy_obs and 'team_points' in numpy_obs['player_0']:
                points = numpy_obs['player_0']['team_points']
                episode_points['player_0'] = points[0] if len(points) > 0 else 0
                episode_points['player_1'] = points[1] if len(points) > 1 else 0
            
            step_count += 1
            done = match_ended
            
            if match_ended:
                reward_calculator.reset_for_match()
                if use_baseline:
                    baseline_agent.reset_match()  # Just reset match state
        
        # Learning phase (only for player_0)
        loss, policy_loss, value_loss, entropy_val = 0.0, 0.0, 0.0, 0.0
        current_memory = len(ppo.memory)
        
        if current_memory >= config.batch_size:
            print(f"\n🎯 Episode {episode}: LEARNING with {current_memory} experiences!")
            loss, policy_loss, value_loss, entropy_val = ppo.update()
            print(f"📊 Loss: {loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy_val:.4f}")
        
        # Episode summary
        win_status = "🏆 WON" if episode_points['player_0'] > episode_points['player_1'] else "💥 LOST" if episode_points['player_0'] < episode_points['player_1'] else "🤝 DRAW"
        
        print(f"\n{'='*50}")
        print(f"📈 EPISODE {episode} SUMMARY [{mode}]:")
        print(f"   {win_status} | Steps: {step_count}, Experiences: {total_experiences}")
        print(f"   Points: P0={episode_points['player_0']}, P1={episode_points['player_1']}")
        print(f"   Reward P0: {episode_rewards['player_0']:.3f}")
        print(f"   Memory: {current_memory}/{config.batch_size}")
        if loss > 0:
            print(f"   Loss: {loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        print(f"{'='*50}")
        
        # Save model at transition point
        if episode == baseline_training_episodes:
            torch.save(model.state_dict(), f"baseline_trained_{episode}.pth")
            print(f"🚀 TRANSITION: Switching to self-play at episode {episode}")
        
        # Save model periodically
        if episode % 10 == 0:
            torch.save(model.state_dict(), f"working_model_{episode}.pth")
            print(f"💾 Saved model at episode {episode}")


def extract_observation_data(obs_dict):
    """
    Extract actual observation data from Lux AI S3 observation dict,
    recursively converting all nested JAX/Lux objects to dicts/NumPy arrays.
    """
    
    def convert_to_dict(obj):
        """Recursively convert objects with __dict__ to dicts and JAX arrays to numpy."""
        if isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        
        # Check for JAX/NumPy arrays
        if hasattr(obj, '__array__'):
            return np.array(obj)
        
        # Check for Lux objects (UnitState, MapFeatures, etc.)
        if hasattr(obj, '__dict__'):
            data = {}
            for attr in dir(obj):
                # Only include public, non-callable attributes
                if not attr.startswith('_') and not callable(getattr(obj, attr)):
                    value = getattr(obj, attr)
                    data[attr] = convert_to_dict(value)
            return data
        
        return obj

    processed_obs = {}
    for player_key, player_obs in obs_dict.items():
        # Only convert player-specific observations
        if player_key.startswith('player'):
             processed_obs[player_key] = convert_to_dict(player_obs)
    
    return processed_obs


def store_learning_experiences(agent, training_data_p0, reward_p0, next_obs, feature_engineer):
    """Store experiences ONLY for the learning agent (player_0)"""
    experiences_stored = 0
    
    actions, log_probs, values, features = training_data_p0
    
    if not features['unit_info']:
        return 0
    
    # Get next state features for player_0
    next_features = feature_engineer.process(next_obs['player_0'], player_id=0)
    
    # Prepare batch data for all units
    unit_positions = []
    unit_energies = []
    unit_actions = []
    unit_log_probs = []
    
    for unit in features['unit_info']:
        unit_id = unit['id']
        
        if unit_id in actions and unit_id in log_probs:
            unit_positions.append(unit['position'])
            unit_energies.append([unit['energy']])
            unit_actions.append(actions[unit_id])
            unit_log_probs.append(log_probs[unit_id])
    
    if not unit_positions:
        return 0
    
    # Create batch tensors
    spatial_tensor = torch.FloatTensor(features['spatial']).unsqueeze(0)
    global_tensor = torch.FloatTensor(features['global']).unsqueeze(0)
    positions_tensor = torch.FloatTensor(unit_positions).unsqueeze(0)
    energies_tensor = torch.FloatTensor(unit_energies).unsqueeze(0)
    actions_tensor = torch.LongTensor(unit_actions).unsqueeze(0)
    log_probs_tensor = torch.stack(unit_log_probs).unsqueeze(0)
    
    # Get next value estimate
    with torch.no_grad():
        next_unit_positions = []
        next_unit_energies = []
        
        for unit in next_features['unit_info']:
            next_unit_positions.append(unit['position'])
            next_unit_energies.append([unit['energy']])
        
        if next_unit_positions:
            next_positions_tensor = torch.FloatTensor(next_unit_positions).unsqueeze(0)
            next_energies_tensor = torch.FloatTensor(next_unit_energies).unsqueeze(0)
            next_spatial = torch.FloatTensor(next_features['spatial']).unsqueeze(0)
            next_global = torch.FloatTensor(next_features['global']).unsqueeze(0)
            
            _, _, next_value = agent.model(next_spatial, next_global, next_positions_tensor, next_energies_tensor)
            next_value = next_value.item()
        else:
            next_value = 0.0
    
    # Each unit gets equal share of team reward
    unit_reward = reward_p0 / max(1, len(unit_positions))
    
    # Store the experience
    agent.ppo.remember(
        spatial_tensor,
        global_tensor,
        positions_tensor,
        energies_tensor,
        actions_tensor,
        torch.zeros_like(actions_tensor),  # SAP placeholder
        log_probs_tensor,
        unit_reward,
        next_value,
        False
    )
    experiences_stored += 1
    
    return experiences_stored


if __name__ == "__main__":
    main()