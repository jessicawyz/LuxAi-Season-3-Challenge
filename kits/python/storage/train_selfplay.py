import jax
import jax.numpy as jnp
import torch
import numpy as np
import random
import os
import json
from datetime import datetime
from pathlib import Path
from config import Config
from feature_engineer import WorkingFeatureEngineer  # Fixed import
from model import WorkingModel  # Fixed import
from ppo import WorkingPPO  # Fixed import
from agent import SimpleAgent
from lux_train_env import LuxTrainEnv  # Use your wrapper
from reward import WorkingRewardCalculator  # Fixed import
from action_masking import get_valid_actions  # Direct import


def safe_convert(value):
    """Safely convert JAX/NumPy values to Python types"""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
        try:
            return [safe_convert(v) for v in value]
        except (TypeError, AttributeError):
            pass
    return value


def save_step_checkpoint(episode, step, obs_before, obs_after, training_data, checkpoint_dir="checkpoints/debug_steps"):
    """Save checkpoint for a specific step"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    step_data = {
        'episode': episode,
        'step': step,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Track units for each player
    for player in ['player_0', 'player_1']:
        player_units = []
        
        # Get units from before observation
        units_before = {}
        if player in obs_before and 'units' in obs_before[player]:
            positions = obs_before[player]['units'].get('position', [])
            energies = obs_before[player]['units'].get('energy', [])
            mask = obs_before[player].get('units_mask', [True] * len(positions))
            
            for i, (pos, energy, valid) in enumerate(zip(positions, energies, mask)):
                # Safe boolean check
                is_valid = bool(valid) if hasattr(valid, '__bool__') else valid
                if is_valid:
                    units_before[i] = {
                        'position': safe_convert(pos),
                        'energy': safe_convert(energy)
                    }
        
        # Get units from after observation
        units_after = {}
        if player in obs_after and 'units' in obs_after[player]:
            positions = obs_after[player]['units'].get('position', [])
            energies = obs_after[player]['units'].get('energy', [])
            mask = obs_after[player].get('units_mask', [True] * len(positions))
            
            for i, (pos, energy, valid) in enumerate(zip(positions, energies, mask)):
                # Safe boolean check
                is_valid = bool(valid) if hasattr(valid, '__bool__') else valid
                if is_valid:
                    units_after[i] = {
                        'position': safe_convert(pos),
                        'energy': safe_convert(energy)
                    }
        
        # Get actions
        actions = {}
        if player in training_data:
            actions = training_data[player][0]
        
        # Compare before/after
        for unit_id in units_before:
            unit_data = {
                'id': unit_id,
                'position_before': units_before[unit_id]['position'],
                'energy_before': units_before[unit_id]['energy'],
                'action': int(actions.get(unit_id, 0))
            }
            
            if unit_id in units_after:
                unit_data['alive'] = True
                unit_data['position_after'] = units_after[unit_id]['position']
                unit_data['energy_after'] = units_after[unit_id]['energy']
                unit_data['energy_change'] = units_after[unit_id]['energy'] - units_before[unit_id]['energy']
            else:
                unit_data['alive'] = False
                unit_data['position_after'] = [-1, -1]
                unit_data['energy_after'] = 0
            
            player_units.append(unit_data)
        
        step_data[player] = player_units
    
    # Save to JSON
    checkpoint_file = os.path.join(checkpoint_dir, f"ep{episode}_step{step}.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(step_data, f, indent=2)
    
    # Also log to text file
    log_file = os.path.join(checkpoint_dir, f"ep{episode}_log.txt")
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Episode {episode}, Step {step}\n")
        f.write(f"{'='*60}\n")
        
        for player in ['player_0', 'player_1']:
            units = step_data.get(player, [])
            f.write(f"\n{player.upper()}: {len(units)} units\n")
            alive = sum(1 for u in units if u.get('alive', True))
            dead = len(units) - alive
            f.write(f"  Alive: {alive}, Dead: {dead}\n")
            
            for unit in units:
                status = "ALIVE" if unit.get('alive', True) else "DEAD"
                action_names = ["NO_OP", "UP", "RIGHT", "DOWN", "LEFT", "SAP"]
                action_idx = unit.get('action', 0)
                action_name = action_names[action_idx] if 0 <= action_idx < len(action_names) else f"UNK({action_idx})"
                f.write(f"  Unit {unit['id']}: {status}, pos={unit['position_before']}, "
                       f"energy={unit['energy_before']:.1f}, action={action_name}\n")


def main():
    config = Config()
    
    # Initialize components with correct classes
    feature_engineer = WorkingFeatureEngineer()
    model = WorkingModel(
        spatial_channels=24,  # 4 temporal × 3 history + 12 static
        global_dim=20,
        n_main_actions=config.n_main_actions,
        hidden_dim=config.hidden_dim
    )
    ppo = WorkingPPO(model, config)
    agent = SimpleAgent(config, model, ppo)
    reward_calculator = WorkingRewardCalculator(use_sparse=config.use_sparse_rewards)
    
    # Use your training environment wrapper
    env = LuxTrainEnv()
    
    print("=== FIXED TRAINING WITH PROPER REWARD COMPUTATION ===")
    print(f"Batch size: {config.batch_size}")
    print("=" * 60)
    
    for episode in range(config.n_episodes):
        key = jax.random.PRNGKey(episode + 1000)
        key, subkey = jax.random.split(key)
        obs_dict, state = env.reset(subkey)
        feature_engineer.reset()
        reward_calculator.reset_for_game()
        
        episode_rewards = {'player_0': 0, 'player_1': 0}
        episode_points = {'player_0': 0, 'player_1': 0}
        done = False
        step_count = 0
        total_experiences = 0
        
        print(f"\n🔄 Starting Episode {episode}")
        
        while not done and step_count < config.max_steps_per_match:
            # Convert observation to proper format
            numpy_obs = extract_observation_data(obs_dict)
            
            # Get actions for both players
            try:
                env_actions, training_data = agent.act_for_env(numpy_obs, feature_engineer)
            except Exception as e:
                print(f"⚠️  Error getting actions: {e}")
                env_actions = {
                    'player_0': jnp.zeros((16, 3), dtype=jnp.int32),
                    'player_1': jnp.zeros((16, 3), dtype=jnp.int32)
                }
                training_data = {
                    'player_0': ({}, {}, {}, {'unit_info': []}),
                    'player_1': ({}, {}, {}, {'unit_info': []})
                }
            
            # DEBUG: Check unit counts
            p0_units = len(training_data['player_0'][3]['unit_info']) if 'player_0' in training_data else 0
            p1_units = len(training_data['player_1'][3]['unit_info']) if 'player_1' in training_data else 0
            
            if step_count <= 5 or (step_count % 20 == 0) or p0_units > 0 or p1_units > 0:
                print(f"\nEpisode {episode}, Step {step_count}:")
                print(f"  Units - P0: {p0_units}, P1: {p1_units}")
                
                # Show unit details if we have units
                for player in ['player_0', 'player_1']:
                    if player in training_data:
                        unit_info = training_data[player][3]['unit_info']
                        actions = training_data[player][0]
                        if unit_info:
                            print(f"  {player.upper()} has {len(unit_info)} units:")
                            for unit in unit_info[:2]:  # Show first 2 units
                                unit_id = unit['id']
                                if unit_id in actions:
                                    action_names = ["NO_OP", "UP", "RIGHT", "DOWN", "LEFT", "SAP"]
                                    action_name = action_names[actions[unit_id]] if actions[unit_id] < len(action_names) else f"UNK({actions[unit_id]})"
                                    player_prefix = "P0" if player == 'player_0' else "P1"
                                    print(f"    {player_prefix} Unit {unit_id}: pos={unit['position']}, energy={unit['energy']:.1f}, action={action_name}")
            
            # Step environment
            key, subkey = jax.random.split(key)
            obs_dict_next, state_next, rewards, terminated_dict, truncated_dict, info = env.step(
                subkey, state, env_actions
            )
            
            # Convert next observation to proper format
            numpy_obs_next = extract_observation_data(obs_dict_next)
            
            # 💾 SAVE CHECKPOINT EVERY 10 STEPS
            if step_count % 10 == 0:
                try:
                    save_step_checkpoint(episode, step_count, numpy_obs, numpy_obs_next, training_data)
                    print(f"  💾 Saved checkpoint for step {step_count}")
                except Exception as e:
                    print(f"  ⚠️  Failed to save checkpoint: {e}")
            
            # ✅ COMPUTE STEP REWARDS for each player
            step_rewards = {}
            match_ended = terminated_dict['player_0'] or truncated_dict['player_0']
            match_winner = None
            
            # Determine match winner if match ended
            if match_ended:
                if 'team_points' in numpy_obs_next['player_0']:
                    p0_points = numpy_obs_next['player_0']['team_points'][0]
                    p1_points = numpy_obs_next['player_0']['team_points'][1]
                    match_winner = 'player_0' if p0_points > p1_points else 'player_1' if p1_points > p0_points else None
            
            # Calculate rewards for each player
            for player in ['player_0', 'player_1']:
                reward = reward_calculator.compute_reward(
                    player=player,
                    obs=numpy_obs[player],
                    next_obs=numpy_obs_next[player],
                    match_ended=match_ended,
                    match_winner=match_winner
                )
                step_rewards[player] = reward
                episode_rewards[player] += reward
            
            # Store experiences with computed rewards
            step_experiences = store_experiences(
                agent, training_data, step_rewards, numpy_obs_next, 
                feature_engineer, numpy_obs, reward_calculator
            )
            total_experiences += step_experiences
            
            if step_experiences > 0 and step_count <= 3:
                print(f"  ✓ Stored {step_experiences} experiences")
                print(f"  Step Rewards: P0={step_rewards['player_0']:.3f}, P1={step_rewards['player_1']:.3f}")
            
            # Update for next step
            obs_dict = obs_dict_next
            state = state_next
            
            # Track points from observation
            if 'player_0' in numpy_obs and 'team_points' in numpy_obs['player_0']:
                points = numpy_obs['player_0']['team_points']
                episode_points['player_0'] = points[0] if len(points) > 0 else 0
                episode_points['player_1'] = points[1] if len(points) > 1 else 0
            
            step_count += 1
            done = match_ended
            
            # Reset reward calculator for new match if current one ended
            if match_ended:
                reward_calculator.reset_for_match()
        
        # Learning phase
        loss, policy_loss, value_loss, entropy_val = 0.0, 0.0, 0.0, 0.0
        current_memory = len(ppo.memory)
        
        if current_memory >= config.batch_size:
            print(f"\n🎯 Episode {episode}: LEARNING with {current_memory} experiences!")
            loss, policy_loss, value_loss, entropy_val = ppo.update()
            
            print(f"📊 Loss: {loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy_val:.4f}")
        
        # Episode summary
        print(f"\n{'='*50}")
        print(f"📈 EPISODE {episode} SUMMARY:")
        print(f"   Steps: {step_count}, Experiences: {total_experiences}")
        print(f"   Memory: {current_memory}/{config.batch_size}")
        print(f"   Points: P0={episode_points['player_0']}, P1={episode_points['player_1']}")
        print(f"   Rewards: P0={episode_rewards['player_0']:.3f}, P1={episode_rewards['player_1']:.3f}")
        if loss > 0:
            print(f"   Loss: {loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        print(f"{'='*50}")
        
        # Save model periodically
        if episode % 10 == 0:
            torch.save(model.state_dict(), f"working_model_{episode}.pth")
            print(f"💾 Saved model at episode {episode}")
            
        if loss > 0 and episode >= 3:
            print("🎉 SUCCESS! Model is learning!")
            torch.save(model.state_dict(), "final_trained_model.pth")
    
    print("\n🏁 Training completed!")


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


def store_experiences(agent, training_data, step_rewards, next_obs, feature_engineer, current_obs, reward_calculator):
    """Store experiences with step rewards for each unit - FIXED DIMENSIONS"""
    experiences_stored = 0
    
    for player in ['player_0', 'player_1']:
        if player not in training_data:
            continue
            
        actions, log_probs, values, features = training_data[player]
        player_id = 0 if player == 'player_0' else 1
        
        if not features['unit_info']:
            continue
            
        # Get next state features
        next_features = feature_engineer.process(next_obs[player], player_id=player_id)
        
        # Distribute team reward to all units of this player
        team_reward = step_rewards[player]
        
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
            continue
            
        # Create batch tensors with proper dimensions
        spatial_tensor = torch.FloatTensor(features['spatial']).unsqueeze(0)
        global_tensor = torch.FloatTensor(features['global']).unsqueeze(0)
        positions_tensor = torch.FloatTensor(unit_positions).unsqueeze(0)  # (1, n_units, 2)
        energies_tensor = torch.FloatTensor(unit_energies).unsqueeze(0)    # (1, n_units, 1)
        actions_tensor = torch.LongTensor(unit_actions).unsqueeze(0)       # (1, n_units)
        log_probs_tensor = torch.stack(unit_log_probs).unsqueeze(0)        # (1, n_units)
        
        # Get next value estimate
        with torch.no_grad():
            # Prepare next state
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
        
        # Each unit gets the team reward
        unit_reward = team_reward / max(1, len(unit_positions))
        
        # Store the experience for all units in this batch
        agent.ppo.remember(
            spatial_tensor,
            global_tensor,
            positions_tensor,
            energies_tensor,
            actions_tensor,
            torch.zeros_like(actions_tensor),  # SAP action placeholder
            log_probs_tensor,
            unit_reward,
            next_value,
            False  # done
        )
        experiences_stored += 1
        
        # Print debug info for significant rewards
        if abs(unit_reward) > 0.1:
            print(f"💰 {player}: {len(unit_positions)} units → team reward {unit_reward:.3f} each")
    
    return experiences_stored


if __name__ == "__main__":
    main()