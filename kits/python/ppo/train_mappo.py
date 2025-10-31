"""
MAPPO Self-Play Training Script for Lux AI S3
REFACTORED VERSION with per-team storage and frequent checkpointing
"""

import jax
import os
import jax.numpy as jnp
import torch
import numpy as np
import sys
import glob
from tqdm import tqdm
from collections import defaultdict

from config import Config
from feature_engineer import AdvancedFeatureEngineer
from model import LuxResNetCNN
from mappo import MAPPO
from agent_mappo import MAPPOAgent
from lux_train_env import LuxTrainEnv
from reward import create_reward_calculator


def extract_observation_data(obs_dict):
    """
    Extract actual observation data from Lux AI S3 observation dict,
    recursively converting all nested JAX/Lux objects to dicts/NumPy arrays.
    
    Enhanced to ensure compatibility with SimplifiedRewardCalculator.
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
            obs_data = convert_to_dict(player_obs)
            
            # Ensure required fields exist for reward calculator
            if 'team_points' not in obs_data:
                obs_data['team_points'] = [0, 0]
            if 'team_wins' not in obs_data:
                obs_data['team_wins'] = [0, 0]
            if 'match_steps' not in obs_data:
                obs_data['match_steps'] = 0
            if 'steps' not in obs_data:
                obs_data['steps'] = 0
            if 'units_mask' not in obs_data:
                obs_data['units_mask'] = [[], []]
            if 'relic_nodes_mask' not in obs_data:
                obs_data['relic_nodes_mask'] = []
            if 'sensor_mask' not in obs_data:
                obs_data['sensor_mask'] = []
            if 'relic_nodes' not in obs_data:
                obs_data['relic_nodes'] = []
            
            processed_obs[player_key] = obs_data
    
    return processed_obs


def train_mappo_self_play(config, resume_from=None):
    """
    Train using MAPPO with self-play and per-team storage
    
    Key changes:
    - Store ONE transition per team per step (not per-unit)
    - Update policy after N matches (configurable)
    - Clear buffers after each update (on-policy learning)
    - Save checkpoints every 5 episodes
    """
    print("\n" + "="*60)
    print("🚀 STARTING MAPPO SELF-PLAY TRAINING FOR LUX AI S3")
    print("="*60)
    
    # Initialize environment
    env = LuxTrainEnv()
    
    # Initialize models (shared for self-play)
    print("\n📦 Initializing models...")
    model = LuxResNetCNN(
        spatial_channels=80,
        global_dim=115,
        n_main_actions=config.n_main_actions,
        hidden_dim=config.hidden_dim,
        n_blocks=config.cnn_blocks
    )
    print(f"   Model: {config.cnn_blocks} ResNet blocks, {config.hidden_dim} hidden dim")
    
    # Initialize MAPPO with shared model
    mappo = MAPPO(model, model, config, shared_model=True)
    
    # Load checkpoint if resuming
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Loading checkpoint from {resume_from}")
        mappo.load(resume_from)
        checkpoint = torch.load(resume_from, map_location='cpu')
        start_episode = checkpoint.get('episode', 0)
        print(f"   ✅ Resuming from episode {start_episode}")
    
    # Initialize agent
    agent = MAPPOAgent(config, model, model, mappo)
    
    # Initialize feature engineers for both players
    feature_engineers = {
        'player_0': AdvancedFeatureEngineer(),
        'player_1': AdvancedFeatureEngineer()
    }
    
    # Initialize reward calculators with config and team_id
    reward_calculators = {
        'player_0': create_reward_calculator(config, team_id=0),
        'player_1': create_reward_calculator(config, team_id=1)
    }
    
    # Training metrics
    match_rewards = defaultdict(list)
    episode_rewards = defaultdict(list)
    win_rates = {'player_0': [], 'player_1': []}
    recent_losses = []
    
    # Training parameters
    matches_per_update = config.matches_per_update  # How many matches before updating policy
    checkpoint_interval = 1  # Save every 5 episodes
    
    print(f"\n📊 Training Configuration:")
    print(f"   Episodes: {config.n_episodes}")
    print(f"   Matches per episode: {config.matches_per_game}")
    print(f"   Matches per update: {matches_per_update}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   PPO epochs: {config.n_ppo_epochs}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Sparse rewards: {config.use_sparse_rewards}")
    print(f"   Checkpoint interval: Every {checkpoint_interval} episodes")
    print(f"   Shared model: Self-play")
    print(f"\n   ⚠️  IMPORTANT: Per-team storage (not per-unit)")
    
    # Main training loop
    matches_since_update = 0
    
    for episode in range(start_episode, config.n_episodes):
        print(f"\n{'='*50}")
        print(f"📋 Episode {episode + 1}/{config.n_episodes}")
        print(f"{'='*50}")
        
        # Episode-level metrics
        episode_reward = {'player_0': 0.0, 'player_1': 0.0}
        episode_wins = {'player_0': 0, 'player_1': 0}
        matches_played = 0
        
        # Play multiple matches per episode
        while matches_played < config.matches_per_game:
            print(f"\n🎮 Match {matches_played + 1}/{config.matches_per_game} [SELF-PLAY]")
            
            # Reset environment for new match
            key = jax.random.PRNGKey(episode * config.matches_per_game + matches_played + 42)
            key, reset_key = jax.random.split(key)
            obs_dict, state = env.reset(reset_key)
            
            # Extract observations properly
            numpy_obs = extract_observation_data(obs_dict)
            
            # Reset feature engineers
            for fe in feature_engineers.values():
                fe.reset()
            
            # Reset reward calculators
            for rc in reward_calculators.values():
                rc.reset()
            
            # Match storage
            match_reward = {'player_0': 0.0, 'player_1': 0.0}
            step_count = 0
            match_ended = False
            
            with tqdm(total=config.max_steps_per_match, desc="Match Progress", leave=False) as pbar:
                while not match_ended and step_count < config.max_steps_per_match:
                    step_count += 1
                    pbar.update(1)
                    
                    # Get actions from both players (self-play)
                    try:
                        lux_actions, training_data = agent.act_for_env(numpy_obs, feature_engineers)

                        # Debug output for early steps
                        if step_count <= 3:
                            print(f"\n🔍 DEBUG Step {step_count}:")
                            for player_id in [0, 1]:
                                player_key = f'player_{player_id}'
                                if player_key in training_data:
                                    td = training_data[player_key]
                                    features = td['features']
                                    unit_info = features.get('unit_info', [])
                                    
                                    team_points = numpy_obs[player_key].get('team_points', [0, 0])
                                    my_points = team_points[player_id] if len(team_points) > player_id else 0
                                    
                                    print(f"  {player_key.upper()}: {len(unit_info)} active units, "
                                          f"points={my_points}, reward={match_reward[player_key]:.2f}")
                        
                        # ✅ NEW: Store ONE transition per team per step (not per unit!)
                        for player_id in [0, 1]:
                            player_key = f'player_{player_id}'
                            if player_key in training_data:
                                td = training_data[player_key]
                                features = td['features']
                                
                                # Prepare state for storage
                                state_tuple = (
                                    torch.FloatTensor(features['spatial']).unsqueeze(0),
                                    torch.FloatTensor(features['global']).unsqueeze(0),
                                    torch.zeros(1, config.max_units, 2),  # Placeholder
                                    torch.zeros(1, config.max_units, 1)   # Placeholder
                                )
                                
                                # Convert dict actions/log_probs to tensors for ALL units
                                actions_dict = td['actions']
                                log_probs_dict = td['log_probs']
                                
                                # Create tensors for all units (max_units=16)
                                actions_tensor = torch.zeros(config.max_units, dtype=torch.long)
                                log_probs_tensor = torch.zeros(config.max_units)
                                unit_mask_tensor = torch.zeros(config.max_units, dtype=torch.bool)
                                
                                # Fill in data for active units
                                for unit_id, action in actions_dict.items():
                                    if unit_id < config.max_units:
                                        actions_tensor[unit_id] = action
                                        log_probs_tensor[unit_id] = log_probs_dict[unit_id]
                                        unit_mask_tensor[unit_id] = True
                                
                                # Store this state for later (after we get reward and next state)
                                if player_id == 0:
                                    stored_p0 = {
                                        'state': state_tuple,
                                        'actions': actions_tensor,
                                        'log_probs': log_probs_tensor,
                                        'unit_mask': unit_mask_tensor,
                                        'info': features
                                    }
                                else:
                                    stored_p1 = {
                                        'state': state_tuple,
                                        'actions': actions_tensor,
                                        'log_probs': log_probs_tensor,
                                        'unit_mask': unit_mask_tensor,
                                        'info': features
                                    }
                        
                    except Exception as e:
                        print(f"⚠️ Error getting actions: {e}")
                        import traceback
                        traceback.print_exc()
                        # Default actions
                        lux_actions = {
                            'player_0': jnp.zeros((16, 3), dtype=jnp.int32),
                            'player_1': jnp.zeros((16, 3), dtype=jnp.int32)
                        }
                        stored_p0 = None
                        stored_p1 = None
                    
                    # Step environment
                    key, subkey = jax.random.split(key)
                    obs_dict_next, state_next, rewards, terminated_dict, truncated_dict, info = env.step(
                        subkey, state, lux_actions
                    )
                    
                    # Extract next observations
                    numpy_obs_next = extract_observation_data(obs_dict_next)
                    
                    # Use environment's termination signals
                    match_ended = (terminated_dict.get('player_0', False) or 
                                 truncated_dict.get('player_0', False) or 
                                 step_count >= config.max_steps_per_match)
                    
                    # ✅ NEW: Calculate team-level rewards and store ONE transition per team
                    for player_id in [0, 1]:
                        player_key = f'player_{player_id}'
                        
                        # Get environment reward
                        env_reward = float(rewards.get(player_key, 0.0))
                        
                        # Calculate shaped reward (team-level)
                        team_reward = reward_calculators[player_key].calculate_reward(
                            numpy_obs[player_key],
                            numpy_obs_next[player_key],
                            env_reward,
                            match_ended,
                            info
                        )
                        match_reward[player_key] += team_reward
                        
                        # Store ONE transition for the entire team
                        if player_id == 0 and stored_p0 is not None:
                            # Get next state
                            next_features = feature_engineers[player_key].process(
                                numpy_obs_next[player_key], player_id
                            )
                            next_state_tuple = (
                                torch.FloatTensor(next_features['spatial']).unsqueeze(0),
                                torch.FloatTensor(next_features['global']).unsqueeze(0),
                                torch.zeros(1, config.max_units, 2),
                                torch.zeros(1, config.max_units, 1)
                            )
                            
                            # Store single team-level transition
                            mappo.store_transition(
                                player_id,
                                stored_p0['state'],
                                stored_p0['actions'],      # All unit actions (16,)
                                stored_p0['log_probs'],    # All unit log probs (16,)
                                stored_p0['unit_mask'],    # Which units active (16,)
                                team_reward,               # Single team reward
                                next_state_tuple,
                                match_ended,
                                stored_p0['info']
                            )
                        
                        elif player_id == 1 and stored_p1 is not None:
                            # Get next state
                            next_features = feature_engineers[player_key].process(
                                numpy_obs_next[player_key], player_id
                            )
                            next_state_tuple = (
                                torch.FloatTensor(next_features['spatial']).unsqueeze(0),
                                torch.FloatTensor(next_features['global']).unsqueeze(0),
                                torch.zeros(1, config.max_units, 2),
                                torch.zeros(1, config.max_units, 1)
                            )
                            
                            # Store single team-level transition
                            mappo.store_transition(
                                player_id,
                                stored_p1['state'],
                                stored_p1['actions'],
                                stored_p1['log_probs'],
                                stored_p1['unit_mask'],
                                team_reward,
                                next_state_tuple,
                                match_ended,
                                stored_p1['info']
                            )
                    
                    # Update for next iteration
                    obs_dict = obs_dict_next
                    numpy_obs = numpy_obs_next
                    state = state_next
                    
                    # Break if match ended
                    if match_ended:
                        break
            
            # Record match metrics
            matches_played += 1
            matches_since_update += 1
            match_rewards['player_0'].append(match_reward['player_0'])
            match_rewards['player_1'].append(match_reward['player_1'])
            
            # Determine match winner
            if match_reward['player_0'] > match_reward['player_1']:
                winner = 'player_0'
                episode_wins['player_0'] += 1
            elif match_reward['player_1'] > match_reward['player_0']:
                winner = 'player_1'
                episode_wins['player_1'] += 1
            else:
                winner = 'draw'
            
            # Update win rates
            win_rates['player_0'].append(1 if winner == 'player_0' else 0)
            win_rates['player_1'].append(1 if winner == 'player_1' else 0)
            
            print(f"   Match {matches_played}: {winner} (P0: {match_reward['player_0']:.1f} | P1: {match_reward['player_1']:.1f})")
            print(f"   Buffer sizes: P0={mappo.buffers['player_0'].size()}, P1={mappo.buffers['player_1'].size()}")
            
            # Accumulate episode rewards
            episode_reward['player_0'] += match_reward['player_0']
            episode_reward['player_1'] += match_reward['player_1']
            
            # ✅ NEW: Update policy after N matches
            if matches_since_update >= matches_per_update:
                print(f"\n🎯 Updating policy after {matches_since_update} matches...")
                metrics = agent.learn()
                
                if metrics.get('total_loss', 0) > 0:
                    recent_losses.append(metrics['total_loss'])
                    print(f"   Total loss: {metrics['total_loss']:.4f}")
                    print(f"   Policy loss: {metrics['policy_loss']:.4f}")
                    print(f"   Value loss: {metrics['value_loss']:.4f}")
                    print(f"   Entropy: {metrics['entropy']:.4f}")
                
                # ✅ CRITICAL: Clear buffers after update (on-policy learning)
                mappo.buffers['player_0'].clear()
                mappo.buffers['player_1'].clear()
                matches_since_update = 0
                print(f"   ✅ Buffers cleared (on-policy learning)")
        
        # Print episode summary
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"   Matches played: {matches_played}")
        print(f"   Player 0 wins: {episode_wins['player_0']}/{matches_played}")
        print(f"   Player 1 wins: {episode_wins['player_1']}/{matches_played}")
        print(f"   Player 0 total reward: {episode_reward['player_0']:.2f}")
        print(f"   Player 1 total reward: {episode_reward['player_1']:.2f}")
        
        # Calculate moving averages
        window = min(10, len(win_rates['player_0']))
        if window > 0:
            avg_reward_p0 = np.mean(match_rewards['player_0'][-window:])
            avg_reward_p1 = np.mean(match_rewards['player_1'][-window:])
            win_rate_p0 = np.mean(win_rates['player_0'][-window:]) * 100
            win_rate_p1 = np.mean(win_rates['player_1'][-window:]) * 100
            
            print(f"\n📈 Last {window} Matches:")
            print(f"   Player 0: Avg Reward={avg_reward_p0:.2f}, Win Rate={win_rate_p0:.1f}%")
            print(f"   Player 1: Avg Reward={avg_reward_p1:.2f}, Win Rate={win_rate_p1:.1f}%")
            
            if len(recent_losses) > 0:
                print(f"   Recent avg loss: {np.mean(recent_losses[-10:]):.4f}")
        
        # ✅ NEW: Final update at episode end if there's remaining data
        if matches_since_update > 0:
            print(f"\n🎯 Episode-end update (remaining {matches_since_update} matches):")
            metrics = agent.learn()
            if metrics.get('total_loss', 0) > 0:
                print(f"   Total loss: {metrics['total_loss']:.4f}")
                print(f"   Policy loss: {metrics['policy_loss']:.4f}")
                print(f"   Value loss: {metrics['value_loss']:.4f}")
                print(f"   Entropy: {metrics['entropy']:.4f}")
            
            # Clear buffers
            mappo.buffers['player_0'].clear()
            mappo.buffers['player_1'].clear()
            matches_since_update = 0
        
        # ✅ NEW: Save checkpoint every 5 episodes (more frequent!)
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = f"ppo_checkpoint_ep_{episode+1}.pth"
            checkpoint = {
                'model_p0_state': mappo.model_p0.state_dict(),
                'model_p1_state': mappo.model_p1.state_dict(),
                'optimizer_state': mappo.optimizer.state_dict(),
                'update_count': mappo.update_count,
                'episode': episode + 1,
                'win_rate_p0': np.mean(win_rates['player_0'][-window:]) * 100 if window > 0 else 0,
                'win_rate_p1': np.mean(win_rates['player_1'][-window:]) * 100 if window > 0 else 0,
                'avg_reward_p0': avg_reward_p0 if window > 0 else 0,
                'avg_reward_p1': avg_reward_p1 if window > 0 else 0,
                'config': config.__dict__
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
            
            # Also save as "latest" for easy resuming
            latest_path = "ppo_checkpoint_latest.pth"
            torch.save(checkpoint, latest_path)
            print(f"💾 Latest checkpoint saved: {latest_path}")
    
    # Save final model
    final_path = "ppo_final_model.pth"
    checkpoint = {
        'model_p0_state': mappo.model_p0.state_dict(),
        'model_p1_state': mappo.model_p1.state_dict(),
        'optimizer_state': mappo.optimizer.state_dict(),
        'update_count': mappo.update_count,
        'episode': config.n_episodes,
        'config': config.__dict__
    }
    torch.save(checkpoint, final_path)
    print(f"\n✅ Training complete! Final model saved: {final_path}")
    
    # Print final statistics
    window = min(50, len(win_rates['player_0']))
    if window > 0:
        print(f"\n📊 Final Training Statistics:")
        print(f"   Total episodes: {config.n_episodes}")
        print(f"   Final win rate (P0, last {window}): {np.mean(win_rates['player_0'][-window:])*100:.1f}%")
        print(f"   Final win rate (P1, last {window}): {np.mean(win_rates['player_1'][-window:])*100:.1f}%")
        print(f"   Final avg reward (P0): {np.mean(match_rewards['player_0'][-window:]):.2f}")
        print(f"   Final avg reward (P1): {np.mean(match_rewards['player_1'][-window:]):.2f}")


if __name__ == "__main__":
    config = Config()
    
    # Check for existing checkpoint
    checkpoints = sorted(glob.glob("ppo_checkpoint_ep_*.pth"))
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"Found checkpoint: {latest_checkpoint}")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() == 'y':
            train_mappo_self_play(config, resume_from=latest_checkpoint)
        else:
            train_mappo_self_play(config)
    else:
        train_mappo_self_play(config)