"""
FIXED TRAIN VS BASELINE - STORE EXPERIENCES EVERY STEP

KEY FIX: Store experiences at EVERY step, not just at match end!
This gives the agent 100x more training data.
"""

import jax
import os
import jax.numpy as jnp
import torch
import numpy as np
import sys

from config import Config
from feature_engineer import AdvancedFeatureEngineer
from model import LuxResNetCNN
from ppo import ImprovedPPO
from agent import AdvancedAgent
from luxai_s3.env import LuxAIS3Env
from reward import create_reward_calculator
import copy
from lux_train_env import LuxTrainEnv
# Add parent directory to path to import baseline_agent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from baseline_agent import BaselineAgent


def find_latest_checkpoint_episode():
    """Find the latest checkpoint episode number"""
    import glob
    checkpoint_files = glob.glob("checkpoint_ep_*.pth")
    if not checkpoint_files:
        return 0
    
    episodes = []
    for file in checkpoint_files:
        try:
            ep_num = int(file.split('_')[-1].split('.')[0])
            episodes.append(ep_num)
        except:
            continue
    
    return max(episodes) if episodes else 0

def reset_environment(env, key, steps_to_spawn=1):
    """Reset environment and step until units spawn properly"""
    obs, state = env.reset(key)
    
    for _ in range(steps_to_spawn):
        key, subkey = jax.random.split(key)
        actions = {
            'player_0': jnp.zeros((16, 3), dtype=jnp.int32),
            'player_1': jnp.zeros((16, 3), dtype=jnp.int32)
        }
        obs, state, rewards, terminated, truncated, info = env.step(subkey, state, actions)
    
    return obs, state, key

def main():
    """
    Training loop with RL agent vs baseline
    FIXED: Stores experiences EVERY STEP
    """
    config = Config()
    
    # Initialize RL agent (Player 0)
    feature_engineer = AdvancedFeatureEngineer(history_len=config.temporal_history_len)
    
    spatial_channels = config.n_spatial_channels * config.temporal_history_len
    global_dim = config.n_global_features * config.temporal_history_len
    
    model = LuxResNetCNN(
        spatial_channels=spatial_channels,
        global_dim=global_dim,
        n_main_actions=config.n_main_actions,
        hidden_dim=config.hidden_dim,
        n_blocks=config.cnn_blocks
    )
    
    # Load previous model if exists
    if os.path.exists('final_trained_model.pth'):
        try:
            model.load_state_dict(torch.load('final_trained_model.pth'))
            latest_episode = find_latest_checkpoint_episode()
            start_episode = latest_episode + 1
            print(f"✅ Loaded final_trained_model.pth")
            print(f"📈 Continuing training from episode {start_episode}")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            start_episode = 0
            print("🆕 Starting with fresh model due to load error")
    else:
        start_episode = 0
        print("🆕 Starting with fresh model - no previous model found")
    
    ppo = ImprovedPPO(model, config)
    rl_agent = AdvancedAgent(config, model, ppo)
    reward_calculator = create_reward_calculator()
    
    # Initialize baseline agent (Player 1)
    baseline_agent = BaselineAgent(
        player_id=1,
        env_cfg={
            "max_units": 16,
            "map_width": 24,
            "map_height": 24,
            "max_relic_nodes": 10
        }
    )
    
    env = LuxTrainEnv()
    
    print("=" * 70)
    print("🚀 TRAINING VS BASELINE AGENT - FIXED")
    print("=" * 70)
    print("Player 0: RL Agent (learning)")
    print("Player 1: Baseline Agent (scripted)")
    print("FIX: Storing experiences EVERY STEP (not just match end)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {config.batch_size}")
    print("=" * 70)
    
    episode_stats = {
        'wins': 0,
        'losses': 0,
        'total_matches': 0
    }
    
    for episode in range(start_episode, config.n_episodes):
        reward_calculator.update_curriculum(episode)
        
        key = jax.random.PRNGKey(episode + 1000)
        key, subkey = jax.random.split(key)
        obs_dict, state, key = reset_environment(env, key, steps_to_spawn=3)
        
        # Reset for new game
        feature_engineer.reset()
        reward_calculator.reset_for_new_game()
        baseline_agent.reset()
        
        game_stats = {
            'player_0': {'match_wins': 0, 'total_score': 0},
            'player_1': {'match_wins': 0, 'total_score': 0}
        }
        
        match_count = 0
        step_count = 0
        total_experiences = 0
        done = False
        
        # 🆕 NEW: Track previous observation for step rewards
        prev_numpy_obs = None
        prev_training_data = None
        
        print(f"\n{'='*70}")
        print(f"🎮 EPISODE {episode} - RL vs Baseline")
        print(f"{'='*70}")
        
        while not done and match_count < config.matches_per_game:
            numpy_obs = extract_observation_data(obs_dict)
            
            # Get actions for BOTH players
            try:
                env_actions, training_data = rl_agent.act_for_env(numpy_obs, feature_engineer)
                baseline_actions = baseline_agent.act(numpy_obs['player_1'])
                env_actions['player_1'] = baseline_actions
                
            except Exception as e:
                print(f"⚠️ Error getting actions: {e}")
                import traceback
                traceback.print_exc()
                env_actions = {
                    'player_0': jnp.zeros((16, 3), dtype=jnp.int32),
                    'player_1': jnp.zeros((16, 3), dtype=jnp.int32)
                }
                training_data = {
                    'player_0': ({}, {}, {}, {}, {'unit_info': []}),
                    'player_1': ({}, {}, {}, {}, {'unit_info': []})
                }
            
            # Step environment
            key, subkey = jax.random.split(key)
            obs_dict_st, state_st, rewards, terminated_dict, truncated_dict, info = env.step(
                subkey, state, env_actions
            )
            
            # 🆕 NEW: Store experiences EVERY STEP (not just at match end!)
            if prev_numpy_obs is not None and prev_training_data is not None and check_memory_safety(rl_agent, config):
                step_experiences = store_step_experiences(
                    agent=rl_agent,
                    training_data=prev_training_data,
                    current_obs=prev_numpy_obs,
                    next_obs=numpy_obs,
                    feature_engineer=feature_engineer,
                    reward_calculator=reward_calculator,
                    episode=episode,
                    is_match_end=False
                )
                total_experiences += step_experiences
            
            match_just_ended = state.match_steps >= 100 and state_st.match_steps == 0
            
            if match_just_ended:
                match_count += 1
                reward_calculator.reset_for_new_match()
                baseline_agent.reset_match()
                
                # Determine match winner
                p0_score = numpy_obs['player_0'].get('team_points', [0, 0])[0]
                p1_score = numpy_obs['player_0'].get('team_points', [0, 0])[1]
                
                if p0_score > p1_score:
                    winner_p0 = 'player_0'
                    game_stats['player_0']['match_wins'] += 1
                elif p1_score > p0_score:
                    winner_p0 = None
                    game_stats['player_1']['match_wins'] += 1
                else:
                    winner_p0 = None
                
                game_stats['player_0']['total_score'] += p0_score
                game_stats['player_1']['total_score'] += p1_score
                
                print(f"\n🏁 MATCH {match_count} ENDED - Step {step_count}")
                print(f"   Score: RL={p0_score}, Baseline={p1_score}")
                print(f"   Winner: {'RL' if winner_p0 else 'Baseline' if p1_score > p0_score else 'TIE'}")
                print(f"   Match wins: RL={game_stats['player_0']['match_wins']}, Baseline={game_stats['player_1']['match_wins']}")
                print(f"   Experiences this match: ~{step_count - total_experiences + 100}")
                
                # 🆕 NEW: Store FINAL step with match-end bonus
                if training_data and check_memory_safety(rl_agent, config):
                    final_experiences = store_step_experiences(
                        agent=rl_agent,
                        training_data=training_data,
                        current_obs=numpy_obs,
                        next_obs=extract_observation_data(obs_dict_st),
                        feature_engineer=feature_engineer,
                        reward_calculator=reward_calculator,
                        episode=episode,
                        is_match_end=True,
                        winner=winner_p0,
                        my_score=p0_score,
                        opp_score=p1_score
                    )
                    total_experiences += final_experiences
                
                # Update stats
                if winner_p0 == 'player_0':
                    episode_stats['wins'] += 1
                elif p1_score > p0_score:
                    episode_stats['losses'] += 1
                episode_stats['total_matches'] += 1
                
                if game_stats['player_0']['match_wins'] >= 3:
                    print(f"\n🎉 RL AGENT WINS THE SERIES 3-{game_stats['player_1']['match_wins']}!")
                    done = True
                elif game_stats['player_1']['match_wins'] >= 3:
                    print(f"\n😔 BASELINE WINS THE SERIES 3-{game_stats['player_0']['match_wins']}!")
                    done = True
                
                # Reset tracking for new match
                prev_numpy_obs = None
                prev_training_data = None
            else:
                # Save for next step
                prev_numpy_obs = numpy_obs
                prev_training_data = training_data
            
            # Update state for next step
            obs_dict = obs_dict_st
            state = state_st
            step_count += 1
        
        # Learning phase
        loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy_val = 0.0
        current_memory = len(rl_agent.ppo.memory)
        
        if current_memory >= config.batch_size:
            print(f"\n🎯 Episode {episode}: LEARNING with {current_memory} experiences!")
            loss_info = rl_agent.learn()
            if isinstance(loss_info, tuple):
                loss, policy_loss, value_loss, entropy_val = loss_info
            else:
                loss = loss_info
            
            print(f"📊 Loss: {loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy_val:.4f}")
            rl_agent.ppo.clear_memory()
        
        # Episode summary
        winrate = episode_stats['wins'] / max(1, episode_stats['total_matches'])
        avg_score_rl = game_stats['player_0']['total_score'] / max(1, match_count)
        avg_score_baseline = game_stats['player_1']['total_score'] / max(1, match_count)
        
        print(f"\n{'='*70}")
        print(f"📈 EPISODE {episode} SUMMARY:")
        print(f"   Matches played: {match_count}")
        print(f"   Total steps: {step_count}")
        print(f"   Experiences stored: {total_experiences}")
        print(f"   Memory: {current_memory}/{config.batch_size}")
        print(f"   Match wins: RL={game_stats['player_0']['match_wins']}, Baseline={game_stats['player_1']['match_wins']}")
        print(f"   Avg score: RL={avg_score_rl:.1f}, Baseline={avg_score_baseline:.1f}")
        print(f"   Overall winrate vs Baseline: {winrate:.2%} ({episode_stats['wins']}/{episode_stats['total_matches']})")
        if loss > 0:
            print(f"   Loss: {loss:.4f}")
        print(f"{'='*70}")
        
        # Save model periodically
        if episode > 0 and episode % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_ep_{episode}.pth")
            print(f"💾 Saved checkpoint at episode {episode}")
        
        torch.save(model.state_dict(), "final_trained_model.pth")
        if episode % 10 == 0:
            print(f"💾 Updated final_trained_model.pth at episode {episode}")
    
    winrate = episode_stats['wins'] / max(1, episode_stats['total_matches'])
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("="*70)
    print(f"Final winrate vs Baseline: {episode_stats['wins']}/{episode_stats['total_matches']} = {winrate:.2%}")
    print(f"Total matches: {episode_stats['total_matches']}")
    print("="*70)


def extract_observation_data(obs_dict):
    """Simple observation extraction"""
    processed_obs = {}
    
    for player_key, player_obs in obs_dict.items():
        if player_key.startswith('player'):
            player_data = {}
            
            if hasattr(player_obs, 'units'):
                player_data['units'] = {
                    'energy': np.array(player_obs.units.energy),
                    'position': np.array(player_obs.units.position)
                }
            
            if hasattr(player_obs, 'units_mask'):
                player_data['units_mask'] = np.array(player_obs.units_mask)
            
            for attr in ['team_points', 'steps', 'match_steps', 'sensor_mask']:
                if hasattr(player_obs, attr):
                    player_data[attr] = np.array(getattr(player_obs, attr))
            
            if hasattr(player_obs, 'map_features'):
                player_data['map_features'] = {
                    'energy': np.array(player_obs.map_features.energy),
                    'tile_type': np.array(player_obs.map_features.tile_type)
                }
            
            if hasattr(player_obs, 'relic_nodes'):
                player_data['relic_nodes'] = np.array(player_obs.relic_nodes)
                player_data['relic_nodes_mask'] = np.array(player_obs.relic_nodes_mask)
            
            processed_obs[player_key] = player_data
    
    return processed_obs


def store_step_experiences(agent, training_data, current_obs, next_obs, 
                           feature_engineer, reward_calculator, episode,
                           is_match_end=False, winner=None, my_score=0, opp_score=0):
    """
    🆕 NEW: Store experiences for EVERY STEP
    
    Args:
        is_match_end: If True, add big win/loss bonus
        winner: Match winner (if is_match_end)
        my_score, opp_score: Final scores (if is_match_end)
    """
    experiences_stored = 0
    player = 'player_0'
    
    if player not in training_data:
        return 0
    
    actions, sap_targets, log_probs, values, features = training_data[player]
    player_id = 0
    
    # Get next state
    next_features = feature_engineer.process(next_obs[player], player_id=player_id)
    
    # Calculate rewards
    match_info = {
        'match_ended': is_match_end,
        'winner': winner if is_match_end else None,
        'my_score': my_score if is_match_end else current_obs[player].get('team_points', [0, 0])[player_id],
        'opp_score': opp_score if is_match_end else current_obs[player].get('team_points', [0, 0])[1 - player_id]
    }
    
    # Get rewards from reward calculator
    step_rewards, _ = reward_calculator.compute_all_rewards(
        player=player,
        actions=actions,
        current_obs=current_obs[player],
        next_obs=next_obs[player],
        match_info=match_info
    )
    
    # Store ONE experience PER UNIT
    for unit_id in range(16):
        if unit_id in actions:
            try:
                state_tuple = create_single_unit_state(features, unit_id)
                next_state_tuple = create_single_unit_state(next_features, unit_id)
                
                action = torch.tensor(actions[unit_id], dtype=torch.int64)
                log_prob = log_probs[unit_id]
                reward = step_rewards.get(unit_id, 0.0)
                
                agent.ppo.remember(
                    state_tuple,
                    action,
                    {},
                    log_prob,
                    reward,
                    next_state_tuple,
                    is_match_end  # Done flag
                )
                experiences_stored += 1
            except Exception as e:
                if experiences_stored == 0:  # Only print first error
                    print(f"⚠️ Error storing experience for unit {unit_id}: {e}")
                continue
    
    return experiences_stored


def create_single_unit_state(features, unit_id):
    """Create state tuple for a single unit"""
    spatial = torch.FloatTensor(features['spatial']).unsqueeze(0)
    global_obs = torch.FloatTensor(features['global']).unsqueeze(0)
    
    # Single unit: shape (1, 1, 2)
    unit_positions = torch.zeros(1, 1, 2)
    unit_energies = torch.zeros(1, 1, 1)
    
    if unit_id < len(features['unit_info']):
        for u in features['unit_info']:
            if u['id'] == unit_id:
                unit_positions[0, 0] = torch.FloatTensor(u['position'])
                unit_energies[0, 0] = torch.FloatTensor([u['energy'] / 400.0])
                break
    
    return (spatial, global_obs, unit_positions, unit_energies)


def check_memory_safety(agent, config):
    """Check if it's safe to store more experiences"""
    current_memory = len(agent.ppo.memory)
    return current_memory < config.batch_size * 2  # Allow 2x buffer


if __name__ == "__main__":
    main()