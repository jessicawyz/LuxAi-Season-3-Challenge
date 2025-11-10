import os
import numpy as np
import torch
import jax
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from envs.vec_env import make_vec_env
from selfplay import OpponentPool

@dataclass
class EvalConfig:
    
    num_eval_episodes: int = 10
    num_envs: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deterministic: bool = True



def load_checkpoint(checkpoint_path: str, model_type: str, device: str):
    if model_type == 'maddpg':
        from maddpg import MADDPGActor as MARLActor
        from maddpg import MADDPGConfig as MARLConfig
    elif model_type == 'mappo':
        from mappo import MAPPOActor as MARLActor
        from mappo import MAPPOConfig as MARLConfig
    else:
        raise Exception("MARL algorithm not found.")
    
    # Load checkpoint    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = MARLConfig(**config_dict)
    else:
        config = MARLConfig()
    
    # Create actors
    actor_0 = MARLActor(config).to(device)
    actor_1 = MARLActor(config).to(device)
    
    # Load weights
    actor_0.load_state_dict(checkpoint["actor_0"])
    actor_1.load_state_dict(checkpoint["actor_1"])
    
    actor_0.eval()
    actor_1.eval()
    
    return actor_0, actor_1, config

def evaluate_vs_opponent(
    agent_actors: Tuple,
    opponent_actors: Tuple,
    config: EvalConfig,
    reward_mode: str = "sparse",
) -> Dict[str, float]:
    agent_actor_0, agent_actor_1, agent_config = agent_actors
    opp_actor_0, opp_actor_1, opp_config = opponent_actors
    
    # Create environment
    env = make_vec_env(
        num_envs=config.num_envs,
        reward_mode=reward_mode,
        device=config.device
    )
    
    # Statistics
    agent_wins = 0
    opponent_wins = 0
    total_episodes = 0
    agent_match_wins = 0
    opponent_match_wins = 0
    total_matches = 0
    
    episodes_per_batch = config.num_envs
    num_batches = (config.num_eval_episodes + episodes_per_batch - 1) // episodes_per_batch
    
    for batch_idx in range(num_batches):
        # Reset 
        obs = env.reset(seed=batch_idx * 1000)
        done = torch.zeros(config.num_envs, dtype=torch.bool, device=config.device)
        
        batch_agent_wins = torch.zeros(config.num_envs, dtype=torch.long)
        batch_opponent_wins = torch.zeros(config.num_envs, dtype=torch.long)
        
        step = 0
        while not done.all():
            with torch.no_grad():
                # Agent: player_0
                actions_0, _, _ = agent_actor_0(
                    spatial_features=obs["team_0"]["spatial_features"],
                    unit_features=obs["team_0"]["unit_features"],
                    unit_mask=obs["team_0"]["unit_mask"],
                    global_features=obs["team_0"]["global_features"],
                    unit_energies=obs["team_0"]["unit_features"][:, :, 2] * 400,
                    unit_positions=(obs["team_0"]["unit_features"][:, :, :2] *
                                  torch.tensor([agent_config.map_width, agent_config.map_height], 
                                              device=config.device)).long(),
                    tile_types=obs["team_0"]["spatial_features"][:, 1] * 2
                )
                
                # Opponent: player_1
                actions_1, _, _ = opp_actor_1(
                    spatial_features=obs["team_1"]["spatial_features"],
                    unit_features=obs["team_1"]["unit_features"],
                    unit_mask=obs["team_1"]["unit_mask"],
                    global_features=obs["team_1"]["global_features"],
                    unit_energies=obs["team_1"]["unit_features"][:, :, 2] * 400,
                    unit_positions=(obs["team_1"]["unit_features"][:, :, :2] *
                                  torch.tensor([opp_config.map_width, opp_config.map_height], 
                                              device=config.device)).long(),
                    tile_types=obs["team_1"]["spatial_features"][:, 1] * 2
                )
            
            # Step 
            obs, rewards, new_done, truncated, infos = env.step(
                {"player_0": actions_0, "player_1": actions_1},
                seed=batch_idx * 10000 + step
            )
            
            # Track match wins 
            for i, info in enumerate(infos):
                if info.get("match_changed", False) and not done[i]:
                    if rewards[i, 0] > rewards[i, 1]:
                        batch_agent_wins[i] += 1
                    elif rewards[i, 1] > rewards[i, 0]:
                        batch_opponent_wins[i] += 1
            
            done = done | new_done | truncated
            step += 1
        
        # Count episode wins
        for i in range(config.num_envs):
            if total_episodes >= config.num_eval_episodes:
                break
            
            if batch_agent_wins[i] > batch_opponent_wins[i]:
                agent_wins += 1
            elif batch_opponent_wins[i] > batch_agent_wins[i]:
                opponent_wins += 1
                
            agent_match_wins += batch_agent_wins[i].item()
            opponent_match_wins += batch_opponent_wins[i].item()
            total_matches += (batch_agent_wins[i] + batch_opponent_wins[i]).item()
            total_episodes += 1
    
    env.close()
    
    # Compute metrics
    agent_win_rate = agent_wins / total_episodes if total_episodes > 0 else 0.0
    agent_match_win_rate = agent_match_wins / total_matches if total_matches > 0 else 0.0
    
    return {
        "agent_wins": agent_wins,
        "opponent_wins": opponent_wins,
        "total_episodes": total_episodes,
        "agent_win_rate": agent_win_rate,
        "agent_match_wins": agent_match_wins,
        "opponent_match_wins": opponent_match_wins,
        "total_matches": total_matches,
        "agent_match_win_rate": agent_match_win_rate,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    model_type: str,
    opponent_pool: OpponentPool,
    config: EvalConfig,
    num_opponents: int = 5,
) -> Dict[str, float]:
    """
    Evaluate a checkpoint against multiple opponents from the pool.
    """
    # Load checkpoint
    agent_actors = load_checkpoint(checkpoint_path, model_type, config.device)
    
    all_results = []
    total_wins = 0
    total_losses = 0
    
    print(f"Evaluating {checkpoint_path} against {num_opponents} opponents...")
    
    for i in range(num_opponents):
        # Sample opponent
        try:
            opponent = opponent_pool.sample_opponent(mode="elo")
            opponent_actors = load_checkpoint(
                opponent.path,
                opponent.model_type,
                config.device
            )
        except Exception as e:
            print(f"Warning: Could not load opponent {i}: {e}")
            continue
        
        # Evaluate
        results = evaluate_vs_opponent(
            agent_actors,
            opponent_actors,
            config,
        )
        
        all_results.append({
            "opponent_path": opponent.path,
            "opponent_elo": opponent.elo_rating,
            **results
        })
        
        total_wins += results["agent_wins"]
        total_losses += results["opponent_wins"]
        
        print(f"  vs {os.path.basename(opponent.path)} (ELO: {opponent.elo_rating:.1f}): "
              f"WR={results['agent_win_rate']:.2%}")
    
    # Aggregate results
    if len(all_results) == 0:
        return {
            "total_wins": 0,
            "total_losses": 0,
            "overall_win_rate": 0.0,
            "estimated_elo": 1500.0,
            "num_opponents": 0,
        }
    
    overall_win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.5
    
    # Estimate ELO based on performance vs opponents
    # Opponent ELOs, weighted by win probability
    opponent_elos = [r["opponent_elo"] for r in all_results]
    mean_opponent_elo = np.mean(opponent_elos)
    
    # Simple ELO estimation: if win_rate > 0.5, stronger
    if overall_win_rate > 0.01 and overall_win_rate < 0.99:
        estimated_elo = mean_opponent_elo - 400 * np.log10((1 - overall_win_rate) / overall_win_rate)
    elif overall_win_rate >= 0.99:
        estimated_elo = mean_opponent_elo + 400  # Much stronger
    else:
        estimated_elo = mean_opponent_elo - 400  # Much weaker
    
    return {
        "total_wins": total_wins,
        "total_losses": total_losses,
        "overall_win_rate": overall_win_rate,
        "estimated_elo": estimated_elo,
        "mean_opponent_elo": mean_opponent_elo,
        "num_opponents": len(all_results),
        "detailed_results": all_results,
    }


def evaluate_and_update_pool(
    checkpoint_path: str,
    model_type: str,
    opponent_pool: OpponentPool,
    config: EvalConfig,
    num_opponents: int = 5,
    update_pool: bool = True,
) -> Tuple[Dict[str, float], float]:
    eval_results = evaluate_checkpoint(
        checkpoint_path,
        model_type,
        opponent_pool,
        config,
        num_opponents,
    )
    
    if update_pool:
        # Update opponent pool with match results
        for result in eval_results.get("detailed_results", []):
            opponent_path = result["opponent_path"]
            agent_wins = result["agent_wins"]
            opponent_wins = result["opponent_wins"]
            
            # Find opponent in pool
            opponent = None
            for opp in opponent_pool.opponents:
                if opp.path == opponent_path:
                    opponent = opp
                    break
            
            if opponent:
                opponent_pool.update_after_match(
                    opponent,
                    current_wins=agent_wins,
                    opponent_wins=opponent_wins,
                    current_checkpoint_path=checkpoint_path,
                )
    
    return eval_results, eval_results["estimated_elo"]


if __name__ == "__main__":
    import argparse
    from selfplay import create_selfplay_pool
    
    parser = argparse.ArgumentParser(description="Evaluate Lux AI S3 checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint to evaluate")
    parser.add_argument("--model-type", type=str, default="maddpg", choices=["maddpg", "mappo"])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--num-opponents", type=int, default=5, help="Number of opponents to evaluate against")
    parser.add_argument("--num-episodes", type=int, default=10, help="Episodes per opponent")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments")
    
    args = parser.parse_args()
    
    # Create opponent pool
    pool = create_selfplay_pool(
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
        max_pool_size=20
    )
    
    print(f"Loaded opponent pool with {len(pool.opponents)} opponents")
    
    # Create eval config
    eval_config = EvalConfig(
        num_eval_episodes=args.num_episodes,
        num_envs=args.num_envs,
    )
    
    # Evaluate
    results, elo = evaluate_and_update_pool(
        args.checkpoint,
        args.model_type,
        pool,
        eval_config,
        num_opponents=args.num_opponents,
        update_pool=False,
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Overall Win Rate: {results['overall_win_rate']:.2%}")
    print(f"Total Wins: {results['total_wins']}")
    print(f"Total Losses: {results['total_losses']}")
    print(f"Estimated ELO: {results['estimated_elo']:.1f}")
    print(f"Mean Opponent ELO: {results['mean_opponent_elo']:.1f}")
    print("="*60)