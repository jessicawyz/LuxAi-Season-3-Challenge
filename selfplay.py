import os
import glob
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random
from collections import defaultdict, deque
import math


import math


@dataclass
class PFSPConfig:
    pfsp_temperature: float = 1.0
    nash_target_range: Tuple[float, float] = (0.4, 0.6)
    uncertainty_bonus: float = 0.5
    recency_decay: float = 100_000
    diversity_window: int = 50
    pfsp_ratio: float = 0.8
    min_games_for_priority: int = 3


@dataclass
class OpponentCheckpoint:
    
    path: str
    
    # Elo tracking (for absolute strength)
    elo_rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    model_type: str = "maddpg" 
    
    # PFSP tracking (for intelligent sampling)
    recent_matches_vs_current: deque = field(default_factory=lambda: deque(maxlen=20))
    games_vs_current_policy: int = 0
    wins_vs_current_policy: int = 0
    last_sampled_step: int = 0
    times_sampled_this_window: int = 0
    checkpoint_step: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played
    
    @property
    def recent_win_rate_vs_current(self) -> float:
        """Calculate recent win rate vs current policy."""
        if not self.recent_matches_vs_current:
            return 0.5 
        return sum(self.recent_matches_vs_current) / len(self.recent_matches_vs_current)


class EloRating:
    """Elo rating system"""
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float
    ) -> Tuple[float, float]:
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1.0 - score_a) - expected_b)
        
        return new_rating_a, new_rating_b


class OpponentPool:
    """
    Hybrid Elo + PFSP 
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_pool_size: int = 20,
        elo_k_factor: float = 32.0,
        temperature: float = 1.0,
        pfsp_config: Optional[PFSPConfig] = None,
    ):
        """
        Initialize opponent pool.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_pool_size = max_pool_size
        self.temperature = temperature
        self.elo_system = EloRating(k_factor=elo_k_factor)
        self.pfsp_config = pfsp_config or PFSPConfig()
        
        self.opponents: List[OpponentCheckpoint] = []
        self.current_checkpoint: Optional[OpponentCheckpoint] = None
        self.current_training_step: int = 0
        
        # Stats tracking
        self.match_history = []
        self.sampling_history = deque(maxlen=self.pfsp_config.diversity_window)
    
    def add_checkpoint(
        self,
        checkpoint_path: str,
        model_type: str = "mappo",
        initial_rating: Optional[float] = None,
        checkpoint_step: int = 0,
    ):
        """
        Add a new checkpoint to the pool.
        """
        if initial_rating is None:
            initial_rating = self.elo_system.initial_rating
        
        opponent = OpponentCheckpoint(
            path=checkpoint_path,
            elo_rating=initial_rating,
            model_type=model_type,
            checkpoint_step=checkpoint_step,
        )
        
        self.opponents.append(opponent)
        
        # Prune pool if too large
        if len(self.opponents) > self.max_pool_size:
            self._prune_pool()
    
    def compute_pfsp_priority(
        self,
        opponent: OpponentCheckpoint,
        current_step: int,
    ) -> float:
        """
        Compute PFSP priority for an opponent.
        """
        
        if opponent.games_vs_current_policy < self.pfsp_config.min_games_for_priority:
            return 1.0  
        
        
        win_rate = opponent.recent_win_rate_vs_current
        nash_distance = 4 * win_rate * (1 - win_rate)  
        
        
        if self.pfsp_config.nash_target_range[0] <= win_rate <= self.pfsp_config.nash_target_range[1]:
            nash_distance *= 1.5
        
        
        uncertainty = 1.0 / math.sqrt(opponent.games_vs_current_policy + 1)
        uncertainty_bonus = 1.0 + self.pfsp_config.uncertainty_bonus * uncertainty
        
        
        age = current_step - opponent.checkpoint_step
        recency = math.exp(-age / self.pfsp_config.recency_decay)
        
        
        diversity = 1.0 / (opponent.times_sampled_this_window + 1)
        
        
        priority = nash_distance * uncertainty_bonus * recency * diversity
        
        return max(priority, 0.01)  
    
    def _prune_pool(self):
        """Remove weakest checkpoints from pool."""
        
        self.opponents.sort(key=lambda x: x.elo_rating, reverse=True)
        
        
        keep_count = int(self.max_pool_size * 0.7)
        top_opponents = self.opponents[:keep_count]
        
        
        remaining = self.opponents[keep_count:]
        if remaining:
            diversity_count = self.max_pool_size - keep_count
            diversity_sample = random.sample(
                remaining,
                min(diversity_count, len(remaining))
            )
            self.opponents = top_opponents + diversity_sample
        else:
            self.opponents = top_opponents
    
    def sample_opponent(
        self,
        exclude_recent: int = 0,
        mode: str = "pfsp",
        current_step: Optional[int] = None,
    ) -> OpponentCheckpoint:
        """
        Sample an opponent from the pool.
        """
        if not self.opponents:
            raise ValueError("Opponent pool is empty!")
        
        # Get eligible opponents
        eligible = self.opponents[:-exclude_recent] if exclude_recent > 0 else self.opponents
        
        if not eligible:
            eligible = self.opponents
        
        # Use pfsp_ratio to decide sampling method
        if mode == "pfsp" and np.random.random() < self.pfsp_config.pfsp_ratio:
            # PFSP-based sampling
            if current_step is None:
                current_step = self.current_training_step
            
            priorities = np.array([
                self.compute_pfsp_priority(opp, current_step) 
                for opp in eligible
            ])
            
            # Temperature-scaled softmax
            priorities = priorities / self.pfsp_config.pfsp_temperature
            exp_priorities = np.exp(priorities - np.max(priorities))
            probs = exp_priorities / exp_priorities.sum()
            
            idx = np.random.choice(len(eligible), p=probs)
            opponent = eligible[idx]
            
        elif mode == "elo" or (mode == "pfsp" and np.random.random() >= self.pfsp_config.pfsp_ratio):
            # Elo-based sampling 
            ratings = np.array([opp.elo_rating for opp in eligible])
            ratings = ratings / self.temperature
            
            exp_ratings = np.exp(ratings - np.max(ratings))
            probs = exp_ratings / exp_ratings.sum()
            
            idx = np.random.choice(len(eligible), p=probs)
            opponent = eligible[idx]
            
        else:  # uniform
            opponent = random.choice(eligible)
        
        # Update diversity tracking
        opponent.times_sampled_this_window += 1
        opponent.last_sampled_step = current_step if current_step else self.current_training_step
        self.sampling_history.append(opponent.path)
        
        # Reset diversity counters periodically
        if len(self.sampling_history) == self.pfsp_config.diversity_window:
            for opp in self.opponents:
                opp.times_sampled_this_window = 0
        
        return opponent
    
    def update_after_match(
        self,
        opponent: OpponentCheckpoint,
        current_wins: int,
        opponent_wins: int,
        current_checkpoint_path: Optional[str] = None,
        current_step: Optional[int] = None,
    ):
        """
        Update ratings after a match (both Elo and PFSP stats).
        """
        total_matches = current_wins + opponent_wins
        
        if total_matches == 0:
            return
        
        # Update training step
        if current_step is not None:
            self.current_training_step = current_step
        
        # Compute score 
        score = current_wins / total_matches
        
        # Update Elo ratings
        if current_checkpoint_path:
            # Current checkpoint
            current = None
            for opp in self.opponents:
                if opp.path == current_checkpoint_path:
                    current = opp
                    break
            
            if current is None and current_checkpoint_path:
                current = OpponentCheckpoint(
                    path=current_checkpoint_path,
                    elo_rating=self.elo_system.initial_rating
                )
        else:
            current_rating = self.elo_system.initial_rating
            current = None
        
        # Update Elo ratings
        if current:
            new_current_rating, new_opponent_rating = self.elo_system.update_ratings(
                current.elo_rating,
                opponent.elo_rating,
                score
            )
            current.elo_rating = new_current_rating
            current.games_played += total_matches
            current.wins += current_wins
            current.losses += opponent_wins
        else:
            _, new_opponent_rating = self.elo_system.update_ratings(
                self.elo_system.initial_rating,
                opponent.elo_rating,
                score
            )
        
        opponent.elo_rating = new_opponent_rating
        opponent.games_played += total_matches
        opponent.wins += opponent_wins
        opponent.losses += current_wins
        
        # Update PFSP stats
        self.update_pfsp_stats(opponent, current_wins, opponent_wins)
        
        # Record match
        self.match_history.append({
            "current_wins": current_wins,
            "opponent_wins": opponent_wins,
            "opponent_path": opponent.path,
            "opponent_rating_before": opponent.elo_rating - (new_opponent_rating - opponent.elo_rating),
            "opponent_rating_after": opponent.elo_rating,
            "pfsp_priority": self.compute_pfsp_priority(opponent, self.current_training_step),
            "training_step": self.current_training_step,
        })
    
    def update_pfsp_stats(
        self,
        opponent: OpponentCheckpoint,
        current_wins: int,
        opponent_wins: int,
    ):
        """
        Update PFSP-specific statistics for opponent
        """
        total_games = current_wins + opponent_wins
        
        # Update game counts
        opponent.games_vs_current_policy += total_games
        opponent.wins_vs_current_policy += opponent_wins
        
        # Update recent match history
        for _ in range(current_wins):
            opponent.recent_matches_vs_current.append(0)  # Current won (opponent lost)
        for _ in range(opponent_wins):
            opponent.recent_matches_vs_current.append(1)  # Opponent won
    
    def reset_pfsp_stats_for_new_policy(self):
        
        for opponent in self.opponents:
            opponent.recent_matches_vs_current.clear()
            opponent.games_vs_current_policy = 0
            opponent.wins_vs_current_policy = 0
    
    def get_leaderboard(self, top_k: Optional[int] = None) -> List[OpponentCheckpoint]:
        
        sorted_opponents = sorted(self.opponents, key=lambda x: x.elo_rating, reverse=True)
        
        if top_k is not None:
            return sorted_opponents[:top_k]
        
        return sorted_opponents
    
    def get_statistics(self) -> Dict:
        """Get statistics about opponent pool."""
        if not self.opponents:
            return {
                "pool_size": 0,
                "mean_elo": 0.0,
                "std_elo": 0.0,
                "max_elo": 0.0,
                "min_elo": 0.0,
            }
        
        ratings = [opp.elo_rating for opp in self.opponents]
        priorities = [
            self.compute_pfsp_priority(opp, self.current_training_step) 
            for opp in self.opponents
        ]
        recent_win_rates = [opp.recent_win_rate_vs_current for opp in self.opponents]
        
        return {
            "pool_size": len(self.opponents),
            "mean_elo": np.mean(ratings),
            "std_elo": np.std(ratings),
            "max_elo": np.max(ratings),
            "min_elo": np.min(ratings),
            "total_matches": sum(opp.games_played for opp in self.opponents),
            "mean_pfsp_priority": np.mean(priorities),
            "mean_win_rate_vs_current": np.mean(recent_win_rates),
            "optimal_opponents": sum(
                1 for wr in recent_win_rates 
                if self.pfsp_config.nash_target_range[0] <= wr <= self.pfsp_config.nash_target_range[1]
            ),
        }
    
    def save_pool_state(self, path: str):
        """Save opponent pool state."""
        state = {
            "opponents": [
                {
                    "path": opp.path,
                    "elo_rating": opp.elo_rating,
                    "games_played": opp.games_played,
                    "wins": opp.wins,
                    "losses": opp.losses,
                    "model_type": opp.model_type,
                }
                for opp in self.opponents
            ],
            "match_history": self.match_history,
        }
        
        torch.save(state, path)
    
    def load_pool_state(self, path: str):
        """Load opponent pool state."""
        if not os.path.exists(path):
            print(f"Pool state not found at {path}")
            return
        
        state = torch.load(path)
        
        self.opponents = [
            OpponentCheckpoint(**opp_data)
            for opp_data in state["opponents"]
        ]
        
        self.match_history = state.get("match_history", [])
        
        print(f"Loaded pool with {len(self.opponents)} opponents")
    
    def scan_checkpoint_dir(self, model_type: str = "mappo"):
        """
        Scan checkpoint directory and add new checkpoints.
        """
        pattern = os.path.join(self.checkpoint_dir, f"{model_type}*.pt")
        checkpoint_files = glob.glob(pattern)
        
        # Get existing checkpoint paths
        existing_paths = {opp.path for opp in self.opponents}
        
        # Add new checkpoints
        for checkpoint_path in checkpoint_files:
            if checkpoint_path not in existing_paths:
                self.add_checkpoint(checkpoint_path, model_type=model_type)
                print(f"Added checkpoint to pool: {checkpoint_path}")


def create_selfplay_pool(
    checkpoint_dir: str,
    model_type: str = "mappo",
    max_pool_size: int = 20,
    elo_k_factor: float = 32.0,
    temperature: float = 1.0,
    pfsp_config: Optional[PFSPConfig] = None,
) -> OpponentPool:

    pool = OpponentPool(
        checkpoint_dir=checkpoint_dir,
        max_pool_size=max_pool_size,
        elo_k_factor=elo_k_factor,
        temperature=temperature,
        pfsp_config=pfsp_config,
    )
    
    pool.scan_checkpoint_dir(model_type=model_type)
    
    pool_state_path = os.path.join(checkpoint_dir, "opponent_pool.pt")
    pool.load_pool_state(pool_state_path)
    
    return pool


if __name__ == "__main__":
    
    pool = create_selfplay_pool(
        checkpoint_dir="checkpoint",
        model_type="mappo",
        max_pool_size=20
    )
    
    print("Opponent Pool Statistics:")
    stats = pool.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nLeaderboard (Elo):")
    for i, opp in enumerate(pool.get_leaderboard(top_k=5), 1):
        pfsp_priority = pool.compute_pfsp_priority(opp, pool.current_training_step)
        print(f"  {i}. {os.path.basename(opp.path)}")
        print(f"     Elo: {opp.elo_rating:.1f} | W/L: {opp.wins}/{opp.losses} | WR: {opp.win_rate:.2%}")
        print(f"     PFSP Priority: {pfsp_priority:.3f} | Recent WR vs Current: {opp.recent_win_rate_vs_current:.2%}")
    
    print("\nPFSP-based sampling demonstration:")
    print("Sampling 10 opponents with PFSP mode:")
    sample_counts = defaultdict(int)
    for _ in range(10):
        opp = pool.sample_opponent(mode="pfsp", current_step=0)
        sample_counts[os.path.basename(opp.path)] += 1
    
    for path, count in sorted(sample_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {path}: sampled {count} times")
