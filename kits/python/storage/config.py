"""
Working Configuration - Based on successful approaches
Key: Simple, proven hyperparameters
"""
from dataclasses import dataclass

@dataclass
class Config:
    # Training - Proven stable values
    n_episodes: int = 5000
    learning_rate: float = 3e-4
    gamma: float = 0.999  # Not 0.9999 - that's too high for 100-step matches
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    batch_size: int = 512
    ppo_epochs: int = 4
    mini_batch_size: int = 128
    
    # Model - Proven architecture from working approach
    hidden_dim: int = 256
    n_main_actions: int = 6  # NO_OP, 4 moves, SAP
    cnn_blocks: int = 8
    
    # Feature Engineering - SIMPLE temporal
    temporal_history_len: int = 3  # Short history, not 10
    n_base_spatial: int = 16  # Base channels
    n_global_features: int = 20  # Global features
    
    # Environment
    map_size: int = 24
    max_units: int = 16
    max_sap_range: int = 7
    max_steps_per_match: int = 500
    matches_per_game: int = 5
    
    # Energy costs (from working approach)
    unit_move_cost: int = 1
    unit_sap_cost: int = 50
    
    # Reward - SPARSE by default
    use_sparse_rewards: bool = True
    match_win_reward: float = 1.0
    match_loss_reward: float = -1.0