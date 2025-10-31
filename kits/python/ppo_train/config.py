from dataclasses import dataclass

@dataclass
class Config:
    # Training
    n_episodes: int = 3000
    learning_rate: float = 1e-4  # Reduced for stability
    gamma: float = 0.9999
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.05  # Increased for more exploration
    value_coef: float = 0.5
    batch_size: int = 1024  # Increased for more stable updates
    
    # Model - Scaled up like Frog Parade
    hidden_dim: int = 256  # Frog Parade used 256
    n_main_actions: int = 6  # NO_OP, 4 moves, SAP
    n_sap_targets: int = 15  # Max SAP range is 7, so 15x15 grid
    cnn_blocks: int = 8  # Frog Parade used 8 ResNet blocks
    
    # Feature Engineering
    temporal_history_len: int = 5  # Track last 5 frames (Frog Parade used 10)
    n_spatial_channels: int = 16  # Base channels before temporal stacking
    n_global_features: int = 23  # Expanded from 8
    
    # Environment
    map_size: int = 24
    max_units: int = 16
    max_sap_range: int = 7
    max_steps_per_match: int = 100
    init_unit_energy = 100
    matches_per_game: int = 5
    spawn_rate = 3
    
    # Reward
    use_sparse_rewards: bool = True  # CRITICAL: Use win/loss only
    match_win_reward: float = 1.0
    match_loss_reward: float = -1.0