from dataclasses import dataclass

@dataclass
class Config:
    # Training
    n_episodes: int = 50
    learning_rate: float = 1e-5  # Reduced for stability
    gamma: float = 0.9999
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.1  # Increased for more exploration
    value_coef: float = 0.1
    batch_size: int = 64  # Reduced from 128 for faster updates
    
    # MAPPO specific
    buffer_size: int = 50000  # Large enough to hold multiple matches
    shared_model: bool = True  # Whether both players share the same model
    matches_per_update: int = 1  # ✅ NEW: Update policy every 3 matches
    n_ppo_epochs: int = 4  # Number of PPO epochs per update
    
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
    max_steps_per_match: int = 500
    init_unit_energy = 100
    matches_per_game: int = 5  # Matches per episode
    spawn_rate = 3
    
    # Reward - Updated for LuxRewardShaper
    use_sparse_rewards: bool = False  # Use sparse (win/loss only) or dense rewards
    match_win_reward: float = 100.0  # Increased for LuxRewardShaper compatibility
    match_loss_reward: float = -100.0  # Increased for LuxRewardShaper compatibility
    
    # LuxRewardShaper specific parameters
    reward_mode: str = "dense"  # "sparse" or "dense"
    episode_win_bonus: float = 500.0  # Bonus for winning entire episode
    relic_point_reward: float = 1.0
    damage_dealt_reward: float = 0.01
    energy_differential_reward: float = 0.001
    unit_loss_penalty: float = -5.0
    survival_reward: float = 0.05
    exploration_reward: float = 0.5
    relic_discovery_reward: float = 10.0