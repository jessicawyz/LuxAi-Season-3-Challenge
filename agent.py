import sys
import os
import numpy as np
import torch

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.featurize import LuxFeaturizer


class Agent():
    """
    Agent for Lux AI Season 3 that loads trained neural network checkpoint.
    
    Compatible with both MADDPG and MAPPO models.
    """
    
    def __init__(self, player: str, env_cfg, model_name: str = "maddpg", checkpoint_path: str = None) -> None:
        """
        Initialize agent.
        
        Args:
            player: "player_0" or "player_1"
            env_cfg: Environment configuration dict
            model_name: "maddpg" or "mappo"
            checkpoint_path: Path to checkpoint file. If None, uses default path.
        """
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.model_name = model_name
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path()
        
        # print(f"[Agent] Loading {model_name} checkpoint from {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model based on type
        self.actor = self._load_model()
        self.actor.eval()  # Set to evaluation mode
        
        # Initialize featurizer
        self.featurizer = LuxFeaturizer(
            map_width=env_cfg.get("map_width", 24),
            map_height=env_cfg.get("map_height", 24),
            max_units=env_cfg.get("max_units", 16),
            max_relic_nodes=env_cfg.get("max_relic_nodes", 6),
            num_teams=2,
        )
        
        # State for hidden states (if using recurrent models)
        self.spatial_hidden_state = None
        
        # print(f"[Agent] Successfully loaded {model_name} model on {self.device}")
    
    def _get_default_checkpoint_path(self):
        """Get default checkpoint path based on model name"""
        # Look in checkpoint directory relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_dir = os.path.join(base_dir, "python", "checkpoint")
        
        # Try to find best checkpoint
        best_path = os.path.join(checkpoint_dir, f"{self.model_name}_best.pt")
        if os.path.exists(best_path):
            return best_path
        
        # Fallback to latest
        latest_path = os.path.join(checkpoint_dir, f"{self.model_name}_latest.pt")
        if os.path.exists(latest_path):
            return latest_path
        
        raise FileNotFoundError(
            f"No checkpoint found for {self.model_name}. "
            f"Looked in {checkpoint_dir} for {self.model_name}_best.pt or {self.model_name}_latest.pt"
        )
    
    def _load_model(self):
        """Load model based on model type"""
        if self.model_name == "maddpg":
            return self._load_maddpg_model()
        elif self.model_name == "mappo":
            return self._load_mappo_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
    
    def _load_maddpg_model(self):
        """Load MADDPG actor"""
        from maddpg import MADDPGActor, MADDPGConfig
        
        # Create config from checkpoint or use default
        if "config" in self.checkpoint:
            config_dict = self.checkpoint["config"]
            config = MADDPGConfig(**config_dict)
        else:
            config = MADDPGConfig()
        
        # Create actor
        actor = MADDPGActor(config).to(self.device)
        
        # Load weights for the appropriate team
        if self.team_id == 0:
            actor.load_state_dict(self.checkpoint["actor_0"])
        else:
            actor.load_state_dict(self.checkpoint["actor_1"])
        
        return actor
    
    def _load_mappo_model(self):
        """Load MAPPO actor"""
        # TODO: Implement when MAPPO is ready
        raise NotImplementedError("MAPPO loading not yet implemented")
    
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Decide what actions to send to each available unit.
        
        Args:
            step: Current timestep number
            obs: Observation dict from environment
            remainingOverageTime: Remaining overtime in seconds
            
        Returns:
            actions: (max_units, 3) numpy array of actions
        """
        # Convert observations to model format
        features = self.featurizer.featurize(obs, self.team_id, device=self.device)
        
        # Add batch dimension
        spatial_features = features["spatial_features"].unsqueeze(0)  # (1, C, H, W)
        unit_features = features["unit_features"].unsqueeze(0)  # (1, max_units, F)
        unit_mask = features["unit_mask"].unsqueeze(0)  # (1, max_units)
        global_features = features["global_features"].unsqueeze(0)  # (1, G)
        
        # Extract unit energies and positions for action masking
        unit_positions_raw = np.array(obs["units"]["position"][self.team_id])
        unit_energies_raw = np.array(obs["units"]["energy"][self.team_id]).flatten()
        
        unit_positions = torch.from_numpy(unit_positions_raw).long().unsqueeze(0).to(self.device)  # (1, max_units, 2)
        unit_energies = torch.from_numpy(unit_energies_raw).float().unsqueeze(0).to(self.device)  # (1, max_units)
        
        # Get tile types for asteroid blocking
        tile_types_raw = np.array(obs["map_features"]["tile_type"])
        tile_types = torch.from_numpy(tile_types_raw).float().to(self.device)  # (H, W)
        tile_types = tile_types.unsqueeze(0)  # (1, H, W)

        # Get actions from model
        with torch.no_grad():
            actions_tensor, self.spatial_hidden_state, _ = self.actor(
                spatial_features=spatial_features,
                unit_features=unit_features,
                unit_mask=unit_mask,
                global_features=global_features,
                unit_energies=unit_energies,
                unit_positions=unit_positions,
                tile_types=tile_types,
                spatial_hidden_state=self.spatial_hidden_state,
                epsilon=0.5  # Deterministic action selection
            )
        
        # Convert to numpy
        actions = actions_tensor.squeeze(0).cpu().numpy()  # (max_units, 3)
        
        # Ensure correct dtype
        actions = actions.astype(np.int32)
        
        return actions
    
    def reset(self):
        """Reset agent state (e.g., between matches)"""
        self.spatial_hidden_state = None


# Backwards compatibility: Allow specifying model_name via environment variable or default
def create_agent(player: str, env_cfg):
    """
    Factory function to create agent with model specified via environment or default.
    
    Usage:
        export MODEL_NAME=maddpg
        agent = create_agent("player_0", env_cfg)
    """
    model_name = os.environ.get("MODEL_NAME", "maddpg")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)
    return Agent(player, env_cfg, model_name=model_name, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    # Test agent loading
    # print("Testing agent loading...")
    
    # Create dummy env_cfg
    env_cfg = {
        "map_width": 24,
        "map_height": 24,
        "max_units": 16,
        "max_relic_nodes": 6,
    }
    
    # Try to create agent
    try:
        agent = Agent("player_0", env_cfg, model_name="maddpg")
        # print("✓ Agent loaded successfully")
        # print(f"✓ Device: {agent.device}")
        # print(f"✓ Team ID: {agent.team_id}")
    except Exception as e:
        # print(f"✗ Failed to load agent: {e}")
        import traceback
        traceback.print_exc()