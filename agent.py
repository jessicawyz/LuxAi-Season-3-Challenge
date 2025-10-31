import os
import numpy as np
import torch

from envs.featurize import LuxFeaturizer


class Agent():
    
    def __init__(self, player: str, env_cfg, model_name: str = "maddpg", checkpoint_path: str = None) -> None:
        
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
        
        # Load model
        self.actor = self._load_model()
        self.actor.eval()  
        
        # Initialize featurizer
        self.featurizer = LuxFeaturizer(
            map_width=env_cfg.get("map_width", 24),
            map_height=env_cfg.get("map_height", 24),
            max_units=env_cfg.get("max_units", 16),
            max_relic_nodes=env_cfg.get("max_relic_nodes", 6),
            num_teams=2,
        )
        
        self.spatial_hidden_state = None
        
        # Relic memory: track seen relic positions
        map_h = env_cfg.get("map_height", 24)
        map_w = env_cfg.get("map_width", 24)
        self.friendly_relic_memory = torch.zeros((map_h, map_w), dtype=torch.float32, device=self.device)
        self.enemy_relic_memory = torch.zeros((map_h, map_w), dtype=torch.float32, device=self.device)
        
        # print(f"[Agent] Successfully loaded {model_name} model on {self.device}")
    
    def _get_default_checkpoint_path(self):
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_dir = os.path.join(base_dir, "python", "checkpoint")
        
        # Try to find best checkpoint
        best_path = os.path.join(checkpoint_dir, f"{self.model_name}_best.pt")
        if os.path.exists(best_path):
            return best_path
        
        # Fallback to latest checkpoint
        latest_path = os.path.join(checkpoint_dir, f"{self.model_name}_latest.pt")
        if os.path.exists(latest_path):
            return latest_path
        
        raise FileNotFoundError(
            f"No checkpoint found for {self.model_name}. "
            f"Looked in {checkpoint_dir} for {self.model_name}_best.pt or {self.model_name}_latest.pt"
        )
    
    def _load_model(self):
        if self.model_name == 'maddpg':
            from maddpg import MADDPGActor as MARLActor
            from maddpg import MADDPGConfig as MARLConfig
        elif self.model_name == 'mappo':
            from mappo import MAPPOActor as MARLActor
            from mappo import MAPPOConfig as MARLConfig
        else:
            raise Exception("MARL algorithm not found.")
        
        # Create config 
        if "config" in self.checkpoint:
            config_dict = self.checkpoint["config"]
            config = MARLConfig(**config_dict)
        else:
            config = MARLConfig()
        
        # Create actor
        actor = MARLActor(config).to(self.device)
        
        # Load weights 
        if self.team_id == 0:
            actor.load_state_dict(self.checkpoint["actor_0"])
        else:
            actor.load_state_dict(self.checkpoint["actor_1"])
        
        return actor
    
    def act(self, step, obs, remainingOverageTime):
        # Convert observations to model format
        features = self.featurizer.featurize(obs, self.team_id, device=self.device)
        
        # Update relic memory with currently visible relics
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        sensor_mask = np.array(obs["sensor_mask"]) 

        for i in range(len(relic_mask)):
            if relic_mask[i]:
                rx, ry = relic_positions[i]
                if 0 <= rx < self.friendly_relic_memory.shape[1] and 0 <= ry < self.friendly_relic_memory.shape[0]:
                    if sensor_mask[rx, ry]: 
                        self.friendly_relic_memory[rx, ry] = 1.0
        
        # Features
        spatial_features = features["spatial_features"].unsqueeze(0)  # (1, 21, H, W)
        unit_features = features["unit_features"].unsqueeze(0)  # (1, max_units, F)
        unit_mask = features["unit_mask"].unsqueeze(0)  # (1, max_units)
        global_features = features["global_features"].unsqueeze(0)  # (1, G)
        
        # Add relic memory channels to spatial features
        friendly_memory_channel = self.friendly_relic_memory.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        enemy_memory_channel = self.enemy_relic_memory.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        spatial_features = torch.cat([
            spatial_features, 
            friendly_memory_channel, 
            enemy_memory_channel
        ], dim=1)  # (1, 23, H, W)
        
        # Extract unit energies and positions
        unit_positions_raw = np.array(obs["units"]["position"][self.team_id])
        unit_energies_raw = np.array(obs["units"]["energy"][self.team_id]).flatten()
        
        unit_positions = torch.from_numpy(unit_positions_raw).long().unsqueeze(0).to(self.device)  # (1, max_units, 2)
        unit_energies = torch.from_numpy(unit_energies_raw).float().unsqueeze(0).to(self.device)  # (1, max_units)
        
        # Get tile types
        tile_types_raw = np.array(obs["map_features"]["tile_type"])
        tile_types = torch.from_numpy(tile_types_raw).float().to(self.device)  # (H, W)
        tile_types = tile_types.unsqueeze(0)  # (1, H, W)

        # Get actions
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
        actions = actions_tensor.squeeze(0).cpu().numpy()  # (max_units, 3)
        
        return actions
    
    def reset(self):
        self.spatial_hidden_state = None
        self.friendly_relic_memory.zero_()
        self.enemy_relic_memory.zero_()