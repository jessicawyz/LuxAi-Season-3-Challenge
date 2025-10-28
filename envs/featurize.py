import numpy as np
import torch
from typing import Dict, Tuple, Optional
import jax.numpy as jnp

class LuxFeaturizer:
    
    def __init__(
        self,
        map_width: int = 24,
        map_height: int = 24,
        max_units: int = 16,
        max_relic_nodes: int = 6,
        num_teams: int = 2,
    ):
        self.map_width = map_width
        self.map_height = map_height
        self.max_units = max_units
        self.max_relic_nodes = max_relic_nodes
        self.num_teams = num_teams
        
    def featurize(
        self, 
        obs: Dict, 
        team_id: int,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Convert observation dict to tensor features.
            
        Returns:
            - spatial_features: (C, H, W) spatial map features
            - unit_features: (max_units, F) per-unit features
            - unit_mask: (max_units,) mask for valid units
            - global_features: (G,) global state features
            - relic_features: (max_relic_nodes, 3) relic positions + mask
        """

        spatial_features = self._extract_spatial_features(obs, team_id)
        unit_features, unit_mask = self._extract_unit_features(obs, team_id)
        global_features = self._extract_global_features(obs, team_id)
        relic_features = self._extract_relic_features(obs)
        
        return {
            "spatial_features": torch.from_numpy(spatial_features).float().to(device),
            "unit_features": torch.from_numpy(unit_features).float().to(device),
            "unit_mask": torch.from_numpy(unit_mask).bool().to(device),
            "global_features": torch.from_numpy(global_features).float().to(device),
            "relic_features": torch.from_numpy(relic_features).float().to(device),
        }
    
    def _extract_spatial_features(self, obs: Dict, team_id: int) -> np.ndarray:
        """
        Extract spatial map features.
        
        Channels:
        0: Energy field (normalized)
        1: Tile type (empty=0, nebula=1, asteroid=2)
        2: Sensor mask (visible=1, hidden=0)
        3: Friendly unit density
        4: Enemy unit density
        5-10: Friendly unit energy levels (binned)
        11-16: Enemy unit energy levels (binned)
        17: Distance to nearest relic node
        18: Relic node presence
        19: Relic nodes collected by friendly team
        20: Relic nodes collected by enemy team
        """
        H, W = self.map_height, self.map_width
        channels = []
        
        # Channel 0
        energy = np.array(obs["map_features"]["energy"], dtype=np.float32)
        energy = np.where(energy == -1, 0, energy)  # Handle unobserved tiles
        energy = np.clip(energy / 20.0, -1.0, 1.0)  # Normalize assuming max=20
        channels.append(energy)
        
        # Channel 1
        tile_type = np.array(obs["map_features"]["tile_type"], dtype=np.float32)
        tile_type = np.where(tile_type == -1, 0, tile_type)  # Handle unobserved
        tile_type = tile_type / 2.0  # Normalize to [0, 1]
        channels.append(tile_type)
        
        # Channel 2
        sensor_mask = np.array(obs["sensor_mask"], dtype=np.float32)
        channels.append(sensor_mask)
        
        # Channels 3-4
        friendly_density = np.zeros((H, W), dtype=np.float32)
        enemy_density = np.zeros((H, W), dtype=np.float32)
        
        for t in range(self.num_teams):
            positions = np.array(obs["units"]["position"][t])
            unit_mask = np.array(obs["units_mask"][t])
            
            for i in range(self.max_units):
                if unit_mask[i]:
                    x, y = positions[i]
                    if x >= 0 and y >= 0 and x < W and y < H:
                        if t == team_id:
                            friendly_density[x, y] += 1
                        else:
                            enemy_density[x, y] += 1
        
        channels.append(friendly_density / self.max_units)  # normalize
        channels.append(enemy_density / self.max_units)
        
        # Channels 5-10
        # Channels 11-16
        energy_bins = [0, 50, 100, 150, 200, 300, 400]
        for t in range(self.num_teams):
            for bin_idx in range(len(energy_bins) - 1):
                bin_map = np.zeros((H, W), dtype=np.float32)
                positions = np.array(obs["units"]["position"][t])
                energies = np.array(obs["units"]["energy"][t])
                unit_mask = np.array(obs["units_mask"][t])
                
                for i in range(self.max_units):
                    if unit_mask[i]:
                        x, y = positions[i]
                        energy = energies[i]
                        if x >= 0 and y >= 0 and x < W and y < H and energy >= 0:
                            if energy_bins[bin_idx] <= energy < energy_bins[bin_idx + 1]:
                                bin_map[x, y] += 1
                
                channels.append(bin_map / self.max_units)
        
        # Channel 17
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        distance_map = np.ones((H, W), dtype=np.float32) * np.sqrt(H**2 + W**2)
        
        for i in range(self.max_relic_nodes):
            if relic_mask[i]:
                rx, ry = relic_positions[i]
                if rx >= 0 and ry >= 0:
                    # Compute distance field
                    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
                    dist = np.sqrt((x_grid - rx) ** 2 + (y_grid - ry) ** 2)
                    distance_map = np.minimum(distance_map, dist.T)
        
        distance_map = distance_map / np.sqrt(H**2 + W**2)  # normalize
        channels.append(distance_map)
        
        # Channel 18
        relic_map = np.zeros((H, W), dtype=np.float32)
        for i in range(self.max_relic_nodes):
            if relic_mask[i]:
                rx, ry = relic_positions[i]
                if rx >= 0 and ry >= 0 and rx < W and ry < H:
                    relic_map[rx, ry] = 1.0
        channels.append(relic_map)
        
        # Channel 19
        friendly_relic_memory = np.zeros((H, W), dtype=np.float32)
        if "relic_nodes_collected" in obs and team_id in obs["relic_nodes_collected"]:
            collected_positions = obs["relic_nodes_collected"][team_id]
            for pos in collected_positions:
                if len(pos) == 2:
                    rx, ry = pos
                    if rx >= 0 and ry >= 0 and rx < W and ry < H:
                        friendly_relic_memory[rx, ry] = 1.0
        channels.append(friendly_relic_memory)
        
        # Channel 20
        enemy_relic_memory = np.zeros((H, W), dtype=np.float32)
        enemy_team_id = 1 - team_id
        if "relic_nodes_collected" in obs and enemy_team_id in obs["relic_nodes_collected"]:
            collected_positions = obs["relic_nodes_collected"][enemy_team_id]
            for pos in collected_positions:
                if len(pos) == 2:
                    rx, ry = pos
                    if rx >= 0 and ry >= 0 and rx < W and ry < H:
                        enemy_relic_memory[rx, ry] = 1.0
        channels.append(enemy_relic_memory)
        
        # Stack all channels
        spatial_features = np.stack(channels, axis=0)
        
        return spatial_features
    
    def _extract_unit_features(self, obs: Dict, team_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract per-unit features.
        
        Features per unit:
        0-1: Position (x, y) 
        2: Energy 
        3: Is valid unit
        4-5: Distance to nearest enemy 
        6-7: Distance to nearest relic 
        8: Team ID 
        9: Unit ID 
        """
        unit_features = np.zeros((self.max_units, 10), dtype=np.float32)
        unit_mask = np.zeros(self.max_units, dtype=bool)
        
        positions = np.array(obs["units"]["position"][team_id])
        energies = np.array(obs["units"]["energy"][team_id])
        units_mask = np.array(obs["units_mask"][team_id])
        
        # Get enemy positions 
        enemy_team_id = 1 - team_id
        enemy_positions = np.array(obs["units"]["position"][enemy_team_id])
        enemy_mask = np.array(obs["units_mask"][enemy_team_id])
        valid_enemy_positions = enemy_positions[enemy_mask]
        
        # Get relic positions
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        valid_relic_positions = relic_positions[relic_mask]
        
        for i in range(self.max_units):
            if units_mask[i]:
                unit_mask[i] = True
                x, y = positions[i]
                energy = energies[i]
                
                # Position normalized
                unit_features[i, 0] = x / self.map_width
                unit_features[i, 1] = y / self.map_height
                
                # Energy normalized
                unit_features[i, 2] = np.clip(energy / 400.0, -0.5, 1.0)
                
                # Valid flag
                unit_features[i, 3] = 1.0
                
                # Distance to nearest enemy
                if len(valid_enemy_positions) > 0 and x >= 0 and y >= 0:
                    distances = np.sqrt(
                        (valid_enemy_positions[:, 0] - x) ** 2 +
                        (valid_enemy_positions[:, 1] - y) ** 2
                    )
                    min_dist = np.min(distances)
                    unit_features[i, 4] = min_dist / np.sqrt(self.map_width**2 + self.map_height**2)
                    
                    # Direction to nearest enemy (as normalized dx, dy)
                    nearest_idx = np.argmin(distances)
                    dx = valid_enemy_positions[nearest_idx, 0] - x
                    dy = valid_enemy_positions[nearest_idx, 1] - y
                    unit_features[i, 5] = np.arctan2(dy, dx) / np.pi 
                else:
                    unit_features[i, 4] = 1.0
                    unit_features[i, 5] = 0.0
                
                # Distance to nearest relic
                if len(valid_relic_positions) > 0 and x >= 0 and y >= 0:
                    distances = np.sqrt(
                        (valid_relic_positions[:, 0] - x) ** 2 +
                        (valid_relic_positions[:, 1] - y) ** 2
                    )
                    min_dist = np.min(distances)
                    unit_features[i, 6] = min_dist / np.sqrt(self.map_width**2 + self.map_height**2)
                    
                    nearest_idx = np.argmin(distances)
                    dx = valid_relic_positions[nearest_idx, 0] - x
                    dy = valid_relic_positions[nearest_idx, 1] - y
                    unit_features[i, 7] = np.arctan2(dy, dx) / np.pi
                else:
                    unit_features[i, 6] = 1.0
                    unit_features[i, 7] = 0.0
                
                # Team ID
                unit_features[i, 8] = float(team_id)
                
                # Unit ID normalized
                unit_features[i, 9] = i / self.max_units
        
        return unit_features, unit_mask
    
    def _extract_global_features(self, obs: Dict, team_id: int) -> np.ndarray:
        """
        Extract global state features.
        
        Features:
            - Current match step
            - Current episode step
            - Team points
            - Enemy points
            - Team wins
            - Enemy wins
            - Team unit count
            - Enemy unit count
            - Team total energy
            - Enemy total energy
            - Number of visible relic nodes
            - Match number in episode
        """
        global_features = np.zeros(12, dtype=np.float32)
        
        # Time features
        match_steps = obs.get("match_steps", 0)
        total_steps = obs.get("steps", 0)
        max_match_steps = 100 
        max_episode_steps = 500 
        
        global_features[0] = match_steps / max_match_steps
        global_features[1] = total_steps / max_episode_steps
        
        # Score features
        team_points = np.array(obs["team_points"])
        global_features[2] = team_points[team_id] / 100.0 
        global_features[3] = team_points[1 - team_id] / 100.0
        
        # Win count
        team_wins = np.array(obs["team_wins"])
        global_features[4] = team_wins[team_id] / 5.0
        global_features[5] = team_wins[1 - team_id] / 5.0
        
        # Unit counts
        team_unit_count = np.sum(obs["units_mask"][team_id])
        enemy_unit_count = np.sum(obs["units_mask"][1 - team_id])
        global_features[6] = team_unit_count / self.max_units
        global_features[7] = enemy_unit_count / self.max_units
        
        # Total energy
        team_energies = np.array(obs["units"]["energy"][team_id])
        team_mask = np.array(obs["units_mask"][team_id])
        team_total_energy = np.sum(team_energies[team_mask])
        
        enemy_energies = np.array(obs["units"]["energy"][1 - team_id])
        enemy_mask = np.array(obs["units_mask"][1 - team_id])
        enemy_total_energy = np.sum(enemy_energies[enemy_mask])
        
        global_features[8] = team_total_energy / (self.max_units * 400)
        global_features[9] = enemy_total_energy / (self.max_units * 400)
        
        # Relic node visibility
        relic_mask = np.array(obs["relic_nodes_mask"])
        visible_relics = np.sum(relic_mask)
        global_features[10] = visible_relics / self.max_relic_nodes
        
        # Match number in episode 
        match_number = total_steps // max_match_steps
        global_features[11] = match_number / 5.0
        
        return global_features
    
    def _extract_relic_features(self, obs: Dict) -> np.ndarray:
        """
        Extract relic node features
        """
        relic_features = np.zeros((self.max_relic_nodes, 3), dtype=np.float32)
        
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        
        for i in range(self.max_relic_nodes):
            if relic_mask[i]:
                x, y = relic_positions[i]
                if x >= 0 and y >= 0:
                    relic_features[i, 0] = x / self.map_width
                    relic_features[i, 1] = y / self.map_height
                    relic_features[i, 2] = 1.0
        
        return relic_features
    
    def _to_numpy(self, obj):
        
        if isinstance(obj, dict):
            return {k: self._to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._to_numpy(item) for item in obj)
        elif isinstance(obj, jnp.ndarray):
            return np.array(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        else:
            return obj


def create_ctde_observation(
    obs_dict: Dict,
    featurizer: LuxFeaturizer,
    device: str = "cpu"
) -> Dict[str, Dict[str, torch.Tensor]]:
    
    team_0_features = featurizer.featurize(obs_dict, team_id=0, device=device)
    team_1_features = featurizer.featurize(obs_dict, team_id=1, device=device)
    
    # Global features combine both teams spatial features
    global_spatial = torch.cat([
        team_0_features["spatial_features"],
        team_1_features["spatial_features"]
    ], dim=0)  # (2*C, H, W)
    
    global_scalars = torch.cat([
        team_0_features["global_features"],
        team_1_features["global_features"]
    ], dim=0)
    
    return {
        "team_0": team_0_features,
        "team_1": team_1_features,
        "global": {
            "spatial_features": global_spatial,
            "global_features": global_scalars,
            "team_0_units": team_0_features["unit_features"],
            "team_1_units": team_1_features["unit_features"],
            "team_0_unit_mask": team_0_features["unit_mask"],
            "team_1_unit_mask": team_1_features["unit_mask"],
        }
    }