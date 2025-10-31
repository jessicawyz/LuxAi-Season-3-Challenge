import numpy as np
from collections import deque
from action_masking import get_valid_actions

class AdvancedFeatureEngineer:
    """
    Enhanced feature engineer with proper JAX EnvObs handling
    """
    def __init__(self, map_size=24, max_units=16, history_len=5):
        self.map_size = map_size
        self.max_units = max_units
        self.history_len = history_len
        
        # Temporal histories
        self.spatial_history = deque(maxlen=history_len)
        self.global_history = deque(maxlen=history_len)
        
        # Inferred knowledge (persistent across steps)
        self.known_point_tiles = set()
        self.known_non_point_tiles = set()
        self.known_relic_nodes = set()
        self.known_asteroids = set()
        self.known_nebula = set()
        self.explored_tiles = set()
        
        # Previous state tracking for inference
        self.prev_score = {'player_0': 0, 'player_1': 0}
        self.prev_unit_positions = {'player_0': [], 'player_1': []}
    
    def _safe_get(self, obs, key, default=None):
        """Safely get value from EnvObs or dict"""
        try:
            if hasattr(obs, key):
                return getattr(obs, key)
            elif isinstance(obs, dict) and key in obs:
                return obs[key]
            else:
                return default
        except:
            return default
    
    def process(self, obs, player_id=0):
        """Process observation with advanced feature engineering"""
        player_key = f'player_{player_id}'
        
        # Extract current frame features (16 channels)
        current_spatial = self._build_spatial_features(obs, player_id)
        current_global = self._build_global_features(obs, player_id)
        
        # Infer knowledge from observations
        self._infer_point_tiles(obs, player_key)
        self._update_explored_tiles(obs, player_id)
        
        # Add to temporal history
        self.spatial_history.append(current_spatial)
        self.global_history.append(current_global)
        
        # Build temporal features
        spatial_temporal = self._stack_temporal_features(self.spatial_history, current_spatial)
        global_temporal = self._stack_temporal_global(self.global_history, current_global)
        
        # Get unit info and action masks
        unit_info = self._get_unit_info(obs, player_id)
        action_masks = self._build_action_masks(obs, player_id, unit_info)
        
        return {
            'spatial': spatial_temporal,
            'global': global_temporal,
            'unit_info': unit_info,
            'action_masks': action_masks,
            'known_point_tiles': list(self.known_point_tiles),
            'known_relic_nodes': list(self.known_relic_nodes)
        }
    
    def _build_spatial_features(self, obs, player_id):
        """Build 16 spatial channels with ALL inferred knowledge"""
        spatial = np.zeros((16, self.map_size, self.map_size), dtype=np.float32)
        
        # Basic observations
        spatial[0] = self._get_unit_positions(obs, player_id)
        spatial[1] = self._get_unit_positions(obs, 1 - player_id)
        spatial[2] = self._get_relic_positions(obs)
        spatial[3] = self._get_energy_field(obs)
        spatial[4] = self._get_asteroid_map(obs)
        spatial[5] = self._get_nebula_map(obs)
        spatial[6] = self._get_known_point_tiles_map()
        spatial[7] = self._get_unit_energy_map(obs, player_id)
        spatial[8] = self._get_unit_energy_map(obs, 1 - player_id)
        spatial[9] = self._get_explored_tiles_map()
        spatial[10] = self._get_distance_to_center_map()
        spatial[11] = self._get_distance_to_nearest_relic_map()
        
        # Strategic inferred knowledge
        spatial[12] = self._get_known_relics_map()
        spatial[13] = self._get_point_tile_certainty_map()
        spatial[14] = self._get_unexplored_priority_map()
        spatial[15] = self._get_strategic_priority_map()
    
        return spatial
    
    def _get_unit_positions(self, obs, player_id):
        """Extract unit positions - handles JAX EnvObs"""
        unit_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        # Try to get units data
        units = self._safe_get(obs, 'units')
        if units is None:
            return unit_map
        
        # Get positions
        positions = self._safe_get(units, 'position')
        if positions is None:
            return unit_map
        
        # Get mask
        units_mask = self._safe_get(obs, 'units_mask')
        
        # Handle different data structures
        try:
            if positions is not None:
                # Convert JAX array to numpy if needed
                if hasattr(positions, '__array__'):
                    positions = np.array(positions)
                
                # Get player's positions
                if len(positions.shape) >= 2 and positions.shape[0] > player_id:
                    player_positions = positions[player_id]
                    
                    # Get mask if available
                    if units_mask is not None:
                        if hasattr(units_mask, '__array__'):
                            units_mask = np.array(units_mask)
                        if len(units_mask.shape) >= 1 and units_mask.shape[0] > player_id:
                            mask = units_mask[player_id]
                        else:
                            mask = np.ones(len(player_positions), dtype=bool)
                    else:
                        mask = np.ones(len(player_positions), dtype=bool)
                    
                    # Fill unit map
                    for pos, valid in zip(player_positions, mask):
                        if valid and len(pos) >= 2:
                            x, y = int(pos[0]), int(pos[1])
                            if 0 <= x < self.map_size and 0 <= y < self.map_size and x >= 0 and y >= 0:
                                unit_map[y, x] = 1.0
        except Exception as e:
            print(f"Warning: Error getting unit positions: {e}")
        
        return unit_map
    
    def _get_unit_energy_map(self, obs, player_id):
        """Extract unit energy as spatial map"""
        energy_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        units = self._safe_get(obs, 'units')
        if units is None:
            return energy_map
        
        positions = self._safe_get(units, 'position')
        energies = self._safe_get(units, 'energy')
        units_mask = self._safe_get(obs, 'units_mask')
        
        if positions is None or energies is None:
            return energy_map
        
        try:
            # Convert to numpy
            if hasattr(positions, '__array__'):
                positions = np.array(positions)
            if hasattr(energies, '__array__'):
                energies = np.array(energies)
            if units_mask is not None and hasattr(units_mask, '__array__'):
                units_mask = np.array(units_mask)
            
            # Get player data
            if len(positions.shape) >= 2 and positions.shape[0] > player_id:
                player_positions = positions[player_id]
                player_energies = energies[player_id]
                
                if units_mask is not None and len(units_mask.shape) >= 1:
                    mask = units_mask[player_id]
                else:
                    mask = np.ones(len(player_positions), dtype=bool)
                
                # Fill energy map
                for pos, energy, valid in zip(player_positions, player_energies, mask):
                    if valid and len(pos) >= 2:
                        x, y = int(pos[0]), int(pos[1])
                        if 0 <= x < self.map_size and 0 <= y < self.map_size and x >= 0 and y >= 0:
                            energy_map[y, x] = float(energy) / 400.0  # Normalize
        except Exception as e:
            print(f"Warning: Error getting unit energy: {e}")
        
        return energy_map
    
    def _get_relic_positions(self, obs):
        """Extract relic positions"""
        relic_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        relics = self._safe_get(obs, 'relic_nodes')
        mask = self._safe_get(obs, 'relic_nodes_mask')
        
        if relics is None:
            return relic_map
        
        try:
            if hasattr(relics, '__array__'):
                relics = np.array(relics)
            if mask is not None and hasattr(mask, '__array__'):
                mask = np.array(mask)
            
            if mask is None:
                mask = np.ones(len(relics), dtype=bool)
            
            for relic_pos, valid in zip(relics, mask):
                if valid and len(relic_pos) >= 2:
                    x, y = int(relic_pos[0]), int(relic_pos[1])
                    if 0 <= x < self.map_size and 0 <= y < self.map_size and x >= 0 and y >= 0:
                        relic_map[y, x] = 1.0
                        self.known_relic_nodes.add((x, y))
        except Exception as e:
            print(f"Warning: Error getting relic positions: {e}")
        
        return relic_map
    
    def _get_energy_field(self, obs):
        """Extract energy field from map"""
        energy_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        map_features = self._safe_get(obs, 'map_features')
        if map_features is None:
            return energy_map
        
        energy_data = self._safe_get(map_features, 'energy')
        if energy_data is None:
            return energy_map
        
        try:
            if hasattr(energy_data, '__array__'):
                energy_data = np.array(energy_data)
            
            for y in range(min(self.map_size, energy_data.shape[0])):
                for x in range(min(self.map_size, energy_data.shape[1] if len(energy_data.shape) > 1 else 0)):
                    val = energy_data[y][x] if len(energy_data.shape) > 1 else energy_data[y]
                    if val != -1.0:
                        energy_map[y, x] = np.clip(float(val) / 100.0, 0, 1)
        except Exception as e:
            print(f"Warning: Error getting energy field: {e}")
        
        return energy_map
    
    def _get_asteroid_map(self, obs):
        """Extract asteroid positions"""
        asteroid_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        map_features = self._safe_get(obs, 'map_features')
        if map_features is None:
            return asteroid_map
        
        tile_data = self._safe_get(map_features, 'tile_type')
        if tile_data is None:
            return asteroid_map
        
        try:
            if hasattr(tile_data, '__array__'):
                tile_data = np.array(tile_data)
            
            for y in range(min(self.map_size, tile_data.shape[0])):
                for x in range(min(self.map_size, tile_data.shape[1] if len(tile_data.shape) > 1 else 0)):
                    val = tile_data[y][x] if len(tile_data.shape) > 1 else tile_data[y]
                    if val == 1:  # Asteroid
                        asteroid_map[y, x] = 1.0
                        self.known_asteroids.add((x, y))
        except Exception as e:
            print(f"Warning: Error getting asteroids: {e}")
        
        return asteroid_map
    
    def _get_nebula_map(self, obs):
        """Extract nebula positions"""
        nebula_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        map_features = self._safe_get(obs, 'map_features')
        if map_features is None:
            return nebula_map
        
        tile_data = self._safe_get(map_features, 'tile_type')
        if tile_data is None:
            return nebula_map
        
        try:
            if hasattr(tile_data, '__array__'):
                tile_data = np.array(tile_data)
            
            for y in range(min(self.map_size, tile_data.shape[0])):
                for x in range(min(self.map_size, tile_data.shape[1] if len(tile_data.shape) > 1 else 0)):
                    val = tile_data[y][x] if len(tile_data.shape) > 1 else tile_data[y]
                    if val == 2:  # Nebula
                        nebula_map[y, x] = 1.0
                        self.known_nebula.add((x, y))
        except Exception as e:
            print(f"Warning: Error getting nebula: {e}")
        
        return nebula_map
    
    def _build_global_features(self, obs, player_id):
        """Build comprehensive global feature vector"""
        features = []
        
        # Score information (3 features)
        team_points = self._safe_get(obs, 'team_points')
        if team_points is not None:
            if hasattr(team_points, '__array__'):
                team_points = np.array(team_points)
            my_score = float(team_points[player_id]) if len(team_points) > player_id else 0.0
            opp_score = float(team_points[1 - player_id]) if len(team_points) > (1 - player_id) else 0.0
            score_diff = my_score - opp_score
            features.extend([my_score / 100.0, opp_score / 100.0, score_diff / 100.0])
        else:
            features.extend([0, 0, 0])
        
        # Step information (2 features)
        steps = self._safe_get(obs, 'steps', 0)
        match_steps = self._safe_get(obs, 'match_steps', 0)
        features.extend([float(steps) / 100.0, float(match_steps) / 100.0])
        
        # Unit counts (3 features)
        units_mask = self._safe_get(obs, 'units_mask')
        if units_mask is not None:
            if hasattr(units_mask, '__array__'):
                units_mask = np.array(units_mask)
            my_units = float(np.sum(units_mask[player_id])) if len(units_mask) > player_id else 0.0
            opp_units = float(np.sum(units_mask[1 - player_id])) if len(units_mask) > (1 - player_id) else 0.0
        else:
            my_units = 0.0
            opp_units = 0.0
        unit_diff = my_units - opp_units
        features.extend([my_units / 16.0, opp_units / 16.0, unit_diff / 16.0])
        
        # Energy information (3 features)
        my_energy = self._get_total_energy(obs, player_id) / 6400.0
        opp_energy = self._get_total_energy(obs, 1 - player_id) / 6400.0
        energy_diff = my_energy - opp_energy
        features.extend([my_energy, opp_energy, energy_diff])
        
        # Map knowledge (10 features)
        total_tiles = self.map_size * self.map_size
        known_point_ratio = len(self.known_point_tiles) / total_tiles
        known_relic_ratio = len(self.known_relic_nodes) / 10.0
        exploration_ratio = len(self.explored_tiles) / total_tiles
        known_relic_count = len(self.known_relic_nodes) / 10.0
        unexplored_ratio = 1.0 - exploration_ratio
        
        # Simplified additional features
        high_confidence_points = 0.0
        strategic_balance = 0.5
        relic_accessibility = 0.5
        exploration_efficiency = exploration_ratio
        inference_confidence = 0.5
        
        features.extend([
            known_point_ratio, known_relic_ratio, exploration_ratio, known_relic_count,
            unexplored_ratio, high_confidence_points, strategic_balance, 
            relic_accessibility, exploration_efficiency, inference_confidence
        ])
        
        # Match wins (2 features)
        match_wins = self._safe_get(obs, 'match_wins')
        if match_wins is not None:
            if hasattr(match_wins, '__array__'):
                match_wins = np.array(match_wins)
            my_wins = float(match_wins[player_id]) / 3.0 if len(match_wins) > player_id else 0.0
            opp_wins = float(match_wins[1 - player_id]) / 3.0 if len(match_wins) > (1 - player_id) else 0.0
            features.extend([my_wins, opp_wins])
        else:
            features.extend([0, 0])
        
        # Pad to expected size if needed
        while len(features) < 115:
            features.append(0.0)
        
        return np.array(features[:115], dtype=np.float32)
    
    def _get_total_energy(self, obs, player_id):
        """Get total energy for a player"""
        units = self._safe_get(obs, 'units')
        if units is None:
            return 0.0
        
        energies = self._safe_get(units, 'energy')
        units_mask = self._safe_get(obs, 'units_mask')
        
        if energies is None:
            return 0.0
        
        try:
            if hasattr(energies, '__array__'):
                energies = np.array(energies)
            if units_mask is not None and hasattr(units_mask, '__array__'):
                units_mask = np.array(units_mask)
            
            if len(energies.shape) >= 2 and energies.shape[0] > player_id:
                player_energies = energies[player_id]
                
                if units_mask is not None and len(units_mask.shape) >= 1:
                    mask = units_mask[player_id]
                else:
                    mask = np.ones(len(player_energies), dtype=bool)
                
                total = 0.0
                for energy, valid in zip(player_energies, mask):
                    if valid and energy > 0:
                        total += float(energy)
                
                return total
        except Exception as e:
            print(f"Warning: Error getting total energy: {e}")
        
        return 0.0
    
    def _get_unit_info(self, obs, player_id):
        """Extract unit information"""
        unit_info = []
        
        units = self._safe_get(obs, 'units')
        if units is None:
            return unit_info
        
        positions = self._safe_get(units, 'position')
        energies = self._safe_get(units, 'energy')
        units_mask = self._safe_get(obs, 'units_mask')
        
        if positions is None or energies is None:
            return unit_info
        
        try:
            # Convert to numpy
            if hasattr(positions, '__array__'):
                positions = np.array(positions)
            if hasattr(energies, '__array__'):
                energies = np.array(energies)
            if units_mask is not None and hasattr(units_mask, '__array__'):
                units_mask = np.array(units_mask)
            
            # Get player data
            if len(positions.shape) >= 2 and positions.shape[0] > player_id:
                player_positions = positions[player_id]
                player_energies = energies[player_id]
                
                if units_mask is not None and len(units_mask.shape) >= 1:
                    mask = units_mask[player_id]
                else:
                    mask = np.ones(len(player_positions), dtype=bool)
                
                for i, (pos, energy, valid) in enumerate(zip(player_positions, player_energies, mask)):
                    if valid and len(pos) >= 2:
                        x, y = int(pos[0]), int(pos[1])
                        if x >= 0 and y >= 0:  # Allow any position (including respawn -1, -1)
                            unit_info.append({
                                'id': i,
                                'position': [x, y],
                                'energy': max(0, float(energy)),
                                'health': 100
                            })
        except Exception as e:
            print(f"Warning: Error getting unit info: {e}")
        
        return unit_info
    
    def _build_action_masks(self, obs, player_id, unit_info):
        """Build action masks using action_masking.py"""
        action_masks = {}
        
        for unit in unit_info:
            unit_id = unit['id']
            pos = tuple(unit['position'])
            energy = unit['energy']
            
            # Use the imported function from action_masking.py
            mask = get_valid_actions(
                unit_pos=pos,
                energy=energy,
                map_size=self.map_size,
                known_asteroids=self.known_asteroids
            )
            
            action_masks[unit_id] = mask
        
        return action_masks
    
    # Simplified helper methods
    def _infer_point_tiles(self, obs, player_key):
        """Stub for point tile inference"""
        pass
    
    def _update_explored_tiles(self, obs, player_id):
        """Update explored tiles based on unit positions"""
        units = self._safe_get(obs, 'units')
        if units is None:
            return
        
        positions = self._safe_get(units, 'position')
        if positions is None:
            return
        
        try:
            if hasattr(positions, '__array__'):
                positions = np.array(positions)
            
            if len(positions.shape) >= 2 and positions.shape[0] > player_id:
                for pos in positions[player_id]:
                    if len(pos) >= 2:
                        x, y = int(pos[0]), int(pos[1])
                        if 0 <= x < self.map_size and 0 <= y < self.map_size:
                            self.explored_tiles.add((x, y))
        except:
            pass
    
    def _get_known_point_tiles_map(self):
        """Create map of known point tiles"""
        point_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for x, y in self.known_point_tiles:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                point_map[y, x] = 1.0
        return point_map
    
    def _get_explored_tiles_map(self):
        """Create map of explored tiles"""
        explored_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for x, y in self.explored_tiles:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                explored_map[y, x] = 1.0
        return explored_map
    
    def _get_distance_to_center_map(self):
        """Distance gradient to center"""
        dist_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        center = self.map_size / 2.0
        max_dist = self.map_size
        
        for y in range(self.map_size):
            for x in range(self.map_size):
                dist = abs(x - center) + abs(y - center)
                dist_map[y, x] = 1.0 - (dist / max_dist)
        
        return dist_map
    
    def _get_distance_to_nearest_relic_map(self):
        """Distance to nearest known relic"""
        dist_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if not self.known_relic_nodes:
            return dist_map
        
        max_dist = self.map_size * 2
        
        for y in range(self.map_size):
            for x in range(self.map_size):
                min_dist = max_dist
                for relic_x, relic_y in self.known_relic_nodes:
                    dist = abs(x - relic_x) + abs(y - relic_y)
                    min_dist = min(min_dist, dist)
                
                dist_map[y, x] = 1.0 - (min_dist / max_dist)
        
        return dist_map
    
    def _get_known_relics_map(self):
        """Create map of ALL known relics"""
        known_relic_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for x, y in self.known_relic_nodes:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                known_relic_map[y, x] = 1.0
        return known_relic_map
    
    def _get_point_tile_certainty_map(self):
        """Map showing high-confidence point tile locations"""
        certainty_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for x, y in self.known_point_tiles:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                certainty_map[y, x] = 1.0
        return certainty_map
    
    def _get_unexplored_priority_map(self):
        """Priority map for unexplored areas"""
        priority_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for y in range(self.map_size):
            for x in range(self.map_size):
                if (x, y) not in self.explored_tiles:
                    center_dist = abs(x - 11.5) + abs(y - 11.5)
                    priority = 1.0 - (center_dist / (self.map_size * 2))
                    priority_map[y, x] = priority
        return priority_map
    
    def _get_strategic_priority_map(self):
        """Combined strategic priority map"""
        return np.zeros((self.map_size, self.map_size), dtype=np.float32)
    
    def _stack_temporal_features(self, history, current):
        """Stack temporal history for spatial features"""
        # Start with current frame
        stacked = [current]
        
        # Add historical frames (most recent first)
        for i in range(len(history) - 1, max(-1, len(history) - self.history_len), -1):
            if i >= 0:
                stacked.append(history[i])
        
        # Pad if not enough history
        while len(stacked) < self.history_len:
            stacked.append(np.zeros_like(current))
        
        # Stack along channel dimension: (history_len, channels, H, W) -> (history_len*channels, H, W)
        return np.concatenate(stacked[:self.history_len], axis=0)
    
    def _stack_temporal_global(self, history, current):
        """Stack temporal history for global features"""
        # For global features, just return current (or could concatenate if needed)
        return current
    
    def reset(self):
        """Reset for new game"""
        self.spatial_history.clear()
        self.global_history.clear()
        self.known_point_tiles.clear()
        self.known_non_point_tiles.clear()
        self.known_relic_nodes.clear()
        self.known_asteroids.clear()
        self.known_nebula.clear()
        self.explored_tiles.clear()
        self.prev_score = {'player_0': 0, 'player_1': 0}
        self.prev_unit_positions = {'player_0': [], 'player_1': []}