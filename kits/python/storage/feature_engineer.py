"""
Working Feature Engineer - Based on successful approaches
Key: Separate temporal and static features, intelligent stacking
"""
import numpy as np
from collections import deque

class WorkingFeatureEngineer:
    """
    Feature engineer that works like Frog Parade winner:
    - Separate temporal (changing) and static (fixed) features
    - Only stack temporal features across history
    - Rich inference (point tiles, relics, map symmetry)
    """
    def __init__(self, map_size=24, max_units=16, history_len=3):
        self.map_size = map_size
        self.max_units = max_units
        self.history_len = history_len
        
        # Temporal histories (only for changing features)
        self.unit_position_history = deque(maxlen=history_len)
        self.score_history = deque(maxlen=history_len)
        
        # Persistent knowledge (like working approach)
        self.known_relic_nodes = set()
        self.known_point_tiles = set()
        self.known_asteroids = set()
        self.explored_tiles = set()
        
        # Score tracking for point tile inference
        self.prev_score = {'player_0': 0, 'player_1': 0}
        self.prev_unit_positions = {'player_0': [], 'player_1': []}
        
    def process(self, obs, player_id=0):
        """Process observation into features"""
        player_key = f'player_{player_id}'
        
        # Build current frame
        temporal_features = self._build_temporal_features(obs, player_id)
        static_features = self._build_static_features(obs, player_id)
        
        # Update histories (only for temporal)
        self.unit_position_history.append(temporal_features)
        if 'team_points' in obs:
            self.score_history.append(obs['team_points'])
        
        # Infer knowledge
        self._infer_point_tiles(obs, player_key)
        self._update_explored_tiles(obs, player_id)
        
        # Stack temporal features
        stacked_temporal = self._stack_temporal(self.unit_position_history, temporal_features)
        
        # Combine: stacked temporal + static (no redundancy!)
        spatial_features = np.concatenate([stacked_temporal, static_features], axis=0)
        
        # Global features
        global_features = self._build_global_features(obs, player_id)
        
        # Unit info and action masks
        unit_info = self._get_unit_info(obs, player_id)
        action_masks = self._build_action_masks(obs, player_id, unit_info)
        
        return {
            'spatial': spatial_features,  # Shape: (C, 24, 24)
            'global': global_features,    # Shape: (G,)
            'unit_info': unit_info,
            'action_masks': action_masks
        }
    
    def _build_temporal_features(self, obs, player_id):
        """
        Build TEMPORAL features (things that change frequently)
        Only these get stacked across history
        """
        # 4 temporal channels
        temporal = np.zeros((4, self.map_size, self.map_size), dtype=np.float32)
        
        # Channel 0: My unit positions
        if 'units' in obs and 'position' in obs['units']:
            positions = obs['units']['position'][player_id]
            mask = obs['units_mask'][player_id]
            for pos, valid in zip(positions, mask):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        temporal[0, y, x] = 1.0
        
        # Channel 1: Opponent unit positions
        opp_id = 1 - player_id
        if 'units' in obs and 'position' in obs['units']:
            positions = obs['units']['position'][opp_id]
            mask = obs['units_mask'][opp_id]
            for pos, valid in zip(positions, mask):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        temporal[1, y, x] = 1.0
        
        # Channel 2: My unit energies (normalized)
        if 'units' in obs and 'energy' in obs['units']:
            positions = obs['units']['position'][player_id]
            energies = obs['units']['energy'][player_id]
            mask = obs['units_mask'][player_id]
            for pos, energy, valid in zip(positions, energies, mask):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        temporal[2, y, x] = np.clip(energy / 400.0, 0, 1)
        
        # Channel 3: Opponent unit energies (normalized)
        if 'units' in obs and 'energy' in obs['units']:
            positions = obs['units']['position'][opp_id]
            energies = obs['units']['energy'][opp_id]
            mask = obs['units_mask'][opp_id]
            for pos, energy, valid in zip(positions, energies, mask):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        temporal[3, y, x] = np.clip(energy / 400.0, 0, 1)
        
        return temporal
    
    def _build_static_features(self, obs, player_id):
        """
        Build STATIC/SLOW-CHANGING features
        These are NOT stacked across history
        """
        # 12 static channels
        static = np.zeros((12, self.map_size, self.map_size), dtype=np.float32)
        
        # Channel 0: Currently visible relics
        if 'relic_nodes' in obs and 'relic_nodes_mask' in obs:
            relics = obs['relic_nodes']
            mask = obs['relic_nodes_mask']
            for relic_pos, valid in zip(relics, mask):
                if valid and relic_pos[0] >= 0:
                    x, y = int(relic_pos[0]), int(relic_pos[1])
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        static[0, y, x] = 1.0
                        self.known_relic_nodes.add((x, y))
                        # Add symmetric relic
                        sx, sy = self._reflect_position((x, y))
                        static[0, sy, sx] = 1.0
                        self.known_relic_nodes.add((sx, sy))
        
        # Channel 1: All known relics (memory)
        for x, y in self.known_relic_nodes:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                static[1, y, x] = 1.0
        
        # Channel 2: Asteroids
        if 'map_features' in obs and 'tile_type' in obs['map_features']:
            tile_data = obs['map_features']['tile_type']
            for y in range(self.map_size):
                for x in range(self.map_size):
                    if tile_data[y][x] == 1:  # Asteroid
                        static[2, y, x] = 1.0
                        self.known_asteroids.add((x, y))
        
        # Channel 3: Nebula
        if 'map_features' in obs and 'tile_type' in obs['map_features']:
            tile_data = obs['map_features']['tile_type']
            for y in range(self.map_size):
                for x in range(self.map_size):
                    if tile_data[y][x] == 2:  # Nebula
                        static[3, y, x] = 1.0
        
        # Channel 4: Energy field (normalized)
        if 'map_features' in obs and 'energy' in obs['map_features']:
            energy_data = obs['map_features']['energy']
            for y in range(self.map_size):
                for x in range(self.map_size):
                    energy_val = energy_data[y][x]
                    if energy_val != -1.0:
                        static[4, y, x] = np.clip(energy_val / 100.0, 0, 1)
        
        # Channel 5: Known point tiles (INFERRED)
        for x, y in self.known_point_tiles:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                static[5, y, x] = 1.0
        
        # Channel 6: Explored tiles
        for x, y in self.explored_tiles:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                static[6, y, x] = 1.0
        
        # Channel 7: Distance to nearest relic (gradient)
        static[7] = self._get_distance_to_nearest_relic_map()
        
        # Channel 8: Distance to center (gradient)
        static[8] = self._get_distance_to_center_map()
        
        # Channels 9-11: Reserved for future features
        
        return static
    
    def _stack_temporal(self, history, current):
        """Stack temporal features across history"""
        if len(history) == 0:
            # No history yet, replicate current
            return np.tile(current, (self.history_len, 1, 1))
        
        # Stack history (oldest to newest)
        stacked = []
        for frame in history:
            stacked.append(frame)
        
        # Pad if needed
        while len(stacked) < self.history_len:
            stacked.insert(0, stacked[0] if stacked else current)
        
        return np.concatenate(stacked, axis=0)
    
    def _build_global_features(self, obs, player_id):
        """Build global feature vector"""
        features = []
        
        # Score information (3 features)
        if 'team_points' in obs:
            my_score = obs['team_points'][player_id]
            opp_score = obs['team_points'][1 - player_id]
            score_diff = my_score - opp_score
            features.extend([my_score / 100.0, opp_score / 100.0, score_diff / 100.0])
        else:
            features.extend([0, 0, 0])
        
        # Step information (2 features)
        current_step = obs.get('steps', 0) / 100.0
        match_step = obs.get('match_steps', 0) / 100.0
        features.extend([current_step, match_step])
        
        # Unit counts (3 features)
        my_units = sum(obs['units_mask'][player_id]) if 'units_mask' in obs else 0
        opp_units = sum(obs['units_mask'][1 - player_id]) if 'units_mask' in obs else 0
        unit_diff = my_units - opp_units
        features.extend([my_units / 16.0, opp_units / 16.0, unit_diff / 16.0])
        
        # Energy information (3 features)
        my_energy = self._get_total_energy(obs, player_id) / 6400.0
        opp_energy = self._get_total_energy(obs, 1 - player_id) / 6400.0
        energy_diff = my_energy - opp_energy
        features.extend([my_energy, opp_energy, energy_diff])
        
        # Knowledge (3 features)
        known_relics = len(self.known_relic_nodes) / 10.0
        known_points = len(self.known_point_tiles) / 50.0
        exploration = len(self.explored_tiles) / (self.map_size * self.map_size)
        features.extend([known_relics, known_points, exploration])
        
        # Match wins (2 features) - if available
        if 'team_wins' in obs:
            my_wins = obs['team_wins'][player_id] / 3.0
            opp_wins = obs['team_wins'][1 - player_id] / 3.0
            features.extend([my_wins, opp_wins])
        else:
            features.extend([0, 0])
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _infer_point_tiles(self, obs, player_key):
        """Infer point tile locations from score changes"""
        player_id = 0 if player_key == 'player_0' else 1
        
        if 'team_points' not in obs:
            return
        
        current_score = obs['team_points'][player_id]
        prev_score = self.prev_score[player_key]
        score_increase = current_score - prev_score
        
        if 'units' in obs and 'position' in obs['units']:
            current_positions = []
            positions = obs['units']['position'][player_id]
            mask = obs['units_mask'][player_id]
            for pos, valid in zip(positions, mask):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    current_positions.append((int(pos[0]), int(pos[1])))
            
            if score_increase > 0:
                # Units are on point tiles
                for pos in current_positions:
                    self.known_point_tiles.add(pos)
                    # Add symmetric position
                    sym_pos = self._reflect_position(pos)
                    self.known_point_tiles.add(sym_pos)
            
            self.prev_unit_positions[player_key] = current_positions
        
        self.prev_score[player_key] = current_score
    
    def _update_explored_tiles(self, obs, player_id):
        """Update explored tiles from sensor mask"""
        if 'sensor_mask' in obs:
            sensor = obs['sensor_mask']
            for y in range(self.map_size):
                for x in range(self.map_size):
                    if sensor[y][x]:
                        self.explored_tiles.add((x, y))
    
    def _reflect_position(self, pos):
        """Reflect position across map diagonal (maps are symmetric)"""
        x, y = pos
        return (self.map_size - 1 - x, self.map_size - 1 - y)
    
    def _get_distance_to_nearest_relic_map(self):
        """Distance gradient to nearest known relic"""
        dist_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if not self.known_relic_nodes:
            return dist_map
        
        max_dist = self.map_size * 2
        
        for y in range(self.map_size):
            for x in range(self.map_size):
                min_dist = max_dist
                for rx, ry in self.known_relic_nodes:
                    dist = abs(x - rx) + abs(y - ry)
                    min_dist = min(min_dist, dist)
                dist_map[y, x] = 1.0 - (min_dist / max_dist)
        
        return dist_map
    
    def _get_distance_to_center_map(self):
        """Distance gradient to map center"""
        dist_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        center = self.map_size / 2.0
        max_dist = self.map_size
        
        for y in range(self.map_size):
            for x in range(self.map_size):
                dist = abs(x - center) + abs(y - center)
                dist_map[y, x] = 1.0 - (dist / max_dist)
        
        return dist_map
    
    def _get_total_energy(self, obs, player_id):
        """Get total energy for a player"""
        total = 0
        if 'units' in obs and 'energy' in obs['units']:
            energies = obs['units']['energy'][player_id]
            mask = obs['units_mask'][player_id]
            for energy, valid in zip(energies, mask):
                if valid and energy > 0:
                    total += energy
        return total
    
    def _get_unit_info(self, obs, player_id):
        """Extract unit information"""
        unit_info = []
        
        if 'units' in obs and 'position' in obs['units']:
            positions = obs['units']['position'][player_id]
            energies = obs['units']['energy'][player_id]
            mask = obs['units_mask'][player_id]
            
            for i, (pos, energy, valid) in enumerate(zip(positions, energies, mask)):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    unit_info.append({
                        'id': i,
                        'position': (int(pos[0]), int(pos[1])),
                        'energy': max(0, int(energy))
                    })
        
        return unit_info
    
    def _build_action_masks(self, obs, player_id, unit_info):
        """Build action masks for each unit"""
        from action_masking import get_valid_actions
        
        action_masks = {}
        for unit in unit_info:
            mask = get_valid_actions(
                unit_pos=unit['position'],
                energy=unit['energy'],
                map_size=self.map_size,
                known_asteroids=self.known_asteroids,
                known_point_tiles=self.known_point_tiles
            )
            action_masks[unit['id']] = mask
        
        return action_masks
    
    def reset(self):
        """Reset for new episode"""
        self.unit_position_history.clear()
        self.score_history.clear()
        self.known_relic_nodes.clear()
        self.known_point_tiles.clear()
        self.known_asteroids.clear()
        self.explored_tiles.clear()
        self.prev_score = {'player_0': 0, 'player_1': 0}
        self.prev_unit_positions = {'player_0': [], 'player_1': []}