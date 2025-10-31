import numpy as np
from collections import deque
from action_masking import get_valid_actions

class AdvancedFeatureEngineer:
    """
    Enhanced feature engineer with 12 base spatial channels
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
        
    def process(self, obs, player_id=0):
        """Process observation with advanced feature engineering"""
        player_key = f'player_{player_id}'
        
        # Extract current frame features (12 channels)
        current_spatial = self._build_spatial_features(obs, player_id)
        current_global = self._build_global_features(obs, player_id)
        
        # Infer knowledge from observations
        self._infer_point_tiles(obs, player_key)
        self._update_explored_tiles(obs, player_id)
        self._infer_map_symmetry(current_spatial)
        
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
        # ✅ INCREASED: 16 channels to include all strategic knowledge
        spatial = np.zeros((16, self.map_size, self.map_size), dtype=np.float32)
        
        # Basic observations (channels 0-11 - same as before)
        spatial[0] = self._get_unit_positions(obs, player_id)
        spatial[1] = self._get_unit_positions(obs, 1 - player_id)
        spatial[2] = self._get_relic_positions(obs)  # Currently visible relics
        spatial[3] = self._get_energy_field(obs)
        spatial[4] = self._get_asteroid_map(obs)
        spatial[5] = self._get_nebula_map(obs)
        spatial[6] = self._get_known_point_tiles_map()
        spatial[7] = self._get_unit_energy_map(obs, player_id)
        spatial[8] = self._get_unit_energy_map(obs, 1 - player_id)
        spatial[9] = self._get_explored_tiles_map()
        spatial[10] = self._get_distance_to_center_map()
        spatial[11] = self._get_distance_to_nearest_relic_map()
        
        # ✅ NEW: Strategic inferred knowledge (channels 12-15)
        spatial[12] = self._get_known_relics_map()  # ALL known relics (memory)
        spatial[13] = self._get_point_tile_certainty_map()  # High-confidence point tiles
        spatial[14] = self._get_unexplored_priority_map()  # Unexplored areas
        spatial[15] = self._get_strategic_priority_map()  # Combined strategic value
    
        return spatial
    
    def _build_global_features(self, obs, player_id):
        """Build comprehensive global feature vector"""
        features = []
        
        # Score information (3 features)
        if 'team_points' in obs:
            my_score = obs['team_points'][player_id]
            opp_score = obs['team_points'][1 - player_id]
            score_diff = my_score - opp_score
            features.extend([my_score / 100.0, opp_score / 100.0, score_diff / 100.0])
        else:
            features.extend([0, 0, 0])
        
        # Normalized step information (2 features)
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
        
        # Map knowledge (10 features)
        total_tiles = self.map_size * self.map_size
        known_point_ratio = len(self.known_point_tiles) / total_tiles
        known_relic_ratio = len(self.known_relic_nodes) / 10.0
        exploration_ratio = len(self.explored_tiles) / total_tiles
        known_relic_count = len(self.known_relic_nodes) / 10.0
        unexplored_ratio = 1.0 - exploration_ratio
        high_confidence_points = len([p for p in self.known_point_tiles if self._is_high_confidence_point(p)]) / 10.0
        strategic_balance = self._calculate_strategic_balance()
        relic_accessibility = self._calculate_relic_accessibility()
        exploration_efficiency = len(self.explored_tiles) / max(1, obs.get('steps', 1))
        inference_confidence = self._calculate_inference_confidence()
        
        features.extend([
            known_point_ratio, known_relic_ratio, exploration_ratio, known_relic_count,
            unexplored_ratio, high_confidence_points, strategic_balance, 
            relic_accessibility, exploration_efficiency, inference_confidence
        ])
        
        # Match win conditions (2 features)
        if 'match_wins' in obs:
            my_wins = obs['match_wins'][player_id] / 3.0
            opp_wins = obs['match_wins'][1 - player_id] / 3.0
            features.extend([my_wins, opp_wins])
        else:
            features.extend([0, 0])
        
        return np.array(features, dtype=np.float32)
    def _get_point_tile_certainty_map(self):
        """Map showing high-confidence point tile locations"""
        certainty_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for x, y in self.known_point_tiles:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                certainty_map[y, x] = 1.0  # We're certain these give points
        return certainty_map

    def _get_unexplored_priority_map(self):
        """Priority map for unexplored areas"""
        priority_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for y in range(self.map_size):
            for x in range(self.map_size):
                if (x, y) not in self.explored_tiles:
                    # Higher priority for central unexplored areas
                    center_dist = abs(x - 11.5) + abs(y - 11.5)
                    priority = 1.0 - (center_dist / (self.map_size * 2))
                    priority_map[y, x] = priority
        return priority_map
    
    def _get_known_relics_map(self):
        """✅ NEW: Create map of ALL known relics (including remembered ones)"""
        known_relic_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for x, y in self.known_relic_nodes:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                known_relic_map[y, x] = 1.0
        return known_relic_map

    def _get_strategic_priority_map(self):
        """Combined strategic priority map"""
        strategic_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        for y in range(self.map_size):
            for x in range(self.map_size):
                priority = 0.0
                
                # Priority from known relics
                for relic_x, relic_y in self.known_relic_nodes:
                    relic_dist = abs(x - relic_x) + abs(y - relic_y)
                    if relic_dist <= 3:
                        priority = max(priority, 0.8 - (relic_dist * 0.2))
                
                # Priority from point tiles
                if (x, y) in self.known_point_tiles:
                    priority = max(priority, 0.6)
                
                # Priority from unexplored areas
                if (x, y) not in self.explored_tiles:
                    priority = max(priority, 0.4)
                
                strategic_map[y, x] = priority
        
        return strategic_map

    # ✅ NEW: Strategic calculation helpers
    def _is_high_confidence_point(self, point):
        """Check if we're highly confident this is a point tile"""
        # Points we actually scored from are high confidence
        # Points inferred from symmetry are lower confidence
        return point in self.known_point_tiles  # Basic version

    def _calculate_strategic_balance(self):
        """Calculate how balanced our map knowledge is"""
        if not self.known_relic_nodes:
            return 0.0
        
        # Check if we know relics on both sides of the map
        left_relics = sum(1 for x, y in self.known_relic_nodes if x < self.map_size // 2)
        right_relics = len(self.known_relic_nodes) - left_relics
        balance = 1.0 - abs(left_relics - right_relics) / len(self.known_relic_nodes)
        return balance

    def _calculate_relic_accessibility(self):
        """Calculate how accessible known relics are"""
        if not self.known_relic_nodes:
            return 0.0
        
        # Simple measure: how many relics are in explored areas
        accessible = sum(1 for relic in self.known_relic_nodes if relic in self.explored_tiles)
        return accessible / len(self.known_relic_nodes)

    def _calculate_inference_confidence(self):
        """Overall confidence in our inferred knowledge"""
        total_inferred = len(self.known_point_tiles) + len(self.known_relic_nodes)
        if total_inferred == 0:
            return 0.0
        
        # Higher confidence when we have more verified knowledge
        confidence = min(1.0, total_inferred / 20.0)  # Scale with knowledge
        return confidence
    def _stack_temporal_features(self, history, current):
        """Stack temporal history into single tensor"""
        if len(history) < self.history_len:
            padding = [current] * (self.history_len - len(history))
            full_history = padding + list(history)
        else:
            full_history = list(history)
        
        stacked = np.concatenate(full_history, axis=0)
        return stacked
    
    def _stack_temporal_global(self, history, current):
        """Stack global temporal history"""
        if len(history) < self.history_len:
            padding = [current] * (self.history_len - len(history))
            full_history = padding + list(history)
        else:
            full_history = list(history)
        
        stacked = np.concatenate(full_history, axis=0)
        return stacked
    
    def _infer_point_tiles(self, obs, player_key):
        """Infer point tile locations from score changes"""
        player_id = 0 if player_key == 'player_0' else 1
        
        current_score = obs.get('team_points', [0, 0])[player_id]
        current_positions = self._get_unit_positions_list(obs, player_id)
        
        score_diff = current_score - self.prev_score[player_key]
        if score_diff > 0 and current_positions:
            for pos in current_positions[:score_diff]:
                pos_tuple = tuple(pos)
                self.known_point_tiles.add(pos_tuple)
                reflected = self._reflect_position(pos_tuple)
                self.known_point_tiles.add(reflected)
        
        elif score_diff == 0 and current_positions:
            for pos in current_positions:
                pos_tuple = tuple(pos)
                if pos_tuple not in self.known_point_tiles:
                    self.known_non_point_tiles.add(pos_tuple)
        
        self.prev_score[player_key] = current_score
        self.prev_unit_positions[player_key] = current_positions
    
    def _update_explored_tiles(self, obs, player_id):
        """Track which tiles we've explored"""
        positions = self._get_unit_positions_list(obs, player_id)
        for pos in positions:
            self.explored_tiles.add(pos)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    exp_pos = (pos[0] + dx, pos[1] + dy)
                    if 0 <= exp_pos[0] < self.map_size and 0 <= exp_pos[1] < self.map_size:
                        self.explored_tiles.add(exp_pos)
    
    def _infer_map_symmetry(self, spatial):
        """Use map symmetry to double our knowledge"""
        relic_reflected = np.flip(np.flip(spatial[2], axis=0), axis=1)
        spatial[2] = np.maximum(spatial[2], relic_reflected)
        
        asteroid_reflected = np.flip(np.flip(spatial[4], axis=0), axis=1)
        spatial[4] = np.maximum(spatial[4], asteroid_reflected)
        
        nebula_reflected = np.flip(np.flip(spatial[5], axis=0), axis=1)
        spatial[5] = np.maximum(spatial[5], nebula_reflected)
    
    def _reflect_position(self, pos):
        """Reflect position across map center"""
        return (self.map_size - 1 - pos[0], self.map_size - 1 - pos[1])
    
    def _get_unit_positions(self, obs, player_id):
        """Extract unit positions for a player"""
        unit_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if 'units' in obs and 'position' in obs['units']:
            positions = obs['units']['position'][player_id]
            mask = obs['units_mask'][player_id] if 'units_mask' in obs else [True] * len(positions)
            
            for i, (pos, valid) in enumerate(zip(positions, mask)):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    x, y = pos
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        unit_map[y, x] = 1
        
        return unit_map
    
    def _get_unit_energy_map(self, obs, player_id):
        """Map of unit energy levels"""
        energy_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if 'units' in obs and 'position' in obs['units'] and 'energy' in obs['units']:
            positions = obs['units']['position'][player_id]
            energies = obs['units']['energy'][player_id]
            mask = obs['units_mask'][player_id] if 'units_mask' in obs else [True] * len(positions)
            
            for pos, energy, valid in zip(positions, energies, mask):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    x, y = pos
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        energy_map[y, x] = max(0, energy) / 400.0
        
        return energy_map
    
    def _get_unit_positions_list(self, obs, player_id):
        """Get list of unit positions"""
        positions = []
        if 'units' in obs and 'position' in obs['units']:
            unit_positions = obs['units']['position'][player_id]
            mask = obs['units_mask'][player_id] if 'units_mask' in obs else [True] * len(unit_positions)
            for pos, valid in zip(unit_positions, mask):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    positions.append(tuple(pos))
        return positions
    
    def _get_relic_positions(self, obs):
        """Extract and accumulate relic positions"""
        relic_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if 'relic_nodes' in obs and 'relic_nodes_mask' in obs:
            relics = obs['relic_nodes']
            mask = obs['relic_nodes_mask']
            
            for relic_pos, valid in zip(relics, mask):
                if valid and relic_pos[0] >= 0 and relic_pos[1] >= 0:
                    x, y = relic_pos
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        relic_map[y, x] = 1
                        self.known_relic_nodes.add((x, y))
                        rx, ry = self._reflect_position((x, y))
                        relic_map[ry, rx] = 1
                        self.known_relic_nodes.add((rx, ry))
        
        return relic_map
    
    def _get_energy_field(self, obs):
        """Extract and normalize energy field"""
        energy_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if 'map_features' in obs and 'energy' in obs['map_features']:
            energy_data = obs['map_features']['energy']
            for y in range(self.map_size):
                for x in range(self.map_size):
                    energy_val = energy_data[y][x]
                    if energy_val != -1.0:
                        energy_map[y, x] = np.clip(energy_val / 100.0, 0, 1)
        
        return energy_map
    
    def _get_asteroid_map(self, obs):
        """Extract asteroid positions only"""
        asteroid_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if 'map_features' in obs and 'tile_type' in obs['map_features']:
            tile_data = obs['map_features']['tile_type']
            for y in range(self.map_size):
                for x in range(self.map_size):
                    if tile_data[y][x] == 1:
                        asteroid_map[y, x] = 1
                        self.known_asteroids.add((x, y))
        
        return asteroid_map
    
    def _get_nebula_map(self, obs):
        """Extract nebula positions only"""
        nebula_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if 'map_features' in obs and 'tile_type' in obs['map_features']:
            tile_data = obs['map_features']['tile_type']
            for y in range(self.map_size):
                for x in range(self.map_size):
                    if tile_data[y][x] == 2:
                        nebula_map[y, x] = 1
                        self.known_nebula.add((x, y))
        
        return nebula_map
    
    def _get_known_point_tiles_map(self):
        """Create map of known point tiles"""
        point_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for x, y in self.known_point_tiles:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                point_map[y, x] = 1
        return point_map
    
    def _get_explored_tiles_map(self):
        """Create map of explored tiles"""
        explored_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        for x, y in self.explored_tiles:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                explored_map[y, x] = 1
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
    
    def _get_total_energy(self, obs, player_id):
        """Get total energy for a player"""
        total_energy = 0
        if 'units' in obs and 'energy' in obs['units']:
            energies = obs['units']['energy'][player_id]
            mask = obs['units_mask'][player_id] if 'units_mask' in obs else [True] * len(energies)
            for energy, valid in zip(energies, mask):
                if valid and energy > 0:
                    total_energy += energy
        return total_energy
    
    def _get_unit_info(self, obs, player_id):
        """Extract unit information - MATCH MODEL LOGIC"""
        unit_info = []
        
        if 'units' in obs and 'position' in obs['units'] and 'energy' in obs['units']:
            positions = obs['units']['position'][player_id]
            energies = obs['units']['energy'][player_id]
            
            # Get mask if available, otherwise assume all are valid
            if 'units_mask' in obs and player_id < len(obs['units_mask']):
                mask = obs['units_mask'][player_id]
            else:
                mask = [True] * len(positions)
            
            for i, (pos, energy, valid) in enumerate(zip(positions, energies, mask)):
                if valid and pos[0] >= 0 and pos[1] >= 0:
                    unit_info.append({
                        'id': i,
                        'position': pos,
                        'energy': max(0, energy),
                        'health': 100
                    })
        
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