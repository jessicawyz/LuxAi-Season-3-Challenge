"""
HYBRID REWARD CALCULATOR

Combines the best of both approaches:
1. LuxRewardShaper's dense every-step shaping (energy, damage, survival)
2. RewardCalculator's interface and curriculum learning
3. Strong focus on scoring (the actual objective)

This should provide richer training signal like LuxRewardShaper while
maintaining compatibility with existing code.
"""

import numpy as np

class ImprovedRewardCalculator:
    """
    Hybrid reward system with dense shaping + curriculum learning
    Compatible with existing code interface
    """
    def __init__(self):
        # ===== SPARSE REWARDS (Match outcomes) =====
        self.match_win_reward = 100.0
        self.match_loss_reward = -100.0
        
        # ===== DENSE REWARDS (LuxRewardShaper-inspired) =====
        # Scoring (PRIMARY OBJECTIVE)
        self.relic_point_reward = 5.0  # Per point scored (higher than LuxRewardShaper's 1.0)
        
        # Combat/Energy (helps in competitive play)
        self.damage_dealt_reward = 0.01  # Per energy damage to enemies
        self.energy_differential_reward = 0.001  # Energy advantage
        self.unit_loss_penalty = -5.0  # When unit dies
        
        # Survival and activity
        self.survival_reward = 0.05  # Per step unit is alive
        
        # Discovery and exploration
        self.exploration_reward = 0.5  # New tile discovered
        self.relic_discovery_reward = 15.0  # New relic found
        self.relic_proximity_reward = 2.0  # Being near relics (gradient)
        
        # Behavioral shaping
        self.movement_reward = 0.3  # Actually moving
        self.blocked_movement_penalty = -0.2  # Tried to move but blocked
        self.oscillation_penalty = -3.0  # Back-and-forth movement
        
        # Energy management
        self.high_energy_reward = 1.0  # Energy > 200
        self.low_energy_penalty = -2.0  # Energy < 50
        
        # ===== STATE TRACKING =====
        self.prev_my_score = {'player_0': 0, 'player_1': 0}
        self.prev_opp_score = {'player_0': 0, 'player_1': 0}
        self.prev_my_energy = {'player_0': 0, 'player_1': 0}
        self.prev_opp_energy = {'player_0': 0, 'player_1': 0}
        self.prev_unit_count = {'player_0': 0, 'player_1': 0}
        
        # ===== CURRICULUM LEARNING =====
        self.curriculum_step = 0
        
        # ===== MEMORY =====
        self.known_relic_positions = {'player_0': set(), 'player_1': set()}
        self.visited_tiles = {'player_0': set(), 'player_1': set()}
        self.position_history = {
            'player_0': {i: [] for i in range(16)},
            'player_1': {i: [] for i in range(16)}
        }
        
        # Initialize spawn as visited
        self.visited_tiles['player_0'].add((0, 0))
        self.visited_tiles['player_1'].add((23, 23))
    
    def update_curriculum(self, episode):
        """Update curriculum phase"""
        self.curriculum_step = episode
        if episode % 50 == 0:
            if episode < 200:
                print(f"📚 Phase 1: EXPLORATION & BASICS (ep {episode}/200)")
            elif episode < 500:
                print(f"📚 Phase 2: SCORING & COMBAT (ep {episode}/500)")
            else:
                print(f"📚 Phase 3: FULL COMPETITIVE (ep {episode}+)")
    
    def _compute_total_energy(self, obs, player_id):
        """Compute total energy for a team (like LuxRewardShaper)"""
        if 'units' not in obs or 'energy' not in obs['units']:
            return 0.0
        
        energies = np.array(obs['units']['energy'][player_id])
        mask = np.array(obs['units_mask'][player_id])
        
        valid_energies = energies[mask]
        valid_energies = valid_energies[valid_energies >= 0]
        
        return float(np.sum(valid_energies)) if len(valid_energies) > 0 else 0.0
    
    def _count_active_units(self, obs, player_id):
        """Count active units"""
        if 'units_mask' not in obs:
            return 0
        return int(np.sum(obs['units_mask'][player_id]))
    
    def calculate_step_reward(self, player, unit_id, unit, action, current_obs, next_obs):
        """
        Calculate per-step reward with DENSE SHAPING
        Combines original scoring focus with LuxRewardShaper's energy/combat rewards
        """
        reward = 0.0
        player_id = 0 if player == 'player_0' else 1
        enemy_id = 1 - player_id
        next_pos = tuple(unit['position'])
        current_pos = tuple(unit.get('prev_position', unit['position']))
        
        # ===== 1. ANTI-OSCILLATION PENALTY =====
        unit_history = self.position_history[player][unit_id]
        unit_history.append(next_pos)
        if len(unit_history) > 4:
            unit_history.pop(0)
        
        if len(unit_history) >= 4:
            if (unit_history[0] == unit_history[2] and 
                unit_history[1] == unit_history[3] and
                unit_history[0] != unit_history[1]):
                reward += self.oscillation_penalty
        
        # ===== 2. SCORING REWARDS (Primary objective) =====
        if 'team_points' in next_obs:
            current_my_score = next_obs['team_points'][player_id]
            prev_my_score = self.prev_my_score.get(player, 0)
            
            score_increase = current_my_score - prev_my_score
            if score_increase > 0:
                reward += score_increase * self.relic_point_reward
                print(f"💰 {player} Unit {unit_id} scored +{score_increase}! Reward: +{score_increase * self.relic_point_reward:.1f}")
        
        # ===== 3. RELIC DISCOVERY =====
        if 'relic_nodes' in next_obs and 'relic_nodes_mask' in next_obs:
            relics = next_obs['relic_nodes']
            mask = next_obs['relic_nodes_mask']
            
            for relic_pos, valid in zip(relics, mask):
                if valid and relic_pos[0] >= 0:
                    relic_tuple = tuple(relic_pos)
                    if relic_tuple not in self.known_relic_positions[player]:
                        relic_dist = abs(next_pos[0] - relic_pos[0]) + abs(next_pos[1] - relic_pos[1])
                        if relic_dist <= 3:
                            self.known_relic_positions[player].add(relic_tuple)
                            reward += self.relic_discovery_reward
                            print(f"🏆 {player} Unit {unit_id} DISCOVERED RELIC! +{self.relic_discovery_reward}")
        
        # ===== 4. RELIC PROXIMITY (Gradient reward) =====
        closest_relic_dist = float('inf')
        for relic_pos in self.known_relic_positions[player]:
            dist = abs(next_pos[0] - relic_pos[0]) + abs(next_pos[1] - relic_pos[1])
            closest_relic_dist = min(closest_relic_dist, dist)
        
        if closest_relic_dist < float('inf'):
            if closest_relic_dist == 0:
                reward += self.relic_proximity_reward * 4.0  # On the relic!
            elif closest_relic_dist <= 2:
                reward += self.relic_proximity_reward * 2.0  # Very close
            elif closest_relic_dist <= 4:
                reward += self.relic_proximity_reward  # Close
        
        # ===== 5. EXPLORATION =====
        if next_pos not in self.visited_tiles[player]:
            self.visited_tiles[player].add(next_pos)
            reward += self.exploration_reward
        
        # ===== 6. MOVEMENT REWARDS =====
        if action in [1, 2, 3, 4]:  # Movement actions
            if current_pos != next_pos:
                reward += self.movement_reward
            else:
                reward += self.blocked_movement_penalty
        
        # ===== 7. SURVIVAL REWARD (LuxRewardShaper-style) =====
        # Small constant reward for being alive and active
        reward += self.survival_reward
        
        # ===== 8. ENERGY MANAGEMENT =====
        # energy = unit.get('energy', 0)
        # if energy > 200:
        #     reward += self.high_energy_reward
        # elif energy < 50:
        #     reward += self.low_energy_penalty
        
        return reward
    
    def calculate_team_rewards(self, player, current_obs, next_obs):
        """
        Calculate team-level rewards (LuxRewardShaper-inspired)
        These are global rewards not tied to specific units
        """
        reward = 0.0
        player_id = 0 if player == 'player_0' else 1
        enemy_id = 1 - player_id
        
        # ===== DAMAGE DEALT REWARD =====
        curr_enemy_energy = self._compute_total_energy(current_obs, enemy_id)
        next_enemy_energy = self._compute_total_energy(next_obs, enemy_id)
        damage_dealt = max(0, curr_enemy_energy - next_enemy_energy)
        if damage_dealt > 0:
            reward += damage_dealt * self.damage_dealt_reward
        
        # ===== ENERGY DIFFERENTIAL REWARD =====
        curr_team_energy = self._compute_total_energy(current_obs, player_id)
        next_team_energy = self._compute_total_energy(next_obs, player_id)
        
        # Track for next iteration
        self.prev_my_energy[player] = next_team_energy
        self.prev_opp_energy[player] = next_enemy_energy
        
        curr_differential = curr_team_energy - curr_enemy_energy
        next_differential = next_team_energy - next_enemy_energy
        differential_change = next_differential - curr_differential
        reward += differential_change * self.energy_differential_reward
        
        # ===== UNIT LOSS PENALTY =====
        curr_unit_count = self._count_active_units(current_obs, player_id)
        next_unit_count = self._count_active_units(next_obs, player_id)
        units_lost = max(0, curr_unit_count - next_unit_count)
        if units_lost > 0:
            reward += units_lost * self.unit_loss_penalty
            print(f"💀 {player} lost {units_lost} unit(s)! Penalty: {units_lost * self.unit_loss_penalty:.1f}")
        
        self.prev_unit_count[player] = next_unit_count
        
        return reward
    
    def calculate_match_reward(self, player, match_info):
        """
        Calculate match-end reward based on WIN/LOSS and score margin
        (Same as original)
        """
        if not match_info.get('match_ended', False):
            return 0.0
        
        player_id = 0 if player == 'player_0' else 1
        my_score = match_info.get('my_score', 0)
        opp_score = match_info.get('opp_score', 0)
        winner = match_info.get('winner', None)
        
        reward = 0.0
        
        # 1. WIN/LOSS REWARD
        if winner == player:
            reward += self.match_win_reward
        elif winner is not None:
            reward += self.match_loss_reward
        
        # 2. SCORE MARGIN REWARD
        score_diff = my_score - opp_score
        margin_reward = score_diff * 2.0
        reward += margin_reward
        
        return reward
    
    def compute_all_rewards(self, player, actions, current_obs, next_obs, match_info):
        """
        Compute rewards with CURRICULUM LEARNING and DENSE SHAPING
        
        Returns:
            step_rewards: dict mapping unit_id -> reward
            global_reward: total reward for all units
        """
        player_id = 0 if player == 'player_0' else 1
        step_rewards = {}
        
        # Update score tracking
        if 'team_points' in next_obs:
            self.prev_my_score[player] = next_obs['team_points'][player_id]
            self.prev_opp_score[player] = next_obs['team_points'][1 - player_id]
        
        # Get sparse match reward
        match_reward = self.calculate_match_reward(player, match_info)
        
        # Get team-level dense rewards (energy, combat, etc.)
        team_reward = self.calculate_team_rewards(player, current_obs, next_obs)
        
        # ===== CURRICULUM: Balance dense and sparse rewards =====
        if self.curriculum_step < 200:
            # Phase 1: Focus on dense shaping (exploration, movement)
            dense_weight = 1.0
            sparse_weight = 0.2
            team_weight = 0.3  # Low combat focus early
        elif self.curriculum_step < 500:
            # Phase 2: Increase focus on combat and winning
            dense_weight = 0.8
            sparse_weight = 0.5
            team_weight = 0.6  # More combat focus
        else:
            # Phase 3: Fully competitive
            dense_weight = 0.6
            sparse_weight = 1.0
            team_weight = 1.0  # Full combat focus
        
        # Calculate per-unit rewards
        if 'units_mask' in next_obs:
            mask = next_obs['units_mask'][player_id]
            positions = next_obs['units']['position'][player_id]
            energies = next_obs['units']['energy'][player_id]
            prev_positions = current_obs['units']['position'][player_id] if 'units' in current_obs else positions
            
            num_active_units = np.sum(mask)
            team_reward_per_unit = team_reward / max(1, num_active_units)
            
            for unit_id, (is_active, pos, energy, prev_pos) in enumerate(zip(mask, positions, energies, prev_positions)):
                if is_active and pos[0] >= 0:
                    unit = {
                        'id': unit_id,
                        'position': pos,
                        'energy': energy,
                        'prev_position': prev_pos
                    }
                    
                    action = actions.get(unit_id, 0)
                    
                    # Get dense step reward for this unit
                    dense_reward = self.calculate_step_reward(
                        player, unit_id, unit, action, current_obs, next_obs
                    )
                    
                    # Combine all reward sources
                    total_reward = (
                        dense_weight * dense_reward + 
                        sparse_weight * match_reward +
                        team_weight * team_reward_per_unit
                    )
                    step_rewards[unit_id] = total_reward
        
        # Global reward
        global_reward = sum(step_rewards.values()) if step_rewards else 0.0
        
        # Logging
        if match_info.get('match_ended', False):
            my_score = match_info.get('my_score', 0)
            opp_score = match_info.get('opp_score', 0)
            result = "WON" if my_score > opp_score else "LOST" if my_score < opp_score else "TIE"
            
            print(f"\n{'='*60}")
            print(f"🏁 {player} {result} MATCH!")
            print(f"   Score: {my_score} vs {opp_score} (diff: {my_score - opp_score:+d})")
            print(f"   Match reward: {match_reward:.1f}")
            print(f"   Team reward: {team_reward:.1f}")
            print(f"   Weights - Dense: {dense_weight:.1f}, Sparse: {sparse_weight:.1f}, Team: {team_weight:.1f}")
            print(f"   Avg reward/unit: {global_reward/max(1, len(step_rewards)):.1f}")
            print(f"   Known relics: {len(self.known_relic_positions[player])}")
            print(f"   Energy: {self.prev_my_energy[player]:.0f} vs {self.prev_opp_energy[player]:.0f}")
            print(f"{'='*60}\n")
        
        return step_rewards, global_reward
    
    def reset_for_new_game(self):
        """Reset everything for new game (episode)"""
        self.prev_my_score = {'player_0': 0, 'player_1': 0}
        self.prev_opp_score = {'player_0': 0, 'player_1': 0}
        self.prev_my_energy = {'player_0': 0, 'player_1': 0}
        self.prev_opp_energy = {'player_0': 0, 'player_1': 0}
        self.prev_unit_count = {'player_0': 0, 'player_1': 0}
        self.known_relic_positions = {'player_0': set(), 'player_1': set()}
        self.visited_tiles = {'player_0': set(), 'player_1': set()}
        self.position_history = {
            'player_0': {i: [] for i in range(16)},
            'player_1': {i: [] for i in range(16)}
        }
        self.visited_tiles['player_0'].add((0, 0))
        self.visited_tiles['player_1'].add((23, 23))
    
    def reset_for_new_match(self):
        """Reset between matches - KEEP relic memory (strategic advantage)"""
        self.prev_my_score = {'player_0': 0, 'player_1': 0}
        self.prev_opp_score = {'player_0': 0, 'player_1': 0}
        self.prev_my_energy = {'player_0': 0, 'player_1': 0}
        self.prev_opp_energy = {'player_0': 0, 'player_1': 0}
        self.prev_unit_count = {'player_0': 0, 'player_1': 0}
        self.visited_tiles = {'player_0': set(), 'player_1': set()}
        self.position_history = {
            'player_0': {i: [] for i in range(16)},
            'player_1': {i: [] for i in range(16)}
        }
        self.visited_tiles['player_0'].add((0, 0))
        self.visited_tiles['player_1'].add((23, 23))
        
        print(f"🔄 NEW MATCH - P0 remembers {len(self.known_relic_positions['player_0'])} relics, "
              f"P1 remembers {len(self.known_relic_positions['player_1'])} relics")


def create_reward_calculator():
    """Factory function to create improved reward calculator"""
    return ImprovedRewardCalculator()
