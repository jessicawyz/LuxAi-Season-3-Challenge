"""
Working Reward Calculator - Based on successful approaches
Key: Sparse win/loss rewards (Frog Parade) with strategic dense shaping
"""
import numpy as np

class WorkingRewardCalculator:
    """
    Reward system based on successful approaches:
    - Primary: Sparse win/loss (+1/-1) like Frog Parade
    - Secondary: Strategic dense shaping for learning
    """
    def __init__(self, use_sparse=True):
        self.use_sparse = use_sparse
        
        # Sparse rewards (Frog Parade - primary)
        self.match_win_reward = 1.0
        self.match_loss_reward = -1.0
        self.episode_win_bonus = 3.0  # For winning the series
        
        # Strategic dense rewards (secondary - from second approach)
        self.relic_point_reward = 0.6  # Per point scored
        self.damage_dealt_reward = 0.002  # Per energy damage dealt to enemies
        self.energy_differential_reward = 0.0005  # Energy advantage
        self.unit_loss_penalty = -0.5  # When unit is destroyed
        self.survival_reward = 0.01  # Per step with active units
        self.exploration_reward = 0.01  # Per new tile explored
        self.relic_discovery_reward = 0.3  # Per new relic discovered
        
        # State tracking
        self.prev_score = {'player_0': 0, 'player_1': 0}
        self.prev_energy = {'player_0': 0, 'player_1': 0}
        self.known_relics = {'player_0': set(), 'player_1': set()}
        self.explored_tiles = {'player_0': set(), 'player_1': set()}
        self.prev_unit_count = {'player_0': 0, 'player_1': 0}
        
    def compute_reward(self, player, obs, next_obs, match_ended, match_winner):
        """
        Compute reward for a player
        
        Args:
            player: 'player_0' or 'player_1'
            obs: Current observation
            next_obs: Next observation
            match_ended: Whether match ended
            match_winner: Winner of match (if ended)
        
        Returns:
            reward: Scalar reward
        """
        player_id = 0 if player == 'player_0' else 1
        enemy_id = 1 - player_id
        reward = 0.0
        
        # Sparse win/loss reward (primary - Frog Parade)
        if match_ended:
            if match_winner == player:
                reward += self.match_win_reward
                print(f"🎉 {player} won match! +{self.match_win_reward}")
            elif match_winner is not None:
                reward += self.match_loss_reward
                print(f"💥 {player} lost match! {self.match_loss_reward}")
        
        # Strategic dense shaping (secondary - always include for learning)
        reward += self._compute_strategic_rewards(player, player_id, enemy_id, obs, next_obs, match_ended)
        
        return reward
    
    def _compute_strategic_rewards(self, player, player_id, enemy_id, obs, next_obs, match_ended):
        """Compute strategic dense rewards for learning"""
        reward = 0.0
        
        # 1. Relic points gained (direct progress)
        if 'team_points' in next_obs:
            curr_score = obs.get('team_points', [0, 0])[player_id]
            next_score = next_obs.get('team_points', [0, 0])[player_id]
            points_gained = next_score - curr_score
            
            if points_gained > 0:
                reward += points_gained * self.relic_point_reward
                # print(f"📊 {player} scored {points_gained} points! +{points_gained * self.relic_point_reward:.3f}")
            
            self.prev_score[player] = next_score
        
        # 2. Damage dealt to enemy (combat effectiveness)
        curr_enemy_energy = self._compute_total_energy(obs, enemy_id)
        next_enemy_energy = self._compute_total_energy(next_obs, enemy_id)
        damage_dealt = max(0, curr_enemy_energy - next_enemy_energy)
        if damage_dealt > 0:
            reward += damage_dealt * self.damage_dealt_reward
            # print(f"⚔️ {player} dealt {damage_dealt} damage! +{damage_dealt * self.damage_dealt_reward:.3f}")
        
        # 3. Energy differential (resource advantage)
        curr_team_energy = self._compute_total_energy(obs, player_id)
        next_team_energy = self._compute_total_energy(next_obs, player_id)
        
        curr_differential = curr_team_energy - curr_enemy_energy
        next_differential = next_team_energy - next_enemy_energy
        differential_change = next_differential - curr_differential
        reward += differential_change * self.energy_differential_reward
        
        # 4. Unit preservation (avoid losses)
        curr_unit_count = self._count_active_units(obs, player_id)
        next_unit_count = self._count_active_units(next_obs, player_id)
        units_lost = max(0, curr_unit_count - next_unit_count)
        if units_lost > 0 and not match_ended:  # Only penalize during match
            reward += units_lost * self.unit_loss_penalty
            print(f"💀 {player} lost {units_lost} units! {units_lost * self.unit_loss_penalty:.3f}")
        
        # 5. Survival reward (encourage staying alive)
        if not match_ended and next_unit_count > 0:
            reward += self.survival_reward
        
        # 6. Exploration (map control)
        if 'sensor_mask' in next_obs:
            sensor = next_obs['sensor_mask']
            map_size = len(sensor)
            new_tiles = 0
            for y in range(map_size):
                for x in range(map_size):
                    if sensor[y][x] and (x, y) not in self.explored_tiles[player]:
                        self.explored_tiles[player].add((x, y))
                        new_tiles += 1
            if new_tiles > 0:
                reward += new_tiles * self.exploration_reward
                # print(f"🗺️ {player} explored {new_tiles} tiles! +{new_tiles * self.exploration_reward:.3f}")
        
        # 7. Relic discovery (strategic information)
        if 'relic_nodes' in next_obs and 'relic_nodes_mask' in next_obs:
            relics = next_obs['relic_nodes']
            mask = next_obs['relic_nodes_mask']
            
            new_relics = 0
            for relic_pos, valid in zip(relics, mask):
                if valid and relic_pos[0] >= 0:
                    relic_tuple = tuple(relic_pos)
                    if relic_tuple not in self.known_relics[player]:
                        self.known_relics[player].add(relic_tuple)
                        new_relics += 1
            
            if new_relics > 0:
                reward += new_relics * self.relic_discovery_reward
                # print(f"💎 {player} discovered {new_relics} relics! +{new_relics * self.relic_discovery_reward:.3f}")
        
        return reward
    
    def _compute_total_energy(self, obs, team_id):
        """Compute total energy for a team"""
        if 'units' not in obs or 'energy' not in obs['units']:
            return 0.0
        
        energies = np.array(obs['units']['energy'][team_id])
        if 'units_mask' in obs:
            unit_mask = np.array(obs['units_mask'][team_id])
            valid_energies = energies[unit_mask]
        else:
            valid_energies = energies[energies >= 0]
        
        return np.sum(valid_energies) if len(valid_energies) > 0 else 0.0
    
    def _count_active_units(self, obs, team_id):
        """Count active units for a team"""
        if 'units_mask' in obs:
            return np.sum(obs['units_mask'][team_id])
        elif 'units' in obs and 'energy' in obs['units']:
            energies = np.array(obs['units']['energy'][team_id])
            return np.sum(energies >= 0)
        return 0
    
    def reset_for_game(self):
        """Reset for new game/episode"""
        self.prev_score = {'player_0': 0, 'player_1': 0}
        self.prev_energy = {'player_0': 0, 'player_1': 0}
        self.known_relics = {'player_0': set(), 'player_1': set()}
        self.explored_tiles = {'player_0': set(), 'player_1': set()}
        self.prev_unit_count = {'player_0': 0, 'player_1': 0}
        print("🔄 Reward calculator reset for new game")
    
    def reset_for_match(self):
        """Reset for new match (keep relic and exploration memory)"""
        self.prev_score = {'player_0': 0, 'player_1': 0}
        self.prev_energy = {'player_0': 0, 'player_1': 0}
        self.prev_unit_count = {'player_0': 0, 'player_1': 0}
        # Keep relics and exploration memory between matches
        print("🔄 Reward calculator reset for new match (keeping memory)")


def create_reward_calculator(use_sparse=True):
    """Factory function"""
    return WorkingRewardCalculator(use_sparse=use_sparse)