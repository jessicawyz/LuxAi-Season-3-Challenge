"""
LUX REWARD SHAPER for MAPPO Training
Based on the comprehensive LuxRewardShaper class
"""

import numpy as np
from typing import Dict, Tuple, Optional
import torch


class LuxRewardShaper:
    
    def __init__(
        self,
        reward_mode: str = "dense",  # "sparse" or "dense"

        # Sparse reward weights
        match_win_bonus: float = 10.0,
        episode_win_bonus: float = 50.0,
        
        # Dense reward weights
        relic_point_reward: float = 1.0,
        damage_dealt_reward: float = 0.01,
        energy_differential_reward: float = 0.001,
        unit_loss_penalty: float = -5.0,
        survival_reward: float = 0.05,
        exploration_reward: float = 0.5,
        relic_discovery_reward: float = 10.0,
    ):
        """
            reward_mode: "sparse" (only wins) or "dense" (wins + shaped rewards)
            match_win_bonus: Reward for winning a match
            episode_win_bonus: Reward for winning the episode (3 or more matches)
            relic_point_reward: Reward per relic point scored
            damage_dealt_reward: Reward per energy damage dealt to enemies
            energy_differential_reward: Reward for energy advantage
            unit_loss_penalty: Penalty when a unit is destroyed
            survival_reward: Small constant reward per step
            exploration_reward: Reward for exploring new tiles
            relic_discovery_reward: Reward for discovering new relic nodes
        """
        
        self.reward_mode = reward_mode
        
        # Sparse rewards
        self.match_win_bonus = match_win_bonus
        self.episode_win_bonus = episode_win_bonus
        
        # Dense rewards
        self.relic_point_reward = relic_point_reward
        self.damage_dealt_reward = damage_dealt_reward
        self.energy_differential_reward = energy_differential_reward
        self.unit_loss_penalty = unit_loss_penalty
        self.survival_reward = survival_reward
        self.exploration_reward = exploration_reward
        self.relic_discovery_reward = relic_discovery_reward
        
        # State tracking for dense rewards
        self.prev_state = None
    
    def compute_reward(
        self,
        obs: Dict,
        next_obs: Dict,
        done: bool,
        info: Dict,
        team_id: int,
    ) -> float:
        
        ## reward per team
        if self.reward_mode == "sparse":
            return self._compute_sparse_reward(obs, next_obs, done, info, team_id)
        else:
            return self._compute_dense_reward(obs, next_obs, done, info, team_id)
    
    def _compute_sparse_reward(
        self,
        obs: Dict,
        next_obs: Dict,
        done: bool,
        info: Dict,
        team_id: int,
    ) -> float:
        
        reward = 0.0
        
        # Check if match ended 
        curr_match_steps = obs.get("match_steps", 0)
        next_match_steps = next_obs.get("match_steps", 0)
        match_ended = next_match_steps < curr_match_steps
        
        if match_ended:
            # Match ended - check if win
            curr_wins = obs["team_wins"][team_id]
            next_wins = next_obs["team_wins"][team_id]
            
            if next_wins > curr_wins:
                reward += self.match_win_bonus
            else:
                reward -= self.match_win_bonus
        
        # Check if episode ended
        if done:
            final_wins = next_obs["team_wins"]
            if final_wins[team_id] > final_wins[1 - team_id]:
                # Won the episode
                reward += self.episode_win_bonus
            elif final_wins[team_id] < final_wins[1 - team_id]:
                # Lost the episode
                reward -= self.episode_win_bonus
        
        return reward
    
    def _compute_dense_reward(
        self,
        obs: Dict,
        next_obs: Dict,
        done: bool,
        info: Dict,
        team_id: int,
    ) -> float:
        
        reward = 0.0
        enemy_id = 1 - team_id
        
        # Start with sparse rewards
        reward += self._compute_sparse_reward(obs, next_obs, done, info, team_id)
        
        # Relic points gained
        curr_points = obs["team_points"][team_id]
        next_points = next_obs["team_points"][team_id]
        points_gained = next_points - curr_points
        reward += points_gained * self.relic_point_reward
        
        # Damage dealt reward 
        curr_enemy_energy = self._compute_total_energy(obs, enemy_id)
        next_enemy_energy = self._compute_total_energy(next_obs, enemy_id)
        damage_dealt = max(0, curr_enemy_energy - next_enemy_energy)
        reward += damage_dealt * self.damage_dealt_reward
        
        # Energy differential reward
        curr_team_energy = self._compute_total_energy(obs, team_id)
        next_team_energy = self._compute_total_energy(next_obs, team_id)
        
        curr_differential = curr_team_energy - curr_enemy_energy
        next_differential = next_team_energy - next_enemy_energy
        differential_change = next_differential - curr_differential
        reward += differential_change * self.energy_differential_reward
        
        # Unit loss penalty
        curr_unit_count = np.sum(obs["units_mask"][team_id])
        next_unit_count = np.sum(next_obs["units_mask"][team_id])
        units_lost = max(0, curr_unit_count - next_unit_count)
        reward += units_lost * self.unit_loss_penalty
        
        # Survival reward
        match_steps = next_obs.get("match_steps", 0)
        if match_steps > 0:  # Match is ongoing
            reward += self.survival_reward
        
        # Exploration reward 
        if self.exploration_reward != 0:
            curr_visible = np.sum(obs["sensor_mask"])
            next_visible = np.sum(next_obs["sensor_mask"])
            new_tiles = max(0, next_visible - curr_visible)
            reward += new_tiles * self.exploration_reward
        
        # Relic discovery reward
        if self.relic_discovery_reward != 0:
            curr_relics = np.sum(obs["relic_nodes_mask"])
            next_relics = np.sum(next_obs["relic_nodes_mask"])
            new_relics = max(0, next_relics - curr_relics)
            reward += new_relics * self.relic_discovery_reward
        
        return reward
    
    def _compute_total_energy(self, obs: Dict, team_id: int) -> float:
        """Compute total energy for a team"""
        if "units" not in obs or "energy" not in obs["units"]:
            return 0.0
            
        energies = np.array(obs["units"]["energy"][team_id])
        unit_mask = np.array(obs["units_mask"][team_id])
        
        valid_energies = energies[unit_mask]
        valid_energies = valid_energies[valid_energies >= 0]
        
        return np.sum(valid_energies) if len(valid_energies) > 0 else 0.0
    
    def reset(self):
        self.prev_state = None


class RelicMemory:
    """
    Remember relic nodes, persists within an episode, resets between episodes
    """
    
    def __init__(self, max_relic_nodes: int = 6, relic_config_size: int = 5):
        self.max_relic_nodes = max_relic_nodes
        self.relic_config_size = relic_config_size
        self.reset()
    
    def reset(self):
        self.discovered_relics = {} 
        self.relic_fragments = {} 
        self.last_match_num = -1
    
    def update(self, obs: Dict, team_points_gained: int, unit_positions: np.ndarray):
        # Detect match transitions
        curr_match_num = obs.get("steps", 0) // 100
        if curr_match_num != self.last_match_num:
            self.last_match_num = curr_match_num
        
        # Update discovered relic positions
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        
        for i in range(self.max_relic_nodes):
            if relic_mask[i]:
                x, y = relic_positions[i]
                if x >= 0 and y >= 0:
                    if i not in self.discovered_relics:
                        self.discovered_relics[i] = (int(x), int(y))
                        if i not in self.relic_fragments:
                            self.relic_fragments[i] = set()
        
        if team_points_gained > 0:
            # Check unit positions against known relic nodes
            for relic_id, (rx, ry) in self.discovered_relics.items():
                if relic_id not in self.relic_fragments:
                    self.relic_fragments[relic_id] = set()
                
                # Check each unit position relative to relic
                for pos in unit_positions:
                    x, y = pos
                    if x >= 0 and y >= 0:
                        dx = int(x - rx)
                        dy = int(y - ry)
                        
                        # Only consider positions within range
                        if abs(dx) <= self.relic_config_size // 2 and abs(dy) <= self.relic_config_size // 2:
                            self.relic_fragments[relic_id].add((dx, dy))
    
    def get_known_relics(self) -> Dict[int, Tuple[int, int]]:
        return self.discovered_relics.copy()
    
    def get_relic_fragments(self, relic_id: int) -> set:
        return self.relic_fragments.get(relic_id, set()).copy()
    
    def is_likely_scoring_position(self, x: int, y: int) -> bool:
        for relic_id, (rx, ry) in self.discovered_relics.items():
            dx = x - rx
            dy = y - ry
            if (dx, dy) in self.relic_fragments.get(relic_id, set()):
                return True
        return False
    
    def get_memory_features(self, map_width: int, map_height: int) -> np.ndarray:
        memory_map = np.zeros((2, map_width, map_height), dtype=np.float32)
        
        # Relic positions
        for relic_id, (rx, ry) in self.discovered_relics.items():
            if 0 <= rx < map_width and 0 <= ry < map_height:
                memory_map[0, rx, ry] = 1.0
        
        # Scoring tiles
        for relic_id, (rx, ry) in self.discovered_relics.items():
            fragments = self.relic_fragments.get(relic_id, set())
            for dx, dy in fragments:
                x, y = rx + dx, ry + dy
                if 0 <= x < map_width and 0 <= y < map_height:
                    memory_map[1, x, y] = 1.0
        
        return memory_map


class SimplifiedRewardCalculator:
    """
    Reward calculator optimized for MAPPO training using LuxRewardShaper
    - Simple interface compatible with train_mappo.py
    - Uses comprehensive LuxRewardShaper for reward calculation
    """
    def __init__(self, config, team_id: int):
        self.config = config
        self.team_id = team_id
        
        # Initialize the comprehensive reward shaper
        self.reward_shaper = LuxRewardShaper(
            reward_mode="sparse" if config.use_sparse_rewards else "dense",
            match_win_bonus=config.match_win_reward,
            episode_win_bonus=config.episode_win_bonus,
            relic_point_reward=config.relic_point_reward,
            damage_dealt_reward=config.damage_dealt_reward,
            energy_differential_reward=config.energy_differential_reward,
            unit_loss_penalty=config.unit_loss_penalty,
            survival_reward=config.survival_reward,
            exploration_reward=config.exploration_reward,
            relic_discovery_reward=config.relic_discovery_reward,
        )
        
        # Initialize relic memory for dense rewards
        self.relic_memory = RelicMemory()
        
    def calculate_reward(self, obs, next_obs, env_reward, done, info):
        """
        Calculate reward for a single step using LuxRewardShaper
        
        Args:
            obs: Current observation (EnvObs object)
            next_obs: Next observation (EnvObs object)  
            env_reward: Raw reward from environment
            done: Whether episode is done
            info: Additional info from environment
            
        Returns:
            float: Shaped reward
        """
        # Convert EnvObs to dictionary format
        obs_dict = self._extract_obs_dict(obs)
        next_obs_dict = self._extract_obs_dict(next_obs)
        
        # Calculate reward using the comprehensive shaper
        reward = self.reward_shaper.compute_reward(
            obs_dict, next_obs_dict, done, info, self.team_id
        )
        
        # Update relic memory for dense rewards (if not using sparse)
        if not self.config.use_sparse_rewards:
            self._update_relic_memory(obs_dict, next_obs_dict)
        
        return reward
    
    def _extract_obs_dict(self, obs):
        """Extract observation data from EnvObs object to dictionary"""
        if hasattr(obs, '__dict__'):  # It's an object with attributes
            obs_dict = {}
            
            # Extract basic team info
            if hasattr(obs, 'team_points'):
                obs_dict['team_points'] = obs.team_points
            else:
                obs_dict['team_points'] = [0, 0]
                
            if hasattr(obs, 'team_wins'):
                obs_dict['team_wins'] = obs.team_wins
            else:
                obs_dict['team_wins'] = [0, 0]
            
            # Extract units data
            obs_dict['units'] = {}
            if hasattr(obs, 'units'):
                obs_dict['units']['energy'] = getattr(obs.units, 'energy', [[], []])
                obs_dict['units']['position'] = getattr(obs.units, 'position', [[], []])
            else:
                obs_dict['units']['energy'] = [[], []]
                obs_dict['units']['position'] = [[], []]
            
            # Extract masks
            obs_dict['units_mask'] = getattr(obs, 'units_mask', [[], []])
            obs_dict['relic_nodes_mask'] = getattr(obs, 'relic_nodes_mask', [])
            obs_dict['sensor_mask'] = getattr(obs, 'sensor_mask', [])
            
            # Extract relic nodes
            obs_dict['relic_nodes'] = getattr(obs, 'relic_nodes', [])
            
            # Extract step information
            obs_dict['match_steps'] = getattr(obs, 'match_steps', 0)
            obs_dict['steps'] = getattr(obs, 'steps', 0)
            
        elif isinstance(obs, dict):  # It's already a dictionary
            obs_dict = obs
        else:  # Fallback
            obs_dict = {
                'team_points': [0, 0],
                'team_wins': [0, 0],
                'units': {
                    'energy': [[], []],
                    'position': [[], []]
                },
                'units_mask': [[], []],
                'relic_nodes': [],
                'relic_nodes_mask': [],
                'sensor_mask': [],
                'match_steps': 0,
                'steps': 0
            }
        
        return obs_dict
    
    def _update_relic_memory(self, obs, next_obs):
        """Update relic memory with current observations"""
        team_points_gained = next_obs["team_points"][self.team_id] - obs["team_points"][self.team_id]
        
        # Extract unit positions
        unit_positions = []
        positions = obs["units"]["position"][self.team_id]
        mask = obs["units_mask"][self.team_id]
        
        # Handle different position formats
        for i, pos in enumerate(positions):
            if i < len(mask) and mask[i]:
                if hasattr(pos, '__len__') and len(pos) >= 2:
                    unit_positions.append([float(pos[0]), float(pos[1])])
                elif hasattr(pos, 'x') and hasattr(pos, 'y'):
                    unit_positions.append([float(pos.x), float(pos.y)])
        
        self.relic_memory.update(next_obs, team_points_gained, np.array(unit_positions))
    
    def reset(self):
        """Reset state tracking"""
        self.reward_shaper.reset()
        self.relic_memory.reset()


def create_reward_calculator(config, team_id: int = 0):
    """Factory function to create reward calculator with config"""
    return SimplifiedRewardCalculator(config, team_id)