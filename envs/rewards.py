import numpy as np
from typing import Dict, Tuple, Optional
import torch

try:
    from envs.il_rewards import ILRewardShaper
    IL_AVAILABLE = True
except ImportError:
    IL_AVAILABLE = False
    print("[Rewards] IL reward module not available - IL rewards disabled")


class LuxRewardShaper:
    
    def __init__(
        self,
        reward_mode: str = "dense",  # "sparse" or "dense"

        # Sparse reward weights
        match_win_bonus: float = 10.0,
        episode_win_bonus: float = 50.0,
        
        # Dense reward weights
        relic_point_reward: float = 5.0,
        damage_dealt_reward: float = 0.01,
        energy_differential_reward: float = 0.001,
        unit_loss_penalty: float = -1.0,
        survival_reward: float = 0.05,
        exploration_reward: float = 0.1,
        relic_discovery_reward: float = 5.0,
        on_relic_bonus: float = 0.1,
        
        # IL reward shaping
        il_reward_shaper = None,  # Optional ILRewardShaper instance
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
            on_relic_bonus: Bonus per unit standing on relic tiles per step
            il_reward_shaper: Optional ILRewardShaper for imitation learning rewards
        """
        
        self.reward_mode = reward_mode
        self.il_reward_shaper = il_reward_shaper
        
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
        self.on_relic_bonus = on_relic_bonus
        
        # State tracking for dense rewards
        self.prev_state = None
    
    def compute_reward(
        self,
        obs: Dict,
        next_obs: Dict,
        done: bool,
        info: Dict,
        team_id: int,
        actions: Optional[np.ndarray] = None,
        global_step: Optional[int] = None,
    ) -> float:
        """
        Compute reward per team
        
        Args:
            obs: Current observation
            next_obs: Next observation
            done: Episode done flag
            info: Info dict
            team_id: Team ID (0 or 1)
            actions: Optional actions for IL reward (max_units, 3)
            global_step: Optional global step for IL annealing
        
        Returns:
            reward: Total reward (environment + IL)
        """
        ## reward per team
        if self.reward_mode == "sparse":
            env_reward = self._compute_sparse_reward(obs, next_obs, done, info, team_id)
        else:
            env_reward = self._compute_dense_reward(obs, next_obs, done, info, team_id)
        
        # IL rewards are handled in the training loop for efficiency
        # This method just returns the environment reward
        
        return env_reward
    
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
                # Won the match
                reward += self.match_win_bonus
            # else:
            #     # Lost the match
            #     reward -= self.match_win_bonus
        
        # Check if episode ended
        if done:
            final_wins = next_obs["team_wins"]
            if final_wins[team_id] > final_wins[1 - team_id]:
                # Won the episode
                reward += self.episode_win_bonus
            # elif final_wins[team_id] < final_wins[1 - team_id]:
            #     # Lost the episode
            #     reward -= self.episode_win_bonus
        
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
        
        # Exploration reward (decays with match progress)
        if self.exploration_reward != 0:
            curr_visible = np.sum(obs["sensor_mask"])
            next_visible = np.sum(next_obs["sensor_mask"])
            new_tiles = max(0, next_visible - curr_visible)
            
            # Decay exploration reward based on match progress
            # Early match (0-200 steps): full exploration reward
            # Mid-Late match (200-500 steps): exponentially decay to 0
            match_steps = next_obs.get("match_steps", 0)
            if match_steps <= 200:
                exploration_scale = 1.0
            else:
                # Exponential decay: goes from 1.0 at step 200 to ~0.01 at step 500
                progress = (match_steps - 200) / 300  # 0 to 1
                exploration_scale = np.exp(-5 * progress)  # e^(-5x)
            
            reward += new_tiles * self.exploration_reward * exploration_scale
        
        # Relic discovery reward
        if self.relic_discovery_reward != 0:
            curr_relics = np.sum(obs["relic_nodes_mask"])
            next_relics = np.sum(next_obs["relic_nodes_mask"])
            new_relics = max(0, next_relics - curr_relics)
            reward += new_relics * self.relic_discovery_reward
        
        # On-relic bonus
        if self.on_relic_bonus != 0 and match_steps > 0:
            # Count units near visible relic nodes
            unit_positions = np.array(next_obs["units"]["position"][team_id])
            unit_mask = np.array(next_obs["units_mask"][team_id])
            relic_positions = np.array(next_obs["relic_nodes"])
            relic_mask = np.array(next_obs["relic_nodes_mask"])
            
            units_on_relics = 0
            relic_config_radius = 2  # 5x5 config has radius of 2
            
            for i, is_active in enumerate(unit_mask):
                if is_active:
                    ux, uy = unit_positions[i]
                    
                    # Check if unit is near any visible relic
                    for j, relic_visible in enumerate(relic_mask):
                        if relic_visible:
                            rx, ry = relic_positions[j]
                            if rx >= 0 and ry >= 0:
                                # Manhattan distance to relic center
                                dx = abs(ux - rx)
                                dy = abs(uy - ry)
                                
                                # Within relic config area (5x5 grid)
                                if dx <= relic_config_radius and dy <= relic_config_radius:
                                    units_on_relics += 1
                                    break  # Count each unit only once
            
            reward += units_on_relics * self.on_relic_bonus
        
        return reward

        return reward
    
    def _compute_total_energy(self, obs: Dict, team_id: int) -> float:
        
        # total energy for team
        energies = np.array(obs["units"]["energy"][team_id])
        unit_mask = np.array(obs["units_mask"][team_id])
        
        valid_energies = energies[unit_mask]
        valid_energies = valid_energies[valid_energies >= 0]
        
        return np.sum(valid_energies) if len(valid_energies) > 0 else 0.0
    
    def reset(self):
        
        self.prev_state = None
    
    def compute_il_rewards_batch(
        self,
        obs_batch: list,
        team_ids: list,
        actions_batch: np.ndarray,
        unit_masks: np.ndarray,
        game_params: dict,
        global_step: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute IL rewards for a batch of observations (for efficiency)
        
        Args:
            obs_batch: List of observation dicts
            team_ids: List of team IDs
            actions_batch: (batch_size, max_units, 3) action arrays
            unit_masks: (batch_size, max_units) unit validity masks
            game_params: Game parameters dict
            global_step: Current training step
        
        Returns:
            il_rewards: (batch_size,) IL rewards
            info: Dictionary with IL statistics
        """
        if self.il_reward_shaper is None:
            # Return zeros if no IL shaper
            return np.zeros(len(obs_batch), dtype=np.float32), {}
        
        il_rewards, info = self.il_reward_shaper.compute_rewards(
            obs_batch=obs_batch,
            team_ids=team_ids,
            rl_actions=actions_batch,
            unit_masks=unit_masks,
            game_params=game_params,
            global_step=global_step,
        )
        
        return il_rewards, info
    
    def reset_il(self, env_idx: Optional[int] = None):
        """Reset IL state for specific or all environments"""
        if self.il_reward_shaper is not None:
            self.il_reward_shaper.reset(env_idx)
    
    def reset_il_statistics(self):
        """Reset IL statistics"""
        if self.il_reward_shaper is not None:
            self.il_reward_shaper.reset_statistics()


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