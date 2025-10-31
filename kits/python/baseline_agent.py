"""
Baseline Agent Wrapper for Training

This wraps the simple baseline agent to work with your training environment.
Your RL agent will train against this to get a clear learning signal.
"""

import numpy as np
import jax.numpy as jnp


class BaselineAgent:
    """
    Simple but effective baseline agent that:
    1. Explores randomly to find relics
    2. Moves toward relics once found
    3. Hovers near relics to score points
    """
    def __init__(self, player="player_1", env_cfg=None):
        self.player = player
        self.player_id = 1 if player == "player_1" else 0
        self.team_id = self.player_id
        self.opp_team_id = 1 - self.player_id
        
        # Use default config if not provided
        if env_cfg is None:
            env_cfg = {
                "max_units": 16,
                "map_width": 24,
                "map_height": 24,
                "max_relic_nodes": 10
            }
        self.env_cfg = env_cfg
        
        # Memory
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = {}
        self.step_counter = 0
        
        np.random.seed(42 + self.player_id)
    
    def reset(self):
        """Reset for new game"""
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = {}
        self.step_counter = 0
    
    def reset_match(self):
        """Reset between matches (keep relic knowledge)"""
        self.unit_explore_locations = {}
        # Keep relic_node_positions for next match!
    
    def direction_to(self, pos1, pos2):
        """Calculate direction from pos1 to pos2 (0=NO_OP, 1=UP, 2=RIGHT, 3=DOWN, 4=LEFT)"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        if dx == 0 and dy == 0:
            return 0  # NO_OP
        
        # Prefer larger axis
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 4  # RIGHT or LEFT
        else:
            return 1 if dy < 0 else 3  # UP or DOWN
    
    def _extract_observation_data(self, obs):
        """
        Extract observation data from various formats:
        - JAX arrays
        - EnvObs objects  
        - Dict format
        """
        # Convert JAX arrays to numpy if needed
        if hasattr(obs, '__array__'):
            obs = np.array(obs)
        
        # Handle numpy array observation
        if isinstance(obs, np.ndarray):
            # If it's already a processed array, return empty data
            return {
                'unit_mask': np.ones(16, dtype=bool),
                'unit_positions': np.zeros((16, 2)),
                'unit_energies': np.zeros(16),
                'relic_positions': np.zeros((10, 2)),
                'relic_mask': np.zeros(10, dtype=bool)
            }
        
        # Handle dictionary format
        if isinstance(obs, dict):
            unit_mask = np.array(obs.get("units_mask", [True] * 16))
            if len(unit_mask.shape) == 2:
                unit_mask = unit_mask[self.team_id]
            
            # Get unit positions
            if "units" in obs and "position" in obs["units"]:
                unit_positions = np.array(obs["units"]["position"])
                if len(unit_positions.shape) == 3:
                    unit_positions = unit_positions[self.team_id]
            else:
                unit_positions = np.zeros((16, 2))
            
            # Get unit energies
            if "units" in obs and "energy" in obs["units"]:
                unit_energies = np.array(obs["units"]["energy"])
                if len(unit_energies.shape) == 2:
                    unit_energies = unit_energies[self.team_id]
            else:
                unit_energies = np.zeros(16)
            
            # Get relic nodes
            if "relic_nodes" in obs:
                relic_positions = np.array(obs["relic_nodes"])
                relic_mask = np.array(obs.get("relic_nodes_mask", [True] * len(relic_positions)))
            else:
                relic_positions = np.zeros((10, 2))
                relic_mask = np.zeros(10, dtype=bool)
            
            return {
                'unit_mask': unit_mask,
                'unit_positions': unit_positions,
                'unit_energies': unit_energies,
                'relic_positions': relic_positions,
                'relic_mask': relic_mask
            }
        
        # Handle EnvObs format (with attributes)
        unit_mask = np.ones(16, dtype=bool)
        unit_positions = np.zeros((16, 2))
        unit_energies = np.zeros(16)
        relic_positions = np.zeros((10, 2))
        relic_mask = np.zeros(10, dtype=bool)
        
        try:
            if hasattr(obs, 'units'):
                # Convert JAX arrays to numpy
                if hasattr(obs.units, 'mask'):
                    mask_data = obs.units.mask
                    if hasattr(mask_data, '__array__'):
                        unit_mask = np.array(mask_data)[self.team_id]
                
                if hasattr(obs.units, 'position'):
                    pos_data = obs.units.position
                    if hasattr(pos_data, '__array__'):
                        unit_positions = np.array(pos_data)[self.team_id]
                
                if hasattr(obs.units, 'energy'):
                    energy_data = obs.units.energy
                    if hasattr(energy_data, '__array__'):
                        unit_energies = np.array(energy_data)[self.team_id]
            
            if hasattr(obs, 'relic_nodes'):
                if hasattr(obs.relic_nodes, 'position'):
                    relic_data = obs.relic_nodes.position
                    if hasattr(relic_data, '__array__'):
                        relic_positions = np.array(relic_data)
                
                if hasattr(obs.relic_nodes, 'mask'):
                    relic_mask_data = obs.relic_nodes.mask
                    if hasattr(relic_mask_data, '__array__'):
                        relic_mask = np.array(relic_mask_data)
        except Exception as e:
            print(f"⚠️ Observation extraction error: {e}")
        
        return {
            'unit_mask': unit_mask,
            'unit_positions': unit_positions,
            'unit_energies': unit_energies,
            'relic_positions': relic_positions,
            'relic_mask': relic_mask
        }
    
    def act(self, step, obs, remaining_overage_time=60):
        """
        Generate actions from observation
        
        Args:
            step: Current step number
            obs: Observation dict or EnvObs object
            remaining_overage_time: Time remaining
        
        Returns:
            actions: (max_units, 3) array of actions
        """
        self.step_counter = step
        
        # Extract observation data
        obs_data = self._extract_observation_data(obs)
        
        unit_mask = obs_data['unit_mask']
        unit_positions = obs_data['unit_positions']
        unit_energies = obs_data['unit_energies']
        observed_relic_node_positions = obs_data['relic_positions']
        observed_relic_nodes_mask = obs_data['relic_mask']
        
        # Find available units
        available_unit_ids = np.where(unit_mask)[0]
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        # Initialize actions
        actions = np.zeros((16, 3), dtype=np.int32)
        
        # Save newly discovered relic nodes
        for relic_id in visible_relic_node_ids:
            if relic_id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(relic_id)
                relic_pos = observed_relic_node_positions[relic_id]
                if relic_pos[0] >= 0 and relic_pos[1] >= 0:  # Valid position
                    self.relic_node_positions.append(relic_pos)
        
        # Generate actions for each unit
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            
            # Skip if invalid position (respawning)
            if unit_pos[0] < 0 or unit_pos[1] < 0:
                actions[unit_id] = [0, 0, 0]  # NO_OP
                continue
            
            if len(self.relic_node_positions) > 0:
                # Strategy: Move toward nearest relic
                nearest_relic = min(self.relic_node_positions, 
                                  key=lambda r: abs(unit_pos[0] - r[0]) + abs(unit_pos[1] - r[1]))
                manhattan_distance = abs(unit_pos[0] - nearest_relic[0]) + abs(unit_pos[1] - nearest_relic[1])
                
                if manhattan_distance <= 4:
                    # Close to relic: hover around it randomly
                    if np.random.random() < 0.7:  # 70% chance to stay near relic
                        # Small random movements around relic
                        random_direction = np.random.randint(1, 5)  # 1-4 for movement
                        actions[unit_id] = [random_direction, 0, 0]
                    else:
                        actions[unit_id] = [0, 0, 0]  # NO_OP sometimes
                else:
                    # Far from relic: move toward it
                    direction = self.direction_to(unit_pos, nearest_relic)
                    if direction != 0:  # Only move if we have a valid direction
                        actions[unit_id] = [direction, 0, 0]
                    else:
                        actions[unit_id] = [0, 0, 0]
            else:
                # No relics found yet: explore randomly
                if self.step_counter % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (
                        np.random.randint(0, 24),
                        np.random.randint(0, 24)
                    )
                    self.unit_explore_locations[unit_id] = rand_loc
                
                target_loc = self.unit_explore_locations[unit_id]
                direction = self.direction_to(unit_pos, target_loc)
                if direction != 0:  # Only move if we have a valid direction
                    actions[unit_id] = [direction, 0, 0]
                else:
                    # Reached target, choose new random direction
                    random_direction = np.random.randint(1, 5)
                    actions[unit_id] = [random_direction, 0, 0]
        
        return actions


def create_baseline_agent(player="player_1", env_cfg=None):
    """Factory function to create baseline agent"""
    return BaselineAgent(player=player, env_cfg=env_cfg)