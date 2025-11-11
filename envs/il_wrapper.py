"""
IL Wrapper: Loads pretrained Unit-UNet and provides predictions for reward shaping
Uses State class to maintain exact feature construction matching IL training
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from scipy.signal import convolve2d

# Import from project dependencies
from IL.agent.path import Action, ActionType
from IL.agent.base import (
    Global,
    SPACE_SIZE,
    transpose,
    get_opposite,
    get_nebula_tile_drift_speed,
)
from IL.agent.space import NodeType
from IL.agent.fleet import find_nebula_vision_reduction
from IL.agent.state import State


class ILWrapper:
    """
    Wrapper for IL model that maintains State objects per environment
    and provides unit-based action predictions for reward shaping
    """
    
    def __init__(
        self,
        model_path: str,
        num_envs: int,
        device: str = "cuda",
        map_size: int = 24,
        max_units: int = 16,
    ):
        self.device = device
        self.num_envs = num_envs
        self.map_size = map_size
        self.max_units = max_units
        
        # Load IL model
        print(f"[IL Wrapper] Loading IL model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"IL model not found at {model_path}")
        
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        print(f"[IL Wrapper] IL model loaded successfully")

        self._initialize_global_params()
        
        # State objects per environment (per team)
        # states[env_idx][team_id] = State object
        self.states: List[Dict[int, State]] = [
            {0: None, 1: None} for _ in range(num_envs)
        ]
        
        # History buffers per environment per team
        self.previous_step_units = [
            {0: np.zeros((4, SPACE_SIZE, SPACE_SIZE), dtype=np.float16),
             1: np.zeros((4, SPACE_SIZE, SPACE_SIZE), dtype=np.float16)}
            for _ in range(num_envs)
        ]
        
        self.previous_step_sap = [
            {0: np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float16),
             1: np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float16)}
            for _ in range(num_envs)
        ]
        
        self.previous_step_sap_positions = [
            {0: set(), 1: set()} for _ in range(num_envs)
        ]
        
        self.previous_step_opp_ships = [
            {0: set(), 1: set()} for _ in range(num_envs)
        ]
        
        # SAP kernel for convolution
        r = Global.UNIT_SAP_RANGE * 2 + 1
        self.sap_kernel = np.ones((r, r), dtype=np.int32)
        
        # Track if we need to reset state on next update
        self.needs_reset = [{0: True, 1: True} for _ in range(num_envs)]
        
        # Track environments where State sync failed (to avoid spam warnings)
        self._skipped_envs = set()
    
    def _initialize_global_params(self):
        """
        Initialize Global parameters that IL agent's State/Space code expects.
        These must be set before creating State objects.
        """
        # Set default values for Global parameters
        # These will be updated with actual game params in predict()
        
        # Obstacle movement direction - set to None initially
        if not hasattr(Global, 'OBSTACLE_MOVEMENT_DIRECTION'):
            Global.OBSTACLE_MOVEMENT_DIRECTION = None
            Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = False
        
        # Other Global parameters that might be needed
        if not hasattr(Global, 'MAX_UNIT_ENERGY'):
            Global.MAX_UNIT_ENERGY = 400
        
        if not hasattr(Global, 'UNIT_MOVE_COST'):
            Global.UNIT_MOVE_COST = 1
        
        if not hasattr(Global, 'UNIT_SAP_COST'):
            Global.UNIT_SAP_COST = 10
        
        if not hasattr(Global, 'UNIT_SAP_RANGE'):
            Global.UNIT_SAP_RANGE = 4
        
        if not hasattr(Global, 'UNIT_SENSOR_RANGE'):
            Global.UNIT_SENSOR_RANGE = 7
        
        if not hasattr(Global, 'MAX_STEPS_IN_MATCH'):
            Global.MAX_STEPS_IN_MATCH = 500
        
        if not hasattr(Global, 'NUM_MATCHES_IN_GAME'):
            Global.NUM_MATCHES_IN_GAME = 3
        
        if not hasattr(Global, 'UNIT_SAP_DROPOFF_FACTOR'):
            Global.UNIT_SAP_DROPOFF_FACTOR = 0.5
            Global.UNIT_SAP_DROPOFF_FACTOR_FOUND = False
        
        if not hasattr(Global, 'RELIC_RESULTS'):
            Global.RELIC_RESULTS = []
        
        if not hasattr(Global, 'NEBULA_VISION_REDUCTION_OPTIONS'):
            Global.NEBULA_VISION_REDUCTION_OPTIONS = []
        
        print(f"[IL Wrapper] Initialized Global parameters")
    
    def reset(self, env_idx: Optional[int] = None):
        """Reset state for specific environment or all environments"""
        if env_idx is not None:
            self.states[env_idx] = {0: None, 1: None}
            self.previous_step_units[env_idx][0][:] = 0
            self.previous_step_units[env_idx][1][:] = 0
            self.previous_step_sap[env_idx][0][:] = 0
            self.previous_step_sap[env_idx][1][:] = 0
            self.previous_step_sap_positions[env_idx][0].clear()
            self.previous_step_sap_positions[env_idx][1].clear()
            self.previous_step_opp_ships[env_idx][0].clear()
            self.previous_step_opp_ships[env_idx][1].clear()
            self.needs_reset[env_idx] = {0: True, 1: True}
            # Clear skipped flag - allow retry after reset
            self._skipped_envs.discard(env_idx)
        else:
            for i in range(self.num_envs):
                self.reset(i)
    
    def _convert_to_state_obs(self, obs: Dict, team_id: int, state: 'State') -> Dict:
        """Convert vectorized obs dict to format expected by State class"""
        # This mimics convert_episode_step from convert_episodes.py
        sensor_mask = np.array(obs["sensor_mask"], dtype=np.int8)
        
        obs_energy = np.array(obs["map_features"]["energy"], dtype=np.int8)
        obs_energy[sensor_mask == 0] = -1
        
        tile_type = np.array(obs["map_features"]["tile_type"], dtype=np.int8)
        tile_type[sensor_mask == 0] = -1
        
        # Relic nodes
        relic_nodes = []
        relic_nodes_mask = []
        for i in range(6):
            x, y = obs["relic_nodes"][i]
            is_visible = obs["relic_nodes_mask"][i]
            if is_visible and x >= 0 and y >= 0:
                if sensor_mask[int(x), int(y)]:
                    relic_nodes.append([int(x), int(y)])
                    relic_nodes_mask.append(True)
                else:
                    relic_nodes.append([-1, -1])
                    relic_nodes_mask.append(False)
            else:
                relic_nodes.append([-1, -1])
                relic_nodes_mask.append(False)
        
        # Units
        units_mask = [[], []]
        units_energy = [[], []]
        units_position = [[], []]
        
        for t in range(2):
            for i in range(self.max_units):
                x, y = obs["units"]["position"][t][i]
                energy = obs["units"]["energy"][t][i]
                is_visible = obs["units_mask"][t][i]
                
                if is_visible:
                    if t == team_id or (sensor_mask[int(x), int(y)] if x >= 0 and y >= 0 else False):
                        units_mask[t].append(True)
                        units_energy[t].append(int(energy))
                        units_position[t].append([int(x), int(y)])
                    else:
                        units_mask[t].append(False)
                        units_energy[t].append(-1)
                        units_position[t].append([-1, -1])
                else:
                    units_mask[t].append(False)
                    units_energy[t].append(-1)
                    units_position[t].append([-1, -1])
        
        state_obs = {
            "steps": state.global_step,  # Use State's own step counter, not env step
            "match_steps": state.match_step,
            "team_wins": [int(obs["team_wins"][0]), int(obs["team_wins"][1])],
            "team_points": [int(obs["team_points"][0]), int(obs["team_points"][1])],
            "sensor_mask": sensor_mask,
            "relic_nodes": relic_nodes,
            "relic_nodes_mask": relic_nodes_mask,
            "map_features": {"energy": obs_energy, "tile_type": tile_type},
            "units_mask": units_mask,
            "units": {"energy": units_energy, "position": units_position},
        }
        
        return state_obs
    
    def _build_features(
        self,
        state: State,
        env_idx: int,
        team_id: int,
        game_params: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 28 feature channels and 17 global features
        Adapted from pars_obs() in convert_episodes.py
        """
        d = np.zeros((28, SPACE_SIZE, SPACE_SIZE), dtype=np.float16)
        
        # Channels 0-3: Current unit positions and energies
        for unit in state.fleet:
            if unit.energy >= 0:
                x, y = unit.coordinates
                d[0, y, x] += 1
                d[1, y, x] += unit.energy
        
        for unit in state.opp_fleet:
            if unit.energy >= 0:
                x, y = unit.coordinates
                d[2, y, x] += 1
                d[3, y, x] += unit.energy
        
        d[0] /= 10
        d[1] /= Global.MAX_UNIT_ENERGY
        d[2] /= 10
        d[3] /= Global.MAX_UNIT_ENERGY
        
        # Channels 4-7: Previous step (filled from history)
        d[4:8] = self.previous_step_units[env_idx][team_id]
        
        # Channel 8: Previous step SAP positions
        d[8] = self.previous_step_sap[env_idx][team_id].copy()
        
        # Channels 9-21: Field features
        f = state.field
        d[9] = f.vision
        d[10] = f.energy / Global.MAX_UNIT_ENERGY
        d[11] = f.asteroid
        d[12] = f.nebulae
        d[13] = f.relic
        d[14] = f.reward
        d[15] = f.need_to_explore_for_relic
        d[16] = f.need_to_explore_for_reward
        d[17] = f.num_units_in_sap_range / 10
        d[18] = f.num_opp_units_in_sap_range / 10
        d[19] = f.fleet_vision(state.opp_fleet, min(Global.NEBULA_VISION_REDUCTION_OPTIONS))
        d[20] = (state.global_step - f.last_relic_check) / Global.MAX_STEPS_IN_MATCH
        d[21] = (state.global_step - f.last_step_in_vision) / Global.MAX_STEPS_IN_MATCH
        
        # Channels 22-23: Out of vision opponent units
        for unit in state.opp_fleet.ships:
            if (
                unit.node is not None
                and unit.energy >= 0
                and unit.steps_since_last_seen > 0
            ):
                x, y = unit.coordinates
                d[22, y, x] += 1
                d[23, y, x] += unit.energy
        
        d[22] /= 10
        d[23] /= Global.MAX_UNIT_ENERGY
        
        # Channels 24-27: Unit action capabilities
        for unit in state.fleet:
            x, y = unit.coordinates
            if unit.energy < 0:
                d[24, y, x] += 1
            if unit.energy >= Global.UNIT_MOVE_COST:
                d[25, y, x] += 1
            if unit.energy >= Global.UNIT_SAP_COST:
                d[26, y, x] += 1
                d[27, y, x] += 1
        
        d[27] = convolve2d(
            d[27],
            self.sap_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        
        d[24] /= 10
        d[25] /= 10
        d[26] /= 10
        d[27] /= 10
        
        # Global features (17 features)
        if (
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
        ):
            nebula_tile_drift_direction = (
                1 if get_nebula_tile_drift_speed() > 0 else -1
            )
            num_steps_before_obstacle_movement = (
                state.num_steps_before_obstacle_movement()
            )
        else:
            nebula_tile_drift_direction = 0
            num_steps_before_obstacle_movement = -Global.MAX_STEPS_IN_MATCH
        
        gf = np.array([
            nebula_tile_drift_direction,
            (
                game_params.get("nebula_tile_energy_reduction", 0) / Global.MAX_UNIT_ENERGY
                if Global.NEBULA_ENERGY_REDUCTION_FOUND
                else -1
            ),
            Global.UNIT_MOVE_COST / Global.MAX_UNIT_ENERGY,
            Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY,
            Global.UNIT_SAP_RANGE / SPACE_SIZE,
            (
                game_params.get("unit_sap_dropoff_factor", 0)
                if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND
                else -1
            ),
            (
                game_params.get("unit_energy_void_factor", 0)
                if Global.UNIT_ENERGY_VOID_FACTOR_FOUND
                else -1
            ),
            state.match_step / Global.MAX_STEPS_IN_MATCH,
            state.match_number / Global.NUM_MATCHES_IN_GAME,
            num_steps_before_obstacle_movement / Global.MAX_STEPS_IN_MATCH,
            state.fleet.points / 1000,
            state.opp_fleet.points / 1000,
            state.fleet.reward / 1000,
            state.opp_fleet.reward / 1000,
            sum(Global.RELIC_RESULTS) / 3,
            min(Global.NEBULA_VISION_REDUCTION_OPTIONS) / 8 if Global.NEBULA_VISION_REDUCTION_OPTIONS else 0,
            float(Global.ALL_RELICS_FOUND),
        ], dtype=np.float32)
        
        return d, gf
    
    def _update_history(
        self,
        state: State,
        env_idx: int,
        team_id: int,
        current_features: np.ndarray,
        rl_actions: Optional[np.ndarray] = None,
    ):
        """Update history buffers for next step"""
        # Store current units as previous for next step
        self.previous_step_units[env_idx][team_id] = current_features[:4].copy()
        
        # Update SAP positions if we have actions
        if rl_actions is not None:
            unit_sap_dropoff_factor = Global.UNIT_SAP_DROPOFF_FACTOR if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND else 0.5
            
            # Decay previous SAP
            self.previous_step_sap[env_idx][team_id] *= unit_sap_dropoff_factor
            
            # Add new SAP positions from actions
            for i, unit in enumerate(state.fleet.ships):
                if i < len(rl_actions) and unit.node is not None and unit.can_sap():
                    action_type = rl_actions[i, 0]
                    if action_type == 5:  # SAP action
                        # Get SAP target from RL action
                        dx = int(rl_actions[i, 1]) - Global.UNIT_SAP_RANGE
                        dy = int(rl_actions[i, 2]) - Global.UNIT_SAP_RANGE
                        sap_x = unit.node.x + dx
                        sap_y = unit.node.y + dy
                        
                        if 0 <= sap_x < SPACE_SIZE and 0 <= sap_y < SPACE_SIZE:
                            self.previous_step_sap[env_idx][team_id][sap_y, sap_x] += 1
    
    def predict(
        self,
        obs_batch: List[Dict],
        team_ids: List[int],
        game_params: Dict,
        rl_actions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get IL predictions for batch of observations
        
        Args:
            obs_batch: List of observation dicts (one per env)
            team_ids: List of team IDs (one per env)
            game_params: Game parameters dict
            rl_actions: Optional RL actions for history update (num_envs, max_units, 3)
        
        Returns:
            il_actions: (num_envs, max_units, 3) - action_type, sap_dx, sap_dy
            il_agreement_mask: (num_envs, max_units) - bool mask for valid predictions
        """
        batch_size = len(obs_batch)
        
        # Prepare batch tensors
        states_batch = []
        gfs_batch = []
        valid_envs = []
        
        for env_idx, (obs, team_id) in enumerate(zip(obs_batch, team_ids)):
            # Initialize or reset state if needed
            if self.states[env_idx][team_id] is None or self.needs_reset[env_idx][team_id]:
                self.states[env_idx][team_id] = State(team_id)
                self.needs_reset[env_idx][team_id] = False
                
                # Reset history
                self.previous_step_units[env_idx][team_id][:] = 0
                self.previous_step_sap[env_idx][team_id][:] = 0
            
            # Check for match reset
            state = self.states[env_idx][team_id]
            if obs.get("match_steps", 0) == 0:
                # Match just reset
                self.previous_step_units[env_idx][team_id][:] = 0
                self.previous_step_sap[env_idx][team_id][:] = 0
            
            # Convert obs and update state
            state_obs = self._convert_to_state_obs(obs, team_id, state)
            
            # Try to update state
            try:
                state.update(state_obs)
                
            except (AssertionError, Exception) as e:
                # State update failed (likely step mismatch) - skip IL for this env
                # This happens when State's internal counters can't sync with current env step
                if env_idx not in self._skipped_envs:
                    print(f"[IL Wrapper] Warning: Skipping env {env_idx} team {team_id} - State sync failed: {e}")
                    print(f"state global_steps {state_obs['steps']}")
                    print(f"state match_steps {state_obs['match_steps']}")
                    print(f"IL    global_steps {state.global_step}")
                    print(f"IL    match_steps {state.match_step}")
                    self._skipped_envs.add(env_idx)
                continue
            
            # Build features
            features, gf = self._build_features(state, env_idx, team_id, game_params)
            
            # Handle team_id == 1 mirroring (IL trained on team 0 spawn)
            if team_id == 1:
                features = transpose(features, reflective=True)
            
            states_batch.append(features)
            gfs_batch.append(gf)
            valid_envs.append(env_idx)
            
            # Update history
            if rl_actions is not None:
                self._update_history(state, env_idx, team_id, features, rl_actions[env_idx])
        
        # Handle empty batch (all envs skipped)
        if len(states_batch) == 0:
            print("[IL Wrapper] Warning: All environments skipped - no IL predictions")
            il_actions = np.zeros((batch_size, self.max_units, 3), dtype=np.int32)
            il_agreement_mask = np.zeros((batch_size, self.max_units), dtype=bool)
            return il_actions, il_agreement_mask
        
        # Convert to tensors
        actual_batch_size = len(states_batch)
        states_tensor = torch.from_numpy(np.stack(states_batch)).float().to(self.device)
        
        # GF needs to be (B, 17, 3, 3) for the model
        gfs_array = np.stack(gfs_batch)  # (actual_batch_size, 17)
        gfs_tensor = torch.zeros((actual_batch_size, 17, 3, 3), dtype=torch.float32, device=self.device)
        for i in range(17):
            gfs_tensor[:, i, :, :] = torch.from_numpy(gfs_array[:, i:i+1]).reshape(-1, 1, 1)
        
        # Run IL model
        with torch.no_grad():
            logits = self.model(states_tensor, gfs_tensor)  # (actual_batch_size, 6, H, W)
            probs = torch.softmax(logits, dim=1)
            action_types = torch.argmax(probs, dim=1)  # (actual_batch_size, H, W)
        
        action_types_np = action_types.cpu().numpy()
        
        # Convert position-based predictions to unit-based actions
        # Initialize outputs for ALL environments (not just valid ones)
        il_actions = np.zeros((batch_size, self.max_units, 3), dtype=np.int32)
        il_agreement_mask = np.zeros((batch_size, self.max_units), dtype=bool)
        
        # Fill in predictions only for valid environments
        for batch_idx, (env_idx, team_id) in enumerate(zip(valid_envs, [team_ids[i] for i in valid_envs])):
            state = self.states[env_idx][team_id]
            if state is None:
                # State is None - skip this environment
                continue
            
            obs = obs_batch[env_idx]
            
            # Get unit positions
            unit_positions = {}  # position -> list of unit indices
            for i, unit in enumerate(state.fleet.ships):
                if i < self.max_units and unit.node is not None and unit.is_visible:
                    pos = unit.coordinates
                    if pos not in unit_positions:
                        unit_positions[pos] = []
                    unit_positions[pos].append((i, unit.energy))
            
            # Get IL predictions for each position (use batch_idx, not env_idx)
            action_map = action_types_np[batch_idx]
            
            # Handle team_id == 1 mirroring back
            if team_id == 1:
                action_map = transpose(action_map)
            
            for pos, units_at_pos in unit_positions.items():
                x, y = pos
                if 0 <= x < SPACE_SIZE and 0 <= y < SPACE_SIZE:
                    il_action_type = int(action_map[y, x])
                    
                    # Sort units by energy (descending)
                    units_at_pos.sort(key=lambda u: u[1], reverse=True)
                    
                    # Assign IL action to top half of units (following IL training approach)
                    num_units = len(units_at_pos)
                    num_to_assign = max(1, num_units // 2) if num_units > 1 else 1
                    
                    for idx in range(num_to_assign):
                        unit_idx, _ = units_at_pos[idx]
                        il_actions[env_idx, unit_idx, 0] = il_action_type
                        
                        # For SAP, set target to center (can be improved with SAP-UNet)
                        if il_action_type == 5:
                            il_actions[env_idx, unit_idx, 1] = Global.UNIT_SAP_RANGE
                            il_actions[env_idx, unit_idx, 2] = Global.UNIT_SAP_RANGE
                        
                        il_agreement_mask[env_idx, unit_idx] = True
        
        return il_actions, il_agreement_mask