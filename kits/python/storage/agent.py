import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import jax.numpy as jnp
import os

class SimpleAgent:
    def __init__(self, config, model, ppo):
        self.config = config
        self.model = model
        self.ppo = ppo
        self.max_units = 16
        self.max_sap_range = config.max_sap_range
        
    def act(self, features, unit_info, player_id=0):
        """Returns actions, log_probs, and values for PPO training for a specific player"""
        spatial_obs = torch.FloatTensor(features['spatial']).unsqueeze(0)
        global_obs = torch.FloatTensor(features['global']).unsqueeze(0)
        
        actions = {}
        log_probs = {}
        values = {}
        
        # Prepare unit positions and energies for the model
        unit_positions = []
        unit_energies = []
        
        for unit in unit_info:
            unit_positions.append(unit['position'])
            unit_energies.append([unit['energy']])
        
        if unit_positions:
            unit_positions_tensor = torch.FloatTensor(unit_positions).unsqueeze(0)  # (1, n_units, 2)
            unit_energies_tensor = torch.FloatTensor(unit_energies).unsqueeze(0)    # (1, n_units, 1)
        else:
            # Handle no units case
            unit_positions_tensor = torch.zeros((1, 1, 2))
            unit_energies_tensor = torch.zeros((1, 1, 1))
        
        # Process all units at once for efficiency
        with torch.no_grad():
            main_action_logits, sap_logits, value = self.model(
                spatial_obs, global_obs, unit_positions_tensor, unit_energies_tensor
            )
        
        # Extract actions for each unit
        for i, unit in enumerate(unit_info):
            if i < main_action_logits.shape[1]:  # Make sure we don't exceed batch
                unit_logits = main_action_logits[0, i]  # Get logits for this unit
                
                # Apply action masks
                if unit['id'] in features['action_masks']:
                    mask = torch.BoolTensor(features['action_masks'][unit['id']])
                    # Ensure mask length matches action space
                    if len(mask) >= len(unit_logits):
                        unit_logits[~mask[:len(unit_logits)]] = -1e10
                    else:
                        # Pad mask if needed
                        padded_mask = torch.ones(len(unit_logits), dtype=torch.bool)
                        padded_mask[:len(mask)] = mask
                        unit_logits[~padded_mask] = -1e10
                
                dist = Categorical(logits=unit_logits.unsqueeze(0))
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                actions[unit['id']] = action.item()
                log_probs[unit['id']] = log_prob
                values[unit['id']] = value
        
        return actions, log_probs, values
    
    def act_for_env(self, obs, feature_engineer):
        """Returns actions in Lux environment format for both players"""
        # Process observations for both players
        features_p0 = feature_engineer.process(obs['player_0'], player_id=0)
        features_p1 = feature_engineer.process(obs['player_1'], player_id=1)
        
        # Get actions for both players
        actions_p0, log_probs_p0, values_p0 = self.act(features_p0, features_p0['unit_info'], player_id=0)
        actions_p1, log_probs_p1, values_p1 = self.act(features_p1, features_p1['unit_info'], player_id=1)
        
        # Convert to Lux environment format with smart SAP targeting
        player_0_actions = self._create_lux_action_array(actions_p0, features_p0['unit_info'], obs['player_0'], 0)
        player_1_actions = self._create_lux_action_array(actions_p1, features_p1['unit_info'], obs['player_1'], 1)
        
        lux_actions = {
            "player_0": player_0_actions,
            "player_1": player_1_actions
        }
        
        training_data = {
            "player_0": (actions_p0, log_probs_p0, values_p0, features_p0),
            "player_1": (actions_p1, log_probs_p1, values_p1, features_p1)
        }
        
        return lux_actions, training_data
    
    def _create_lux_action_array(self, actions_dict, unit_info, current_obs, player_id):
        """Create Lux action array with smart SAP targeting"""
        action_array = jnp.zeros((self.max_units, 3), dtype=jnp.int32)
        
        # Fill in actions for existing units
        for unit in unit_info:
            unit_id = unit['id']
            if unit_id in actions_dict:
                action_val = actions_dict[unit_id]
                
                # Convert our action to Lux format (action_type, dx, dy)
                if action_val == 0:  # NO_OP
                    lux_action = [0, 0, 0]
                elif 1 <= action_val <= 4:  # Movement 
                    lux_action = [action_val, 0, 0]
                else:  # SAP action (action_val = 5)
                    # Use smart targeting instead of random
                    sap_dx, sap_dy = self._get_sap_target(
                        unit['position'], 
                        current_obs, 
                        player_id
                    )
                    lux_action = [5, sap_dx, sap_dy]
                
                action_array = action_array.at[unit_id].set(jnp.array(lux_action, dtype=jnp.int32))
        
        return action_array
    
    def _get_sap_target(self, unit_pos, current_obs, player_id):
        """Smart SAP targeting - find best enemy to hit"""
        best_target = (0, 0)
        best_score = -float('inf')
        
        # Get enemy units from observation
        enemy_player_id = 1 - player_id
        if 'units' in current_obs and 'position' in current_obs['units']:
            enemy_positions = current_obs['units']['position'][enemy_player_id]
            enemy_energies = current_obs['units']['energy'][enemy_player_id]
            enemy_mask = current_obs['units_mask'][enemy_player_id] if 'units_mask' in current_obs else [True] * len(enemy_positions)
            
            for i, (enemy_pos, enemy_energy, valid) in enumerate(zip(enemy_positions, enemy_energies, enemy_mask)):
                if valid and enemy_pos[0] >= 0 and enemy_pos[1] >= 0:
                    dx = enemy_pos[0] - unit_pos[0]
                    dy = enemy_pos[1] - unit_pos[1]
                    distance = abs(dx) + abs(dy)  # Manhattan distance
                    
                    # Check if within SAP range
                    if distance <= self.max_sap_range:
                        # Score based on energy and distance
                        # Higher energy enemies are better targets
                        # Closer enemies are better targets
                        score = enemy_energy * 0.1 - distance * 2
                        
                        if score > best_score:
                            best_score = score
                            best_target = (dx, dy)
        
        return best_target
    
    def learn(self):
        return self.ppo.update()
    
    def _get_unit_info(self, obs, player_id):
        """Get unit information for Lux AI S3 format"""
        unit_info = []
        
        if 'units' in obs and 'position' in obs['units']:
            positions = obs['units']['position'][player_id]
            energies = obs['units']['energy'][player_id]
            mask = obs['units_mask'][player_id] if 'units_mask' in obs else [True] * len(positions)
            
            for i, (pos, energy, valid) in enumerate(zip(positions, energies, mask)):
                if valid and pos[0] >= 0 and pos[1] >= 0:  # Valid unit
                    unit_info.append({
                        'id': i,  # Use index as ID
                        'position': pos,
                        'energy': energy if energy > 0 else 0,
                        'health': 100
                    })
        
        return unit_info

# Kaggle submission interface
class Agent:
    def __init__(self, player: str, env_cfg: dict):
        self.player = player
        self.env_cfg = env_cfg
        self.max_units = 16
        self.max_sap_range = 7  # Maximum possible
        
        # Extract player ID from string like 'player_0' or 'player_1'
        self.player_id = int(player.split('_')[1])
        
        # Initialize your SimpleAgent components - USE WorkingModel instead of SimpleLuxCNN
        from config import Config
        from model import WorkingModel  # Changed from SimpleLuxCNN
        from ppo import WorkingPPO     # Changed from SimplePPO
        
        config = Config()
        model = WorkingModel(
            spatial_channels=24,  # 4 temporal × 3 history + 12 static
            global_dim=20,
            n_main_actions=config.n_main_actions,
            hidden_dim=config.hidden_dim
        )
        ppo = WorkingPPO(model, config)
        
        # Load the trained model
        model_path = '../../../final_trained_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        else:
            print(f"Warning: Trained model not found at {model_path}, using untrained model")
        
        self.simple_agent = SimpleAgent(config, model, ppo)
        from feature_engineer import WorkingFeatureEngineer  # Use WorkingFeatureEngineer
        self.feature_engineer = WorkingFeatureEngineer()
    
    def _create_kaggle_action_array(self, actions_dict, unit_info, obs):
        """Create action array for Kaggle with smart SAP"""
        action_array = np.zeros((self.max_units, 3), dtype=np.int32)
        
        for unit in unit_info:
            unit_id = unit['id']
            if unit_id in actions_dict:
                action_val = actions_dict[unit_id]
                
                if action_val == 0:  # NO_OP
                    action_array[unit_id] = [0, 0, 0]
                elif 1 <= action_val <= 4:  # Movement
                    action_array[unit_id] = [action_val, 0, 0]
                else:  # SAP action
                    sap_dx, sap_dy = self.simple_agent._get_sap_target(unit['position'], obs, self.player_id)
                    action_array[unit_id] = [5, sap_dx, sap_dy]
        
        return action_array
    
    def act(self, step: int, obs: dict, remaining_overage_time: int):
        """Main interface for Kaggle"""
        features = self.feature_engineer.process(obs, self.player_id)
        actions, _, _ = self.simple_agent.act(features, features['unit_info'], self.player_id)
        
        # Use smart SAP targeting
        lux_actions = self._create_kaggle_action_array(actions, features['unit_info'], obs)
        
        return lux_actions