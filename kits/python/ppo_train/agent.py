import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import jax.numpy as jnp
import os

class AdvancedAgent:
    """
    FIXED: Agent that handles single-unit states properly
    - act() now processes one unit at a time
    - Model expects variable n_units (1 for training, multiple for inference)
    """
    def __init__(self, config, model, ppo):
        self.config = config
        self.model = model
        self.ppo = ppo
        self.max_units = 16
        self.max_sap_range = config.max_sap_range
        
    def act(self, features, unit_info, player_id=0, deterministic=False):
        """
        Generate actions for all units
        
        Returns:
            main_actions: dict mapping unit_id -> action (0-5)
            sap_targets: dict mapping unit_id -> (dx, dy) for units that SAP
            log_probs: dict mapping unit_id -> log probability
            values: dict mapping unit_id -> value estimate
        """
        if not unit_info:
            return {}, {}, {}, {}
        
        spatial_obs = torch.FloatTensor(features['spatial']).unsqueeze(0)
        global_obs = torch.FloatTensor(features['global']).unsqueeze(0)
        
        # FIXED: Prepare unit positions and energies for ALL units
        n_units = len(unit_info)
        unit_positions = torch.zeros(1, n_units, 2)  # Variable n_units
        unit_energies = torch.zeros(1, n_units, 1)
        
        unit_id_to_idx = {}
        for idx, unit in enumerate(unit_info):
            unit_id = unit['id']
            unit_id_to_idx[unit_id] = idx
            unit_positions[0, idx] = torch.FloatTensor(unit['position'])
            unit_energies[0, idx] = torch.FloatTensor([unit['energy'] / 400.0])
        
        # Forward pass
        with torch.no_grad():
            main_action_logits, sap_target_logits, value = self.model(
                spatial_obs, global_obs, unit_positions, unit_energies
            )
        
        main_actions = {}
        sap_targets = {}
        log_probs = {}
        values = {}
        
        for unit in unit_info:
            unit_id = unit['id']
            idx = unit_id_to_idx[unit_id]
            
            # Get action mask
            action_mask = features['action_masks'].get(unit_id, np.ones(6, dtype=bool))
            
            # Sample main action
            unit_logits = main_action_logits[0, idx]
            
            # Apply mask
            masked_logits = unit_logits.clone()
            masked_logits[~torch.BoolTensor(action_mask)] = -1e10
            
            dist = Categorical(logits=masked_logits)
            
            if deterministic:
                action = dist.probs.argmax()
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            main_actions[unit_id] = action.item()
            log_probs[unit_id] = log_prob
            values[unit_id] = value[0]
            
            # If action is SAP (action 5), determine target
            if action.item() == 5:
                sap_target = self._sample_sap_target(
                    unit['position'],
                    sap_target_logits[0],
                    player_id,
                    deterministic=deterministic
                )
                sap_targets[unit_id] = sap_target
            else:
                sap_targets[unit_id] = (0, 0)
        
        return main_actions, sap_targets, log_probs, values
    
    def _sample_sap_target(self, unit_pos, sap_logits, player_id, deterministic=False):
        """
        Sample SAP target from spatial logits
        
        Args:
            unit_pos: (x, y) position of unit
            sap_logits: (24, 24) tensor of logits for each map position
            player_id: 0 or 1
            deterministic: if True, take argmax instead of sampling
        
        Returns:
            (dx, dy): relative offset to target
        """
        # Create mask for valid SAP targets (within range)
        mask = torch.zeros(24, 24, dtype=torch.bool)
        x, y = int(unit_pos[0]), int(unit_pos[1])
        
        for tx in range(24):
            for ty in range(24):
                dx = tx - x
                dy = ty - y
                if abs(dx) <= self.max_sap_range and abs(dy) <= self.max_sap_range:
                    mask[ty, tx] = True
        
        # Apply mask
        masked_logits = sap_logits.clone()
        masked_logits[~mask] = -1e10
        
        # Sample target
        flat_logits = masked_logits.view(-1)
        dist = Categorical(logits=flat_logits)
        
        if deterministic:
            target_idx = dist.probs.argmax()
        else:
            target_idx = dist.sample()
        
        # Convert flat index to (y, x)
        target_y = target_idx // 24
        target_x = target_idx % 24
        
        # Convert to relative offset
        dx = target_x - x
        dy = target_y - y
        
        return (int(dx), int(dy))
    
    def act_for_env(self, obs, feature_engineer):
        """
        Generate actions for both players in environment format
        
        Returns:
            lux_actions: dict with 'player_0' and 'player_1' action arrays
            training_data: dict with data for PPO training
        """
        # Process observations for both players
        features_p0 = feature_engineer.process(obs['player_0'], player_id=0)
        features_p1 = feature_engineer.process(obs['player_1'], player_id=1)
        
        # Get actions for both players
        actions_p0, sap_p0, log_probs_p0, values_p0 = self.act(
            features_p0, features_p0['unit_info'], player_id=0
        )
        actions_p1, sap_p1, log_probs_p1, values_p1 = self.act(
            features_p1, features_p1['unit_info'], player_id=1
        )
        
        # Convert to Lux environment format
        lux_actions = {
            "player_0": self._create_lux_action_array(
                actions_p0, sap_p0, features_p0['unit_info'], obs['player_0'], 0
            ),
            "player_1": self._create_lux_action_array(
                actions_p1, sap_p1, features_p1['unit_info'], obs['player_1'], 1
            )
        }
        
        training_data = {
            "player_0": (actions_p0, sap_p0, log_probs_p0, values_p0, features_p0),
            "player_1": (actions_p1, sap_p1, log_probs_p1, values_p1, features_p1)
        }
        
        return lux_actions, training_data
    
    def _create_lux_action_array(self, actions_dict, sap_dict, unit_info, obs, player_id):
        """
        Create Lux action array format: (max_units, 3)
        Each action is [action_type, dx, dy]
        """
        action_array = jnp.zeros((self.max_units, 3), dtype=jnp.int32)
        
        for unit in unit_info:
            unit_id = unit['id']
            if unit_id not in actions_dict:
                continue
            
            action_val = actions_dict[unit_id]
            
            if action_val == 0:  # NO_OP
                lux_action = [0, 0, 0]
            elif 1 <= action_val <= 4:  # Movement
                lux_action = [action_val, 0, 0]
            elif action_val == 5:  # SAP
                dx, dy = sap_dict.get(unit_id, (0, 0))
                lux_action = [5, dx, dy]
            else:
                lux_action = [0, 0, 0]
            
            action_array = action_array.at[unit_id].set(jnp.array(lux_action, dtype=jnp.int32))
        
        return action_array
    
    def learn(self):
        """Update policy using PPO"""
        return self.ppo.update()


# Kaggle submission interface
class Agent:
    """Kaggle submission wrapper"""
    def __init__(self, player: str, env_cfg: dict):
        self.player = player
        self.env_cfg = env_cfg
        self.max_units = 16
        self.max_sap_range = 7
        
        self.player_id = int(player.split('_')[1])
        
        # Initialize components
        from config import Config
        from model import LuxResNetCNN
        from ppo import ImprovedPPO
        
        config = Config()
        
        # Model expects temporal features: 6 base * 5 history = 30 spatial channels
        model = LuxResNetCNN(
            spatial_channels=80,
            global_dim=115,
            n_main_actions=config.n_main_actions,
            hidden_dim=config.hidden_dim,
            n_blocks=config.cnn_blocks
        )
        ppo = ImprovedPPO(model, config)
        
        # Load trained model
        model_path = '../../../final_trained_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        else:
            print(f"⚠️ Warning: Model not found at {model_path}")
        
        self.advanced_agent = AdvancedAgent(config, model, ppo)
        
        from feature_engineer import AdvancedFeatureEngineer
        self.feature_engineer = AdvancedFeatureEngineer()
    
    def act(self, step: int, obs: dict, remaining_overage_time: int):
        """Main interface for Kaggle submission"""
        features = self.feature_engineer.process(obs, self.player_id)
        
        # Use stochastic policy at test time (better for blind SAP)
        actions, sap_targets, _, _ = self.advanced_agent.act(
            features, 
            features['unit_info'], 
            self.player_id,
            deterministic=False  # Stochastic is better!
        )
        
        # Convert to Kaggle format
        lux_actions = self._create_kaggle_action_array(actions, sap_targets, features['unit_info'])
        
        return lux_actions
    
    def _create_kaggle_action_array(self, actions_dict, sap_dict, unit_info):
        """Create action array for Kaggle"""
        action_array = np.zeros((self.max_units, 3), dtype=np.int32)
        
        for unit in unit_info:
            unit_id = unit['id']
            if unit_id not in actions_dict:
                continue
            
            action_val = actions_dict[unit_id]
            
            if action_val == 0:
                action_array[unit_id] = [0, 0, 0]
            elif 1 <= action_val <= 4:
                action_array[unit_id] = [action_val, 0, 0]
            elif action_val == 5:
                dx, dy = sap_dict.get(unit_id, (0, 0))
                action_array[unit_id] = [5, dx, dy]
        
        return action_array