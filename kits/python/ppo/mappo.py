"""
TRUE Multi-Agent PPO (MAPPO) for Lux AI S3
Centralized Training, Decentralized Execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import defaultdict

class MAPPO:
    """
    TRUE Multi-Agent PPO with centralized critic
    - Centralized Training: Critic sees both players' states
    - Decentralized Execution: Each policy only sees own state
    - Proper credit assignment across agents
    """
    def __init__(self, model_p0, model_p1, config, shared_model=True):
        """
        Args:
            model_p0: Model for player 0
            model_p1: Model for player 1 (can be same as model_p0 for parameter sharing)
            config: Config object
            shared_model: Whether both players share the same model
        """
        self.config = config
        self.shared_model = shared_model
        
        if shared_model:
            self.model_p0 = model_p0
            self.model_p1 = model_p0  # Same model
            self.optimizer = torch.optim.Adam(
                model_p0.parameters(),
                lr=config.learning_rate,
                eps=1e-5
            )
        else:
            self.model_p0 = model_p0
            self.model_p1 = model_p1
            # Combined optimizer for both models
            params = list(model_p0.parameters()) + list(model_p1.parameters())
            self.optimizer = torch.optim.Adam(
                params,
                lr=config.learning_rate,
                eps=1e-5
            )
        
        # Separate buffers for each player
        self.buffers = {
            'player_0': ReplayBuffer(config.buffer_size),
            'player_1': ReplayBuffer(config.buffer_size)
        }
        
        # Training stats
        self.update_count = 0
        
        # GAE parameters
        self.gamma = config.gamma
        self.gae_lambda = 0.95
        
    def get_model(self, player_id):
        """Get the model for a specific player"""
        if player_id == 0:
            return self.model_p0
        else:
            return self.model_p1
    
    def store_transition(self, player_id, state, actions, log_probs, unit_mask, reward, next_state, done, info=None):
        """
        Store a transition for a specific player - MULTI-UNIT VERSION
        
        Args:
            player_id: 0 or 1
            state: (spatial, global, unit_positions, unit_energies)
            actions: (max_units,) tensor - actions for ALL units
            log_probs: (max_units,) tensor - log probs for ALL units
            unit_mask: (max_units,) bool tensor - which units are valid
            reward: reward received
            next_state: next state
            done: episode done flag
            info: additional info (e.g., action masks)
        """
        player_key = f'player_{player_id}'
        self.buffers[player_key].push(state, actions, log_probs, unit_mask, reward, next_state, done, info)
    
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        batch_size = len(rewards)
        advantages = torch.zeros(batch_size)
        returns = torch.zeros(batch_size)
        
        # Bootstrap from last value
        last_advantage = 0
        last_value = values[-1] if len(values) > 0 else 0
        
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0 if dones[t] else last_value
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage
            
            # Returns
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _process_unit_data(self, states, max_units=16):
        """
        Process unit data with variable number of units per state
        Pads to max_units for batching
        """
        batch_size = len(states)
        
        # Initialize padded tensors
        unit_positions = torch.zeros(batch_size, max_units, 2)
        unit_energies = torch.zeros(batch_size, max_units, 1)
        unit_masks = torch.zeros(batch_size, max_units, dtype=torch.bool)
        
        for i, state in enumerate(states):
            positions = state[2].squeeze(0)  # (n_units, 2)
            energies = state[3].squeeze(0)   # (n_units, 1)
            
            n_units = positions.shape[0]
            if n_units > max_units:
                n_units = max_units  # Truncate if too many units
            
            unit_positions[i, :n_units] = positions[:n_units]
            unit_energies[i, :n_units] = energies[:n_units]
            unit_masks[i, :n_units] = True
        
        return unit_positions, unit_energies, unit_masks
    
    def update(self, n_epochs=4):
        """
        ✅ TRUE MAPPO UPDATE with Centralized Critic
        """
        metrics = {
            'total_loss': 0,
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'player_0_loss': 0,
            'player_1_loss': 0
        }
        
        # Check if we have enough data
        min_samples = self.config.batch_size
        if (self.buffers['player_0'].size() < min_samples or 
            self.buffers['player_1'].size() < min_samples):
            return metrics
        
        self.update_count += 1
        print(f"\n🎯 TRUE MAPPO Update #{self.update_count} (Centralized Critic)")
        print(f"   Buffer sizes: P0={self.buffers['player_0'].size()}, P1={self.buffers['player_1'].size()}")
        
        # ✅ STEP 1: Sample batches from BOTH players (for centralized critic)
        batch_p0 = self.buffers['player_0'].sample(self.config.batch_size)
        batch_p1 = self.buffers['player_1'].sample(self.config.batch_size)
        
        if batch_p0 is None or batch_p1 is None:
            return metrics
        
        # Unpack both player batches
        states_p0, actions_p0, log_probs_p0, unit_masks_p0, rewards_p0, next_states_p0, dones_p0, infos_p0 = batch_p0
        states_p1, actions_p1, log_probs_p1, unit_masks_p1, rewards_p1, next_states_p1, dones_p1, infos_p1 = batch_p1
        
        print(f"   Batch sizes - P0: {len(states_p0)}, P1: {len(states_p1)}")
        
        # ✅ STEP 2: Prepare observations for both players
        spatial_obs_p0 = torch.stack([s[0].squeeze(0) for s in states_p0])
        global_obs_p0 = torch.stack([s[1].squeeze(0) for s in states_p0])
        unit_positions_p0, unit_energies_p0, _ = self._process_unit_data(states_p0)
        
        spatial_obs_p1 = torch.stack([s[0].squeeze(0) for s in states_p1])
        global_obs_p1 = torch.stack([s[1].squeeze(0) for s in states_p1])
        unit_positions_p1, unit_energies_p1, _ = self._process_unit_data(states_p1)
        
        # Prepare next states
        next_spatial_p0 = torch.stack([s[0].squeeze(0) for s in next_states_p0])
        next_global_p0 = torch.stack([s[1].squeeze(0) for s in next_states_p0])
        next_positions_p0, next_energies_p0, _ = self._process_unit_data(next_states_p0)
        
        next_spatial_p1 = torch.stack([s[0].squeeze(0) for s in next_states_p1])
        next_global_p1 = torch.stack([s[1].squeeze(0) for s in next_states_p1])
        next_positions_p1, next_energies_p1, _ = self._process_unit_data(next_states_p1)
        
        # Prepare stored masks and actions
        stored_unit_masks_p0 = torch.stack([mask for mask in unit_masks_p0])
        stored_unit_masks_p1 = torch.stack([mask for mask in unit_masks_p1])
        
        batch_actions_p0 = torch.stack([act for act in actions_p0])
        batch_actions_p1 = torch.stack([act for act in actions_p1])
        
        batch_log_probs_p0 = torch.stack([lp for lp in log_probs_p0])
        batch_log_probs_p1 = torch.stack([lp for lp in log_probs_p1])
        
        # Convert rewards and dones
        def to_tensor(data):
            tensor = torch.zeros(len(data), dtype=torch.float32)
            for i, val in enumerate(data):
                if hasattr(val, '__array__'):
                    tensor[i] = np.array(val).item() if hasattr(val, 'item') else float(val)
                else:
                    tensor[i] = float(val)
            return tensor
        
        rewards_tensor_p0 = to_tensor(rewards_p0)
        rewards_tensor_p1 = to_tensor(rewards_p1)
        dones_tensor_p0 = to_tensor(dones_p0)
        dones_tensor_p1 = to_tensor(dones_p1)
        
        # ✅ Normalize rewards to prevent value explosion
        rewards_tensor_p0 = torch.clamp(rewards_tensor_p0, -10, 10)
        rewards_tensor_p1 = torch.clamp(rewards_tensor_p1, -10, 10)
        
        # ✅ STEP 3: CENTRALIZED CRITIC - Value sees both players
        with torch.no_grad():
            model = self.model_p0  # Use shared model
            
            # Get centralized values (critic sees both players)
            if hasattr(model, 'get_value_for_both_players'):
                # Use the centralized critic method
                values_p0, values_p1 = model.get_value_for_both_players(
                    spatial_obs_p0, global_obs_p0,
                    spatial_obs_p1, global_obs_p1
                )
                values_p0 = values_p0.squeeze(-1)
                values_p1 = values_p1.squeeze(-1)
                
                # Next values
                next_values_p0, next_values_p1 = model.get_value_for_both_players(
                    next_spatial_p0, next_global_p0,
                    next_spatial_p1, next_global_p1
                )
                next_values_p0 = next_values_p0.squeeze(-1)
                next_values_p1 = next_values_p1.squeeze(-1)
                
                print("   ✅ Using centralized critic (sees both players)")
            else:
                # Fallback: separate value predictions
                _, _, values_p0 = model(spatial_obs_p0, global_obs_p0, unit_positions_p0, unit_energies_p0)
                _, _, values_p1 = model(spatial_obs_p1, global_obs_p1, unit_positions_p1, unit_energies_p1)
                values_p0 = values_p0.squeeze(-1)
                values_p1 = values_p1.squeeze(-1)
                
                _, _, next_values_p0 = model(next_spatial_p0, next_global_p0, next_positions_p0, next_energies_p0)
                _, _, next_values_p1 = model(next_spatial_p1, next_global_p1, next_positions_p1, next_energies_p1)
                next_values_p0 = next_values_p0.squeeze(-1)
                next_values_p1 = next_values_p1.squeeze(-1)
                
                print("   ⚠️  Using separate critics (not true MAPPO)")
        
        # ✅ Clip values to prevent explosion
        values_p0 = torch.clamp(values_p0, -50, 50)
        values_p1 = torch.clamp(values_p1, -50, 50)
        next_values_p0 = torch.clamp(next_values_p0, -50, 50)
        next_values_p1 = torch.clamp(next_values_p1, -50, 50)
        
        # ✅ STEP 4: Compute advantages and returns for both players
        advantages_p0, returns_p0 = self.compute_gae(rewards_tensor_p0, values_p0, dones_tensor_p0)
        advantages_p1, returns_p1 = self.compute_gae(rewards_tensor_p1, values_p1, dones_tensor_p1)
        
        # ✅ STEP 5: PPO Update for both players
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_updates = 0
        
        # Update both players together (true multi-agent)
        for epoch in range(n_epochs):
            # Shuffle indices
            indices = torch.randperm(self.config.batch_size)
            
            mini_batch_size = min(32, self.config.batch_size // 4)
            
            for start_idx in range(0, self.config.batch_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, self.config.batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # ✅ PLAYER 0 POLICY UPDATE (Decentralized - only sees own state)
                mb_spatial_p0 = spatial_obs_p0[mb_indices]
                mb_global_p0 = global_obs_p0[mb_indices]
                mb_positions_p0 = unit_positions_p0[mb_indices]
                mb_energies_p0 = unit_energies_p0[mb_indices]
                mb_masks_p0 = stored_unit_masks_p0[mb_indices]
                mb_actions_p0 = batch_actions_p0[mb_indices]
                mb_old_log_probs_p0 = batch_log_probs_p0[mb_indices]
                mb_advantages_p0 = advantages_p0[mb_indices]
                mb_returns_p0 = returns_p0[mb_indices]
                
                # ✅ PLAYER 1 POLICY UPDATE (Decentralized - only sees own state)
                mb_spatial_p1 = spatial_obs_p1[mb_indices]
                mb_global_p1 = global_obs_p1[mb_indices]
                mb_positions_p1 = unit_positions_p1[mb_indices]
                mb_energies_p1 = unit_energies_p1[mb_indices]
                mb_masks_p1 = stored_unit_masks_p1[mb_indices]
                mb_actions_p1 = batch_actions_p1[mb_indices]
                mb_old_log_probs_p1 = batch_log_probs_p1[mb_indices]
                mb_advantages_p1 = advantages_p1[mb_indices]
                mb_returns_p1 = returns_p1[mb_indices]
                
                # Forward pass for both players
                action_logits_p0, _, values_pred_p0 = self.model_p0(
                    mb_spatial_p0, mb_global_p0, mb_positions_p0, mb_energies_p0
                )
                action_logits_p1, _, values_pred_p1 = self.model_p1(
                    mb_spatial_p1, mb_global_p1, mb_positions_p1, mb_energies_p1
                )
                
                # Compute policy losses for both players
                policy_loss_p0, entropy_p0 = self._compute_policy_loss(
                    action_logits_p0, mb_actions_p0, mb_old_log_probs_p0, 
                    mb_masks_p0, mb_advantages_p0, infos_p0, mb_indices
                )
                
                policy_loss_p1, entropy_p1 = self._compute_policy_loss(
                    action_logits_p1, mb_actions_p1, mb_old_log_probs_p1, 
                    mb_masks_p1, mb_advantages_p1, infos_p1, mb_indices
                )
                
                # ✅ Compute value loss (centralized or separate)
                values_pred_p0 = values_pred_p0.squeeze(-1)
                values_pred_p1 = values_pred_p1.squeeze(-1)
                
                # Clip predictions and returns
                values_pred_p0 = torch.clamp(values_pred_p0, -50, 50)
                values_pred_p1 = torch.clamp(values_pred_p1, -50, 50)
                mb_returns_p0_clipped = torch.clamp(mb_returns_p0, -50, 50)
                mb_returns_p1_clipped = torch.clamp(mb_returns_p1, -50, 50)
                
                value_loss_p0 = F.mse_loss(values_pred_p0, mb_returns_p0_clipped)
                value_loss_p1 = F.mse_loss(values_pred_p1, mb_returns_p1_clipped)
                
                # Combined loss
                avg_policy_loss = (policy_loss_p0 + policy_loss_p1) / 2
                avg_value_loss = (value_loss_p0 + value_loss_p1) / 2
                avg_entropy = (entropy_p0 + entropy_p1) / 2
                
                loss = (avg_policy_loss + 
                       self.config.value_coef * avg_value_loss - 
                       self.config.entropy_coef * avg_entropy)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_p0.parameters(), 0.5)
                self.optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_policy_loss += avg_policy_loss.item()
                total_value_loss += avg_value_loss.item()
                total_entropy += avg_entropy.item()
                total_updates += 1
        
        # Average metrics
        if total_updates > 0:
            metrics['total_loss'] = total_loss / total_updates
            metrics['policy_loss'] = total_policy_loss / total_updates
            metrics['value_loss'] = total_value_loss / total_updates
            metrics['entropy'] = total_entropy / total_updates
        
        # Don't clear buffers here - let the training script do it
        
        print(f"   ✅ Update complete - Loss: {metrics['total_loss']:.4f}")
        print(f"      Policy: {metrics['policy_loss']:.4f}, Value: {metrics['value_loss']:.4f}, Entropy: {metrics['entropy']:.4f}")
        
        return metrics
    
    def _compute_policy_loss(self, action_logits, actions, old_log_probs, masks, advantages, infos, mb_indices):
        """
        Compute policy loss for a player with proper unit masking
        """
        batch_size, n_units, n_actions = action_logits.shape
        
        # Apply unit masks
        valid_units = masks.unsqueeze(-1).expand(-1, -1, n_actions)
        action_logits = action_logits * valid_units.float()
        
        # Apply action masks if available
        if infos and infos[0] is not None and 'action_masks' in infos[0]:
            for i, info_idx in enumerate(mb_indices):
                if info_idx < len(infos):
                    info = infos[info_idx]
                    if info and 'action_masks' in info:
                        masks_dict = info['action_masks']
                        for unit_id, mask in masks_dict.items():
                            if unit_id < n_units and masks[i, unit_id]:
                                action_logits[i, unit_id, ~torch.BoolTensor(mask)] = -1e10
        
        # Compute policy loss for all valid units
        total_policy_loss = 0
        total_entropy = 0
        total_valid_units = 0
        
        for i in range(batch_size):
            valid_unit_indices = torch.where(masks[i])[0]
            
            if len(valid_unit_indices) == 0:
                continue
            
            unit_logits = action_logits[i, valid_unit_indices]
            unit_actions = actions[i, valid_unit_indices]
            unit_old_log_probs = old_log_probs[i, valid_unit_indices]
            
            dist = Categorical(logits=unit_logits)
            new_log_probs = dist.log_prob(unit_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - unit_old_log_probs)
            avg_advantage = advantages[i]
            
            surr1 = ratio * avg_advantage
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * avg_advantage
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            total_policy_loss += policy_loss
            total_entropy += entropy
            total_valid_units += len(valid_unit_indices)
        
        if total_valid_units == 0:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        avg_policy_loss = total_policy_loss / batch_size
        avg_entropy = total_entropy / batch_size
        
        return avg_policy_loss, avg_entropy
    
    def save(self, path):
        """Save models and optimizer state"""
        checkpoint = {
            'model_p0_state': self.model_p0.state_dict(),
            'model_p1_state': self.model_p1.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'update_count': self.update_count
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load models and optimizer state"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model_p0.load_state_dict(checkpoint['model_p0_state'])
        if not self.shared_model:
            self.model_p1.load_state_dict(checkpoint['model_p1_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.update_count = checkpoint.get('update_count', 0)


class ReplayBuffer:
    """
    Experience replay buffer for MAPPO - Multi-unit version
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, actions, log_probs, unit_mask, reward, next_state, done, info=None):
        """Store a transition with multi-unit data"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (
            state, actions, log_probs, unit_mask, reward, next_state, done, info
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        if len(self.buffer) < batch_size:
            return None
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Gather samples
        batch = [self.buffer[i] for i in indices]
        
        # Unzip - now with unit_mask
        states, actions, log_probs, unit_masks, rewards, next_states, dones, infos = zip(*batch)
        
        return (states, actions, log_probs, unit_masks, rewards, next_states, dones, infos)
    
    def size(self):
        """Get current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.position = 0
    
    def clear_old(self, keep_recent=1000):
        """Keep only the most recent experiences"""
        if len(self.buffer) > keep_recent:
            self.buffer = self.buffer[-keep_recent:]
            self.position = len(self.buffer) % self.capacity