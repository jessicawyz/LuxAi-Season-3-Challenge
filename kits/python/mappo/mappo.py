"""
Multi-Agent PPO (MAPPO) for Lux AI S3
Handles both cooperative and self-play training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import defaultdict

class MAPPO:
    """
    Multi-Agent PPO with centralized training and decentralized execution
    - Supports variable number of agents (units)
    - Handles both players simultaneously
    - Uses GAE for advantage estimation
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
        Update both agents using MAPPO
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
        print(f"\n🎯 MAPPO Update #{self.update_count}")
        print(f"   Buffer sizes: P0={self.buffers['player_0'].size()}, P1={self.buffers['player_1'].size()}")
        
        # Update each player
        for player_id in [0, 1]:
            player_key = f'player_{player_id}'
            model = self.get_model(player_id)
            buffer = self.buffers[player_key]
            
            # Sample batch
            batch = buffer.sample(self.config.batch_size)
            if batch is None:
                continue
            
            states, actions, log_probs, unit_masks, rewards, next_states, dones, infos = batch
            
            # Debug: Check shapes
            print(f"   Batch sizes - states: {len(states)}, actions: {len(actions)}, unit_masks: {len(unit_masks)}")
            
            # Unpack states
            spatial_obs = torch.stack([s[0].squeeze(0) for s in states])
            global_obs = torch.stack([s[1].squeeze(0) for s in states])
            
            # Process unit data with padding (from states, which already have proper shapes)
            unit_positions, unit_energies, _ = self._process_unit_data(states)
            
            # ✅ Use stored unit_masks from buffer (not derived from positions)
            stored_unit_masks = torch.stack([mask for mask in unit_masks])  # (batch_size, max_units)
            
            # Stack actions and log_probs (already padded to max_units in storage)
            batch_actions = torch.stack([act for act in actions])  # (batch_size, max_units)
            batch_log_probs = torch.stack([lp for lp in log_probs])  # (batch_size, max_units)
            
            # Compute values for advantage estimation
            with torch.no_grad():
                _, _, values = model(spatial_obs, global_obs, unit_positions, unit_energies)
                values = values.squeeze(-1)
                
                # Compute next values
                next_spatial = torch.stack([s[0].squeeze(0) for s in next_states])
                next_global = torch.stack([s[1].squeeze(0) for s in next_states])
                next_positions, next_energies, _ = self._process_unit_data(next_states)
                
                _, _, next_values = model(next_spatial, next_global, next_positions, next_energies)
                next_values = next_values.squeeze(-1)
            
            # Convert dones and rewards to tensor
            dones_tensor = torch.zeros(len(dones), dtype=torch.float32)
            rewards_tensor = torch.zeros(len(rewards), dtype=torch.float32)
            
            for i, (done, reward) in enumerate(zip(dones, rewards)):
                # Handle JAX arrays and other types
                if hasattr(done, '__array__'):
                    done_val = np.array(done).item() if hasattr(done, 'item') else float(done)
                else:
                    done_val = float(done)
                    
                if hasattr(reward, '__array__'):
                    reward_val = np.array(reward).item() if hasattr(reward, 'item') else float(reward)
                else:
                    reward_val = float(reward)
                    
                dones_tensor[i] = done_val
                rewards_tensor[i] = reward_val
            
            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards_tensor, values, dones_tensor)
            
            # PPO epochs
            player_total_loss = 0
            player_policy_loss = 0
            player_value_loss = 0
            player_entropy = 0
            
            update_count = 0
            
            for epoch in range(n_epochs):
                # Shuffle indices
                indices = torch.randperm(self.config.batch_size)
                
                # Process in mini-batches
                mini_batch_size = min(64, self.config.batch_size // 4)
                
                for start_idx in range(0, self.config.batch_size, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, self.config.batch_size)
                    mb_indices = indices[start_idx:end_idx]
                    
                    # Mini-batch data
                    mb_spatial = spatial_obs[mb_indices]
                    mb_global = global_obs[mb_indices]
                    mb_positions = unit_positions[mb_indices]
                    mb_energies = unit_energies[mb_indices]
                    mb_masks = stored_unit_masks[mb_indices]  # ✅ Use stored masks
                    mb_actions = batch_actions[mb_indices]  # ✅ (mini_batch, max_units)
                    mb_old_log_probs = batch_log_probs[mb_indices]  # ✅ (mini_batch, max_units)
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]
                    
                    # Forward pass
                    action_logits, sap_logits, values_pred = model(
                        mb_spatial, mb_global, mb_positions, mb_energies
                    )
                    
                    # Handle action probabilities with variable units
                    batch_size, n_units, n_actions = action_logits.shape
                    
                    # Apply unit masks to ignore padded units
                    valid_units = mb_masks.unsqueeze(-1).expand(-1, -1, n_actions)
                    action_logits = action_logits * valid_units.float()
                    
                    # Apply action masks if available
                    if infos and infos[0] is not None and 'action_masks' in infos[0]:
                        for i, info_idx in enumerate(mb_indices):
                            info = infos[info_idx]
                            if info and 'action_masks' in info:
                                masks = info['action_masks']
                                for unit_id, mask in masks.items():
                                    if unit_id < n_units and mb_masks[i, unit_id]:
                                        action_logits[i, unit_id, ~torch.BoolTensor(mask)] = -1e10
                    
                    # ✅ PROPER MULTI-AGENT UPDATE: Process ALL units with masking
                    total_valid_units = 0
                    batch_policy_loss = 0
                    batch_value_loss = 0
                    batch_entropy = 0
                    
                    for i in range(batch_size):
                        # Get valid units for this batch element
                        valid_unit_indices = torch.where(mb_masks[i])[0]
                        
                        if len(valid_unit_indices) == 0:
                            continue  # Skip if no valid units
                        
                        # Get logits for all valid units
                        unit_logits = action_logits[i, valid_unit_indices]  # (n_valid_units, n_actions)
                        
                        # ✅ Get actions and log probs for ALL valid units (not just first)
                        unit_actions = mb_actions[i, valid_unit_indices]  # (n_valid_units,)
                        unit_old_log_probs = mb_old_log_probs[i, valid_unit_indices]  # (n_valid_units,)
                        
                        # Compute distributions for all valid units
                        dist = Categorical(logits=unit_logits)
                        new_log_probs = dist.log_prob(unit_actions)
                        entropy = dist.entropy().mean()
                        
                        # PPO loss for all valid units
                        ratio = torch.exp(new_log_probs - unit_old_log_probs)
                        avg_advantage = mb_advantages[i]  # Single advantage for whole transition
                        
                        surr1 = ratio * avg_advantage
                        surr2 = torch.clamp(ratio, 
                                        1 - self.config.clip_epsilon,
                                        1 + self.config.clip_epsilon) * avg_advantage
                        
                        # Average policy loss over valid units
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        batch_policy_loss += policy_loss
                        batch_entropy += entropy
                        total_valid_units += len(valid_unit_indices)
                    
                    if total_valid_units == 0:
                        continue  # Skip if no valid units in entire mini-batch
                    
                    # Average policy loss and entropy
                    avg_policy_loss = batch_policy_loss / batch_size
                    avg_entropy = batch_entropy / batch_size
                    
                    # Value loss (computed on entire batch)
                    values_pred = values_pred.squeeze(-1)
                    value_loss = F.mse_loss(values_pred, mb_returns)
                    
                    # Total loss
                    loss = (avg_policy_loss + 
                        self.config.value_coef * value_loss - 
                        self.config.entropy_coef * avg_entropy)
                    
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    if hasattr(model, 'value_head'):
                        torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), max_norm=0.1)
                    self.optimizer.step()
                    
                    # Accumulate metrics
                    player_total_loss += loss.item()
                    player_policy_loss += avg_policy_loss.item()
                    player_value_loss += value_loss.item()
                    player_entropy += avg_entropy.item()
                    update_count += 1
            
            # Store player metrics
            if update_count > 0:
                metrics[f'player_{player_id}_loss'] = player_total_loss / update_count
                metrics['policy_loss'] += player_policy_loss / update_count
                metrics['value_loss'] += player_value_loss / update_count
                metrics['entropy'] += player_entropy / update_count
        
        # Average metrics across players
        if metrics['player_0_loss'] > 0 and metrics['player_1_loss'] > 0:
            metrics['total_loss'] = (metrics['player_0_loss'] + metrics['player_1_loss']) / 2
            metrics['policy_loss'] /= 2
            metrics['value_loss'] /= 2
            metrics['entropy'] /= 2
        else:
            metrics['total_loss'] = 0
        
        # Clear old experiences
        for buffer in self.buffers.values():
            buffer.clear_old(keep_recent=self.config.batch_size * 2)
        
        print(f"   ✅ Update complete - Loss: {metrics['total_loss']:.4f}")
        
        return metrics
    
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