import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ImprovedPPO:
    """
    FIXED PPO implementation that handles variable n_units
    - Works with both single-unit training (n_units=1)
    - And multi-unit inference (n_units=variable)
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            eps=1e-5
        )
        self.memory = []
        
        # GAE-Lambda parameters
        self.gamma = config.gamma  # 0.9999 for long-term planning
        self.gae_lambda = 0.95

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Proper GAE implementation based on Frog Parade's bootstrap_value function
        """
        steps = len(rewards)
        advantages = np.zeros(steps, dtype=np.float32)
        returns = np.zeros(steps, dtype=np.float32)
        
        last_gae_lambda = 0.0
        
        # Work backwards through steps
        for t in reversed(range(steps)):
            if t == steps - 1:
                next_values = next_value
                next_not_done = 1.0 - dones[t]
            else:
                next_values = values[t + 1]
                next_not_done = 1.0 - dones[t + 1]
            
            # TD residual: δ = r + γV(s') - V(s)
            delta = rewards[t] + self.gamma * next_values * next_not_done - values[t]
            
            # GAE: λ-return
            last_gae_lambda = delta + self.gamma * self.gae_lambda * next_not_done * last_gae_lambda
            advantages[t] = last_gae_lambda
            
            # Returns = advantages + values
            returns[t] = advantages[t] + values[t]
        
        # Convert to tensors and normalize advantages
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages (important for stability)
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        return advantages_tensor, returns_tensor

    def update(self):
        if len(self.memory) < self.config.batch_size:
            return 0.0, 0.0, 0.0, 0.0
        
        print(f"\n🎯 PPO Update with {len(self.memory)} experiences")
        
        # Safe batch sizes
        chunk_size = min(64, self.config.batch_size)
        micro_batch_size = min(16, chunk_size)
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_epochs = 4
        
        # Only use exactly batch_size experiences
        experiences = self.memory[:self.config.batch_size]
        
        # Pre-compute advantages
        print("  📊 Computing advantages...")
        states, main_actions, sap_targets, old_log_probs, rewards, next_states, dones = zip(*experiences)
        
        # Convert to tensors for advantage calculation
        spatial_states = torch.cat([s[0] for s in states])
        global_states = torch.cat([s[1] for s in states])
        unit_positions = torch.cat([s[2] for s in states])
        unit_energies = torch.cat([s[3] for s in states])
        
        # Compute current values for advantage calculation
        with torch.no_grad():
            _, _, current_values = self.model(spatial_states, global_states, unit_positions, unit_energies)
            current_values = current_values.squeeze(-1).numpy()
        
        # Compute next values for bootstrapping
        next_spatial = torch.cat([s[0] for s in next_states])
        next_global = torch.cat([s[1] for s in next_states])
        next_positions = torch.cat([s[2] for s in next_states])
        next_energies = torch.cat([s[3] for s in next_states])
        
        with torch.no_grad():
            _, _, next_value_estimates = self.model(next_spatial, next_global, next_positions, next_energies)
            next_values = next_value_estimates.squeeze(-1).numpy()
        
        next_value = next_values[-1] if len(next_values) > 0 else 0.0
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            np.array([r.item() if hasattr(r, 'item') else r for r in rewards]),
            current_values,
            np.array([d.item() if hasattr(d, 'item') else d for d in dones]),
            next_value
        )
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Clear large tensors immediately
        del spatial_states, global_states, unit_positions, unit_energies
        del next_spatial, next_global, next_positions, next_energies
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  🔄 Training with {len(experiences)} experiences")
        print(f"  📦 Chunk size: {chunk_size}, Micro-batch: {micro_batch_size}")
        
        # Process in MICRO-BATCHES
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            num_micro_batches = 0
            
            # Shuffle indices for this epoch
            indices = torch.randperm(len(experiences))
            
            # Process in chunks
            for chunk_start in range(0, len(experiences), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(experiences))
                chunk_indices = indices[chunk_start:chunk_end]
                
                # Extract chunk data
                chunk_states = [experiences[i][0] for i in chunk_indices]
                chunk_main_actions = torch.cat([experiences[i][1].unsqueeze(0) for i in chunk_indices])
                chunk_old_log_probs = torch.cat([experiences[i][3].unsqueeze(0) for i in chunk_indices])
                chunk_advantages = advantages[chunk_indices]
                chunk_returns = returns[chunk_indices]
                
                # Convert chunk states to tensors
                chunk_spatial = torch.cat([s[0] for s in chunk_states])
                chunk_global = torch.cat([s[1] for s in chunk_states])
                chunk_positions = torch.cat([s[2] for s in chunk_states])
                chunk_energies = torch.cat([s[3] for s in chunk_states])
                
                # Process in MICRO-BATCHES within each chunk
                for micro_start in range(0, len(chunk_indices), micro_batch_size):
                    micro_end = min(micro_start + micro_batch_size, len(chunk_indices))
                    
                    # Extract micro-batch data
                    micro_spatial = chunk_spatial[micro_start:micro_end]
                    micro_global = chunk_global[micro_start:micro_end]
                    micro_positions = chunk_positions[micro_start:micro_end]
                    micro_energies = chunk_energies[micro_start:micro_end]
                    micro_actions = chunk_main_actions[micro_start:micro_end]
                    micro_old_log_probs = chunk_old_log_probs[micro_start:micro_end]
                    micro_advantages = chunk_advantages[micro_start:micro_end]
                    micro_returns = chunk_returns[micro_start:micro_end]
                    
                    # Forward pass on micro-batch
                    main_action_logits, sap_target_logits, values_pred = self.model(
                        micro_spatial, micro_global, micro_positions, micro_energies
                    )
                    
                    # ✅ FIXED: Handle variable n_units properly
                    if len(main_action_logits.shape) == 3:
                        # Multi-unit case: (batch, n_units, n_actions)
                        batch_size, n_units, n_actions = main_action_logits.shape
                        
                        if n_units == 1:
                            # Single unit: squeeze the unit dimension
                            main_action_logits_flat = main_action_logits.squeeze(1)  # (batch, n_actions)
                            micro_actions_flat = micro_actions.squeeze(1) if len(micro_actions.shape) > 1 else micro_actions
                        else:
                            # Multiple units: flatten batch and unit dims
                            main_action_logits_flat = main_action_logits.reshape(-1, n_actions)
                            micro_actions_flat = micro_actions.reshape(-1)
                    elif len(main_action_logits.shape) == 2:
                        # Already flat: (batch, n_actions)
                        main_action_logits_flat = main_action_logits
                        micro_actions_flat = micro_actions
                    else:
                        raise ValueError(f"Unexpected logits shape: {main_action_logits.shape}")
                    
                    # Debug shapes on first batch
                    if micro_start == 0 and epoch == 0:
                        print(f"    🔍 Micro-batch shapes:")
                        print(f"      Input positions: {micro_positions.shape}")
                        print(f"      Logits: {main_action_logits.shape} -> {main_action_logits_flat.shape}")
                        print(f"      Actions: {micro_actions.shape} -> {micro_actions_flat.shape}")
                        print(f"      Old log probs: {micro_old_log_probs.shape}")
                    
                    # Main action distribution
                    main_dist = Categorical(logits=main_action_logits_flat)
                    new_log_probs = main_dist.log_prob(micro_actions_flat)
                    entropy = main_dist.entropy().mean()
                    
                    # Handle advantages shape if needed
                    if len(micro_advantages.shape) > len(new_log_probs.shape):
                        micro_advantages = micro_advantages.squeeze()
                    elif len(micro_advantages.shape) < len(new_log_probs.shape):
                        micro_advantages = micro_advantages.unsqueeze(-1)
                    
                    # PPO clipped objective
                    ratio = (new_log_probs - micro_old_log_probs).exp()
                    surr1 = ratio * micro_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * micro_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    values_pred = values_pred.squeeze(-1)
                    value_loss = F.mse_loss(values_pred, micro_returns)
                    
                    # Combined loss
                    loss = (policy_loss + 
                            self.config.value_coef * value_loss - 
                            self.config.entropy_coef * entropy)
                    
                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    
                    # Accumulate for logging
                    epoch_loss += loss.item()
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_entropy += entropy.item()
                    num_micro_batches += 1
                    
                    # Clear micro-batch tensors
                    del micro_spatial, micro_global, micro_positions, micro_energies
                    del main_action_logits, sap_target_logits, values_pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Clear chunk tensors
                del chunk_spatial, chunk_global, chunk_positions, chunk_energies
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average over micro-batches
            if num_micro_batches > 0:
                total_loss += epoch_loss / num_micro_batches
                total_policy_loss += epoch_policy_loss / num_micro_batches
                total_value_loss += epoch_value_loss / num_micro_batches
                total_entropy += epoch_entropy / num_micro_batches
        
        # Clear used experiences
        self.memory = self.memory[self.config.batch_size:]
        
        print(f"  ✅ Update complete - Memory freed: {self.config.batch_size} experiences")
        
        if n_epochs > 0 and num_micro_batches > 0:
            return (
                total_loss / n_epochs,
                total_policy_loss / n_epochs,
                total_value_loss / n_epochs,
                total_entropy / n_epochs
            )
        else:
            return 0.0, 0.0, 0.0, 0.0
    
    def remember(self, state, main_action, sap_target, log_prob, reward, next_state, done):
        """
        Store experience in memory
        
        Args:
            state: tuple of (spatial, global, unit_positions, unit_energies)
            main_action: action index tensor
            sap_target: (dx, dy) relative target for SAP
            log_prob: log probability of action
            reward: scalar reward
            next_state: tuple of next state
            done: boolean indicating episode end
        """
        self.memory.append((state, main_action, sap_target, log_prob, reward, next_state, done))

    def clear_memory(self):
        self.memory = []