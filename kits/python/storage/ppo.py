"""
Working PPO Implementation - Based on successful approaches
Key: Team-level rewards, proper GAE, joint log probability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class WorkingPPO:
    """
    PPO implementation based on successful approaches:
    - Team-level rewards (Frog Parade)
    - Proper GAE implementation
    - Joint log probability across units
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
        
        # GAE parameters
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        
    def compute_gae(self, rewards, values, dones):
        """
        Proper GAE implementation
        
        Args:
            rewards: list of rewards
            values: list of value estimates
            dones: list of done flags
        
        Returns:
            advantages: tensor of advantages
            returns: tensor of returns
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0  # Terminal value is 0
        
        # Work backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        # Convert to tensors and normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """PPO update with proper batch handling"""
        if len(self.memory) < self.config.batch_size:
            return 0.0, 0.0, 0.0, 0.0
        
        print(f"\n🎯 PPO Update with {len(self.memory)} experiences")
        
        # Take exactly batch_size experiences
        experiences = self.memory[:self.config.batch_size]
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Shuffle experiences
            indices = torch.randperm(len(experiences))
            
            # Process in mini-batches
            for start_idx in range(0, len(experiences), self.config.mini_batch_size):
                end_idx = min(start_idx + self.config.mini_batch_size, len(experiences))
                batch_indices = indices[start_idx:end_idx]
                
                if len(batch_indices) == 0:
                    continue
                    
                # Collect batch data
                batch_spatial = []
                batch_global = []
                batch_positions = []
                batch_energies = []
                batch_main_actions = []
                batch_old_log_probs = []
                batch_rewards = []
                batch_next_values = []
                batch_dones = []
                
                for idx in batch_indices:
                    exp = experiences[idx]
                    batch_spatial.append(exp[0])
                    batch_global.append(exp[1])
                    batch_positions.append(exp[2])
                    batch_energies.append(exp[3])
                    batch_main_actions.append(exp[4])
                    batch_old_log_probs.append(exp[6])
                    batch_rewards.append(exp[7])
                    batch_next_values.append(exp[8])
                    batch_dones.append(exp[9])
                
                # Stack batch data
                spatial_batch = torch.cat(batch_spatial, dim=0)
                global_batch = torch.cat(batch_global, dim=0)
                
                # Handle variable unit counts by processing each experience separately in the batch
                for i, idx in enumerate(batch_indices):
                    exp = experiences[idx]
                    
                    # Unpack single experience
                    (spatial_single, global_single, positions_single, energies_single,
                    main_actions_single, _, old_log_prob_single,
                    reward_single, next_value_single, done_single) = exp
                    
                    # Get current value for this state
                    with torch.no_grad():
                        _, _, current_value = self.model(
                            spatial_single, global_single, positions_single, energies_single
                        )
                        current_value = current_value.squeeze(-1)
                    
                    # Compute advantage
                    advantage = reward_single + self.gamma * next_value_single * (1 - done_single) - current_value.item()
                    advantage = torch.tensor([advantage], dtype=torch.float32)
                    returns = torch.tensor([reward_single + self.gamma * next_value_single * (1 - done_single)], dtype=torch.float32)
                    
                    # Forward pass
                    main_logits, _, values_pred = self.model(
                        spatial_single, global_single, positions_single, energies_single
                    )
                    
                    # Handle the action predictions
                    batch_size, n_units, n_actions = main_logits.shape
                    main_logits_flat = main_logits.reshape(-1, n_actions)
                    main_actions_flat = main_actions_single.reshape(-1)
                    old_log_prob_flat = old_log_prob_single.reshape(-1)
                    
                    # Compute new log probabilities
                    main_dist = Categorical(logits=main_logits_flat)
                    new_log_probs = main_dist.log_prob(main_actions_flat)
                    entropy = main_dist.entropy().mean()
                    
                    # PPO clipped objective
                    ratio = (new_log_probs - old_log_prob_flat).exp()
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    values_pred = values_pred.squeeze(-1)
                    value_loss = F.mse_loss(values_pred, returns)
                    
                    # Combined loss
                    loss = (policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy)
                    
                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    
                    # Track metrics
                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    n_updates += 1
        
        # Clear used experiences
        self.memory = self.memory[self.config.batch_size:]
        
        # Return average losses
        if n_updates > 0:
            return (
                total_loss / n_updates,
                total_policy_loss / n_updates,
                total_value_loss / n_updates,
                total_entropy / n_updates
            )
        return 0.0, 0.0, 0.0, 0.0
    
    def remember(self, state_spatial, state_global, state_positions, state_energies,
                 main_action, sap_action, log_prob, reward, next_value, done):
        """Store experience in memory"""
        self.memory.append((
            state_spatial, state_global, state_positions, state_energies,
            main_action, sap_action, log_prob,
            reward, next_value, done
        ))
    
    def clear_memory(self):
        """Clear all memory"""
        self.memory = []