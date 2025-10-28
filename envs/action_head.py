import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class HierarchicalActionHead(nn.Module):
    """
    Hierarchical action head
    
    Action space:
        - First level: Action type (0-5: stay, up, down, left, right, sap)
        - Second level: Sap target (if sap selected)
    """
    
    def __init__(
        self,
        feature_dim: int,
        max_units: int = 16,
        unit_sap_range: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.max_units = max_units
        self.unit_sap_range = unit_sap_range
        self.num_action_types = 6
        
        # num of sap targets within range
        self.num_sap_targets = (2 * unit_sap_range + 1) ** 2
        
        # action type head
        self.action_type_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_action_types)
        )
        
        # sap target head
        self.sap_target_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_sap_targets)
        )
    
    def forward(
        self,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
        unit_energies: torch.Tensor,
        unit_positions: torch.Tensor,
        map_width: int,
        map_height: int,
        tile_types: torch.Tensor,
        unit_move_cost: int,
        unit_sap_cost: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass to compute action logits with masking.
        """
        batch_size = unit_features.shape[0]
        
        # action type logits
        action_type_logits = self.action_type_net(unit_features)  # (B, N, 6)
        
        # sap target logits
        sap_target_logits = self.sap_target_net(unit_features)  # (B, N, num_sap_targets)
        
        # action masks
        action_type_mask = self._compute_action_type_mask(
            unit_mask,
            unit_energies,
            unit_positions,
            map_width,
            map_height,
            tile_types,
            unit_move_cost,
            unit_sap_cost,
        )  # (B, N, 6)
        
        sap_target_mask = self._compute_sap_target_mask(
            unit_positions,
            map_width,
            map_height,
        )  # (B, N, num_sap_targets)
        
        # apply masks (invalid actions as -inf)
        action_type_logits = torch.where(
            action_type_mask,
            action_type_logits,
            torch.tensor(-1e9, dtype=action_type_logits.dtype, device=action_type_logits.device)
        )
        
        sap_target_logits = torch.where(
            sap_target_mask,
            sap_target_logits,
            torch.tensor(-1e9, dtype=sap_target_logits.dtype, device=sap_target_logits.device)
        )
        
        info = {
            "action_type_mask": action_type_mask,
            "sap_target_mask": sap_target_mask,
        }
        
        return action_type_logits, sap_target_logits, info
    
    def _compute_action_type_mask(
        self,
        unit_mask: torch.Tensor,
        unit_energies: torch.Tensor,
        unit_positions: torch.Tensor,
        map_width: int,
        map_height: int,
        tile_types: torch.Tensor,
        unit_move_cost: int,
        unit_sap_cost: int,
    ) -> torch.Tensor:
        
        batch_size = unit_mask.shape[0]
        mask = torch.zeros(
            (batch_size, self.max_units, self.num_action_types),
            dtype=torch.bool,
            device=unit_mask.device
        )
        
        # Action: stay
        mask[:, :, 0] = unit_mask
        
        # Check if enough energy
        can_move = (unit_energies >= unit_move_cost) & unit_mask
        
        # movement directions: up, right, down, left
        directions = torch.tensor(
            [[0, -1], [1, 0], [0, 1], [-1, 0]],
            dtype=torch.long,
            device=unit_positions.device
        )
        
        # check each movement direction
        for i, direction in enumerate(directions):
            action_idx = i + 1  # Actions 1-4 are moves
            
            # compute new positions
            new_positions = unit_positions + direction.unsqueeze(0).unsqueeze(0)  # (B, N, 2)
            
            # check if new position is within map bounds
            in_bounds = (
                (new_positions[:, :, 0] >= 0) &
                (new_positions[:, :, 0] < map_width) &
                (new_positions[:, :, 1] >= 0) &
                (new_positions[:, :, 1] < map_height)
            )
            
            # check if destination is asteroid
            not_blocked = torch.ones_like(in_bounds, dtype=torch.bool)
            
            for b in range(batch_size):
                for u in range(self.max_units):
                    if in_bounds[b, u]:
                        x, y = new_positions[b, u]
                        x, y = x.item(), y.item()
                        if 0 <= x < map_width and 0 <= y < map_height:
                            not_blocked[b, u] = (tile_types[b, x, y] != 2)
            
            # move is valid if: unit exists, has energy, in bounds, not blocked
            mask[:, :, action_idx] = can_move & in_bounds & not_blocked
        
        # Action: sap
        # requires enough energy
        mask[:, :, 5] = (unit_energies >= unit_sap_cost) & unit_mask
        
        return mask
    
    def _compute_sap_target_mask(
        self,
        unit_positions: torch.Tensor,
        map_width: int,
        map_height: int,
    ) -> torch.Tensor:
        """
        Compute mask for valid sap targets.
        """
        batch_size, num_units = unit_positions.shape[:2]
        mask = torch.zeros(
            (batch_size, num_units, self.num_sap_targets),
            dtype=torch.bool,
            device=unit_positions.device
        )
        
        offsets = []
        for dx in range(-self.unit_sap_range, self.unit_sap_range + 1):
            for dy in range(-self.unit_sap_range, self.unit_sap_range + 1):
                offsets.append((dx, dy))
        
        offsets = torch.tensor(offsets, dtype=torch.long, device=unit_positions.device)
        
        # For each unit, check which sap targets are valid
        for target_idx, offset in enumerate(offsets):
            # Compute sap target positions
            target_positions = unit_positions + offset.unsqueeze(0).unsqueeze(0)
            
            # Check if target is within bounds
            in_bounds = (
                (target_positions[:, :, 0] >= 0) &
                (target_positions[:, :, 0] < map_width) &
                (target_positions[:, :, 1] >= 0) &
                (target_positions[:, :, 1] < map_height)
            )
            
            mask[:, :, target_idx] = in_bounds
        
        return mask
    
    def sample_actions(
        self,
        action_type_logits: torch.Tensor,
        sap_target_logits: torch.Tensor,
        action_type_mask: torch.Tensor,
        sap_target_mask: torch.Tensor,
        unit_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample actions from logits.
        """
        batch_size = action_type_logits.shape[0]
        
        # Sample action types
        if deterministic:
            action_types = torch.argmax(action_type_logits, dim=-1)  # (B, N)
        else:
            action_type_dist = torch.distributions.Categorical(logits=action_type_logits)
            action_types = action_type_dist.sample()  # (B, N)
        
        # Sample sap targets
        if deterministic:
            sap_targets = torch.argmax(sap_target_logits, dim=-1)  # (B, N)
        else:
            sap_target_dist = torch.distributions.Categorical(logits=sap_target_logits)
            sap_targets = sap_target_dist.sample()  # (B, N)
        
        sap_offsets = self._sap_index_to_offset(sap_targets)  # (B, N, 2)
        
        actions = torch.zeros(
            (batch_size, self.max_units, 3),
            dtype=torch.long,
            device=action_type_logits.device
        )
        actions[:, :, 0] = action_types
        actions[:, :, 1:] = sap_offsets
        
        # invalid units
        actions = actions * unit_mask.unsqueeze(-1).long()
        
        # log probabilities
        action_type_dist = torch.distributions.Categorical(logits=action_type_logits)
        action_type_log_probs = action_type_dist.log_prob(action_types)  # (B, N)
        
        sap_target_dist = torch.distributions.Categorical(logits=sap_target_logits)
        sap_target_log_probs = sap_target_dist.log_prob(sap_targets)  # (B, N)
        
        # Only use sap log probs if action is sap
        is_sap = (action_types == 5).float()
        total_log_probs = action_type_log_probs + is_sap * sap_target_log_probs
        
        # Mask invalid units
        total_log_probs = total_log_probs * unit_mask
        
        info = {
            "action_type_log_probs": action_type_log_probs,
            "sap_target_log_probs": sap_target_log_probs,
            "total_log_probs": total_log_probs,
            "action_type_entropy": action_type_dist.entropy(),
            "sap_target_entropy": sap_target_dist.entropy(),
        }
        
        return actions, info
    
    def _sap_index_to_offset(self, sap_indices: torch.Tensor) -> torch.Tensor:
        
        batch_size, num_units = sap_indices.shape
        
        offsets_list = []
        for dx in range(-self.unit_sap_range, self.unit_sap_range + 1):
            for dy in range(-self.unit_sap_range, self.unit_sap_range + 1):
                offsets_list.append([dx, dy])
        
        offset_table = torch.tensor(
            offsets_list,
            dtype=torch.long,
            device=sap_indices.device
        ) 

        offsets = offset_table[sap_indices] 
        
        return offsets
    
    def get_log_probs(
        self,
        action_type_logits: torch.Tensor,
        sap_target_logits: torch.Tensor,
        actions: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        action_types = actions[:, :, 0]  # (B, N)
        sap_offsets = actions[:, :, 1:]  # (B, N, 2)
        
        sap_indices = self._sap_offset_to_index(sap_offsets)  # (B, N)
        
        # Compute log probs
        action_type_dist = torch.distributions.Categorical(logits=action_type_logits)
        action_type_log_probs = action_type_dist.log_prob(action_types)
        
        sap_target_dist = torch.distributions.Categorical(logits=sap_target_logits)
        sap_target_log_probs = sap_target_dist.log_prob(sap_indices)
        
        # Only sap log probs if action is sap
        is_sap = (action_types == 5).float()
        total_log_probs = action_type_log_probs + is_sap * sap_target_log_probs
        
        # Mask invalid units
        total_log_probs = total_log_probs * unit_mask
        
        return total_log_probs
    
    def _sap_offset_to_index(self, sap_offsets: torch.Tensor) -> torch.Tensor:
        
        dx = sap_offsets[:, :, 0] + self.unit_sap_range
        dy = sap_offsets[:, :, 1] + self.unit_sap_range
        indices = dx * (2 * self.unit_sap_range + 1) + dy
        return indices.long()
