"""
Working Model Architecture - Based on successful approaches
Key: ResNet with squeeze-excitation, proper action heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with squeeze-excitation (from Frog Parade)"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-excitation
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(channels, channels // 4)
        self.se_fc2 = nn.Linear(channels // 4, channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Squeeze-excitation
        se = self.se_pool(out).view(out.size(0), -1)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        se = se.view(out.size(0), out.size(1), 1, 1)
        out = out * se
        
        out = out + residual
        out = F.relu(out)
        return out


class WorkingModel(nn.Module):
    """
    Working model architecture combining:
    - Frog Parade's ResNet with squeeze-excitation
    - Working MADDPG's hierarchical action space
    """
    def __init__(self, 
                 spatial_channels=24,  # 4 temporal × 3 history + 12 static
                 global_dim=20,
                 n_main_actions=6,
                 hidden_dim=256,
                 n_blocks=8,
                 max_sap_range=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_main_actions = n_main_actions
        self.max_sap_range = max_sap_range
        
        # SAP target space: (2*range+1)^2 offsets
        self.num_sap_targets = (2 * max_sap_range + 1) ** 2  # 15×15 = 225
        
        # Spatial input projection
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(spatial_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Global input projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Core ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        
        # Main action head (per-unit)
        self.main_action_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for unit energy
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_main_actions)
        )
        
        # SAP target head (per-unit, offset-based)
        self.sap_target_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_sap_targets)
        )
        
        # Value head (team-level, can see both players - Frog Parade trick)
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, spatial_obs, global_obs, unit_positions=None, unit_energies=None):
        """
        Forward pass with fixed dimension handling
        """
        batch_size = spatial_obs.shape[0]
        
        # Process spatial features
        spatial_features = self.spatial_proj(spatial_obs)  # (batch, hidden_dim, 24, 24)
        
        # Process global features and broadcast
        global_features = self.global_proj(global_obs)  # (batch, hidden_dim)
        global_features = global_features.view(batch_size, self.hidden_dim, 1, 1)
        global_features = global_features.expand(-1, -1, 24, 24)
        
        # Combine spatial and global
        combined = spatial_features + global_features
        
        # Pass through ResNet blocks
        core_output = combined
        for block in self.res_blocks:
            core_output = block(core_output)
        
        # Value prediction (team-level)
        value = self.value_head(core_output)
        
        # Action predictions (per-unit)
        main_action_logits = None
        sap_target_logits = None
        
        if unit_positions is not None and unit_energies is not None:
            n_units = unit_positions.shape[1]
            unit_features_list = []
            
            # Extract features for each unit
            for b in range(batch_size):
                batch_features = []
                for u in range(n_units):
                    x, y = unit_positions[b, u]
                    x, y = int(x), int(y)
                    if 0 <= x < 24 and 0 <= y < 24:
                        unit_feat = core_output[b, :, y, x]
                    else:
                        # Invalid position (respawning)
                        unit_feat = torch.zeros(self.hidden_dim, device=core_output.device)
                    batch_features.append(unit_feat)
                unit_features_list.append(torch.stack(batch_features))
            
            unit_features = torch.stack(unit_features_list)  # (batch, n_units, hidden_dim)
            
            # FIX: Ensure unit_energies has correct dimensions
            # unit_energies should be (batch, n_units, 1)
            if unit_energies.dim() == 2:
                unit_energies = unit_energies.unsqueeze(-1)  # Add feature dimension
            
            # Concatenate with unit energy
            unit_input = torch.cat([unit_features, unit_energies], dim=-1)  # (batch, n_units, hidden_dim+1)
            
            # Flatten for MLP
            unit_input_flat = unit_input.view(-1, self.hidden_dim + 1)  # (batch*n_units, hidden_dim+1)
            
            # Main actions
            main_action_logits_flat = self.main_action_head(unit_input_flat)
            main_action_logits = main_action_logits_flat.view(batch_size, n_units, self.n_main_actions)
            
            # SAP targets
            sap_target_logits_flat = self.sap_target_head(unit_input_flat)
            sap_target_logits = sap_target_logits_flat.view(batch_size, n_units, self.num_sap_targets)
        
        return main_action_logits, sap_target_logits, value
    
    def sap_index_to_offset(self, sap_indices):
        """Convert SAP index to (dx, dy) offset"""
        # Build offset table
        offsets_list = []
        for dx in range(-self.max_sap_range, self.max_sap_range + 1):
            for dy in range(-self.max_sap_range, self.max_sap_range + 1):
                offsets_list.append([dx, dy])
        
        offset_table = torch.tensor(
            offsets_list,
            dtype=torch.long,
            device=sap_indices.device
        )
        
        return offset_table[sap_indices]
    
    def sap_offset_to_index(self, sap_offsets):
        """Convert (dx, dy) offset to SAP index"""
        dx = sap_offsets[..., 0] + self.max_sap_range
        dy = sap_offsets[..., 1] + self.max_sap_range
        indices = dx * (2 * self.max_sap_range + 1) + dy
        return indices.long()