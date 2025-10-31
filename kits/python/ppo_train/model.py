import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with squeeze-excitation (like Frog Parade)"""
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


class LuxResNetCNN(nn.Module):
    """
    ✅ FIXED: Handles variable n_units flexibly
    - Can process 1 unit (training) or multiple units (inference)
    - No assumptions about fixed unit count
    """
    def __init__(self, 
                 spatial_channels=80,
                 global_dim=115,
                 n_main_actions=6, 
                 hidden_dim=256,
                 n_blocks=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_main_actions = n_main_actions
        
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
        
        # Main actor head (NO_OP, 4 moves, SAP yes/no)
        self.main_actor_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for unit energy
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_main_actions)
        )
        
        # SAP target head (outputs spatial logits for where to SAP)
        self.sap_target_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, 1)  # (batch, 1, 24, 24)
        )
        
        # Value head (predicts win probability)
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
        ✅ FIXED: Flexible n_units handling
        
        Args:
            spatial_obs: (batch, spatial_channels, 24, 24)
            global_obs: (batch, global_dim)
            unit_positions: (batch, n_units, 2) - VARIABLE n_units!
            unit_energies: (batch, n_units, 1) - VARIABLE n_units!
        
        Returns:
            main_action_logits: (batch, n_units, n_main_actions) - matches input n_units
            sap_target_logits: (batch, 24, 24)
            value: (batch, 1)
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
        
        # Value prediction
        value = self.value_head(core_output)
        
        # SAP target prediction (spatial)
        sap_target_logits = self.sap_target_head(core_output).squeeze(1)  # (batch, 24, 24)
        
        # Main action prediction (per-unit) - ✅ HANDLES VARIABLE n_units
        main_action_logits = None
        if unit_positions is not None and unit_energies is not None:
            # ✅ Get actual number of units from input (could be 1, 16, or anything!)
            n_units = unit_positions.shape[1]
            unit_features_list = []
            
            for b in range(batch_size):
                batch_features = []
                for u in range(n_units):
                    x, y = unit_positions[b, u]
                    x, y = int(x), int(y)
                    if 0 <= x < 24 and 0 <= y < 24:
                        unit_feat = core_output[b, :, y, x]  # (hidden_dim,)
                    else:
                        # Invalid position (e.g., respawning) - use zero features
                        unit_feat = torch.zeros(self.hidden_dim, device=core_output.device)
                    batch_features.append(unit_feat)
                unit_features_list.append(torch.stack(batch_features))
            
            unit_features = torch.stack(unit_features_list)  # (batch, n_units, hidden_dim)
            
            # Concatenate with unit energy
            unit_input = torch.cat([unit_features, unit_energies], dim=-1)  # (batch, n_units, hidden_dim+1)
            
            # Reshape for MLP: flatten batch and unit dims
            unit_input_flat = unit_input.view(-1, self.hidden_dim + 1)  # (batch*n_units, hidden_dim+1)
            main_action_logits_flat = self.main_actor_head(unit_input_flat)  # (batch*n_units, n_actions)
            
            # Reshape back to (batch, n_units, n_actions)
            main_action_logits = main_action_logits_flat.view(batch_size, n_units, self.n_main_actions)
        
        return main_action_logits, sap_target_logits, value
    
    def get_value_for_both_players(self, spatial_obs_p0, global_obs_p0, 
                                    spatial_obs_p1, global_obs_p1):
        """
        Training trick from Frog Parade: value can see both players
        This helps stabilize training by providing a clearer signal
        
        Returns normalized win probability for both players
        """
        _, _, value_p0 = self.forward(spatial_obs_p0, global_obs_p0)
        _, _, value_p1 = self.forward(spatial_obs_p1, global_obs_p1)
        
        # Softmax to get win probabilities
        values_combined = torch.cat([value_p0, value_p1], dim=-1)
        win_probs = F.softmax(values_combined, dim=-1)
        
        return win_probs[:, 0:1], win_probs[:, 1:2]  # (batch, 1) each