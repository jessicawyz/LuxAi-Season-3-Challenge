import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(self, x, hidden_state):
        h, c = hidden_state
        
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class SpatialEncoder(nn.Module):
    
    def __init__(self, input_channels: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.convlstm = ConvLSTMCell(128, hidden_dim, kernel_size=3)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, spatial_features, hidden_state=None):
        batch_size = spatial_features.shape[0]
        
        x = F.relu(self.conv1(spatial_features))
        x = F.relu(self.conv2(x))
        
        if hidden_state is None:
            h = torch.zeros(
                batch_size, self.hidden_dim,
                spatial_features.shape[2], spatial_features.shape[3],
                device=spatial_features.device
            )
            c = torch.zeros_like(h)
            hidden_state = (h, c)
        
        h, new_hidden_state = self.convlstm(x, hidden_state)
        
        pooled = self.pool(h).flatten(1)
        encoded = self.fc(pooled)
        
        return encoded, new_hidden_state


class TransformerUnitEncoder(nn.Module):
        
    def __init__(
        self,
        unit_feature_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(unit_feature_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, unit_features, unit_mask):
        batch_size = unit_features.shape[0]
        num_units = unit_features.shape[1]
        
        has_valid_units = unit_mask.any(dim=1)
        
        x = self.input_proj(unit_features)
        
        encoded = torch.zeros_like(x)
        aggregated = torch.zeros(batch_size, x.shape[-1], device=x.device)
        
        if has_valid_units.any():
            attn_mask = ~unit_mask
            
            valid_indices = torch.where(has_valid_units)[0]
            
            if len(valid_indices) > 0:
                x_valid = x[valid_indices]
                attn_mask_valid = attn_mask[valid_indices]
                
                encoded_valid = self.transformer(x_valid, src_key_padding_mask=attn_mask_valid)
                
                encoded_valid = self.output_proj(encoded_valid)
                
                encoded[valid_indices] = encoded_valid
                
                mask_expanded = unit_mask[valid_indices].unsqueeze(-1).float()
                aggregated_valid = (encoded_valid * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
                aggregated[valid_indices] = aggregated_valid
        
        return encoded, aggregated

