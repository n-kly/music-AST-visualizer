import torch
import torch.nn as nn

class AudioTransformerModel(nn.Module):
    def __init__(self, patch_size, num_layers, num_heads, d_model, dim_feedforward, dropout=0.1):
        super(AudioTransformerModel, self).__init__()

        self.conv2d = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=(2, 2), stride=2)
        self.gelu = nn.GELU()
        self.linear_proj = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 258, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, 768)

    def forward(self, x):
        batch_size, num_patches, mel_bins, time = x.size()
        
        # Reshape input to (batch_size * num_patches, 1, mel_bins, time)
        x = x.view(batch_size * num_patches, 1, mel_bins, time)
        
        # Apply 2D convolution
        x = self.conv2d(x)
        x = self.gelu(x)
        
        # Reshape to (batch_size, num_patches, d_model, new_height * new_width)
        _, d_model, new_height, new_width = x.size()
        x = x.view(batch_size, num_patches, d_model, -1).mean(dim=-1)
        
        # Linear projection
        x = self.linear_proj(x)

        # Layer normalization
        x = self.layer_norm1(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :num_patches, :]
        
        # Transformer encoder
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Layer normalization
        x = self.layer_norm2(x)
        
        # Output layer
        x = self.output_layer(x.mean(dim=1))  # Global average pooling
        # print(x)
        return x