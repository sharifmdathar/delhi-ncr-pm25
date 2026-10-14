import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphWaveNet(nn.Module):
    """
    Simplified Graph WaveNet for spatiotemporal forecasting.
    
    Based on: https://github.com/nnzhan/Graph-WaveNet
    Adapted for frozen .npz dataset format.
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.name = "graph_wavenet"
        self.config = config or {}
        
        # Model hyperparameters
        self.hidden_size = self.config.get("hidden_size", 64)
        self.num_timesteps_input = self.config.get("num_timesteps_input", 72)
        self.num_timesteps_output = self.config.get("num_timesteps_output", 48)
        self.graph_type = self.config.get("graph_type", "wind_aware")
        self.residual_channels = self.config.get("residual_channels", 32)
        self.dilation_channels = self.config.get("dilation_channels", 32)
        self.skip_channels = self.config.get("skip_channels", 64)
        self.end_channels = self.config.get("end_channels", 128)
        self.blocks = self.config.get("blocks", 2)
        self.layers = self.config.get("layers", 4)
        self.dropout = self.config.get("dropout", 0.3)
        
        self.num_nodes = self.config.get("num_nodes", 70)  # dynamic, default 70
        self.num_features = 1  # PM2.5 only for now; can extend
        
        # Graph adjacency (will be set from data)
        self.register_buffer('adjacency', torch.eye(self.num_nodes))
        
        # Simplified architecture: 1D temporal convolutions + graph convolution
        self.start_conv = nn.Conv1d(
            in_channels=self.num_features,
            out_channels=self.residual_channels,
            kernel_size=1
        )
        
        # Dilated causal convolutions
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        self.receptive_field = 1
        
        for b in range(self.blocks):
            new_dilation = 1
            for i in range(self.layers):
                # Dilated convolutions (1D temporal) - use padding to maintain length
                self.filter_convs.append(nn.Conv1d(
                    in_channels=self.residual_channels,
                    out_channels=self.dilation_channels,
                    kernel_size=3,
                    dilation=new_dilation,
                    padding=new_dilation
                ))
                
                self.gate_convs.append(nn.Conv1d(
                    in_channels=self.residual_channels,
                    out_channels=self.dilation_channels,
                    kernel_size=3,
                    dilation=new_dilation,
                    padding=new_dilation
                ))
                
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(
                    in_channels=self.dilation_channels,
                    out_channels=self.residual_channels,
                    kernel_size=1
                ))
                
                self.skip_convs.append(nn.Conv1d(
                    in_channels=self.dilation_channels,
                    out_channels=self.skip_channels,
                    kernel_size=1
                ))
                
                self.bn.append(nn.BatchNorm1d(self.residual_channels))
                
                new_dilation *= 2
                self.receptive_field += (3 - 1) * new_dilation // 2
                
        self.end_conv_1 = nn.Conv1d(
            in_channels=self.skip_channels,
            out_channels=self.end_channels,
            kernel_size=1,
            bias=True
        )
        
        self.end_conv_2 = nn.Conv1d(
            in_channels=self.end_channels,
            out_channels=self.num_timesteps_output,
            kernel_size=1,
            bias=True
        )
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Adaptive adjacency for Graph WaveNet
        if self.graph_type == "learned_adaptive":
            self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes), requires_grad=True)
    
    def set_adjacency(self, adj: torch.Tensor):
        """Set the adjacency matrix (computed from training data)."""
        self.register_buffer('adjacency', adj)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: (batch, seq_len, num_nodes, num_features)
            
        Returns:
            output: (batch, num_timesteps_output, num_nodes)
        """
        batch_size = input.size(0)
        # input: (B, T, N, F) -> (B, N, T) for 1D conv
        x = input.transpose(1, 2).reshape(batch_size * self.num_nodes, self.num_features, -1)
        
        # Pad if needed
        in_len = x.size(2)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0))
        
        x = self.start_conv(x)  # (B*N, residual_channels, T)
        skip = 0
        
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            
            # Dilated convolution with kernel_size=3, padding=dilation (maintains length)
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            
            # Skip connection
            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip
            
            # Residual connection with length matching
            x = self.residual_convs[i](x)
            x = self.bn[i](x)
            if x.size(2) != residual.size(2):
                residual = residual[:, :, -x.size(2):]
            x = residual + x
        
        # Output - apply to skip connection
        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # (B*N, output_horizons, T)
        
        # Take the last time step for each node
        x = x[:, :, -1]  # (B*N, output_horizons)
        
        # Reshape to (batch, output_timesteps, num_nodes)
        x = x.view(batch_size, self.num_timesteps_output, self.num_nodes)
        return x


def build_graph_wavenet_model(config=None):
    """Build Graph WaveNet model from config."""
    return GraphWaveNet(config)