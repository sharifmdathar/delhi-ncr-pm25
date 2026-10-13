import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet for spatiotemporal forecasting.
    
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
        
        self.num_nodes = 70  # fixed for Delhi-NCR
        self.num_features = 1  # PM2.5 only for now; can extend
        
        # Graph adjacency (will be set from data)
        self.register_buffer('adjacency', torch.eye(self.num_nodes))
        
        # Input projection
        self.start_conv = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=self.residual_channels,
            kernel_size=(1, 1)
        )
        
        # Dilated convolutions
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        self.receptive_field = 1
        
        for b in range(self.blocks):
            additional_scope = self.num_timesteps_input - 1
            new_dilation = 1
            for i in range(self.layers):
                # Dilated convolutions
                self.filter_convs.append(nn.Conv2d(
                    in_channels=self.residual_channels,
                    out_channels=self.dilation_channels,
                    kernel_size=(1, 2),
                    dilation=new_dilation
                ))
                
                self.gate_convs.append(nn.Conv2d(
                    in_channels=self.residual_channels,
                    out_channels=self.dilation_channels,
                    kernel_size=(1, 2),
                    dilation=new_dilation
                ))
                
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(
                    in_channels=self.dilation_channels,
                    out_channels=self.residual_channels,
                    kernel_size=(1, 1)
                ))
                
                self.skip_convs.append(nn.Conv1d(
                    in_channels=self.dilation_channels,
                    out_channels=self.skip_channels,
                    kernel_size=(1, 1)
                ))
                
                self.bn.append(nn.BatchNorm1d(self.residual_channels))
                
                new_dilation *= 2
                self.receptive_field += additional_scope
                additional_scope //= 2
                
        self.end_conv_1 = nn.Conv1d(
            in_channels=self.skip_channels,
            out_channels=self.end_channels,
            kernel_size=(1, 1),
            bias=True
        )
        
        self.end_conv_2 = nn.Conv1d(
            in_channels=self.end_channels,
            out_channels=self.num_timesteps_output * self.num_nodes,
            kernel_size=(1, 1),
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
        input = input.transpose(1, 3)  # (batch, num_features, num_nodes, seq_len)
        
        # Input convolution
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
            
        x = self.start_conv(x)
        skip = 0
        
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            
            # Dilated convolution
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            
            # Parametric skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            
            # Adaptive adjacency if enabled
            if self.graph_type == "learned_adaptive":
                adj = self.adjacency + torch.relu(torch.mm(self.nodevec1, self.nodevec2))
                adj = torch.softmax(adj, dim=-1)
            else:
                adj = self.adjacency
            
            # Graph convolution
            x = torch.einsum('bnnc, nw-> bncw', x, adj)
            x = x.transpose(2, 1)
            
            # Residual connection
            x = self.residual_convs[i](x)
            x = self.bn[i](x)
            x = residual + x
            x = x.transpose(2, 1)
            
        # Output
        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        # Reshape to (batch, output_timesteps, num_nodes)
        x = x.view(batch_size, self.num_timesteps_output, self.num_nodes)
        return x


def build_graph_wavenet_model(config=None):
    """Build Graph WaveNet model from config."""
    return GraphWaveNet(config)