import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool


class FeatureNormalizer:
    """Normalizes node and edge features to improve training stability"""
    def __init__(self):
        self.node_mean = None
        self.node_std = None
        self.edge_mean = None
        self.edge_std = None
        
    def fit(self, data_list):
        """Compute normalization statistics from a list of graph data objects"""
        # Collect all node features
        node_features = []
        edge_features = []
        
        for data in data_list:
            if data.x is not None:
                node_features.append(data.x)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_features.append(data.edge_attr)
                
        # Compute mean and std for node features
        if node_features:
            node_features = torch.cat(node_features, dim=0)
            self.node_mean = torch.mean(node_features, dim=0)
            self.node_std = torch.std(node_features, dim=0)
            self.node_std[self.node_std < 1e-6] = 1.0  # Prevent division by zero
            
        # Compute mean and std for edge features
        if edge_features:
            edge_features = torch.cat(edge_features, dim=0)
            self.edge_mean = torch.mean(edge_features, dim=0)
            self.edge_std = torch.std(edge_features, dim=0)
            self.edge_std[self.edge_std < 1e-6] = 1.0  # Prevent division by zero
            
    def transform(self, x, edge_attr=None):
        """Normalize features using precomputed statistics"""
        if x is not None and self.node_mean is not None:
            x = (x - self.node_mean) / self.node_std
            
        if edge_attr is not None and self.edge_mean is not None:
            edge_attr = (edge_attr - self.edge_mean) / self.edge_std
            
        return x, edge_attr


class SimpleQuantumGNN(nn.Module):
    """
    A simpler quantum-aware GNN model that works with the existing data format
    """
    def __init__(self, 
                 in_channels, 
                 hidden_channels=256, 
                 num_layers=5, 
                 dropout=0.25,
                 heads=8,
                 edge_dim=None,
                 readout_layers=3,
                 readout_mode='combined'):
        """
        Initialize the simple quantum GNN
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden channels
            num_layers: Number of GNN layers
            dropout: Dropout rate
            heads: Number of attention heads for GAT layers
            edge_dim: Number of edge features (optional)
            readout_layers: Number of readout MLP layers
            readout_mode: Readout aggregation mode ('mean', 'max', 'add', or 'combined')
        """
        super(SimpleQuantumGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.edge_dim = edge_dim
        self.readout_mode = readout_mode
        
        # Feature normalization
        self.feature_normalizer = FeatureNormalizer()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers - alternate between GAT and GCN for diversity
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                # GAT layer with attention
                self.gnn_layers.append(
                    GATConv(
                        hidden_channels, 
                        hidden_channels // heads, 
                        heads=heads, 
                        dropout=dropout,
                        edge_dim=edge_dim
                    )
                )
            else:
                # GCN layer
                self.gnn_layers.append(
                    GCNConv(hidden_channels, hidden_channels)
                )
        
        # Layer normalization after each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        
        # Readout MLP 
        if readout_mode == 'combined':
            readout_in_channels = hidden_channels * 3  # For mean, max, and add pooling
        else:
            readout_in_channels = hidden_channels
        
        # Create MLP layers with residual connections
        mlp_layers = []
        mlp_layers.append(nn.Linear(readout_in_channels, hidden_channels))
        mlp_layers.append(nn.LayerNorm(hidden_channels))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(dropout))
        
        for _ in range(readout_layers - 2):
            mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))
            mlp_layers.append(nn.LayerNorm(hidden_channels))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
        
        mlp_layers.append(nn.Linear(hidden_channels, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def fit_normalizer(self, data_list):
        """Fit the feature normalizer on a list of data objects"""
        self.feature_normalizer.fit(data_list)
        
    def forward(self, data):
        """
        Forward pass through the model
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
            
        Returns:
            out: Predicted correlation energy
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Normalize features
        x, edge_attr = self.feature_normalizer.transform(x, edge_attr)
        
        # Initial projection
        h = self.input_proj(x)
        
        # Apply GNN layers
        for i, (layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            if i % 2 == 0 and edge_attr is not None:
                # GAT layer with edge attributes
                h = layer(h, edge_index, edge_attr=edge_attr)
            else:
                # GCN layer or GAT without edge attributes
                h = layer(h, edge_index)
            
            # Apply layer normalization
            h = norm(h)
            
            # Apply dropout and activation
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Readout phase
        if self.readout_mode == 'mean':
            pooled = global_mean_pool(h, batch)
        elif self.readout_mode == 'max':
            pooled = global_max_pool(h, batch)
        elif self.readout_mode == 'add':
            pooled = global_add_pool(h, batch)
        elif self.readout_mode == 'combined':
            # Combine multiple pooling methods
            mean_pooled = global_mean_pool(h, batch)
            max_pooled = global_max_pool(h, batch)
            add_pooled = global_add_pool(h, batch)
            pooled = torch.cat([mean_pooled, max_pooled, add_pooled], dim=1)
        else:
            raise ValueError(f"Unknown readout mode: {self.readout_mode}")
        
        # Apply MLP
        out = self.mlp(pooled)
        
        return out
    
    def predict_with_uncertainty(self, data, num_samples=30):
        """Make predictions with uncertainty estimation using Monte Carlo dropout"""
        # Enable dropout for uncertainty estimation
        self.train()
        
        # Run multiple forward passes
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self(data)
                predictions.append(pred)
                
        # Stack predictions: [num_samples, batch_size]
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate mean and standard deviation
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # Set model back to evaluation mode
        self.eval()
        
        return mean_pred, std_pred 