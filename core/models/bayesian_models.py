import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops
import numpy as np

# For Bayesian Neural Networks, we'll use the Blitz library
# You'll need to install it with: pip install blitz-bayesian-pytorch
try:
    import blitz.modules as bnn
    import blitz.utils as utils
    from blitz.modules import BayesianLinear
    BLITZ_AVAILABLE = True
except ImportError:
    print("Blitz library not found. Using standard PyTorch layers with MC Dropout instead.")
    BLITZ_AVAILABLE = False
    # Define a fallback BayesianLinear that uses MC Dropout
    class BayesianLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            return self.dropout(self.linear(x))

class FeatureNormalizer:
    def __init__(self):
        self.node_mean = None
        self.node_std = None
        self.edge_mean = None
        self.edge_std = None
        
    def fit(self, data_list):
        """Compute mean and std from a list of graph data objects"""
        node_features = torch.cat([data.x for data in data_list])
        edge_features = torch.cat([data.edge_attr for data in data_list])
        
        self.node_mean = node_features.mean(dim=0)
        self.node_std = node_features.std(dim=0)
        self.edge_mean = edge_features.mean(dim=0)
        self.edge_std = edge_features.std(dim=0)
        
        # Prevent division by zero with small epsilon
        eps = 1e-8
        self.node_std = torch.where(self.node_std > eps, self.node_std, torch.ones_like(self.node_std))
        self.edge_std = torch.where(self.edge_std > eps, self.edge_std, torch.ones_like(self.edge_std))
        
    def transform(self, x, edge_attr):
        """Normalize features"""
        x_norm = (x - self.node_mean) / self.node_std
        edge_attr_norm = (edge_attr - self.edge_mean) / self.edge_std
        return x_norm, edge_attr_norm

class BayesianMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=64):
        super().__init__()
        self.bayes_linear1 = BayesianLinear(in_features, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.bayes_linear2 = BayesianLinear(hidden_size, out_features)
        
    def forward(self, x):
        x = self.bayes_linear1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.bayes_linear2(x)
        return x

class BayesianGNNLayer(MessagePassing):
    def __init__(self, node_features, edge_features, out_features):
        super().__init__(aggr='add')  # Use additive aggregation

        # Attention mechanism
        self.attention = nn.Sequential(
            BayesianLinear(2 * node_features + edge_features, out_features),
            nn.LeakyReLU(0.2),
            BayesianLinear(out_features, 1),
            nn.Sigmoid()
        )

        # Node transformation
        self.node_nn = nn.Sequential(
            BayesianLinear(node_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )

        # Edge transformation
        self.edge_nn = nn.Sequential(
            BayesianLinear(edge_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )

        # Message transformation
        self.message_nn = nn.Sequential(
            BayesianLinear(2 * node_features + edge_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )

        # Update transformation
        self.update_nn = nn.Sequential(
            BayesianLinear(node_features + out_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )

    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        x_transformed = self.node_nn(x)
        
        # Transform edge features
        edge_attr_transformed = self.edge_nn(edge_attr)
        
        # Message passing
        out = self.propagate(edge_index, x=x, x_transformed=x_transformed, 
                            edge_attr=edge_attr_transformed)
        
        # Update with residual connection
        return out + x_transformed

    def message(self, x_i, x_j, edge_attr):
        # Concatenate source and target node features with edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Compute attention weights
        alpha = self.attention(msg_input)
        
        # Apply attention to message
        msg = self.message_nn(msg_input)
        
        return alpha * msg

    def update(self, aggr_out, x):
        # Combine aggregated messages with original node features
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_nn(update_input)

class BayesianDMRGNet(nn.Module):
    def __init__(self, node_features=4, edge_features=4, hidden_size=64, num_layers=3, num_samples=10):
        super().__init__()
        self.feature_normalizer = FeatureNormalizer()
        self.num_samples = num_samples

        # Multiple Bayesian GNN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(BayesianGNNLayer(node_features, edge_features, hidden_size))
        for _ in range(num_layers - 1):
            self.gnn_layers.append(BayesianGNNLayer(hidden_size, edge_features, hidden_size))

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Global pooling transformation
        self.pool_nn = BayesianMLP(hidden_size, hidden_size)

        # Final MLP for prediction
        self.mlp = nn.Sequential(
            BayesianLinear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            BayesianLinear(hidden_size, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, BayesianLinear):
            if hasattr(module, 'weight'):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def fit_normalizer(self, data_list):
        """Fit the feature normalizer on training data"""
        self.feature_normalizer.fit(data_list)

    def forward(self, data):
        # Normalize features
        x_norm, edge_attr_norm = self.feature_normalizer.transform(data.x, data.edge_attr)

        # Initial node features
        x = x_norm

        # GNN layers with residual connections
        for gnn_layer in self.gnn_layers:
            x_new = gnn_layer(x, data.edge_index, edge_attr_norm)
            x = x_new + x  # Residual connection
            x = self.dropout(x)

        # Transform before pooling
        x = self.pool_nn(x)

        # Multiple pooling operations
        x_mean = global_mean_pool(x, data.batch)
        x_sum = global_add_pool(x, data.batch)

        # Combine different pooling results
        x_combined = torch.cat([x_mean, x_sum], dim=-1)

        # Final prediction
        return self.mlp(x_combined)
    
    def predict_with_uncertainty(self, data, num_samples=None):
        """
        Make predictions with uncertainty estimation
        
        Args:
            data: Input graph data
            num_samples: Number of Monte Carlo samples (default: self.num_samples)
            
        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
            samples: All sampled predictions
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        self.train()  # Enable dropout for MC sampling
        
        samples = []
        for _ in range(num_samples):
            prediction = self.forward(data)
            samples.append(prediction)
            
        samples = torch.cat(samples, dim=1)
        mean = samples.mean(dim=1, keepdim=True)
        std = samples.std(dim=1, keepdim=True)
        
        return mean, std, samples 