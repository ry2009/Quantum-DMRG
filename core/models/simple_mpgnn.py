import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Import PyTorch Geometric libraries
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import global_mean_pool

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# FeatureNormalizer: Compute mean and std for node and edge features from a list of Data objects.
class FeatureNormalizer:
    def __init__(self):
        self.node_mean = None
        self.node_std = None
        self.edge_mean = None
        self.edge_std = None
        
    def fit(self, data_list):
        # Concatenate all node features and edge attributes from the dataset
        node_features = []
        edge_features = []
        
        for data in data_list:
            if data.x is not None:
                node_features.append(data.x)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_features.append(data.edge_attr)
        
        if node_features:
            node_features = torch.cat(node_features, dim=0)
            self.node_mean = torch.mean(node_features, dim=0)
            self.node_std = torch.std(node_features, dim=0)
            # Prevent division by zero
            self.node_std[self.node_std < 1e-6] = 1.0
        
        if edge_features:
            edge_features = torch.cat(edge_features, dim=0)
            self.edge_mean = torch.mean(edge_features, dim=0)
            self.edge_std = torch.std(edge_features, dim=0)
            self.edge_std[self.edge_std < 1e-6] = 1.0
        
    def transform(self, x, edge_attr=None):
        if x is not None and self.node_mean is not None:
            x_norm = (x - self.node_mean) / self.node_std
        else:
            x_norm = x
            
        if edge_attr is not None and self.edge_mean is not None:
            edge_attr_norm = (edge_attr - self.edge_mean) / self.edge_std
        else:
            edge_attr_norm = edge_attr
            
        return x_norm, edge_attr_norm


# TargetNormalizer: Normalizes the target values (e.g., energy)
class TargetNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, targets):
        self.mean = targets.mean()
        self.std = targets.std()
        if self.std < 1e-8:
            self.std = 1.0
            
    def transform(self, targets):
        return (targets - self.mean) / self.std
    
    def inverse_transform(self, targets_norm):
        return targets_norm * self.std + self.mean


# A simple Message Passing Graph Neural Network (MPGNN)
class SimpleMPGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim=1):
        """
        Args:
            node_dim: Dimension of node features (e.g., single-orbital entropy)
            edge_dim: Dimension of edge features (e.g., mutual information and occupancy info)
            hidden_dim: Hidden dimension used for internal projections
            out_dim: Output dimension (e.g., energy prediction, default 1)
        """
        super(SimpleMPGNN, self).__init__()
        
        # Input projection for node features
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        # (Optional) Input projection for edge features if needed
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_proj = None
        
        # Message passing layer:
        # In the simplest case, we implement f and g as identity functions.
        # However, to allow learning, we can use a simple one-layer network.
        self.message_fn = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),  # Concatenate: source node, target node, and edge
            nn.ReLU()
        )
        
        # Update function: combine aggregated message with original node features.
        self.update_fn = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Final graph readout: Pool node representations and then MLP to predict energy.
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, data):
        """
        Args:
            data: PyG Data object with attributes:
                x: Node features tensor of shape [num_nodes, node_dim]
                edge_index: Tensor of shape [2, num_edges]
                edge_attr: (Optional) Tensor of shape [num_edges, edge_dim]
                batch: Tensor assigning nodes to graphs for pooling
        """
        # Get input features
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Project node features to hidden_dim
        h = self.node_proj(x)  # shape: [num_nodes, hidden_dim]
        
        # If edge features exist, project them too.
        if edge_attr is not None and self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)  # shape: [num_edges, hidden_dim]
        
        # Message passing: for each edge, create a message.
        # We assume edge_index[0] are source nodes and edge_index[1] are target nodes.
        row, col = edge_index  # row: indices of source nodes; col: target nodes
        
        # Gather the features for source and target nodes.
        h_source = h[row]   # shape: [num_edges, hidden_dim]
        h_target = h[col]   # shape: [num_edges, hidden_dim]
        
        # If edge_attr exists, include it; otherwise, use zeros.
        if edge_attr is None:
            edge_attr = torch.zeros((row.size(0), h.size(1)), device=h.device)
        
        # Concatenate source, target, and edge features.
        msg_input = torch.cat([h_source, h_target, edge_attr], dim=-1)
        
        # Compute message using the message function f (here, a small neural network).
        messages = self.message_fn(msg_input)  # shape: [num_edges, hidden_dim]
        
        # Aggregate messages for each target node: we use mean aggregation.
        # We initialize an aggregation tensor for each node.
        num_nodes = h.size(0)
        aggregated = torch.zeros((num_nodes, h.size(1)), device=h.device)
        # For each edge, add its message to the corresponding target node.
        aggregated = aggregated.index_add_(0, col, messages)
        
        # Count number of messages per node to compute mean
        count = torch.zeros((num_nodes,), device=h.device)
        count = count.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
        # Avoid division by zero
        count = count.unsqueeze(-1).clamp(min=1)
        aggregated = aggregated / count
        
        # Update function: combine original node features with aggregated messages.
        h_updated = self.update_fn(torch.cat([h, aggregated], dim=-1))
        
        # Final graph pooling: mean pooling over nodes for each graph.
        graph_repr = global_mean_pool(h_updated, batch)
        
        # Final prediction using readout MLP.
        out = self.readout_mlp(graph_repr)
        return out


# Training loop for one epoch
def train_epoch(model, loader, optimizer, criterion, feature_normalizer, target_normalizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        # Normalize node and edge features using our FeatureNormalizer
        data.x, data.edge_attr = feature_normalizer.transform(data.x, data.edge_attr)
        
        optimizer.zero_grad()
        output = model(data)  # Forward pass: shape [batch_size, out_dim]
        # Normalize target values
        target = target_normalizer.transform(data.y)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs  # data.num_graphs from batch
    return total_loss / len(loader.dataset)


# Evaluation function: returns loss and predictions for all graphs
def evaluate(model, loader, criterion, feature_normalizer, target_normalizer, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.x, data.edge_attr = feature_normalizer.transform(data.x, data.edge_attr)
            output = model(data)
            target = target_normalizer.transform(data.y)
            loss = criterion(output, target)
            total_loss += loss.item() * data.num_graphs
            
            # Convert normalized predictions back to original scale
            pred = target_normalizer.inverse_transform(output)
            all_preds.append(pred.cpu())
            all_targets.append(data.y.cpu())
            
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics: RMSE, MAE, R²
    mse = F.mse_loss(preds, targets).item()
    rmse = np.sqrt(mse)
    mae = F.l1_loss(preds, targets).item()
    
    # Calculate R² score
    variance = torch.var(targets, unbiased=False)
    r2 = 1 - (mse / variance)
    
    return avg_loss, preds, targets, rmse, mae, r2.item()


# Function to plot predictions vs. true values
def plot_predictions(true_vals, pred_vals, save_path=None):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(true_vals.min().item(), pred_vals.min().item())
    max_val = max(true_vals.max().item(), pred_vals.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    # Add labels and title
    plt.xlabel("True Energy")
    plt.ylabel("Predicted Energy")
    plt.title("Predicted vs True Energies")
    
    # Calculate metrics for display
    mse = ((true_vals - pred_vals) ** 2).mean().item()
    rmse = np.sqrt(mse)
    mae = (true_vals - pred_vals).abs().mean().item()
    
    # Calculate R² score
    variance = torch.var(true_vals, unbiased=False)
    r2 = 1 - (mse / variance)
    
    # Add metrics text box
    plt.text(0.05, 0.95, 
             f"RMSE: {rmse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.6f}",
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show() 