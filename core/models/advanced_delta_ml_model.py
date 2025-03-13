import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import json
from tqdm import tqdm
import time
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx
import copy
import random
import argparse  # Added for command line argument support

from improved_dmrg_processor import ImprovedDMRGProcessor, DeltaMLDataset

# Add timing utilities
class TimingTracker:
    """Utility class to track and store timing information."""
    
    def __init__(self):
        """Initialize timing tracker."""
        self.timing_data = {
            'system_sizes': [],  # Number of orbitals per system
            'training_times': [],  # Training time per system size
            'inference_times': [],  # Inference time per system size
            'epoch_times': [],  # Time per epoch
            'total_training_time': 0,  # Total training time
            'total_inference_time': 0,  # Total inference time
        }
    
    def add_system_size(self, num_orbitals):
        """Add system size."""
        self.timing_data['system_sizes'].append(num_orbitals)
    
    def add_training_time(self, training_time):
        """Add training time for a system size."""
        self.timing_data['training_times'].append(training_time)
    
    def add_inference_time(self, inference_time):
        """Add inference time for a system size."""
        self.timing_data['inference_times'].append(inference_time)
    
    def add_epoch_time(self, epoch_time):
        """Add time for a single epoch."""
        self.timing_data['epoch_times'].append(epoch_time)
    
    def set_total_training_time(self, total_time):
        """Set total training time."""
        self.timing_data['total_training_time'] = total_time
    
    def set_total_inference_time(self, total_time):
        """Set total inference time."""
        self.timing_data['total_inference_time'] = total_time
    
    def save_timing_data(self, output_dir):
        """Save timing data to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save timing data to JSON
        with open(os.path.join(output_dir, 'timing_data.json'), 'w') as f:
            json.dump(self.timing_data, f, indent=2)
        
        # Create and save plots
        self.plot_timing_data(output_dir)
    
    def plot_timing_data(self, output_dir):
        """Plot timing data."""
        # Plot training time vs system size
        plt.figure(figsize=(10, 6))
        
        # Check if we have system size data
        if len(self.timing_data['system_sizes']) > 0 and len(self.timing_data['training_times']) > 0:
            plt.scatter(self.timing_data['system_sizes'], self.timing_data['training_times'], 
                        label='Training time', s=80, marker='o')
            
            # If we have more than 2 data points, fit a polynomial to estimate scaling
            if len(self.timing_data['system_sizes']) > 2:
                # Try to fit a polynomial (degree 2)
                coeffs = np.polyfit(self.timing_data['system_sizes'], self.timing_data['training_times'], 2)
                poly = np.poly1d(coeffs)
                
                # Create smooth curve for the fit
                x_fit = np.linspace(min(self.timing_data['system_sizes']), 
                                   max(self.timing_data['system_sizes']) * 1.5, 100)
                y_fit = poly(x_fit)
                
                plt.plot(x_fit, y_fit, 'r--', label=f'Polynomial fit: {poly}')
                
                # Estimate maximum feasible system size (assuming 24h training time limit)
                max_time = 24 * 60 * 60  # 24 hours in seconds
                try:
                    # Find roots of polynomial - time limit equation
                    time_limit_poly = poly - max_time
                    roots = np.roots(time_limit_poly)
                    # Find positive real roots
                    feasible_roots = [root.real for root in roots if root.real > 0 and abs(root.imag) < 1e-10]
                    
                    if feasible_roots:
                        max_size = min(feasible_roots)
                        plt.axvline(x=max_size, color='g', linestyle='--', 
                                   label=f'Est. max size: {int(max_size)} orbitals')
                        plt.text(max_size, max_time / 2, f'{int(max_size)} orbitals', 
                                rotation=90, verticalalignment='center')
                except Exception as e:
                    print(f"Could not estimate maximum system size: {e}")
        
        plt.xlabel('Number of Orbitals', fontsize=14)
        plt.ylabel('Training Time (seconds)', fontsize=14)
        plt.title('Training Time vs. System Size', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_time_vs_size.png'), dpi=300)
        plt.close()
        
        # Plot inference time vs system size
        if len(self.timing_data['system_sizes']) > 0 and len(self.timing_data['inference_times']) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.timing_data['system_sizes'], self.timing_data['inference_times'], 
                        label='Inference time', s=80, marker='o')
            
            # If we have more than 2 data points, fit a polynomial
            if len(self.timing_data['system_sizes']) > 2:
                # Try to fit a polynomial (degree 2)
                coeffs = np.polyfit(self.timing_data['system_sizes'], self.timing_data['inference_times'], 2)
                poly = np.poly1d(coeffs)
                
                # Create smooth curve for the fit
                x_fit = np.linspace(min(self.timing_data['system_sizes']), 
                                   max(self.timing_data['system_sizes']) * 1.5, 100)
                y_fit = poly(x_fit)
                
                plt.plot(x_fit, y_fit, 'r--', label=f'Polynomial fit: {poly}')
            
            plt.xlabel('Number of Orbitals', fontsize=14)
            plt.ylabel('Inference Time (seconds)', fontsize=14)
            plt.title('Inference Time vs. System Size', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'inference_time_vs_size.png'), dpi=300)
            plt.close()
        
        # Plot epoch times
        if len(self.timing_data['epoch_times']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.timing_data['epoch_times']) + 1), 
                    self.timing_data['epoch_times'], 'b-o')
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Time (seconds)', fontsize=14)
            plt.title('Training Time per Epoch', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'epoch_times.png'), dpi=300)
            plt.close()
            
        # Create a summary plot
        plt.figure(figsize=(12, 8))
        
        # Plot bar chart of total training and inference time
        plt.bar(['Training', 'Inference'], 
                [self.timing_data['total_training_time'], self.timing_data['total_inference_time']])
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.title('Total Training and Inference Time', fontsize=16)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_timing.png'), dpi=300)
        plt.close()

# Create global timing tracker
timing_tracker = TimingTracker()

class FeatureNormalizer:
    """Normalize node, edge, and global features."""
    
    def __init__(self, dataset):
        """Initialize the normalizer with a dataset."""
        # Extract all features
        x = torch.cat([data.x for data in dataset], dim=0)
        edge_attr = torch.cat([data.edge_attr for data in dataset], dim=0)
        global_feature = torch.cat([data.global_feature for data in dataset], dim=0)
        
        # Compute means and standard deviations
        self.node_mean = x.mean(dim=0)
        self.node_std = x.std(dim=0) + 1e-6  # Add small epsilon to avoid division by zero
        
        self.edge_mean = edge_attr.mean(dim=0)
        self.edge_std = edge_attr.std(dim=0) + 1e-6
        
        self.global_mean = global_feature.mean(dim=0)
        self.global_std = global_feature.std(dim=0) + 1e-6
        
        # For target normalization
        targets = torch.tensor([data.y for data in dataset])
        self.target_mean = targets.mean().item()
        self.target_std = targets.std().item() + 1e-6
        
    def transform(self, data):
        """Normalize the features of a data object."""
        # Normalize node features
        data.x = (data.x - self.node_mean) / self.node_std
        
        # Normalize edge features
        data.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
        
        # Normalize global features
        data.global_feature = (data.global_feature - self.global_mean) / self.global_std
        
        return data
    
    def normalize_target(self, target):
        """Normalize a target value."""
        return (target - self.target_mean) / self.target_std
    
    def denormalize_target(self, normalized_target):
        """Denormalize a normalized target value."""
        return normalized_target * self.target_std + self.target_mean
    
    def print_stats(self):
        """Print the normalization statistics."""
        print("Fitted normalizer:")
        print(f"  Node mean: {self.node_mean}")
        print(f"  Node std: {self.node_std}")
        print(f"  Edge mean: {self.edge_mean}")
        print(f"  Edge std: {self.edge_std}")
        print(f"  Target mean: {self.target_mean}")
        print(f"  Target std: {self.target_std}")
        print(f"  Global mean: {self.global_mean}")
        print(f"  Global std: {self.global_std}")


class DeltaMessagePassing(MessagePassing):
    """
    Custom message passing layer for Δ-ML model.
    Processes node features, edge features, and global features.
    
    Args:
        in_channels: Dimension of input features
        out_channels: Dimension of output features
        edge_dim: Dimension of edge features
        global_dim: Dimension of global features
    """
    def __init__(self, in_channels, out_channels, edge_dim, global_dim):
        """
        Initialize the message passing layer.
        
        Args:
            in_channels: Dimension of input features
            out_channels: Dimension of output features
            edge_dim: Dimension of edge features
            global_dim: Dimension of global features
        """
        super().__init__(aggr='add')
        self.lin_1 = nn.Linear(in_channels, out_channels)
        self.lin_2 = nn.Linear(out_channels + edge_dim, out_channels)
        self.lin_update = nn.Linear(out_channels + global_dim, out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr, global_feature, batch=None):
        # If batch is not provided, assume all nodes belong to a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Propagate with message and update
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, global_feature=global_feature, batch=batch)

    def message(self, x_j, edge_attr):
        # Process node and edge features
        x_j = self.lin_1(x_j)
        edge_features = torch.cat([x_j, edge_attr], dim=1)
        return self.lin_2(edge_features)

    def update(self, aggr_out, x, global_feature, batch):
        # Get the number of unique graphs in the batch
        if global_feature.dim() == 1:
            global_feature = global_feature.unsqueeze(0)
            
        # For each node, get the corresponding global feature based on batch index
        expanded_global = global_feature.index_select(0, batch)
        
        # Combine aggregated messages with global features
        features = torch.cat([aggr_out, expanded_global], dim=1)
        out = self.lin_update(features)
        out = self.layer_norm(out)
        return out


class AdvancedDeltaMLModel(torch.nn.Module):
    """
    Advanced Graph Neural Network for Δ-ML energy prediction.
    Incorporates multiple message-passing layers with residual connections
    and global feature integration.
    """
    
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim=64, n_layers=3, dropout=0.3):
        """
        Initialize the Δ-ML model.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            global_dim: Dimension of global features
            hidden_dim: Dimension of hidden layers
            n_layers: Number of message-passing layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        
        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(n_layers):
            in_channels = node_dim if i == 0 else hidden_dim
            self.mp_layers.append(
                DeltaMessagePassing(in_channels, hidden_dim, edge_dim, global_dim)
            )
        
        # Layer normalization after each message passing layer
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Readout MLP
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        global_feature = data.global_feature
        batch = data.batch
        
        # Apply message passing layers
        for i, mp_layer in enumerate(self.mp_layers):
            x = mp_layer(x, edge_index, edge_attr, global_feature, batch)
            x = self.dropout(x)
        
        # Global pooling (mean of node features)
        pooled = global_mean_pool(x, batch)
        
        # Combine pooled node features with global features
        expanded_global = global_feature
        if len(expanded_global.shape) < 2:
            expanded_global = expanded_global.unsqueeze(0)
        if pooled.size(0) != expanded_global.size(0):
            # Handle batched data - repeat global features for each graph in batch
            expanded_global = expanded_global.repeat(pooled.size(0), 1)
            
        out = torch.cat([pooled, expanded_global], dim=1)
        
        # Pass through readout MLP
        out = self.readout_mlp(out)
        
        return out.squeeze(-1)


def train_epoch(model, data_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    start_time = time.time()  # Start timing
    
    for data in data_loader:
        # Move data to device
        data = data.to(device)
        target = data.y
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Compute loss
        loss = criterion(out, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item() * data.num_graphs
    
    epoch_time = time.time() - start_time  # Calculate epoch time
    timing_tracker.add_epoch_time(epoch_time)  # Track epoch time
    
    # Return average loss
    return total_loss / len(data_loader.dataset)


def validate(model, data_loader, criterion, device, normalizer):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in data_loader:
            # Move data to device
            data = data.to(device)
            target = data.y
            
            # Forward pass
            out = model(data)
            
            # Compute loss
            loss = criterion(out, target)
            
            # Accumulate loss
            total_loss += loss.item() * data.num_graphs
            
            # Transform predictions and targets back to original scale
            preds = normalizer.denormalize_target(out.cpu())
            targets = normalizer.denormalize_target(target.cpu())
            
            # Accumulate results
            all_preds.append(preds)
            all_targets.append(targets)
    
    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    r2 = 1 - np.sum((all_preds - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
    
    # Return metrics and average loss
    metrics = {
        'loss': total_loss / len(data_loader.dataset),
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, all_preds, all_targets


def train_model(model, train_loader, val_loader, optimizer, criterion, device, scheduler=None, n_epochs=100, patience=10, output_dir=None):
    """Train the model."""
    # Initialize variables
    best_val_loss = float('inf')
    best_val_metrics = None
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'learning_rates': []
    }
    
    # Create normalizer for validation
    normalizer = None
    for data in val_loader.dataset:
        if hasattr(data, 'y'):
            normalizer = FeatureNormalizer(val_loader.dataset)
            break
    
    # Start timing for total training time
    start_time = time.time()
    
    # Train for n_epochs
    for epoch in range(n_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics, _, _ = validate(model, val_loader, criterion, device, normalizer)
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs} - "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_metrics['loss']:.6f}, "
              f"Val RMSE: {val_metrics['rmse']:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check if this is the best model so far
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_metrics = val_metrics
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save best model if output_dir is provided
            if output_dir is not None:
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Calculate total training time
    total_training_time = time.time() - start_time
    timing_tracker.set_total_training_time(total_training_time)
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, best_val_metrics


def evaluate_model(model, data_loader, criterion, device, normalizer):
    """Evaluate the model on a dataset."""
    # Start timing for inference
    start_time = time.time()
    
    # Validate
    metrics, all_preds, all_targets = validate(model, data_loader, criterion, device, normalizer)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    timing_tracker.set_total_inference_time(inference_time)
    print(f"Total inference time: {inference_time:.2f} seconds")
    
    # Calculate relative error
    rel_error = np.mean(np.abs(all_preds - all_targets) / (np.abs(all_targets) + 1e-10)) * 100
    metrics['rel_error'] = rel_error
    
    # Return the metrics and predictions in the expected format
    return metrics, {
        'predictions': all_preds.tolist(),
        'targets': all_targets.tolist()
    }


def plot_training_curves(history, output_dir):
    """
    Plot training loss curves and learning rate schedule.
    
    Args:
        history (dict): Dictionary containing training history
        output_dir (str): Directory to save the plots
    """
    plt.figure(figsize=(12, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    if 'val_rmse' in history:
        plt.plot(history['val_rmse'], label='Validation RMSE', color='green')
    if 'val_r2' in history:
        plt.plot(history['val_r2'], label='Validation R2', color='purple')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Metric')
    plt.title('Training and Validation Metrics')
    plt.yscale('log')  # Use log scale for loss
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot learning rate
    plt.subplot(2, 1, 2)
    plt.plot(history['learning_rates'], label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')  # Use log scale for learning rate
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()


def visualize_predictions(model, test_loader, normalizer, systems, config, save_dir="delta_ml_results"):
    """
    Visualize model predictions vs. true values.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        normalizer: Normalizer object
        systems: List of system names
        config: Configuration dictionary
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get device
    device = config.get('device', 'cpu')
    
    model.eval()
    all_preds = []
    all_targets = []
    
    # Get target normalization parameters as floats, not tensors
    target_mean = float(normalizer.target_mean)
    target_std = float(normalizer.target_std)
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch)
            
            # Make sure target has same shape as output for consistency
            target = batch.y.view(-1, 1)
            
            # Denormalize predictions and targets
            # Manual denormalization using the mean and std
            pred = out.cpu().numpy() * target_std + target_mean
            target = target.cpu().numpy() * target_std + target_mean
            
            all_preds.extend(pred.flatten())
            all_targets.extend(target.flatten())
    
    # Convert to numpy arrays for easier handling
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Make sure we don't have more predictions than system names
    num_samples = min(len(all_preds), len(systems))
    
    # Limit predictions/targets to the number of available system names
    all_preds = all_preds[:num_samples]
    all_targets = all_targets[:num_samples]
    
    # Handle case where we have more system names than predictions
    systems = systems[:num_samples]
    
    # Calculate metrics
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
    
    # Handle R² calculation for small sample sizes
    if len(all_preds) <= 1:
        r2 = float('nan')
    else:
        r2 = r2_score(all_targets, all_preds)
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'System': systems,
        'True_Delta_Ha': all_targets,
        'Pred_Delta_Ha': all_preds,
        'Error_Ha': np.abs(all_preds - all_targets),
        'True_Delta_mHa': all_targets * 1000,  # Convert to millihartree
        'Pred_Delta_mHa': all_preds * 1000,    # Convert to millihartree
        'Error_mHa': np.abs(all_preds - all_targets) * 1000  # Convert to millihartree
    })
    results_df.to_csv(f"{save_dir}/prediction_results.csv", index=False)
    
    # Plot predictions vs. true values
    plt.figure(figsize=(10, 8))
    if len(all_preds) == 1:
        # If only one prediction, plot as a single point
        plt.scatter(all_targets, all_preds, s=100, c='blue', marker='o')
        # Add system name annotation
        plt.annotate(systems[0], (all_targets[0], all_preds[0]), 
                     fontsize=12, xytext=(10, 5), textcoords='offset points')
        # Add identity line
        min_val = min(all_targets[0], all_preds[0]) * 0.9
        max_val = max(all_targets[0], all_preds[0]) * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    else:
        # If multiple predictions, plot as scatter plot with annotations
        plt.scatter(all_targets, all_preds, s=100, c='blue', marker='o')
        for i, sys in enumerate(systems):
            plt.annotate(sys, (all_targets[i], all_preds[i]), 
                         fontsize=12, xytext=(10, 5), textcoords='offset points')
        
        # Add identity line and trend line
        all_vals = np.concatenate([all_targets, all_preds])
        min_val = np.min(all_vals) * 1.1 if np.min(all_vals) < 0 else np.min(all_vals) * 0.9
        max_val = np.max(all_vals) * 0.9 if np.max(all_vals) < 0 else np.max(all_vals) * 1.1
        
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Add trend line if there are enough points
        if len(all_preds) > 2:
            z = np.polyfit(all_targets, all_preds, 1)
            p = np.poly1d(z)
            plt.plot(np.linspace(min_val, max_val, 100), p(np.linspace(min_val, max_val, 100)), 'r-', alpha=0.7)
    
    plt.xlabel('True ΔE (Ha)', fontsize=14)
    plt.ylabel('Predicted ΔE (Ha)', fontsize=14)
    
    # Add mHa values to title for clarity
    mae_mha = mae * 1000
    rmse_mha = rmse * 1000
    
    plt.title(f'Delta-ML Predictions vs. True Values\nMAE: {mae:.6f} Ha ({mae_mha:.2f} mHa), RMSE: {rmse:.6f} Ha ({rmse_mha:.2f} mHa), R²: {r2:.3f}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/predictions_vs_true.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error distribution if there are enough samples
    if len(all_preds) > 3:
        errors = all_preds - all_targets
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=min(10, len(errors)), alpha=0.7, color='blue')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.xlabel('Prediction Error (Ha)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of Prediction Errors', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Return metrics
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def save_results(model, history, best_val_metrics, test_metrics, test_predictions, output_dir):
    """
    Save model, history, and results.
    
    Args:
        model: Trained model
        history: Training history
        best_val_metrics: Best validation metrics
        test_metrics: Test metrics
        test_predictions: Test predictions
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    # Helper function to convert NumPy values to Python native types
    def convert_numpy_to_python(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, list):
            return [convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_python(value) for key, value in obj.items()}
        else:
            return obj
    
    # Save training history
    history_dict = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_rmse': history['val_rmse'],
        'val_mae': history['val_mae'],
        'val_r2': history['val_r2'],
        'learning_rates': history['learning_rates']
    }
    
    # Convert all values to JSON-serializable types
    history_dict = convert_numpy_to_python(history_dict)
    
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save best validation metrics (convert NumPy values)
    with open(os.path.join(output_dir, 'best_val_metrics.json'), 'w') as f:
        json.dump(convert_numpy_to_python(best_val_metrics), f, indent=2)
    
    # Save test metrics and predictions (convert NumPy values)
    results = {
        'metrics': test_metrics,
        'predictions': test_predictions['predictions'],
        'targets': test_predictions['targets'],
        'rmse_ha': float(test_metrics['rmse']),
        'mae_ha': float(test_metrics['mae']),
        'r2': float(test_metrics['r2']),
        'rel_error': float(test_metrics['rel_error']),
        'min_delta_ha': float(np.min(test_predictions['targets'])),
        'max_delta_ha': float(np.max(test_predictions['targets']))
    }
    
    # Convert all results to JSON-serializable types
    results = convert_numpy_to_python(results)
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    # Print test metrics
    print(f"Test RMSE: {test_metrics['rmse']:.6f} Ha ({test_metrics['rmse']*1000:.2f} mHa)")
    print(f"Test MAE: {test_metrics['mae']:.6f} Ha ({test_metrics['mae']*1000:.2f} mHa)")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test Relative Error: {test_metrics['rel_error']:.4f}")
    
    print(f"Results saved to {output_dir}")


def plot_predictions(predictions, targets, output_dir):
    """
    Plot predictions vs targets.
    
    Args:
        predictions: Predicted values
        targets: True values
        output_dir: Output directory
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy arrays if they're not already
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    
    # Plot predictions vs targets
    plt.scatter(targets, predictions, alpha=0.7)
    
    # Plot perfect prediction line
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
    
    # Add labels and title
    plt.xlabel('True Delta E (Ha)')
    plt.ylabel('Predicted Delta E (Ha)')
    plt.title(f'Predictions vs Targets\nRMSE: {rmse:.6f} Ha, MAE: {mae:.6f} Ha, R²: {r2:.4f}')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=300)
    plt.close()


def filter_dataset_by_orbital_size(dataset, min_orbitals=None, max_orbitals=None, target_orbitals=None):
    """
    Filter a DeltaMLDataset based on orbital size constraints.
    
    Args:
        dataset: The dataset to filter
        min_orbitals: Minimum number of orbitals (inclusive)
        max_orbitals: Maximum number of orbitals (inclusive)
        target_orbitals: Specific orbital count to target (exact match)
        
    Returns:
        Filtered dataset
    """
    filtered_pairs = []
    
    # Extract orbital sizes for each delta pair
    for pair in dataset.delta_pairs:
        system, low_dim, high_dim, low_file, high_file = pair
        
        try:
            # Load the high-dim file to get orbital count (assumption: both have same orbital count)
            processor = ImprovedDMRGProcessor(high_file, mi_threshold=dataset.mi_threshold)
            num_orbitals = processor.num_orbitals
            
            # Apply filtering
            include = True
            if target_orbitals is not None:
                include = (num_orbitals == target_orbitals)
            else:
                if min_orbitals is not None and num_orbitals < min_orbitals:
                    include = False
                if max_orbitals is not None and num_orbitals > max_orbitals:
                    include = False
            
            if include:
                filtered_pairs.append(pair)
                
        except Exception as e:
            print(f"Warning: Could not process {high_file}: {e}")
    
    # Create new filtered dataset
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset.delta_pairs = filtered_pairs
    
    return filtered_dataset


def main():
    """
    Main function for training and evaluating the model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate the Advanced Delta-ML model")
    
    # Data settings
    parser.add_argument('--train_dir', type=str, default='train',
                        help='Directory containing training data')
    parser.add_argument('--test_dir', type=str, default='test',
                        help='Directory containing test data')
    parser.add_argument('--mi_threshold', type=float, default=0.0,
                        help='Threshold for mutual information filtering')
    
    # System size filtering
    parser.add_argument('--min_orbitals', type=int, default=None,
                        help='Minimum number of orbitals to include')
    parser.add_argument('--max_orbitals', type=int, default=None,
                        help='Maximum number of orbitals to include')
    parser.add_argument('--system_size', type=int, default=None,
                        help='Target specific system size (number of orbitals)')
    
    # Model settings
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for model layers')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of message passing layers')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for L2 regularization')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='advanced_delta_ml_results',
                        help='Directory to save results')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        # Data settings
        'train_dir': args.train_dir,
        'test_dir': args.test_dir,
        'bond_dim_pairs': None,  # Auto-detect all available pairs
        'mi_threshold': args.mi_threshold,
        
        # System size filtering
        'min_orbitals': args.min_orbitals,
        'max_orbitals': args.max_orbitals,
        'system_size': args.system_size,
        
        # Model settings
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        
        # Training settings
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.max_epochs,
        'patience': args.early_stopping,
        'val_ratio': args.val_ratio,
        
        # Output settings
        'output_dir': args.output_dir,
        
        # Random seed
        'seed': args.seed
    }
    
    # Set random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create training dataset
    print("Creating Delta-ML dataset...")
    full_train_dataset = DeltaMLDataset(
        data_dir=config['train_dir'],
        bond_dim_pairs=config['bond_dim_pairs'],
        mi_threshold=config['mi_threshold'],
    )
    print(f"Initial training dataset: {len(full_train_dataset)} examples")
    
    # Apply system size filtering if requested
    if config['min_orbitals'] is not None or config['max_orbitals'] is not None or config['system_size'] is not None:
        print(f"Filtering dataset by orbital count...")
        print(f"  Constraints: min={config['min_orbitals']}, max={config['max_orbitals']}, target={config['system_size']}")
        full_train_dataset = filter_dataset_by_orbital_size(
            full_train_dataset,
            min_orbitals=config['min_orbitals'],
            max_orbitals=config['max_orbitals'],
            target_orbitals=config['system_size']
        )
        print(f"Filtered training dataset: {len(full_train_dataset)} examples")
    
    # Create test dataset
    print("Creating test Delta-ML dataset...")
    test_dataset = DeltaMLDataset(
        data_dir=config['test_dir'],
        bond_dim_pairs=config['bond_dim_pairs'],
        mi_threshold=config['mi_threshold'],
    )
    print(f"Initial test dataset: {len(test_dataset)} examples")
    
    # Apply system size filtering to test dataset if requested
    if config['min_orbitals'] is not None or config['max_orbitals'] is not None or config['system_size'] is not None:
        print(f"Filtering test dataset by orbital count...")
        test_dataset = filter_dataset_by_orbital_size(
            test_dataset,
            min_orbitals=config['min_orbitals'],
            max_orbitals=config['max_orbitals'],
            target_orbitals=config['system_size']
        )
        print(f"Filtered test dataset: {len(test_dataset)} examples")

    # Extract orbital counts from datasets
    train_orbital_counts = extract_orbital_counts(full_train_dataset)
    test_orbital_counts = extract_orbital_counts(test_dataset)
    
    # Track average orbital counts
    avg_train_orbitals = sum(train_orbital_counts) / len(train_orbital_counts) if train_orbital_counts else 0
    avg_test_orbitals = sum(test_orbital_counts) / len(test_orbital_counts) if test_orbital_counts else 0
    
    # Add system sizes to timing tracker
    timing_tracker.add_system_size(avg_train_orbitals)
    
    print(f"Average orbital count in training set: {avg_train_orbitals:.2f}")
    print(f"Average orbital count in test set: {avg_test_orbitals:.2f}")

    # Create feature normalizer and normalize the full training dataset
    normalizer = FeatureNormalizer(full_train_dataset)
    normalizer.print_stats()
    
    # Normalize the full train dataset
    normalized_full_train = []
    for i in range(len(full_train_dataset)):
        data = full_train_dataset[i]
        normalized_full_train.append(normalizer.transform(data))
    
    # Normalize the test dataset
    normalized_test = []
    for i in range(len(test_dataset)):
        data = test_dataset[i]
        normalized_test.append(normalizer.transform(data))
    
    # Split training dataset into training and validation sets
    val_size = max(1, int(len(normalized_full_train) * config['val_ratio']))
    train_size = len(normalized_full_train) - val_size
    
    # Instead of using random_split directly on the dataset, we'll split the normalized data
    indices = list(range(len(normalized_full_train)))
    # Use a fixed seed for reproducibility
    np.random.seed(config['seed'])
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create our own train and validation datasets
    train_normalized = [normalized_full_train[i] for i in train_indices]
    val_normalized = [normalized_full_train[i] for i in val_indices]
    
    print(f"Training split: {len(train_normalized)} examples")
    print(f"Validation split: {len(val_normalized)} examples")
    
    # Get feature dimensions
    sample = train_normalized[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    global_dim = sample.global_feature.shape[0]
    
    print(f"Feature dimensions: node={node_dim}, edge={edge_dim}, global={global_dim}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_normalized,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_normalized,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        normalized_test,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedDeltaMLModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=config['hidden_dim'],
        n_layers=config['num_layers']
    ).to(device)
    
    # Print model architecture
    print("Model architecture:")
    print(model)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=config['patience']//3,
        min_lr=1e-6,
        verbose=True
    )
    
    criterion = nn.MSELoss()
    
    # Train model
    print("Training model...")
    history, best_val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        n_epochs=config['epochs'],
        patience=config['patience'],
        output_dir=config['output_dir']
    )
    
    # Add training time to timing tracker
    training_time = timing_tracker.timing_data['total_training_time']
    timing_tracker.add_training_time(training_time)
    
    # Plot training curves - fixed to pass just the directory
    plot_training_curves(history, config['output_dir'])
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics, test_predictions = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        normalizer=normalizer
    )
    print(f"Test RMSE: {test_metrics['rmse']:.6f} Ha ({test_metrics['rmse']*1000:.2f} mHa)")
    
    # Add inference time to timing tracker
    inference_time = timing_tracker.timing_data['total_inference_time']
    timing_tracker.add_inference_time(inference_time)
    
    # Save results
    save_results(
        model=model,
        history=history,
        best_val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
        output_dir=config['output_dir']
    )
    
    # Plot predictions vs targets
    plot_predictions(
        test_predictions['predictions'], 
        test_predictions['targets'], 
        config['output_dir']
    )
    
    # Save timing data
    timing_tracker.save_timing_data(config['output_dir'])
    
    print(f"\nAll results saved to {config['output_dir']}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f} Ha ({test_metrics['rmse']*1000:.2f} mHa)")
    print(f"Test MAE: {test_metrics['mae']:.6f} Ha ({test_metrics['mae']*1000:.2f} mHa)")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test Relative Error: {test_metrics['rel_error']:.2f}%")
    
    # Print timing summary
    print("\nTiming Summary:")
    print(f"Average orbital count: {avg_train_orbitals:.2f}")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Total inference time: {inference_time:.2f} seconds")


# Add a function to extract the number of orbitals from the datasets
def extract_orbital_counts(dataset):
    """Extract the number of orbitals for each system in the dataset."""
    orbital_counts = []
    
    for data in dataset:
        # Number of nodes represents number of orbitals
        num_orbitals = data.x.shape[0]
        orbital_counts.append(num_orbitals)
    
    return orbital_counts


if __name__ == "__main__":
    main() 