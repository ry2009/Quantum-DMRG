import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import json
import pandas as pd
from tqdm import tqdm

from analyze_dmrg_data import DMRGFileParser

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class FeatureNormalizer:
    """Normalize node features, edge features, and target values."""
    
    def __init__(self):
        self.node_mean = None
        self.node_std = None
        self.edge_mean = None
        self.edge_std = None
        self.target_mean = None
        self.target_std = None
        
    def fit(self, data_list):
        """Compute normalization parameters from a list of Data objects."""
        # Node features
        all_node_features = torch.cat([data.x for data in data_list], dim=0)
        self.node_mean = torch.mean(all_node_features, dim=0)
        self.node_std = torch.std(all_node_features, dim=0)
        self.node_std[self.node_std < 1e-6] = 1.0  # Prevent division by zero
        
        # Edge features (if available)
        if data_list[0].edge_attr is not None:
            all_edge_features = torch.cat([data.edge_attr for data in data_list], dim=0)
            self.edge_mean = torch.mean(all_edge_features, dim=0)
            self.edge_std = torch.std(all_edge_features, dim=0)
            self.edge_std[self.edge_std < 1e-6] = 1.0
        
        # Target values
        all_targets = torch.cat([data.y for data in data_list], dim=0)
        self.target_mean = torch.mean(all_targets)
        self.target_std = torch.std(all_targets)
        if self.target_std < 1e-6:
            self.target_std = 1.0
    
    def transform(self, data):
        """Normalize the features of a single Data object."""
        data.x = (data.x - self.node_mean) / self.node_std
        
        if data.edge_attr is not None and self.edge_mean is not None:
            data.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
            
        data.y = (data.y - self.target_mean) / self.target_std
        
        return data
    
    def inverse_transform_target(self, normalized_target):
        """Convert normalized target back to original scale."""
        return normalized_target * self.target_std + self.target_mean


class DeltaMessagePassing(MessagePassing):
    """Message passing layer for Δ-ML orbital graph."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(DeltaMessagePassing, self).__init__(aggr='add')  # "add" aggregation
        
        # Message function (message from source to target, including edge features)
        self.message_nn = nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function (update node features with aggregated messages)
        self.update_nn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        # x: node features [num_nodes, node_dim]
        # edge_index: graph connectivity [2, num_edges]
        # edge_attr: edge features [num_edges, edge_dim]
        
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features
        # x_j: source node features
        # edge_attr: edge features
        
        # Concatenate source node, target node, and edge features
        message_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Apply message neural network
        return self.message_nn(message_input)
    
    def update(self, aggr_out, x):
        # aggr_out: aggregated messages from neighbors
        # x: original node features
        
        # Concatenate original features with aggregated messages
        update_input = torch.cat([x, aggr_out], dim=1)
        
        # Apply update neural network
        return self.update_nn(update_input)


class DeltaMLModel(nn.Module):
    """
    Graph Neural Network for Δ-ML energy prediction.
    
    This model takes orbital graphs with node, edge, and global features
    and predicts the energy difference between low and high-level calculations.
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, global_dim=1):
        super(DeltaMLModel, self).__init__()
        
        # Input projections
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        self.global_embedding = nn.Linear(global_dim, hidden_dim)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mp_layers.append(DeltaMessagePassing(hidden_dim, hidden_dim, hidden_dim))
        
        # Readout MLP
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Node + global features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Predict single value (delta energy)
        )
        
    def forward(self, data):
        # Get data attributes
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Get global features (truncation error) if available
        global_features = data.global_feature if hasattr(data, 'global_feature') else None
        
        # Initial projections
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Process global features
        if global_features is not None:
            global_vec = self.global_embedding(global_features)
        else:
            # Create a dummy global feature
            batch_size = batch.max().item() + 1
            global_vec = torch.zeros(batch_size, self.global_embedding.weight.shape[1], 
                                    device=x.device)
        
        # Message passing
        for mp_layer in self.mp_layers:
            x = x + mp_layer(x, edge_index, edge_attr)  # Residual connection
        
        # Readout: Pool node features to graph level
        pooled = global_mean_pool(x, batch)
        
        # Repeat global vector for each graph in the batch
        repeated_global = global_vec
        
        # Concatenate node and global representations
        final_repr = torch.cat([pooled, repeated_global], dim=1)
        
        # Final prediction
        delta_energy = self.readout_mlp(final_repr)
        
        return delta_energy


def prepare_delta_ml_dataset(data_path, low_bond_dim, high_bond_dim, mi_threshold=0.004):
    """
    Prepare a dataset for Δ-ML by finding pairs of low and high bond dimension calculations.
    
    Args:
        data_path (str): Path to directory containing DMRG data
        low_bond_dim (int): Bond dimension for E_low
        high_bond_dim (int): Bond dimension for E_high
        mi_threshold (float): Threshold for mutual information
        
    Returns:
        list: PyTorch Geometric Data objects for training
    """
    dataset = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(data_path):
        # Find the low and high bond dimension files for the same system
        low_file = None
        high_file = None
        
        for file in files:
            if file.endswith('_m'):
                bond_dim = int(file.split('_')[0])
                if bond_dim == low_bond_dim:
                    low_file = os.path.join(root, file)
                elif bond_dim == high_bond_dim:
                    high_file = os.path.join(root, file)
        
        # If we found both files, create a data point
        if low_file and high_file:
            try:
                # Parse both files
                low_parser = DMRGFileParser(low_file, mi_threshold)
                high_parser = DMRGFileParser(high_file, mi_threshold)
                
                # Calculate delta energy (target value)
                delta_energy = high_parser.dmrg_energy - low_parser.dmrg_energy
                
                # Convert the low bond dimension data to PyG format
                data = low_parser.to_pyg_data()
                
                # Override the target value with delta energy
                data.y = torch.tensor([[delta_energy]], dtype=torch.float)
                
                dataset.append(data)
                
                print(f"Added data point: {os.path.basename(root)}, "
                      f"Delta E = {delta_energy:.8f} Ha")
                
            except Exception as e:
                print(f"Error processing {root}: {e}")
    
    return dataset


def train_delta_ml_model(train_loader, val_loader, model, optimizer, criterion, 
                       normalizer, device, num_epochs=100, patience=10):
    """
    Train the Δ-ML model.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: DeltaMLModel instance
        optimizer: PyTorch optimizer
        criterion: Loss function
        normalizer: FeatureNormalizer instance
        device: PyTorch device
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        
    Returns:
        dict: Training history
    """
    model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data)
            loss = criterion(out, data.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.num_graphs
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
                
                # De-normalize predictions and targets for metrics
                pred = normalizer.inverse_transform_target(out).cpu()
                target = normalizer.inverse_transform_target(data.y).cpu()
                
                val_preds.append(pred)
                val_targets.append(target)
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Calculate metrics
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        
        val_rmse = torch.sqrt(torch.mean((val_preds - val_targets) ** 2)).item()
        val_mae = torch.mean(torch.abs(val_preds - val_targets)).item()
        
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Val RMSE: {val_rmse:.6f} Ha, "
              f"Val MAE: {val_mae:.6f} Ha")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_delta_ml_model(model, test_loader, normalizer, device):
    """
    Evaluate the trained Δ-ML model on test data.
    
    Args:
        model: Trained DeltaMLModel
        test_loader: DataLoader for test data
        normalizer: FeatureNormalizer
        device: PyTorch device
        
    Returns:
        dict: Test metrics and predictions
    """
    model.eval()
    test_preds = []
    test_targets = []
    test_systems = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            out = model(data)
            
            # De-normalize predictions and targets
            pred = normalizer.inverse_transform_target(out).cpu()
            target = normalizer.inverse_transform_target(data.y).cpu()
            
            test_preds.append(pred)
            test_targets.append(target)
            
            # Store system names if available
            if hasattr(data, 'system_name'):
                test_systems.extend(data.system_name)
            else:
                test_systems.extend([f"System_{i}_{j}" for j in range(data.num_graphs)])
    
    # Convert to tensors
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Calculate metrics
    test_mse = torch.mean((test_preds - test_targets) ** 2).item()
    test_rmse = np.sqrt(test_mse)
    test_mae = torch.mean(torch.abs(test_preds - test_targets)).item()
    
    # Calculate relative error (percentage)
    rel_errors = 100 * torch.abs(test_preds - test_targets) / torch.abs(test_targets)
    mean_rel_error = torch.mean(rel_errors).item()
    
    # Create results dictionary - ensure everything is a list
    predictions = test_preds.squeeze().tolist()
    targets = test_targets.squeeze().tolist()
    
    # Handle single value case
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(targets, list):
        targets = [targets]
    
    results = {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'mean_rel_error': mean_rel_error,
        'predictions': predictions,
        'targets': targets,
        'systems': test_systems
    }
    
    # Print results
    print("\nTest Results:")
    print(f"  RMSE: {test_rmse:.8f} Ha")
    print(f"  MAE: {test_mae:.8f} Ha")
    print(f"  Mean Relative Error: {mean_rel_error:.4f}%")
    
    return results


def plot_delta_ml_results(history, results, output_dir="delta_ml_results"):
    """
    Plot training curves and prediction results.
    
    Args:
        history (dict): Training history
        results (dict): Test results
        output_dir (str): Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_rmse'], label='RMSE')
    plt.plot(history['val_mae'], label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error (Ha)')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Check if we have multiple test points or just one
    if isinstance(results['targets'], list) and len(results['targets']) > 1:
        # Plot predictions vs targets
        plt.figure(figsize=(8, 8))
        plt.scatter(results['targets'], results['predictions'], alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(min(results['targets']), min(results['predictions']))
        max_val = max(max(results['targets']), max(results['predictions']))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        plt.xlabel('True ΔE (Ha)')
        plt.ylabel('Predicted ΔE (Ha)')
        plt.title('Predicted vs True Energy Difference')
        
        # Add metrics textbox
        plt.text(0.05, 0.95, 
                f"RMSE: {results['test_rmse']:.8f} Ha\n"
                f"MAE: {results['test_mae']:.8f} Ha\n"
                f"Rel. Error: {results['mean_rel_error']:.4f}%",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictions_vs_true.png'))
        plt.close()
        
        # Save results to CSV
        df = pd.DataFrame({
            'System': results['systems'],
            'True_Delta': results['targets'],
            'Predicted_Delta': results['predictions'],
            'Absolute_Error': np.abs(np.array(results['predictions']) - np.array(results['targets'])),
            'Relative_Error': np.abs(np.array(results['predictions']) - np.array(results['targets'])) / 
                            np.abs(np.array(results['targets'])) * 100
        })
        df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
    else:
        # For a single test point, just save text summary
        with open(os.path.join(output_dir, 'single_prediction_result.txt'), 'w') as f:
            if isinstance(results['targets'], list):
                true_val = results['targets'][0]
                pred_val = results['predictions'][0]
                system = results['systems'][0]
            else:
                true_val = results['targets']
                pred_val = results['predictions']
                system = results['systems']
                
            f.write(f"System: {system}\n")
            f.write(f"True delta energy: {true_val:.8f} Ha\n")
            f.write(f"Predicted delta energy: {pred_val:.8f} Ha\n")
            f.write(f"Absolute error: {abs(pred_val - true_val):.8f} Ha\n")
            f.write(f"Relative error: {abs(pred_val - true_val) / abs(true_val) * 100:.4f}%\n")
    
    # Save all results as JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'test_rmse': results['test_rmse'],
            'test_mae': results['test_mae'],
            'mean_rel_error': results['mean_rel_error']
        }, f, indent=4)


def main():
    # Configuration
    LOW_BOND_DIM = 256
    HIGH_BOND_DIM = 1024
    MI_THRESHOLD = 0.004
    BATCH_SIZE = 8
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    OUTPUT_DIR = "delta_ml_results"
    VAL_RATIO = 0.2
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataset
    print(f"\nPreparing Δ-ML dataset (Low M = {LOW_BOND_DIM}, High M = {HIGH_BOND_DIM})...")
    
    # First prepare data from both train and test directories
    train_data = prepare_delta_ml_dataset("train", LOW_BOND_DIM, HIGH_BOND_DIM, MI_THRESHOLD)
    test_data = prepare_delta_ml_dataset("test", LOW_BOND_DIM, HIGH_BOND_DIM, MI_THRESHOLD)
    
    # Combine all data
    all_data = train_data + test_data
    print(f"Total data points: {len(all_data)}")
    
    if len(all_data) < 3:
        print("Insufficient data for training. Trying with different bond dimensions...")
        # Try with different bond dimensions
        LOW_BOND_DIM = 128
        HIGH_BOND_DIM = 512
        train_data = prepare_delta_ml_dataset("train", LOW_BOND_DIM, HIGH_BOND_DIM, MI_THRESHOLD)
        test_data = prepare_delta_ml_dataset("test", LOW_BOND_DIM, HIGH_BOND_DIM, MI_THRESHOLD)
        all_data = train_data + test_data
        print(f"With M={LOW_BOND_DIM} to M={HIGH_BOND_DIM}: {len(all_data)} data points")
    
    # Ensure we have enough data
    if len(all_data) < 3:
        print("Not enough data available for training. Exiting.")
        return
    
    # Split into train, validation, and test (use 60/20/20 split)
    np.random.shuffle(all_data)
    test_size = max(1, int(len(all_data) * 0.2))
    val_size = max(1, int(len(all_data) * 0.2))
    train_size = len(all_data) - test_size - val_size
    
    train_dataset = all_data[:train_size]
    val_dataset = all_data[train_size:train_size+val_size]
    test_dataset = all_data[train_size+val_size:]
    
    print(f"Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} validation, "
          f"{len(test_dataset)} test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(val_dataset)), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=min(BATCH_SIZE, len(test_dataset)), shuffle=False)
    
    # Create normalizer and fit on training data
    normalizer = FeatureNormalizer()
    normalizer.fit(train_dataset)
    
    # Apply normalization to all datasets
    for loader in [train_loader, val_loader, test_loader]:
        for data in loader.dataset:
            normalizer.transform(data)
    
    # Get input dimensions from the first data point
    sample_data = train_dataset[0]
    node_dim = sample_data.x.shape[1]
    edge_dim = sample_data.edge_attr.shape[1]
    global_dim = sample_data.global_feature.shape[1] if hasattr(sample_data, 'global_feature') else 1
    
    # Create model
    model = DeltaMLModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        global_dim=global_dim
    )
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Train model
    print("\nTraining model...")
    history = train_delta_ml_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        normalizer=normalizer,
        device=device,
        num_epochs=NUM_EPOCHS
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    results = evaluate_delta_ml_model(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device
    )
    
    # Plot results
    plot_delta_ml_results(history, results, OUTPUT_DIR)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalizer': {
            'node_mean': normalizer.node_mean,
            'node_std': normalizer.node_std,
            'edge_mean': normalizer.edge_mean,
            'edge_std': normalizer.edge_std,
            'target_mean': normalizer.target_mean,
            'target_std': normalizer.target_std
        },
        'model_config': {
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'global_dim': global_dim
        }
    }, os.path.join(OUTPUT_DIR, 'delta_ml_model.pt'))
    
    print(f"\nResults and model saved to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    main() 