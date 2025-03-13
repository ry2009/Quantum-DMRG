import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import time
import json

from simple_mpgnn import SimpleMPGNN, FeatureNormalizer, TargetNormalizer, train_epoch, evaluate, plot_predictions
from dmrg_data_processor import DMRGDataset

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading parameters
    train_dir = 'train'
    test_dir = 'test'
    batch_size = 4
    val_ratio = 0.15
    
    # DMRG dataset parameters
    bond_dims = [512]  # Default bond dimension
    system_type = 'pah'  # Default system type
    max_orbitals = 100  # Default max orbitals
    mi_threshold = 0.01  # Default mutual information threshold
    predict_total_energy = True  # Predict total energy instead of correlation energy
    
    # Model hyperparameters
    hidden_dim = 64
    learning_rate = 0.001
    weight_decay = 5e-4
    num_epochs = 50
    
    # Create output directory
    results_dir = 'results/mpgnn_simple'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading training data from {train_dir}...")
    train_dataset = DMRGDataset(
        root_dir=train_dir,
        bond_dims=bond_dims,
        system_type=system_type,
        max_orbitals=max_orbitals,
        mi_threshold=mi_threshold,
        predict_total_energy=predict_total_energy
    )
    print(f"Loaded {len(train_dataset)} training samples")
    
    print(f"Loading test data from {test_dir}...")
    test_dataset = DMRGDataset(
        root_dir=test_dir,
        bond_dims=bond_dims,
        system_type=system_type,
        max_orbitals=max_orbitals,
        mi_threshold=mi_threshold,
        predict_total_energy=predict_total_energy
    )
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Split train into train/val
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    split = int(np.floor(val_ratio * num_train))
    train_idx, val_idx = indices[split:], indices[:split]
    
    # Create data loaders
    train_loader = DataLoader([train_dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([train_dataset[i] for i in val_idx], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split: {len(train_idx)} train, {len(val_idx)} validation, {len(test_dataset)} test")
    
    # Get feature dimensions from the dataset
    sample_data = train_dataset[0]
    node_dim = sample_data.x.shape[1]
    edge_dim = sample_data.edge_attr.shape[1] if hasattr(sample_data, 'edge_attr') and sample_data.edge_attr is not None else None
    
    print(f"Node feature dimension: {node_dim}")
    print(f"Edge feature dimension: {edge_dim}")
    
    # Initialize feature and target normalizers
    feature_normalizer = FeatureNormalizer()
    target_normalizer = TargetNormalizer()
    
    # Fit normalizers on training data
    print("Fitting normalizers...")
    feature_normalizer.fit([train_dataset[i] for i in train_idx])
    
    train_targets = torch.cat([train_dataset[i].y for i in train_idx], dim=0)
    target_normalizer.fit(train_targets)
    
    print(f"Target normalization - Mean: {target_normalizer.mean:.6f}, Std: {target_normalizer.std:.6f}")
    
    # Initialize model
    model = SimpleMPGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    print(model)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    val_rmses = []
    val_maes = []
    val_r2s = []
    
    best_val_rmse = float('inf')
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, feature_normalizer, target_normalizer, device)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss, val_preds, val_targets, val_rmse, val_mae, val_r2 = evaluate(
            model, val_loader, criterion, feature_normalizer, target_normalizer, device
        )
        
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val RMSE: {val_rmse:.6f} | "
              f"Val MAE: {val_mae:.6f} | "
              f"Val R²: {val_r2:.6f}")
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            print(f"New best model with validation RMSE: {val_rmse:.6f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_loss, test_preds, test_targets, test_rmse, test_mae, test_r2 = evaluate(
        model, test_loader, criterion, feature_normalizer, target_normalizer, device
    )
    
    print("\nTest Set Evaluation:")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test R²: {test_r2:.6f}")
    
    # Save model and results
    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))
    torch.save({
        'feature_normalizer': {
            'node_mean': feature_normalizer.node_mean,
            'node_std': feature_normalizer.node_std,
            'edge_mean': feature_normalizer.edge_mean,
            'edge_std': feature_normalizer.edge_std
        },
        'target_normalizer': {
            'mean': target_normalizer.mean,
            'std': target_normalizer.std
        }
    }, os.path.join(results_dir, 'normalizers.pt'))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # RMSE curve
    plt.subplot(1, 3, 2)
    plt.plot(val_rmses)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE')
    plt.grid(True, alpha=0.3)
    
    # R² curve
    plt.subplot(1, 3, 3)
    plt.plot(val_r2s)
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('Validation R²')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300)
    
    # Plot predictions vs. true values
    plot_predictions(
        test_targets.view(-1),
        test_preds.view(-1),
        save_path=os.path.join(results_dir, 'test_predictions.png')
    )
    
    # Save test results
    test_results = {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'training_time': training_time,
        'num_parameters': total_params,
        'predictions': test_preds.view(-1).tolist(),
        'true_values': test_targets.view(-1).tolist()
    }
    
    with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    print(f"Results saved to {results_dir}")
    
    # Return the final metrics for comparison
    return {
        'model': 'SimpleMPGNN',
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }

if __name__ == '__main__':
    main() 