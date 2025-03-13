import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm
import json

from bayesian_models import BayesianDMRGNet
from spin_data_processor import SpinDataset

class TargetNormalizer:
    """Normalizes target values for better training."""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, targets):
        """Compute mean and std of targets."""
        self.mean = targets.mean()
        self.std = targets.std()
        if self.std < 1e-8:
            self.std = 1.0
            
    def transform(self, targets):
        """Normalize targets."""
        return (targets - self.mean) / self.std
    
    def inverse_transform(self, normalized_targets):
        """Convert normalized targets back to original scale."""
        return normalized_targets * self.std + self.mean

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_epoch(model, loader, optimizer, device, target_normalizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(data)
        
        # Normalize targets
        y = target_normalizer.transform(data.y)
        
        # Calculate loss
        loss = nn.MSELoss()(pred, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, target_normalizer, num_samples=10):
    """Evaluate model with uncertainty estimation."""
    model.eval()
    mse_loss = 0
    predictions = []
    targets = []
    uncertainties = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Get prediction with uncertainty
            mean, std, _ = model.predict_with_uncertainty(data, num_samples)
            
            # Transform back to original scale
            mean = target_normalizer.inverse_transform(mean)
            std = std * target_normalizer.std  # Scale uncertainty
            
            # Calculate MSE
            mse = nn.MSELoss()(mean, data.y)
            mse_loss += mse.item() * data.num_graphs
            
            # Store predictions and targets
            predictions.append(mean.cpu())
            targets.append(data.y.cpu())
            uncertainties.append(std.cpu())
    
    # Concatenate results
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    uncertainties = torch.cat(uncertainties, dim=0)
    
    # Calculate metrics
    mse = mse_loss / len(loader.dataset)
    rmse = np.sqrt(mse)
    mae = nn.L1Loss()(predictions, targets).item()
    
    # Calculate calibration metrics
    # For a well-calibrated model, about 68% of true values should fall within 1 std dev
    within_1std = ((targets >= predictions - uncertainties) & 
                  (targets <= predictions + uncertainties)).float().mean().item()
    
    # About 95% should fall within 2 std devs
    within_2std = ((targets >= predictions - 2*uncertainties) & 
                  (targets <= predictions + 2*uncertainties)).float().mean().item()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'within_1std': within_1std,
        'within_2std': within_2std,
        'mean_uncertainty': uncertainties.mean().item()
    }

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{args.system_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load dataset
    print(f"Loading {args.system_type} dataset...")
    dataset = SpinDataset(
        root_dir=args.data_dir,
        system_type=args.system_type,
        dim=args.dim,
        size=args.size
    )
    
    # Split dataset
    n_samples = len(dataset)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = BayesianDMRGNet(
        node_features=dataset[0].x.size(1),
        edge_features=dataset[0].edge_attr.size(1),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_samples=args.num_samples
    ).to(device)
    
    # Fit feature normalizer
    model.fit_normalizer([data for data in train_dataset])
    
    # Initialize target normalizer
    target_normalizer = TargetNormalizer()
    targets = torch.cat([data.y for data in train_dataset])
    target_normalizer.fit(targets)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    train_losses = []
    val_metrics = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, target_normalizer)
        train_losses.append(train_loss)
        
        # Evaluate
        val_metric = evaluate(model, val_loader, device, target_normalizer, args.num_samples)
        val_metrics.append(val_metric)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val RMSE: {val_metric['rmse']:.6f} | "
              f"Val Uncertainty: {val_metric['mean_uncertainty']:.6f} | "
              f"Val Calibration (1σ): {val_metric['within_1std']:.2f}")
        
        # Update learning rate
        scheduler.step(val_metric['rmse'])
        
        # Check early stopping
        early_stopping(val_metric['rmse'])
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, device, target_normalizer, args.num_samples)
    print("\nTest set evaluation:")
    print(f"RMSE: {test_metrics['rmse']:.6f}")
    print(f"MAE: {test_metrics['mae']:.6f}")
    print(f"Mean uncertainty: {test_metrics['mean_uncertainty']:.6f}")
    print(f"Calibration (1σ): {test_metrics['within_1std']:.2f}")
    print(f"Calibration (2σ): {test_metrics['within_2std']:.2f}")
    
    # Save metrics
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # RMSE curve
    plt.subplot(1, 3, 2)
    plt.plot([m['rmse'] for m in val_metrics], label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE')
    plt.legend()
    
    # Uncertainty curve
    plt.subplot(1, 3, 3)
    plt.plot([m['mean_uncertainty'] for m in val_metrics], label='Val Uncertainty')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Uncertainty')
    plt.title('Validation Uncertainty')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Plot predictions vs targets with uncertainty
    plt.figure(figsize=(10, 6))
    
    # Get predictions on test set
    all_preds = []
    all_targets = []
    all_uncertainties = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            mean, std, _ = model.predict_with_uncertainty(data, args.num_samples)
            mean = target_normalizer.inverse_transform(mean)
            std = std * target_normalizer.std
            
            all_preds.append(mean.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
            all_uncertainties.append(std.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_uncertainties = np.concatenate(all_uncertainties)
    
    # Plot predictions vs targets
    plt.errorbar(all_targets.flatten(), all_preds.flatten(), 
                 yerr=all_uncertainties.flatten(), fmt='o', alpha=0.5)
    
    # Add diagonal line
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title('Predictions vs Targets with Uncertainty')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bayesian GNN for quantum spin systems")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/spin_systems',
                        help='Directory to store/load dataset')
    parser.add_argument('--system_type', type=str, default='heisenberg',
                        choices=['heisenberg', 'ising', 'hubbard'],
                        help='Type of quantum spin system')
    parser.add_argument('--dim', type=int, default=1, choices=[1, 2],
                        help='Dimension of the system (1D or 2D)')
    parser.add_argument('--size', type=int, default=10,
                        help='Size of the system (number of sites)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='bayesian_gnn',
                        help='Type of model to train')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of MC samples for uncertainty estimation')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    main(args) 