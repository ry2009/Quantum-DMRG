import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
from tqdm import tqdm

from models import DMRGNet
from dmrg_data_processor import DMRGDataset

class TargetNormalizer:
    """Normalizes target values for better training."""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, targets):
        """Compute statistics from targets"""
        self.mean = targets.mean()
        self.std = targets.std()
        if self.std < 1e-8:
            self.std = 1.0
            
    def transform(self, targets):
        """Normalize targets"""
        return (targets - self.mean) / self.std
    
    def inverse_transform(self, normalized_targets):
        """Convert back to original scale"""
        return normalized_targets * self.std + self.mean

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0):
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
        return self.early_stop

def train_epoch(model, train_loader, optimizer, device, target_normalizer, scheduler=None, clip_grad=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(data)
        
        # Normalize targets
        y = target_normalizer.transform(data.y)
        
        # Compute loss
        loss = nn.functional.mse_loss(pred, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    # Step scheduler if provided
    if scheduler is not None:
        scheduler.step()
        
    return total_loss / len(train_loader.dataset)

def evaluate(model, loader, device, target_normalizer):
    """Evaluate the model"""
    model.eval()
    mse = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            
            # Convert back to original scale
            pred = target_normalizer.inverse_transform(pred)
            
            # Compute MSE
            mse += ((pred - data.y) ** 2).sum().item()
    
    # Compute RMSE
    rmse = np.sqrt(mse / len(loader.dataset))
    return rmse

def plot_training_curves(train_losses, val_rmses, output_dir):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    # Plot validation RMSE
    plt.subplot(1, 2, 2)
    plt.plot(val_rmses, 'r-', label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (Ha)')
    plt.title('Validation RMSE Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load dataset
    print("Loading DMRG dataset...")
    dataset = DMRGDataset(
        root_dir=args.data_dir,
        bond_dims=[args.bond_dim],
        system_type=args.system_type,
        max_orbitals=args.max_orbitals,
        mi_threshold=args.mi_threshold,
        predict_total_energy=args.predict_total_energy  # Use total energy as target
    )
    
    # Get some basic statistics
    if args.predict_total_energy:
        energies = torch.stack([data.y for data in dataset])
        min_energy = energies.min().item()
        max_energy = energies.max().item()
        mean_energy = energies.mean().item()
        std_energy = energies.std().item()
        print(f"Energy statistics: Min={min_energy:.2f}, Max={max_energy:.2f}, Mean={mean_energy:.2f}, Std={std_energy:.2f}")
    
    # Split dataset
    dataset_size = len(dataset)
    if args.test_split > 0:
        test_size = int(dataset_size * args.test_split)
        train_val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [dataset_size - test_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        train_val_dataset = dataset
        test_dataset = None
    
    # Split train/val
    train_val_size = len(train_val_dataset)
    val_size = int(train_val_size * args.val_split)
    train_size = train_val_size - val_size
    
    # Create splits with fixed random seed for reproducibility
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize target normalizer
    target_normalizer = TargetNormalizer()
    targets = torch.cat([dataset[idx].y for idx in train_dataset.indices])
    target_normalizer.fit(targets)
    
    # Create model
    model = DMRGNet(
        node_features=2,  # occupation and entropy
        edge_features=1,  # mutual information
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        heads=args.attention_heads,
        dropout=args.dropout,
        pooling=args.pooling
    ).to(device)
    
    # Fit feature normalizer
    train_data_list = [dataset[idx] for idx in train_dataset.indices]
    model.fit_normalizer(train_data_list)
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
    else:
        scheduler = None
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    train_losses = []
    val_rmses = []
    best_val_rmse = float('inf')
    best_model_state = None
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            target_normalizer,
            scheduler if args.lr_scheduler_type == 'epoch' else None,
            args.grad_clip
        )
        train_losses.append(train_loss)
        
        # Validate
        val_rmse = evaluate(model, val_loader, device, target_normalizer)
        val_rmses.append(val_rmse)
        
        # Step scheduler if using validation metric
        if scheduler is not None and args.lr_scheduler_type == 'metric':
            scheduler.step(val_rmse)
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()
        
        # Print progress
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.6f}")
        
        # Check early stopping
        if early_stopping(val_rmse):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set if available
    if test_dataset is not None:
        test_rmse = evaluate(model, test_loader, device, target_normalizer)
        print(f"\nTest set evaluation:")
        print(f"RMSE: {test_rmse:.6f}")
        
        # Calculate MAE on test set
        model.eval()
        mae = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = model(data)
                pred = target_normalizer.inverse_transform(pred)
                mae += torch.abs(pred - data.y).sum().item()
        mae /= len(test_dataset)
        print(f"MAE: {mae:.6f}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"standard_gnn_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and args
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Save target normalizer
    torch.save({
        'mean': target_normalizer.mean,
        'std': target_normalizer.std
    }, os.path.join(output_dir, 'target_normalizer.pt'))
    
    # Plot training curves
    plot_training_curves(train_losses, val_rmses, output_dir)
    
    print(f"Results saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN to predict DMRG energies")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='train',
                        help='Directory with DMRG data')
    parser.add_argument('--bond_dim', type=int, default=512,
                        help='Bond dimension for DMRG data')
    parser.add_argument('--system_type', type=str, default='pah',
                        help='Type of molecular system')
    parser.add_argument('--max_orbitals', type=int, default=100,
                        help='Maximum number of orbitals')
    parser.add_argument('--mi_threshold', type=float, default=0.01,
                        help='Mutual information threshold for creating edges')
    parser.add_argument('--predict_total_energy', action='store_true',
                        help='Predict total DMRG energy instead of correlation energy')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size for model layers')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of GNN layers')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='combined', 
                        choices=['mean', 'add', 'max', 'combined'],
                        help='Global pooling method')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for L2 regularization')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio (set to 0 to use all data for training)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help='Use learning rate scheduler')
    parser.add_argument('--lr_scheduler_type', type=str, default='metric',
                        choices=['epoch', 'metric'],
                        help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results/standard_gnn',
                        help='Directory to save results')
    
    args = parser.parse_args()
    main(args) 