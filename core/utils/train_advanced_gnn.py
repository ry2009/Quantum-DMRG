import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

from models import AdvancedDMRGNet as DMRGNet
from dmrg_data_processor import DMRGDataset

class TargetNormalizer:
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
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def restore(self, model):
        """Restore best weights to model"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)

def get_lr_scheduler(scheduler_type, optimizer, args, steps_per_epoch=None):
    """Create learning rate scheduler based on type"""
    if scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            verbose=True,
            min_lr=1e-6
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    else:
        return None

def get_warmup_factor(current_step, warmup_steps):
    """Get learning rate warmup factor"""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, train_loader, optimizer, device, target_normalizer, 
                scheduler=None, grad_clip=1.0, epoch=0, warmup_epochs=0,
                steps_per_epoch=None, scaler=None, use_amp=False):
    """Train for one epoch with advanced techniques"""
    model.train()
    loss_meter = AverageMeter()
    
    # Setup progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
    
    for step, data in enumerate(pbar):
        data = data.to(device)
        
        # Learning rate warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_steps = len(train_loader) * warmup_epochs
            current_step = epoch * len(train_loader) + step
            warmup_factor = get_warmup_factor(current_step, warmup_steps)
            
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward with optional automatic mixed precision
        if use_amp and device != torch.device('cpu'):
            with autocast():
                pred = model(data)
                y = target_normalizer.transform(data.y)
                loss = nn.functional.mse_loss(pred, y)
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Unscale weights for gradient clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Update weights with scaling
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward/backward
            pred = model(data)
            y = target_normalizer.transform(data.y)
            loss = nn.functional.mse_loss(pred, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Update weights
            optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), data.num_graphs)
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss_meter.avg:.6f}"})
    
    # Step epoch-based scheduler if provided
    if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step()
        
    return loss_meter.avg

def evaluate(model, loader, device, target_normalizer, use_amp=False):
    """Evaluate the model with optional mixed precision"""
    model.eval()
    mse = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass with optional mixed precision
            if use_amp and device != torch.device('cpu'):
                with autocast():
                    pred = model(data)
            else:
                pred = model(data)
            
            # Convert back to original scale
            pred = target_normalizer.inverse_transform(pred)
            
            # Compute MSE
            mse += ((pred - data.y) ** 2).sum().item()
    
    # Compute RMSE
    rmse = np.sqrt(mse / len(loader.dataset))
    return rmse

def plot_training_curves(train_losses, val_rmses, lr_history, output_dir):
    """Plot training and validation curves with learning rate"""
    # Setup figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training loss
    axes[0].plot(train_losses, 'b-', label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot validation RMSE
    axes[1].plot(val_rmses, 'r-', label='Validation RMSE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE (Ha)')
    axes[1].set_title('Validation RMSE Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot learning rate
    axes[2].plot(lr_history, 'g-', label='Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))

def setup_data_loaders(args, device):
    """Setup and return data loaders and dataset information"""
    # Load dataset
    print("Loading DMRG dataset...")
    dataset = DMRGDataset(
        root_dir=args.data_dir,
        bond_dims=[args.bond_dim],
        system_type=args.system_type,
        max_orbitals=args.max_orbitals,
        mi_threshold=args.mi_threshold,
        predict_total_energy=False  # Use correlation energy
    )
    
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
    
    # Create data loaders with different batch sizes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=device != torch.device('cpu'),
        num_workers=0  # Set to higher value if needed
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2,  # Larger batches for validation
        pin_memory=device != torch.device('cpu')
    )
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size * 2,
            pin_memory=device != torch.device('cpu')
        )
    else:
        test_loader = None
    
    # Initialize target normalizer
    target_normalizer = TargetNormalizer()
    targets = torch.cat([dataset[idx].y for idx in train_dataset.indices])
    target_normalizer.fit(targets)
    
    # Return all data components
    return {
        'dataset': dataset,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'target_normalizer': target_normalizer
    }

def setup_model_optimizer(args, data_info, device):
    """Setup and return model, optimizer and scheduler"""
    # Create model
    if args.model_type == 'sota':
        from models import StateOfTheArtDMRGNet as ModelClass
        print("Using State-of-the-Art DMRG Net")
    else:
        from models import AdvancedDMRGNet as ModelClass
        print("Using Advanced DMRG Net")
    
    model = ModelClass(
        node_features=2,  # occupation and entropy
        edge_features=1,  # mutual information
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        heads=args.attention_heads,
        dropout=args.dropout,
        pooling=args.pooling,
        gating=args.gating,
        readout_layers=args.readout_layers,
        with_global_features=args.global_features if hasattr(args, 'global_features') else False
    ).to(device)
    
    # Fit feature normalizer
    train_data_list = [data_info['dataset'][idx] for idx in data_info['train_dataset'].indices]
    model.fit_normalizer(train_data_list)
    
    # Define optimizer with weight decay and correct initial learning rate
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Store initial learning rate for warmup
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
    
    # Create learning rate scheduler
    steps_per_epoch = len(data_info['train_loader'])
    if args.use_lr_scheduler:
        scheduler = get_lr_scheduler(
            args.lr_scheduler_type, 
            optimizer, 
            args, 
            steps_per_epoch
        )
    else:
        scheduler = None
    
    return model, optimizer, scheduler

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup data
    data_info = setup_data_loaders(args, device)
    
    # Setup model, optimizer and scheduler
    model, optimizer, scheduler = setup_model_optimizer(args, data_info, device)
    
    # Initialize automatic mixed precision if available
    use_amp = device.type == 'cuda' and args.use_amp
    scaler = GradScaler() if use_amp else None
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        restore_best_weights=True
    )
    
    # Training loop
    train_losses = []
    val_rmses = []
    lr_history = []
    best_val_rmse = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Train one epoch
        train_loss = train_epoch(
            model, 
            data_info['train_loader'], 
            optimizer, 
            device, 
            data_info['target_normalizer'],
            scheduler=scheduler if args.lr_scheduler_type != 'plateau' else None,
            grad_clip=args.grad_clip,
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
            steps_per_epoch=len(data_info['train_loader']),
            scaler=scaler,
            use_amp=use_amp
        )
        train_losses.append(train_loss)
        
        # Validate
        val_rmse = evaluate(
            model, 
            data_info['val_loader'], 
            device, 
            data_info['target_normalizer'],
            use_amp=use_amp
        )
        val_rmses.append(val_rmse)
        
        # Step scheduler if using validation metric
        if scheduler is not None and args.lr_scheduler_type == 'plateau':
            scheduler.step(val_rmse)
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.6f} | LR: {current_lr:.8f}")
        
        # Check early stopping
        if early_stopping(val_rmse, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best weights
    early_stopping.restore(model)
    
    # Evaluate on test set if available
    if data_info['test_dataset'] is not None:
        test_rmse = evaluate(
            model, 
            data_info['test_loader'], 
            device, 
            data_info['target_normalizer'],
            use_amp=use_amp
        )
        print(f"\nTest set evaluation:")
        print(f"RMSE: {test_rmse:.6f}")
        
        # Calculate MAE on test set
        model.eval()
        mae = 0
        with torch.no_grad():
            for data in data_info['test_loader']:
                data = data.to(device)
                pred = model(data)
                pred = data_info['target_normalizer'].inverse_transform(pred)
                mae += torch.abs(pred - data.y).sum().item()
        mae /= len(data_info['test_dataset'])
        print(f"MAE: {mae:.6f}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "sota_gnn" if args.model_type == 'sota' else "advanced_gnn"
    output_dir = os.path.join(args.output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and args
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Save target normalizer
    torch.save({
        'mean': data_info['target_normalizer'].mean,
        'std': data_info['target_normalizer'].std
    }, os.path.join(output_dir, 'target_normalizer.pt'))
    
    # Plot training curves
    plot_training_curves(train_losses, val_rmses, lr_history, output_dir)
    
    print(f"Results saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an advanced GNN to predict DMRG energies")
    
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
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='advanced',
                        choices=['advanced', 'sota'],
                        help='Type of model to use')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size for model layers')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of GNN layers')
    parser.add_argument('--attention_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='combined', 
                        choices=['mean', 'add', 'max', 'combined'],
                        help='Global pooling method')
    parser.add_argument('--gating', type=bool, default=True,
                        help='Use gating mechanism')
    parser.add_argument('--readout_layers', type=int, default=3,
                        help='Number of readout layers')
    parser.add_argument('--global_features', type=bool, default=False,
                        help='Use global molecular features')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for L2 regularization')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=25,
                        help='Patience for early stopping')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio (set to 0 to use all data for training)')
    parser.add_argument('--grad_clip', type=float, default=0.5,
                        help='Gradient clipping value')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help='Use learning rate scheduler')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        choices=['plateau', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results/advanced_gnn',
                        help='Directory to save results')
    
    args = parser.parse_args()
    main(args) 