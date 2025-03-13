import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from torch_geometric.loader import DataLoader

from bayesian_models import BayesianDMRGNet
from spin_data_processor import SpinDataset
from train_bayesian import TargetNormalizer

def load_model(model_path, model_config, device):
    """Load a trained model from a checkpoint."""
    # Create model with the same configuration
    model = BayesianDMRGNet(
        node_features=model_config['node_features'],
        edge_features=model_config['edge_features'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_samples=model_config['num_samples']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model

def predict_with_uncertainty(model, data_loader, target_normalizer, device, num_samples=50):
    """Make predictions with uncertainty estimation."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_params = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # Get prediction with uncertainty
            mean, std, samples = model.predict_with_uncertainty(data, num_samples)
            
            # Transform back to original scale
            mean = target_normalizer.inverse_transform(mean)
            std = std * target_normalizer.std
            
            # Store results
            all_predictions.append(mean.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
            all_uncertainties.append(std.cpu().numpy())
            all_params.append(data.params.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    uncertainties = np.concatenate(all_uncertainties)
    params = np.concatenate(all_params)
    
    return predictions, targets, uncertainties, params

def plot_predictions(predictions, targets, uncertainties, params, system_type, output_dir):
    """Plot predictions vs targets with uncertainty."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.errorbar(targets.flatten(), predictions.flatten(), 
                 yerr=uncertainties.flatten(), fmt='o', alpha=0.5)
    
    # Add diagonal line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title(f'Predictions vs Targets with Uncertainty for {system_type.capitalize()} Model')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    
    # Plot uncertainty vs error
    plt.figure(figsize=(10, 6))
    errors = np.abs(predictions - targets).flatten()
    plt.scatter(uncertainties.flatten(), errors, alpha=0.5)
    
    # Add trend line
    z = np.polyfit(uncertainties.flatten(), errors, 1)
    p = np.poly1d(z)
    plt.plot(uncertainties.flatten(), p(uncertainties.flatten()), "r--")
    
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Absolute Error')
    plt.title('Uncertainty vs Error Correlation')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'uncertainty_vs_error.png'))
    
    # Plot parameter space with uncertainty heatmap
    if system_type in ['heisenberg', 'ising']:
        param_names = ['J', 'h']
    elif system_type == 'hubbard':
        param_names = ['t', 'U']
    else:
        param_names = ['Param1', 'Param2']
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(params[:, 0], params[:, 1], c=uncertainties.flatten(), 
                          cmap='viridis', alpha=0.8, s=50)
    plt.colorbar(scatter, label='Uncertainty')
    plt.xlabel(param_names[0])
    plt.ylabel(param_names[1])
    plt.title(f'Uncertainty across Parameter Space for {system_type.capitalize()} Model')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'parameter_uncertainty.png'))
    
    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    
    # Calculate percentage of points within different sigma ranges
    sigma_ranges = np.linspace(0.1, 3.0, 30)
    within_range = []
    
    for sigma in sigma_ranges:
        within_sigma = ((targets >= predictions - sigma * uncertainties) & 
                        (targets <= predictions + sigma * uncertainties)).mean()
        within_range.append(within_sigma)
    
    plt.plot(sigma_ranges, within_range, 'o-')
    
    # Plot ideal calibration curve (CDF of normal distribution)
    from scipy.stats import norm
    ideal_curve = [2 * norm.cdf(sigma) - 1 for sigma in sigma_ranges]
    plt.plot(sigma_ranges, ideal_curve, 'r--', label='Ideal Calibration')
    
    plt.xlabel('Number of Sigma')
    plt.ylabel('Fraction of Points Within Range')
    plt.title('Calibration Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    
    return

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model configuration
    model_dir = os.path.dirname(args.model_path)
    args_path = os.path.join(model_dir, 'args.json')
    
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            model_args = json.load(f)
        print(f"Loaded model configuration from {args_path}")
    else:
        print(f"Warning: Could not find model configuration at {args_path}")
        print("Using default configuration")
        model_args = {
            'system_type': args.system_type,
            'hidden_size': 64,
            'num_layers': 3,
            'num_samples': 20
        }
    
    # Load dataset
    print(f"Loading {args.system_type} dataset...")
    dataset = SpinDataset(
        root_dir=args.data_dir,
        system_type=args.system_type,
        dim=args.dim,
        size=args.size
    )
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    
    # Get feature dimensions from dataset
    node_features = dataset[0].x.size(1)
    edge_features = dataset[0].edge_attr.size(1)
    
    # Create model configuration
    model_config = {
        'node_features': node_features,
        'edge_features': edge_features,
        'hidden_size': model_args.get('hidden_size', 64),
        'num_layers': model_args.get('num_layers', 3),
        'num_samples': args.num_samples
    }
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, model_config, device)
    
    # Initialize target normalizer
    target_normalizer = TargetNormalizer()
    targets = torch.cat([data.y for data in dataset])
    target_normalizer.fit(targets)
    
    # Fit feature normalizer
    model.fit_normalizer([data for data in dataset])
    
    # Make predictions
    print("Making predictions with uncertainty estimation...")
    predictions, targets, uncertainties, params = predict_with_uncertainty(
        model, data_loader, target_normalizer, device, args.num_samples
    )
    
    # Calculate metrics
    errors = np.abs(predictions - targets)
    rmse = np.sqrt(np.mean(np.square(predictions - targets)))
    mae = np.mean(errors)
    
    # Calculate calibration metrics
    within_1std = ((targets >= predictions - uncertainties) & 
                  (targets <= predictions + uncertainties)).mean()
    within_2std = ((targets >= predictions - 2*uncertainties) & 
                  (targets <= predictions + 2*uncertainties)).mean()
    
    # Print results
    print("\nPrediction Results:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Mean uncertainty: {uncertainties.mean():.6f}")
    print(f"Calibration (1σ): {within_1std:.2f}")
    print(f"Calibration (2σ): {within_2std:.2f}")
    
    # Plot results
    print(f"Plotting results to {args.output_dir}")
    plot_predictions(predictions, targets, uncertainties, params, args.system_type, args.output_dir)
    
    # Save results
    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'mean_uncertainty': float(uncertainties.mean()),
        'within_1std': float(within_1std),
        'within_2std': float(within_2std)
    }
    
    with open(os.path.join(args.output_dir, 'prediction_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with uncertainty using a trained Bayesian GNN")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of MC samples for uncertainty estimation')
    
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
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='prediction_results',
                        help='Directory to save prediction results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction')
    
    args = parser.parse_args()
    main(args) 