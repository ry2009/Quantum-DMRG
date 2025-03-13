#!/usr/bin/env python
"""
Custom Hamiltonian Plugin for Delta-ML

This is a plug-and-play script that allows physicists to experiment with custom Hamiltonians
without modifying the core Delta-ML workflow.

To use:
1. Define your Hamiltonian model in the HAMILTONIAN_MODEL function
2. Set your parameters in the PARAMETERS section
3. Run this script using: python Custom_Hamiltonian_Plugin.py

The script will automatically:
- Load your existing DMRG data
- Apply your custom Hamiltonian modifications
- Train a Delta-ML model
- Evaluate and save the results
"""

import os
import sys
import numpy as np
import torch
import argparse
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# Import the core Delta-ML modules
from improved_dmrg_processor import ImprovedDMRGProcessor, DeltaMLDataset
from advanced_delta_ml_model import (
    AdvancedDeltaMLModel, 
    FeatureNormalizer, 
    train_model, 
    evaluate_model, 
    plot_training_curves, 
    plot_predictions,
    save_results,
    filter_dataset_by_orbital_size
)

#######################################################
# CUSTOMIZE THIS SECTION: DEFINE YOUR HAMILTONIAN MODEL
#######################################################

def HAMILTONIAN_MODEL(n_orbitals, occupations, entropies, bond_dim, **kwargs):
    """
    Define your custom Hamiltonian model here.
    
    This function applies corrections to the DMRG energies based on 
    your custom Hamiltonian physics.
    
    Args:
        n_orbitals (int): Number of orbitals in the system
        occupations (list): List of orbital occupations (0, 1, or 2)
        entropies (dict): Dictionary of single-site entropies (indexed from 1)
        bond_dim (int): DMRG bond dimension used
        **kwargs: Additional custom parameters passed from PARAMETERS
    
    Returns:
        float: Energy correction in Hartrees
    """
    # Example: a combined Hubbard + Heisenberg model
    # Replace this with your own Hamiltonian model
    
    # Get parameters from kwargs or use defaults
    U = kwargs.get('U', 4.0)  # Hubbard U parameter
    J = kwargs.get('J', 0.1)  # Heisenberg J parameter
    
    # Hubbard model contribution (on-site interaction)
    hubbard_contrib = 0.0
    for i in range(n_orbitals):
        if occupations[i] > 0:
            # Use entropy as a proxy for determining double occupation
            entropy = entropies.get(i+1, 0.0)  # +1 because entropies are 1-indexed
            hubbard_contrib += entropy * occupations[i] / 2.0 * U
    
    # Heisenberg model contribution (nearest-neighbor interaction)
    heisenberg_contrib = 0.0
    for i in range(n_orbitals - 1):
        if occupations[i] > 0 and occupations[i+1] > 0:
            # Exchange interaction between neighboring orbitals
            si = entropies.get(i+1, 0.0)  # +1 because entropies are 1-indexed
            sj = entropies.get(i+2, 0.0)  # +1 because entropies are 1-indexed
            heisenberg_contrib += J * si * sj
    
    # Scale correction with bond dimension (smaller correction for higher accuracy)
    scaling = 1.0 / np.sqrt(bond_dim)
    
    # Total correction
    correction = (hubbard_contrib + heisenberg_contrib) * scaling
    
    return correction

#######################################################
# CUSTOMIZE THIS SECTION: SET YOUR PARAMETERS
#######################################################

# Custom Hamiltonian parameters
PARAMETERS = {
    # Hamiltonian physics parameters
    'U': 3.5,        # Hubbard U parameter
    'J': 0.15,       # Heisenberg exchange parameter
    
    # Data selection parameters
    'train_dir': 'train',
    'test_dir': 'test',
    'mi_threshold': 0.004,  # Mutual information threshold for graph construction
    'system_size': None,    # Specific orbital size to use (None to use all)
    'min_orbitals': None,   # Minimum orbital size to include
    'max_orbitals': None,   # Maximum orbital size to include
    
    # Model training parameters
    'hidden_dim': 64,       # Hidden dimension for model layers
    'num_layers': 3,        # Number of message passing layers
    'batch_size': 4,        # Batch size for training
    'learning_rate': 0.0005, # Learning rate
    'weight_decay': 1e-4,   # Weight decay for L2 regularization
    'max_epochs': 100,      # Maximum number of training epochs
    'early_stopping': 15,   # Patience for early stopping
    'val_ratio': 0.1,       # Validation set ratio
    
    # Output settings
    'output_dir': 'custom_hamiltonian_results',  # Directory to save results
    'apply_correction': True,  # Whether to apply the Hamiltonian correction
    'verbose': True,           # Print detailed information during processing
}

#######################################################
# IMPLEMENTATION (You shouldn't need to modify below)
#######################################################

class CustomHamiltonianProcessor(ImprovedDMRGProcessor):
    """
    Extended DMRG processor that applies custom Hamiltonian corrections.
    """
    
    def __init__(self, file_path, mi_threshold=0.004, system_name=None, 
                 hamiltonian_model=None, parameters=None, apply_correction=True):
        """
        Initialize the custom processor.
        
        Args:
            file_path: Path to the DMRG data file
            mi_threshold: Threshold for mutual information
            system_name: Name of the quantum system
            hamiltonian_model: Custom Hamiltonian model function
            parameters: Additional parameters to pass to the Hamiltonian model
            apply_correction: Whether to apply energy corrections
        """
        self.hamiltonian_model = hamiltonian_model
        self.hamiltonian_parameters = parameters if parameters else {}
        self.apply_correction = apply_correction
        self.energy_correction = 0.0  # Store the correction for later analysis
        super().__init__(file_path, mi_threshold, system_name)
    
    def _parse_file(self):
        """
        Parse the DMRG data file and apply custom Hamiltonian modifications.
        """
        # First use the original parsing method
        super()._parse_file()
        
        # Now apply custom Hamiltonian modifications if specified
        if self.hamiltonian_model is not None and self.apply_correction:
            # Extract system parameters
            n_orbitals = len(self.orbital_occupations)
            
            # Apply the custom Hamiltonian model to modify the energies
            self.energy_correction = self.hamiltonian_model(
                n_orbitals=n_orbitals,
                occupations=self.orbital_occupations,
                entropies=self.single_site_entropies,
                bond_dim=self.bond_dim,
                **self.hamiltonian_parameters
            )
            
            # Apply the correction to the DMRG energy
            if PARAMETERS['verbose']:
                print(f"Applying energy correction of {self.energy_correction:.6f} Ha to {self.system_name}")
            
            # Update energies
            self.original_dmrg_energy = self.dmrg_energy  # Store original for reference
            self.dmrg_energy += self.energy_correction
            
            # Recalculate correlation energy
            self.correlation_energy = self.dmrg_energy - self.hf_energy
                
        return self
        
    def to_pyg_data(self):
        """
        Convert the processed data to PyTorch Geometric format.
        """
        # Get the original PyG data object
        data = super().to_pyg_data()
        
        # Add custom Hamiltonian information
        data.custom_hamiltonian = torch.tensor([1.0], dtype=torch.float)
        data.energy_correction = torch.tensor([self.energy_correction], dtype=torch.float)
        
        if hasattr(self, 'original_dmrg_energy'):
            data.original_dmrg_energy = torch.tensor([self.original_dmrg_energy], dtype=torch.float)
        
        return data


class CustomHamiltonianDataset(DeltaMLDataset):
    """
    Dataset that applies custom Hamiltonian models to the Delta-ML data.
    """
    
    def __init__(self, data_dir, hamiltonian_model=None, parameters=None, apply_correction=True, **kwargs):
        """
        Initialize the dataset with custom Hamiltonian processing.
        
        Args:
            data_dir: Directory containing the data
            hamiltonian_model: Function that implements the custom Hamiltonian
            parameters: Parameters to pass to the Hamiltonian model
            apply_correction: Whether to apply the Hamiltonian corrections
            **kwargs: Additional arguments to pass to the parent class
        """
        self.hamiltonian_model = hamiltonian_model
        self.hamiltonian_parameters = parameters if parameters else {}
        self.apply_correction = apply_correction
        
        # We need to override the processor creation in the parent class
        # Store these for use in our custom processor
        self._processor_args = {
            'hamiltonian_model': hamiltonian_model,
            'parameters': parameters,
            'apply_correction': apply_correction
        }
        
        # Initialize with parent class, but don't process immediately
        super().__init__(data_dir, process_immediately=False, **kwargs)
        
        # Now process the data with our custom processor
        self._process_data()
    
    def _create_processor(self, file_path, mi_threshold, system_name):
        """
        Create a custom processor instead of the default one.
        """
        return CustomHamiltonianProcessor(
            file_path=file_path,
            mi_threshold=mi_threshold,
            system_name=system_name,
            **self._processor_args
        )
    
    def _process_data(self):
        """
        Process all data files with the custom processor.
        """
        self.data_list = []
        
        if PARAMETERS['verbose']:
            print(f"Processing data with custom Hamiltonian model...")
            print(f"Data directory: {self.data_dir}")
            print(f"Hamiltonian parameters: {self.hamiltonian_parameters}")
        
        for system_name, low_dim, high_dim, low_file, high_file in tqdm(self.delta_pairs, 
                                                                        desc="Processing DMRG files"):
            try:
                # Create custom processors for both low and high bond dimension files
                low_processor = self._create_processor(
                    low_file, self.mi_threshold, system_name
                )
                
                high_processor = self._create_processor(
                    high_file, self.mi_threshold, system_name
                )
                
                # Create the PyG data object from the low-level calculation
                data = low_processor.to_pyg_data()
                
                # Update the target to be the delta energy (high - low)
                delta_energy = high_processor.dmrg_energy - low_processor.dmrg_energy
                data.y = torch.tensor([delta_energy], dtype=torch.float)
                
                # Store useful metadata
                data.low_bond_dim = torch.tensor([low_dim], dtype=torch.int)
                data.high_bond_dim = torch.tensor([high_dim], dtype=torch.int)
                data.high_dmrg_energy = torch.tensor([high_processor.dmrg_energy], dtype=torch.float)
                data.low_dmrg_energy = torch.tensor([low_processor.dmrg_energy], dtype=torch.float)
                data.high_tre = torch.tensor([high_processor.truncation_error], dtype=torch.float)
                
                # Store custom Hamiltonian information
                data.low_energy_correction = torch.tensor([low_processor.energy_correction], dtype=torch.float)
                data.high_energy_correction = torch.tensor([high_processor.energy_correction], dtype=torch.float)
                
                # Add the processed data to our list
                self.data_list.append(data)
                
            except Exception as e:
                print(f"Error processing {system_name} ({low_dim}→{high_dim}): {e}")
        
        print(f"Dataset contains {len(self.data_list)} examples")


def run_custom_hamiltonian_experiment():
    """
    Run a complete experiment with the custom Hamiltonian model.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    os.makedirs(PARAMETERS['output_dir'], exist_ok=True)
    
    # Save the parameters for reproducibility
    import json
    with open(os.path.join(PARAMETERS['output_dir'], 'parameters.json'), 'w') as f:
        json.dump(PARAMETERS, f, indent=2)
    
    # Create datasets with custom Hamiltonian
    print("\n1. Creating datasets with custom Hamiltonian...")
    
    train_dataset = CustomHamiltonianDataset(
        data_dir=PARAMETERS['train_dir'],
        hamiltonian_model=HAMILTONIAN_MODEL,
        parameters=PARAMETERS,
        apply_correction=PARAMETERS['apply_correction'],
        mi_threshold=PARAMETERS['mi_threshold']
    )
    
    test_dataset = CustomHamiltonianDataset(
        data_dir=PARAMETERS['test_dir'],
        hamiltonian_model=HAMILTONIAN_MODEL,
        parameters=PARAMETERS,
        apply_correction=PARAMETERS['apply_correction'],
        mi_threshold=PARAMETERS['mi_threshold']
    )
    
    # Filter datasets if system size parameters are provided
    if any(p is not None for p in [PARAMETERS['system_size'], PARAMETERS['min_orbitals'], PARAMETERS['max_orbitals']]):
        train_dataset = filter_dataset_by_orbital_size(
            train_dataset,
            min_orbitals=PARAMETERS['min_orbitals'],
            max_orbitals=PARAMETERS['max_orbitals'],
            target_orbitals=PARAMETERS['system_size']
        )
        
        test_dataset = filter_dataset_by_orbital_size(
            test_dataset,
            min_orbitals=PARAMETERS['min_orbitals'],
            max_orbitals=PARAMETERS['max_orbitals'],
            target_orbitals=PARAMETERS['system_size']
        )
    
    print(f"Filtered training dataset: {len(train_dataset)} examples")
    print(f"Filtered test dataset: {len(test_dataset)} examples")
    
    # Create normalizer and split train/val
    print("\n2. Preparing data for model training...")
    normalizer = FeatureNormalizer(train_dataset)
    
    # Apply normalization
    normalized_train = normalizer.transform(train_dataset)
    normalized_test = normalizer.transform(test_dataset)
    
    # Split train/val
    val_size = int(len(normalized_train) * PARAMETERS['val_ratio'])
    indices = torch.randperm(len(normalized_train))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_normalized = [normalized_train[i] for i in train_indices]
    val_normalized = [normalized_train[i] for i in val_indices]
    
    print(f"Training split: {len(train_normalized)} examples")
    print(f"Validation split: {len(val_normalized)} examples")
    
    # Determine feature dimensions from a sample
    sample = normalized_train[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    global_dim = sample.global_feature.shape[1]
    
    print(f"Feature dimensions: node={node_dim}, edge={edge_dim}, global={global_dim}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_normalized,
        batch_size=PARAMETERS['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_normalized,
        batch_size=PARAMETERS['batch_size'],
        shuffle=False
    )
    
    test_loader = DataLoader(
        normalized_test,
        batch_size=PARAMETERS['batch_size'],
        shuffle=False
    )
    
    # Create model
    print("\n3. Creating and training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedDeltaMLModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=PARAMETERS['hidden_dim'],
        n_layers=PARAMETERS['num_layers']
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
        lr=PARAMETERS['learning_rate'],
        weight_decay=PARAMETERS['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=PARAMETERS['early_stopping']//3,
        min_lr=1e-6,
        verbose=True
    )
    
    criterion = torch.nn.MSELoss()
    
    # Train model
    history, best_val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        n_epochs=PARAMETERS['max_epochs'],
        patience=PARAMETERS['early_stopping'],
        output_dir=PARAMETERS['output_dir']
    )
    
    # Plot training curves
    plot_training_curves(history, PARAMETERS['output_dir'])
    
    # Evaluate on test set
    print("\n4. Evaluating model on test set...")
    test_metrics, test_predictions = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        normalizer=normalizer
    )
    
    print(f"Test RMSE: {test_metrics['rmse']:.6f} Ha ({test_metrics['rmse']*1000:.2f} mHa)")
    print(f"Test MAE: {test_metrics['mae']:.6f} Ha ({test_metrics['mae']*1000:.2f} mHa)")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test Relative Error: {test_metrics['rel_error']:.2f}%")
    
    # Save results
    save_results(
        model=model,
        history=history,
        best_val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
        output_dir=PARAMETERS['output_dir']
    )
    
    # Plot predictions vs targets
    plot_predictions(
        test_predictions['predictions'], 
        test_predictions['targets'], 
        PARAMETERS['output_dir']
    )
    
    # Create a custom analysis of Hamiltonian corrections
    analyze_hamiltonian_corrections(test_dataset, PARAMETERS['output_dir'])
    
    print(f"\nAll results saved to {PARAMETERS['output_dir']}")


def analyze_hamiltonian_corrections(dataset, output_dir):
    """
    Analyze the impact of custom Hamiltonian corrections.
    
    Args:
        dataset: Dataset with Hamiltonian corrections
        output_dir: Directory to save results
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Extract correction information
    data = []
    for i, sample in enumerate(dataset):
        if hasattr(sample, 'low_energy_correction') and hasattr(sample, 'high_energy_correction'):
            system_name = sample.system_name if hasattr(sample, 'system_name') else f"System_{i}"
            low_dim = sample.low_bond_dim.item() if hasattr(sample, 'low_bond_dim') else 0
            high_dim = sample.high_bond_dim.item() if hasattr(sample, 'high_bond_dim') else 0
            
            data.append({
                'system': system_name,
                'low_bond_dim': low_dim,
                'high_bond_dim': high_dim,
                'low_correction': sample.low_energy_correction.item(),
                'high_correction': sample.high_energy_correction.item(),
                'delta_correction': sample.high_energy_correction.item() - sample.low_energy_correction.item(),
                'delta_energy': sample.y.item(),
            })
    
    if not data:
        print("No Hamiltonian correction data found for analysis")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'hamiltonian_corrections.csv'), index=False)
    
    # Plot correction distributions
    plt.figure(figsize=(10, 6))
    plt.hist(df['low_correction'], alpha=0.5, label='Low bond dim')
    plt.hist(df['high_correction'], alpha=0.5, label='High bond dim')
    plt.xlabel('Energy correction (Ha)')
    plt.ylabel('Count')
    plt.title('Distribution of Hamiltonian Energy Corrections')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'correction_distribution.png'), dpi=300)
    
    # Plot correction vs bond dimension
    plt.figure(figsize=(10, 6))
    plt.scatter(df['low_bond_dim'], df['low_correction'], alpha=0.7, label='Low bond dim')
    plt.scatter(df['high_bond_dim'], df['high_correction'], alpha=0.7, label='High bond dim')
    plt.xlabel('Bond dimension')
    plt.ylabel('Energy correction (Ha)')
    plt.title('Hamiltonian Corrections vs Bond Dimension')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'correction_vs_bond_dim.png'), dpi=300)
    
    # Plot the relationship between delta correction and delta energy
    plt.figure(figsize=(10, 6))
    plt.scatter(df['delta_correction'], df['delta_energy'], alpha=0.7)
    plt.xlabel('Delta correction (Ha)')
    plt.ylabel('Delta energy (Ha)')
    plt.title('Relationship Between Correction and Energy Differences')
    plt.grid(alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(df['delta_correction'], df['delta_energy'])[0, 1]
    plt.annotate(f'Correlation: {corr:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.savefig(os.path.join(output_dir, 'correction_vs_energy.png'), dpi=300)
    
    print(f"Hamiltonian correction analysis saved to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Custom Hamiltonian Plugin for Delta-ML')
    
    # Basic parameters
    parser.add_argument('--output_dir', type=str, default=PARAMETERS['output_dir'],
                        help='Directory to save results')
    parser.add_argument('--apply_correction', action='store_true', default=PARAMETERS['apply_correction'],
                        help='Apply the Hamiltonian correction')
    parser.add_argument('--verbose', action='store_true', default=PARAMETERS['verbose'],
                        help='Print detailed information during processing')
    
    # Hamiltonian physics parameters
    parser.add_argument('--U', type=float, default=PARAMETERS['U'],
                        help='Hubbard U parameter')
    parser.add_argument('--J', type=float, default=PARAMETERS['J'],
                        help='Heisenberg exchange parameter')
    
    # Override with command line args
    args = parser.parse_args()
    PARAMETERS['output_dir'] = args.output_dir
    PARAMETERS['apply_correction'] = args.apply_correction
    PARAMETERS['verbose'] = args.verbose
    PARAMETERS['U'] = args.U
    PARAMETERS['J'] = args.J
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print("Starting Custom Hamiltonian Experiment...")
    print(f"Physics parameters: U={PARAMETERS['U']}, J={PARAMETERS['J']}")
    print(f"Apply correction: {PARAMETERS['apply_correction']}")
    print(f"Output directory: {PARAMETERS['output_dir']}")
    
    run_custom_hamiltonian_experiment() 