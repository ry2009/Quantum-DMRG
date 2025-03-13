#!/usr/bin/env python
"""
Hamiltonian Research Experiment Runner

This is the main entry point for running Hamiltonian experiments within the Delta-ML framework.
It provides a clean interface to the functionality in the src directory.

Usage:
    python run_experiment.py [--model MODEL] [--U VALUE] [--J VALUE] [--output OUTPUT_DIR]

Example:
    python run_experiment.py --model hubbard --U 4.0 --J 0.1 --output my_results
"""

import os
import sys
import argparse

# Ensure the parent directory is in the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from src directory
from hamiltonian_research.src.hamiltonian_plugin import run_custom_hamiltonian_experiment, PARAMETERS

# Import Hamiltonian models
from hamiltonian_research.examples.hubbard_model import hubbard_hamiltonian
from hamiltonian_research.examples.heisenberg_model import heisenberg_hamiltonian
from hamiltonian_research.examples.combined_model import combined_hamiltonian

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hamiltonian Research Experiment Runner')
    
    # Model selection
    parser.add_argument('--model', type=str, default='combined',
                        choices=['combined', 'hubbard', 'heisenberg', 'custom'],
                        help='Hamiltonian model to use')
    
    # Physics parameters
    parser.add_argument('--U', type=float, default=PARAMETERS['U'],
                        help='Hubbard U parameter')
    parser.add_argument('--J', type=float, default=PARAMETERS['J'],
                        help='Heisenberg exchange parameter')
    parser.add_argument('--t', type=float, default=1.0,
                        help='Hubbard hopping parameter')
    parser.add_argument('--anisotropy', type=float, default=1.0,
                        help='Heisenberg anisotropy parameter')
    parser.add_argument('--next_nearest', type=float, default=0.0,
                        help='Next-nearest neighbor coupling ratio')
    
    # Weight parameters for combined model
    parser.add_argument('--hubbard_weight', type=float, default=1.0,
                        help='Weight for Hubbard model in combined model')
    parser.add_argument('--heisenberg_weight', type=float, default=1.0,
                        help='Weight for Heisenberg model in combined model')
    
    # Output settings
    parser.add_argument('--output', type=str, default=PARAMETERS['output_dir'],
                        help='Directory to save results')
    parser.add_argument('--apply_correction', action='store_true', 
                        default=PARAMETERS['apply_correction'],
                        help='Apply the Hamiltonian correction')
    parser.add_argument('--verbose', action='store_true', 
                        default=PARAMETERS['verbose'],
                        help='Print detailed information during processing')
    
    return parser.parse_args()

def select_hamiltonian_model(model_name):
    """Select the appropriate Hamiltonian model based on the model name."""
    models = {
        'hubbard': hubbard_hamiltonian,
        'heisenberg': heisenberg_hamiltonian,
        'combined': combined_hamiltonian,
        # 'custom' will use the default in hamiltonian_plugin.py
    }
    
    return models.get(model_name, None)

def main():
    """Main entry point for the experiment runner."""
    args = parse_args()
    
    # Update parameters based on command line arguments
    PARAMETERS['output_dir'] = args.output
    PARAMETERS['apply_correction'] = args.apply_correction  
    PARAMETERS['verbose'] = args.verbose
    PARAMETERS['U'] = args.U
    PARAMETERS['J'] = args.J
    PARAMETERS['t'] = args.t
    PARAMETERS['anisotropy'] = args.anisotropy
    PARAMETERS['next_nearest'] = args.next_nearest
    PARAMETERS['hubbard_weight'] = args.hubbard_weight
    PARAMETERS['heisenberg_weight'] = args.heisenberg_weight
    
    # Set the Hamiltonian model based on the selection
    import hamiltonian_research.src.hamiltonian_plugin as plugin
    hamiltonian_model = select_hamiltonian_model(args.model)
    if hamiltonian_model:
        plugin.HAMILTONIAN_MODEL = hamiltonian_model
    
    print("=== Hamiltonian Research Experiment Runner ===")
    print(f"Model: {args.model}")
    print(f"Physics parameters:")
    print(f"  U: {PARAMETERS['U']}")
    print(f"  J: {PARAMETERS['J']}")
    print(f"  t: {PARAMETERS['t']}")
    if args.model in ['heisenberg', 'combined']:
        print(f"  Anisotropy: {PARAMETERS['anisotropy']}")
        print(f"  Next-nearest: {PARAMETERS['next_nearest']}")
    if args.model == 'combined':
        print(f"  Hubbard weight: {PARAMETERS['hubbard_weight']}")
        print(f"  Heisenberg weight: {PARAMETERS['heisenberg_weight']}")
    
    print(f"Apply correction: {PARAMETERS['apply_correction']}")
    print(f"Output directory: {PARAMETERS['output_dir']}")
    print("=" * 45)
    
    # Run the experiment
    run_custom_hamiltonian_experiment()
    
    print("\nExperiment complete! Results saved to:")
    print(f"  {os.path.abspath(PARAMETERS['output_dir'])}")

if __name__ == "__main__":
    main() 