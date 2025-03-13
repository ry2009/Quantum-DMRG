"""
Combined Hubbard-Heisenberg Model for Delta-ML

This module implements a combined Hubbard-Heisenberg model that incorporates both
electronic and spin interactions in a single Hamiltonian.

Usage:
    from hamiltonian_research.examples.combined_model import combined_hamiltonian

    # Configure parameters
    params = {'U': 4.0, 'J': 0.15}
    
    # Use the model
    correction = combined_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **params)
"""

import numpy as np
from hamiltonian_research.examples.hubbard_model import hubbard_hamiltonian
from hamiltonian_research.examples.heisenberg_model import heisenberg_hamiltonian

def combined_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **kwargs):
    """
    Combined Hubbard-Heisenberg Hamiltonian model.
    
    This model applies corrections based on both the Hubbard and Heisenberg models:
    H = H_Hubbard + H_Heisenberg
    
    Args:
        n_orbitals (int): Number of orbitals in the system
        occupations (list): List of orbital occupations (0, 1, or 2)
        entropies (dict): Dictionary of single-site entropies (indexed from 1)
        bond_dim (int): DMRG bond dimension used
        **kwargs: Additional parameters that will be passed to both models
    
    Returns:
        float: Energy correction in Hartrees
    """
    # Calculate contributions from both models
    hubbard_contrib = hubbard_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **kwargs)
    heisenberg_contrib = heisenberg_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **kwargs)
    
    # Get weight parameters from kwargs or use defaults
    hubbard_weight = kwargs.get('hubbard_weight', 1.0)
    heisenberg_weight = kwargs.get('heisenberg_weight', 1.0)
    
    # Combine contributions with optional weights
    combined_correction = (hubbard_weight * hubbard_contrib + 
                          heisenberg_weight * heisenberg_contrib)
    
    return combined_correction 