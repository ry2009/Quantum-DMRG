"""
Hubbard Model for Delta-ML

This module implements a simple Hubbard model for use in the Delta-ML framework.
The Hubbard model describes electrons in a lattice with on-site interactions.

Usage:
    from hamiltonian_research.examples.hubbard_model import hubbard_hamiltonian

    # Configure parameters
    params = {'U': 4.0}
    
    # Use the model
    correction = hubbard_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **params)
"""

import numpy as np

def hubbard_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **kwargs):
    """
    Hubbard Hamiltonian model.
    
    This model applies corrections based on the Hubbard model:
    H = -t ∑ (c†_i σ c_j σ + h.c.) + U ∑ n_i↑ n_i↓
    
    Args:
        n_orbitals (int): Number of orbitals in the system
        occupations (list): List of orbital occupations (0, 1, or 2)
        entropies (dict): Dictionary of single-site entropies (indexed from 1)
        bond_dim (int): DMRG bond dimension used
        **kwargs: Additional parameters:
            - U (float): Hubbard U parameter (on-site interaction)
            - t (float): Hopping parameter (default: 1.0)
    
    Returns:
        float: Energy correction in Hartrees
    """
    # Get parameters from kwargs or use defaults
    U = kwargs.get('U', 4.0)  # Hubbard U parameter
    t = kwargs.get('t', 1.0)  # Hopping parameter
    
    # Hubbard model contribution (on-site interaction)
    hubbard_contrib = 0.0
    for i in range(n_orbitals):
        if occupations[i] > 0:
            # Use entropy as a proxy for determining double occupation probability
            entropy = entropies.get(i+1, 0.0)  # +1 because entropies are 1-indexed
            double_occ_prob = entropy / np.log(2)  # Normalize by log(2) for a two-level system
            hubbard_contrib += double_occ_prob * U
    
    # Scale correction with bond dimension (smaller correction for higher accuracy)
    scaling = 1.0 / np.sqrt(bond_dim)
    
    # Total correction
    correction = hubbard_contrib * scaling
    
    return correction 