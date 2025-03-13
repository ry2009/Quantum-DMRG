"""
Heisenberg Model for Delta-ML

This module implements a Heisenberg spin model for use in the Delta-ML framework.
The Heisenberg model describes interacting quantum spins with exchange interactions.

Usage:
    from hamiltonian_research.examples.heisenberg_model import heisenberg_hamiltonian

    # Configure parameters
    params = {'J': 0.15, 'anisotropy': 0.1}
    
    # Use the model
    correction = heisenberg_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **params)
"""

import numpy as np

def heisenberg_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **kwargs):
    """
    Heisenberg Hamiltonian model.
    
    This model applies corrections based on the Heisenberg model:
    H = J ∑ S_i · S_j
    
    For anisotropic models (XXZ):
    H = J ∑ (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)
    
    Args:
        n_orbitals (int): Number of orbitals in the system
        occupations (list): List of orbital occupations (0, 1, or 2)
        entropies (dict): Dictionary of single-site entropies (indexed from 1)
        bond_dim (int): DMRG bond dimension used
        **kwargs: Additional parameters:
            - J (float): Exchange coupling constant (default: 0.1)
            - anisotropy (float): XXZ anisotropy parameter Δ (default: 1.0, isotropic)
            - next_nearest (float): Next-nearest neighbor coupling ratio (default: 0.0)
    
    Returns:
        float: Energy correction in Hartrees
    """
    # Get parameters from kwargs or use defaults
    J = kwargs.get('J', 0.1)  # Exchange parameter
    anisotropy = kwargs.get('anisotropy', 1.0)  # XXZ anisotropy
    next_nearest = kwargs.get('next_nearest', 0.0)  # Next-nearest neighbor coupling
    
    # Heisenberg model contribution (nearest-neighbor interaction)
    heisenberg_contrib = 0.0
    
    # Nearest neighbor interactions
    for i in range(n_orbitals - 1):
        if occupations[i] > 0 and occupations[i+1] > 0:
            # Exchange interaction between neighboring orbitals
            s_i = entropies.get(i+1, 0.0)  # +1 because entropies are 1-indexed
            s_j = entropies.get(i+2, 0.0)  # +1 because entropies are 1-indexed
            
            # Use entropies as proxy for effective spin magnitude
            # For anisotropic model, xy-plane has different coupling than z-axis
            xy_coupling = (s_i * s_j) 
            z_coupling = (s_i * s_j) * anisotropy
            
            # Combine for total coupling
            heisenberg_contrib += J * (xy_coupling + z_coupling)
    
    # Next-nearest neighbor interactions
    if next_nearest > 0:
        for i in range(n_orbitals - 2):
            if occupations[i] > 0 and occupations[i+2] > 0:
                # Exchange interaction between next-nearest neighbors
                s_i = entropies.get(i+1, 0.0)  # +1 because entropies are 1-indexed
                s_j = entropies.get(i+3, 0.0)  # +1 because entropies are 1-indexed
                
                # Apply the next-nearest neighbor coupling (scaled by factor)
                heisenberg_contrib += J * next_nearest * (s_i * s_j)
    
    # Scale correction with bond dimension (smaller correction for higher accuracy)
    scaling = 1.0 / np.sqrt(bond_dim)
    
    # Total correction
    correction = heisenberg_contrib * scaling
    
    return correction 