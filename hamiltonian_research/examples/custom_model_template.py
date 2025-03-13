"""
Custom Hamiltonian Model Template for Delta-ML

This template shows how to implement a custom Hamiltonian model for the Delta-ML framework.
Copy this file and modify the model function to implement your own Hamiltonian physics.

Usage:
    1. Copy this file to a new file (e.g., my_hamiltonian.py)
    2. Modify the custom_hamiltonian function to implement your physics
    3. Import and use in the main plugin:
    
       from hamiltonian_research.examples.my_hamiltonian import custom_hamiltonian
       
       # Then set in the main script:
       HAMILTONIAN_MODEL = custom_hamiltonian
"""

import numpy as np

def custom_hamiltonian(n_orbitals, occupations, entropies, bond_dim, **kwargs):
    """
    Custom Hamiltonian model template.
    
    Implement your Hamiltonian physics here. This function should calculate
    the energy correction based on your custom Hamiltonian model.
    
    Args:
        n_orbitals (int): Number of orbitals in the system
        occupations (list): List of orbital occupations (0, 1, or 2)
        entropies (dict): Dictionary of single-site entropies (indexed from 1)
        bond_dim (int): DMRG bond dimension used
        **kwargs: Additional custom parameters passed from PARAMETERS
    
    Returns:
        float: Energy correction in Hartrees
    """
    # Get parameters from kwargs or use defaults
    param1 = kwargs.get('param1', 1.0)  # First parameter
    param2 = kwargs.get('param2', 0.5)  # Second parameter
    
    # Initialize energy correction
    energy_correction = 0.0
    
    # Implement your Hamiltonian physics here
    # Example:
    # for i in range(n_orbitals):
    #     # On-site terms
    #     if occupations[i] > 0:
    #         entropy = entropies.get(i+1, 0.0)  # +1 because entropies are 1-indexed
    #         energy_correction += param1 * entropy * occupations[i]
    #
    #     # Interaction terms
    #     if i < n_orbitals - 1 and occupations[i] > 0 and occupations[i+1] > 0:
    #         s_i = entropies.get(i+1, 0.0)  # +1 because entropies are 1-indexed
    #         s_j = entropies.get(i+2, 0.0)  # +1 because entropies are 1-indexed
    #         energy_correction += param2 * s_i * s_j
    
    # Scale correction with bond dimension (smaller correction for higher accuracy)
    scaling = 1.0 / np.sqrt(bond_dim)
    
    # Total correction
    return energy_correction * scaling

# You can also define helper functions for more complex models
def calculate_something(data):
    """Example helper function for the model."""
    return data * 2 