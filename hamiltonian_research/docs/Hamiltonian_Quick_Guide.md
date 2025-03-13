# Quick Guide: Hamiltonian Integration in Delta-ML

## Key Points for Physicists

1. **The Hamiltonian's Role**: The Hamiltonian is **not directly used** in our ML model. Instead, it's part of the upstream DMRG calculations that generate our training/test data.

2. **Delta Learning Approach**: Our model learns the *difference* between energies calculated at different accuracies (bond dimensions):
   * Low-accuracy DMRG calculation → initial energy
   * High-accuracy DMRG calculation → reference energy
   * Delta-ML predicts: ΔE = E_high - E_low

3. **Data Flow**:
   * DMRG calculations (external to our code) solve the Hamiltonian eigenvalue problem
   * Our data files contain these pre-computed energies (HF and DMRG energies)
   * Our model extracts features from the low-accuracy calculation
   * Our model predicts the energy difference (correction)

## Where to Look in the Code

1. **Data Parsing** (where energies are extracted):
   * `improved_dmrg_processor.py` → `_parse_file()` method
   * Data format: First two lines of each file contain HF and DMRG energies

2. **Target Creation** (where ΔE is calculated):
   * `improved_dmrg_processor.py` → `DeltaMLDataset` class
   * Key line: `delta_energy = high_processor.dmrg_energy - low_processor.dmrg_energy`

3. **Example Data File**:
   ```
   -840.12276929       # HF energy
   -840.35297292       # DMRG energy
   0.00051949409500    # Truncation error
   2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0  # Orbital occupations
   1  0.282408         # Single-site entropy values
   ...
   ```

## How to Modify the Hamiltonian

### Option 1: Replace DMRG Data Files
Generate new data files with the same format but using your modified Hamiltonian in the DMRG calculations.

### Option 2: Create Custom Energy Parser
```python
class CustomHamiltonianProcessor(ImprovedDMRGProcessor):
    def _parse_file(self):
        # First use the original parsing
        super()._parse_file()
        
        # Apply your Hamiltonian modifications
        self.dmrg_energy = my_hamiltonian_function(
            self.dmrg_energy, 
            self.orbital_occupations, 
            self.single_site_entropies
        )
        
        # Update correlation energy
        self.correlation_energy = self.dmrg_energy - self.hf_energy
```

### Option 3: Implement On-the-fly Calculations
Integrate a quantum chemistry library (PySCF, OpenFermion, etc.) to perform Hamiltonian calculations during data loading instead of using pre-computed values.

## Hamiltonian Types You Could Investigate

1. **Modified Hubbard Models**: 
   * Change U/t ratio
   * Add next-nearest neighbor hopping
   * Add site-dependent potentials

2. **Different Exchange-Correlation Functionals**:
   * Different DFT approximations
   * Range-separated functionals 

3. **Extended System-Bath Couplings**:
   * Add environmental coupling terms
   * Simulate open quantum systems

## Files Provided

1. `Hamiltonian_Integration_Guide.md`: Detailed explanation of Hamiltonian integration
2. `Hamiltonian_Integration_Example.py`: Code example showing how to implement custom Hamiltonians
3. This quick guide

## Next Steps

1. Examine one of the DMRG data files to understand the format
2. Try the code example to experiment with energy modifications
3. Consider which Hamiltonian modifications would be most interesting to explore

If you have questions, reach out to the ML team. We'll focus on improving the ML architecture while you explore the quantum chemical aspects. 