# Hamiltonian Research Guide for Delta-ML

## Overview

This guide explains how to modify and experiment with different Hamiltonians in our Delta-ML framework. The code is set up to allow parallel research - while the ML team works on improving the machine learning architecture, you can explore different Hamiltonian models without needing to modify the core codebase.

## Directory Structure

```
hamiltonian_research/
├── README.md                     # This file
├── run_experiment.py             # Main entry point for running experiments
├── docs/                         # Documentation
│   ├── Hamiltonian_Integration_Guide.md  # Detailed technical explanation
│   └── Hamiltonian_Quick_Guide.md        # Quick summary for reference
├── examples/                     # Example Hamiltonian implementations
│   ├── hubbard_model.py          # Hubbard model implementation
│   ├── heisenberg_model.py       # Heisenberg model implementation
│   ├── combined_model.py         # Combined Hubbard-Heisenberg model
│   └── custom_model_template.py  # Template for creating your own models
└── src/                          # Source code
    ├── __init__.py               # Package initialization
    └── hamiltonian_plugin.py     # Main implementation
```

## Getting Started (5-minute setup)

1. **Understand the basics**: Our Delta-ML method predicts the energy difference between low and high bond dimension DMRG calculations. The Hamiltonian is used in the upstream DMRG calculations that generate our data files.

2. **Examine a data file**:
   ```bash
   head -n 10 test/1.C22H12_triangulene_s0/0256_m
   ```
   You'll see the format:
   ```
   -840.12276929       # HF energy
   -840.35297292       # DMRG energy
   0.00051949409500    # Truncation error
   2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0  # Orbital occupations
   1  0.282408         # Single-site entropy (orbital index, entropy value)
   2  0.384339
   ...
   ```

3. **Run an experiment with a pre-built Hamiltonian model**:
   ```bash
   python run_experiment.py --model combined --U 4.0 --J 0.15 --output combined_model_results
   ```
   This will:
   - Load your DMRG data
   - Apply the combined Hubbard-Heisenberg model
   - Train a Delta-ML model
   - Evaluate and save results to `combined_model_results/`

## Available Models

We provide several ready-to-use Hamiltonian models:

1. **Hubbard Model** (`--model hubbard`):
   - Parameters: `--U`, `--t`
   - Description: On-site electronic interactions

2. **Heisenberg Model** (`--model heisenberg`):
   - Parameters: `--J`, `--anisotropy`, `--next_nearest`
   - Description: Spin exchange interactions between sites

3. **Combined Model** (`--model combined`):
   - Parameters: All of the above plus `--hubbard_weight`, `--heisenberg_weight`
   - Description: Combination of Hubbard and Heisenberg models

4. **Custom Model** (`--model custom`):
   - Uses the default model in src/hamiltonian_plugin.py

## Creating Your Own Hamiltonian Model

You can easily create your own Hamiltonian model:

1. **Copy the template**:
   ```bash
   cp examples/custom_model_template.py examples/my_model.py
   ```

2. **Edit your model**:
   Modify the `custom_hamiltonian` function to implement your physics.

3. **Use your model**:
   ```python
   # In src/hamiltonian_plugin.py
   from hamiltonian_research.examples.my_model import custom_hamiltonian
   HAMILTONIAN_MODEL = custom_hamiltonian
   ```

## Analyzing Results

After running an experiment, check the output directory for:

1. **Training performance**: Review `training_curves.png`
2. **Prediction accuracy**: Examine `predictions.png`
3. **Test metrics**: See `test_results.json`
4. **Hamiltonian analysis**: Look at the Hamiltonian-specific plots:
   - `correction_distribution.png` - How your corrections are distributed
   - `correction_vs_bond_dim.png` - Correction vs. bond dimension
   - `correction_vs_energy.png` - Relationship between corrections and energy differences
5. **Raw data**: Examine `hamiltonian_corrections.csv` for detailed analysis

## Advanced Usage

For more complex Hamiltonian implementations:

1. **Add new parameters** to the run_experiment.py and pass them to your model
2. **Implement complex physics** by creating new functions in your model file
3. **Extend the analysis** by modifying the analysis functions in src/hamiltonian_plugin.py

## Documentation

For more details, see the documentation in the `docs/` directory:

- `Hamiltonian_Integration_Guide.md`: Detailed technical explanation
- `Hamiltonian_Quick_Guide.md`: Quick summary for reference

## Questions?

If you have questions or need help implementing a specific Hamiltonian, let us know. While we focus on improving the ML side, we're happy to help with the Hamiltonian integration.

Happy researching! 