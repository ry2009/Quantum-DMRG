import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def read_dmrg_file(file_path):
    """Read comprehensive information from a DMRG file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract energies
    hf_energy = float(lines[0].strip())
    dmrg_energy = float(lines[1].strip())
    correlation_energy = dmrg_energy - hf_energy
    truncation_error = float(lines[2].strip())
    
    # Extract occupations
    occupations = list(map(int, lines[3].strip().split()))
    
    # Extract single-site entropies
    norbs = len(occupations)
    entrop1 = np.zeros(norbs)
    for i in range(norbs):
        parts = lines[4 + i].strip().split()
        orbital_idx = int(parts[0]) - 1  # Convert to 0-indexed
        entropy = float(parts[1])
        entrop1[orbital_idx] = entropy
    
    # Calculate some features
    n_electrons = sum(occupations)
    max_entropy = max(entrop1)
    avg_entropy = np.mean(entrop1)
    
    # Extract system information from the path
    system_info = os.path.basename(os.path.dirname(file_path))
    # Try to extract molecule formula using regex
    formula_match = re.search(r'C(\d+)H(\d+)', system_info)
    if formula_match:
        n_carbon = int(formula_match.group(1))
        n_hydrogen = int(formula_match.group(2))
    else:
        n_carbon = 0
        n_hydrogen = 0
    
    return {
        'file_path': file_path,
        'system_name': system_info,
        'hf_energy': hf_energy,
        'dmrg_energy': dmrg_energy,
        'correlation_energy': correlation_energy,
        'truncation_error': truncation_error,
        'n_orbitals': norbs,
        'n_electrons': n_electrons,
        'max_entropy': max_entropy,
        'avg_entropy': avg_entropy,
        'n_carbon': n_carbon,
        'n_hydrogen': n_hydrogen
    }

def load_predictions(json_path):
    """Load prediction results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

def main():
    # Find all test DMRG files
    dmrg_files = glob.glob('test/*/0512_m')
    
    # Get our model's predictions
    predictions_path = 'results/standard_gnn/standard_gnn_20250306_145007/test_predictions/prediction_results.json'
    predictions = load_predictions(predictions_path)
    
    # Extract the prediction results into a dictionary keyed by file path
    pred_dict = {p['file_path']: p for p in predictions['predictions']}
    
    # Create a table of results
    print("\nDetailed Analysis of Prediction Results")
    print("-" * 125)
    print(f"{'System':<20} {'Corr. Energy':<15} {'Predicted':<15} {'Error':<15} {'Rel Error %':<12} {'#Orbitals':<10} {'#Electrons':<10} {'MaxEntropy':<12} {'TruncErr':<12}")
    print("-" * 125)
    
    # Collect data for visualization
    systems = []
    actual_energies = []
    predicted_energies = []
    errors = []
    rel_errors = []
    n_orbitals = []
    n_electrons = []
    n_carbons = []
    max_entropies = []
    truncation_errors = []
    
    for dmrg_file in dmrg_files:
        # Read DMRG data
        dmrg_data = read_dmrg_file(dmrg_file)
        
        # Get corresponding prediction
        if dmrg_file in pred_dict:
            pred = pred_dict[dmrg_file]
            true_energy = pred['true_energy']
            predicted_energy = pred['predicted_energy']
            error = pred['error']
            rel_error_pct = 100 * abs(error) / abs(true_energy) if abs(true_energy) > 1e-10 else float('inf')
            
            # Add to data collection
            systems.append(dmrg_data['system_name'])
            actual_energies.append(true_energy)
            predicted_energies.append(predicted_energy)
            errors.append(error)
            rel_errors.append(rel_error_pct)
            n_orbitals.append(dmrg_data['n_orbitals'])
            n_electrons.append(dmrg_data['n_electrons'])
            n_carbons.append(dmrg_data['n_carbon'])
            max_entropies.append(dmrg_data['max_entropy'])
            truncation_errors.append(dmrg_data['truncation_error'])
            
            # Print table row
            print(f"{dmrg_data['system_name']:<20} {true_energy:<15.6f} {predicted_energy:<15.6f} {error:<15.6f} {rel_error_pct:<12.2f} {dmrg_data['n_orbitals']:<10d} {dmrg_data['n_electrons']:<10d} {dmrg_data['max_entropy']:<12.4f} {dmrg_data['truncation_error']:<12.8f}")
    
    # Calculate overall metrics
    rmse = np.sqrt(np.mean(np.square(np.array(predicted_energies) - np.array(actual_energies))))
    mae = np.mean(np.abs(np.array(predicted_energies) - np.array(actual_energies)))
    r2 = np.corrcoef(actual_energies, predicted_energies)[0, 1]**2
    
    print("-" * 125)
    print("\nOverall Metrics:")
    print(f"RMSE: {rmse:.6f} Ha")
    print(f"MAE: {mae:.6f} Ha")
    print(f"R²: {r2:.6f}")
    
    # Patterns in the data
    print("\nData Patterns and Observations:")
    
    # Group by molecule type
    molecule_types = {}
    for i, system in enumerate(systems):
        mol_type = system.split("_")[0]  # Extract the base type
        if mol_type not in molecule_types:
            molecule_types[mol_type] = []
        molecule_types[mol_type].append((system, errors[i], rel_errors[i]))
    
    # Print patterns
    print("Performance by Molecule Type:")
    for mol_type, data in molecule_types.items():
        avg_error = np.mean([abs(err) for _, err, _ in data])
        avg_rel_error = np.mean([rel for _, _, rel in data])
        print(f"  {mol_type:<20}: Avg Error: {avg_error:.6f} Ha, Avg Rel Error: {avg_rel_error:.2f}%")
    
    # Check correlation between system size and error
    size_error_corr = np.corrcoef(n_orbitals, np.abs(errors))[0, 1]
    entropy_error_corr = np.corrcoef(max_entropies, np.abs(errors))[0, 1]
    print(f"\nCorrelation between system size and error: {size_error_corr:.4f}")
    print(f"Correlation between max entropy and error: {entropy_error_corr:.4f}")
    
    # Create visualizations
    
    # 1. Bar chart comparing actual vs predicted energies
    plt.figure(figsize=(12, 6))
    x = np.arange(len(systems))
    width = 0.35
    
    plt.bar(x - width/2, actual_energies, width, label='Actual Correlation Energy')
    plt.bar(x + width/2, predicted_energies, width, label='Predicted Correlation Energy')
    
    plt.xlabel('System')
    plt.ylabel('Correlation Energy (Ha)')
    plt.title('Comparison of Actual vs. Predicted Correlation Energies')
    plt.xticks(x, systems, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_comparison.png')
    
    # 2. Error vs system size (number of orbitals)
    plt.figure(figsize=(10, 6))
    plt.scatter(n_orbitals, np.abs(errors), c=rel_errors, cmap='viridis', s=100)
    plt.colorbar(label='Relative Error (%)')
    plt.xlabel('Number of Orbitals')
    plt.ylabel('Absolute Error (Ha)')
    plt.title('Prediction Error vs. System Size')
    for i, system in enumerate(systems):
        plt.annotate(system, (n_orbitals[i], np.abs(errors[i])), 
                     xytext=(5, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('error_vs_size.png')
    
    # 3. Correlation between error and truncation error
    plt.figure(figsize=(10, 6))
    plt.scatter(truncation_errors, np.abs(errors), c=rel_errors, cmap='viridis', s=100)
    plt.colorbar(label='Relative Error (%)')
    plt.xlabel('DMRG Truncation Error')
    plt.ylabel('Absolute Prediction Error (Ha)')
    plt.title('Prediction Error vs. DMRG Truncation Error')
    for i, system in enumerate(systems):
        plt.annotate(system, (truncation_errors[i], np.abs(errors[i])), 
                     xytext=(5, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('error_vs_truncation.png')
    
    # 4. Correlation between error and maximum entropy
    plt.figure(figsize=(10, 6))
    plt.scatter(max_entropies, np.abs(errors), c=rel_errors, cmap='viridis', s=100)
    plt.colorbar(label='Relative Error (%)')
    plt.xlabel('Maximum Orbital Entropy')
    plt.ylabel('Absolute Prediction Error (Ha)')
    plt.title('Prediction Error vs. Maximum Orbital Entropy')
    for i, system in enumerate(systems):
        plt.annotate(system, (max_entropies[i], np.abs(errors[i])), 
                     xytext=(5, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('error_vs_entropy.png')
    
    # 5. Correlation between actual and predicted energies
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_energies, predicted_energies, c=rel_errors, cmap='viridis', s=100)
    plt.colorbar(label='Relative Error (%)')
    
    # Add diagonal line
    min_val = min(min(actual_energies), min(predicted_energies))
    max_val = max(max(actual_energies), max(predicted_energies))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Actual Correlation Energy (Ha)')
    plt.ylabel('Predicted Correlation Energy (Ha)')
    plt.title(f'Correlation between Actual and Predicted Energies (R² = {r2:.4f})')
    for i, system in enumerate(systems):
        plt.annotate(system, (actual_energies[i], predicted_energies[i]), 
                     xytext=(5, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    
    print("\nAnalysis plots saved as PNG files.")

if __name__ == "__main__":
    main() 