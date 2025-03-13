import os
import sys
from tqdm import tqdm

# Add helpful debug color printing
def print_color(text, color="white"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "white": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['white']}")

def find_available_bond_dims(data_dir):
    """Find all systems and their available bond dimensions."""
    results = {}
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print_color(f"Error: Directory '{data_dir}' does not exist!", "red")
        return results
    
    for root, dirs, files in os.walk(data_dir):
        # Filter for DMRG files
        dmrg_files = [f for f in files if f.endswith('_m')]
        if not dmrg_files:
            continue
        
        # Extract system name and bond dimensions
        system_name = os.path.basename(root)
        bond_dims = []
        
        for file in dmrg_files:
            try:
                bond_dim = int(file.split('_')[0])
                bond_dims.append(bond_dim)
            except:
                pass
        
        if bond_dims:
            results[system_name] = sorted(bond_dims)
    
    return results

def test_file_parsing(file_path):
    """Try to parse a DMRG file and report success/failure."""
    try:
        # First check if we can import the module
        try:
            from improved_dmrg_processor import ImprovedDMRGProcessor
        except ImportError:
            print_color("  Error: Cannot import ImprovedDMRGProcessor. Make sure the file exists in the current directory.", "red")
            return False
        
        processor = ImprovedDMRGProcessor(file_path)
        print_color("  Success parsing file!", "green")
        print(f"    HF Energy: {processor.hf_energy}")
        print(f"    DMRG Energy: {processor.dmrg_energy}")
        print(f"    Truncation Error: {processor.truncation_error}")
        print(f"    Orbitals: {processor.num_orbitals}")
        print(f"    Edges (MI > 0.004): {sum(processor.mutual_info > 0.004) // 2}")
        return True
    except Exception as e:
        print_color(f"  Failed to parse file: {e}", "red")
        
        # Try to open the file and print some content for debugging
        try:
            with open(file_path, 'r') as f:
                content = f.read(500)  # First 500 chars
            print_color("  First 500 characters of file:", "yellow")
            print("    " + content.replace("\n", "\n    "))
        except Exception as read_err:
            print_color(f"  Could not even read file: {read_err}", "red")
        
        return False

def main():
    # Check current directory and Python environment
    print_color("Working directory:", "blue")
    print(f"  {os.getcwd()}")
    
    print_color("\nPython environment:", "blue")
    print(f"  Python version: {sys.version}")
    print(f"  Modules available: os, sys, tqdm")
    
    # Check if directories exist
    print_color("\nChecking directories:", "blue")
    for directory in ["test", "train"]:
        if os.path.exists(directory):
            print_color(f"  ✓ {directory} directory exists", "green")
            # Count subdirectories that might contain DMRG files
            subdirs = [d for d in os.listdir(directory) 
                      if os.path.isdir(os.path.join(directory, d)) and 
                      not d.startswith('.')]
            print(f"    Found {len(subdirs)} potential system directories")
        else:
            print_color(f"  ✗ {directory} directory not found!", "red")
    
    # Check available bond dimensions
    print_color("\nAvailable bond dimensions in test directory:", "blue")
    test_results = find_available_bond_dims("test")
    if not test_results:
        print_color("  No systems with DMRG files found in test directory!", "red")
    else:
        for system, dims in test_results.items():
            print(f"  {system}: {dims}")
    
    print_color("\nAvailable bond dimensions in train directory:", "blue")
    train_results = find_available_bond_dims("train")
    if not train_results:
        print_color("  No systems with DMRG files found in train directory!", "red")
    else:
        for system, dims in train_results.items():
            print(f"  {system}: {dims}")
    
    # Find possible pairs for Delta-ML
    print_color("\nPossible Δ-ML pairs:", "blue")
    pair_found = False
    
    for low_dim in [128, 256, 512]:
        for high_dim in [1024, 2048, 3072]:
            if low_dim >= high_dim:
                continue
                
            # Count systems with both dimensions
            test_count = sum(1 for dims in test_results.values() if low_dim in dims and high_dim in dims)
            train_count = sum(1 for dims in train_results.values() if low_dim in dims and high_dim in dims)
            
            if test_count > 0 or train_count > 0:
                pair_found = True
                if test_count > 0 and train_count > 0:
                    print_color(f"  ✓ M={low_dim} → M={high_dim}: {test_count} test systems, {train_count} train systems", "green")
                else:
                    print(f"  M={low_dim} → M={high_dim}: {test_count} test systems, {train_count} train systems")
    
    if not pair_found:
        print_color("  No valid bond dimension pairs found for Delta-ML!", "red")
    
    # Test file parsing on a few example files
    print_color("\nTesting file parsing on sample files:", "blue")
    parsing_success = 0
    parsing_failure = 0
    
    for data_dir in ["test", "train"]:
        if not os.path.exists(data_dir):
            continue
            
        for root, dirs, files in os.walk(data_dir):
            dmrg_files = [f for f in files if f.endswith('_m')]
            if not dmrg_files:
                continue
            
            # Get the first file to test
            system_name = os.path.basename(root)
            file_path = os.path.join(root, dmrg_files[0])
            
            print(f"\nSystem {system_name}:")
            if test_file_parsing(file_path):
                parsing_success += 1
            else:
                parsing_failure += 1
                
            if parsing_success + parsing_failure >= 3:  # Limit to testing a few files
                break
        
        if parsing_success + parsing_failure >= 3:
            break
    
    # Print summary and recommendations
    print_color("\nSummary:", "blue")
    print(f"  Successfully parsed: {parsing_success} files")
    print(f"  Failed to parse: {parsing_failure} files")
    
    if parsing_failure > 0:
        print_color("\nPossible issues:", "yellow")
        print("  1. The DMRG files don't match the expected format")
        print("  2. The regular expressions in ImprovedDMRGProcessor might need adjustment")
        print("  3. There could be encoding issues with the files")
    
    print_color("\nRecommended changes:", "blue")
    if pair_found:
        # Find the best pair that has the most data
        best_pair = (256, 1024)  # Default
        best_count = 0
        
        for low_dim in [128, 256, 512]:
            for high_dim in [1024, 2048, 3072]:
                if low_dim >= high_dim:
                    continue
                    
                count = sum(1 for dims in test_results.values() if low_dim in dims and high_dim in dims) + \
                        sum(1 for dims in train_results.values() if low_dim in dims and high_dim in dims)
                
                if count > best_count:
                    best_count = count
                    best_pair = (low_dim, high_dim)
        
        print_color(f"  1. Update bond dimensions in the code to: low_bond_dim={best_pair[0]}, high_bond_dim={best_pair[1]}", "green")
    else:
        print_color("  1. Check that your DMRG files are named correctly (should end with _m)", "yellow")
        print_color("  2. Verify that the directory structure matches what the code expects", "yellow")
    
    print("  3. If file parsing is failing, inspect the files and update the parsing code as needed")

if __name__ == "__main__":
    main() 