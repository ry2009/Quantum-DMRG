import os
from pathlib import Path

def analyze_directory(base_path):
    """Analyze the contents of the training directory structure."""
    base_path = Path(base_path)
    
    # Statistics
    total_folders = 0
    total_files = 0
    bond_dims_found = set()
    systems_by_type = {}
    
    # Walk through all directories
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        
        # Skip the base directory itself
        if root_path == base_path:
            continue
            
        # Count folders
        if dirs:
            total_folders += len(dirs)
            
        # Look for bond dimension files
        for file in files:
            if file.startswith('0') and file.endswith('_m'):
                total_files += 1
                bond_dims_found.add(file)
                
                # Categorize system
                parent_dir = root_path.parent.name
                if parent_dir not in systems_by_type:
                    systems_by_type[parent_dir] = set()
                systems_by_type[parent_dir].add(root_path.name)
    
    # Print results
    print(f"\nData Analysis Results:")
    print(f"=====================")
    print(f"Total folders found: {total_folders}")
    print(f"Total bond dimension files: {total_files}")
    print(f"\nBond dimensions found: {sorted(list(bond_dims_found))}")
    
    print(f"\nSystems by type:")
    for system_type, systems in systems_by_type.items():
        print(f"\n{system_type}:")
        print(f"- Number of systems: {len(systems)}")
        print(f"- Example systems: {sorted(list(systems))[:5]}")

if __name__ == "__main__":
    analyze_directory('train') 