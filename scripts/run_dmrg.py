import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime
import multiprocessing
from typing import Optional, List, Tuple
import json

def backup_file(file_path: Path) -> Path:
    """
    Create a backup of a file with timestamp.
    
    Args:
        file_path: Path to file to backup
        
    Returns:
        Path to backup file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path.parent / f"{file_path.name}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def read_dmrg_file(file_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Read a DMRG file and separate its contents into header, single-site, and two-site data.
    
    Returns:
        Tuple of (header_lines, single_site_lines, two_site_lines)
    """
    with open(file_path) as f:
        lines = [line.rstrip() for line in f.readlines()]
    
    # First 4 lines are always header
    header = lines[:4]
    
    # Find where two-site entropies start (first line with 3 values)
    two_site_start = 4
    for i, line in enumerate(lines[4:], start=4):
        parts = line.strip().split()
        if len(parts) == 3:
            two_site_start = i
            break
    
    single_site = lines[4:two_site_start]
    two_site = lines[two_site_start:]
    
    return header, single_site, two_site

def process_single_dir(dir_path: Path) -> bool:
    """Process a single directory to update its 4096_m file with entropy data from 0512_m."""
    try:
        # Check if we have both required files
        m4096_path = dir_path / "4096_m"
        m0512_path = dir_path / "0512_m"
        
        if not (m4096_path.exists() and m0512_path.exists()):
            return False
        
        # Read both files
        m4096_header, _, _ = read_dmrg_file(m4096_path)
        _, m0512_single, m0512_two = read_dmrg_file(m0512_path)
        
        # Combine data
        output_lines = m4096_header + m0512_single + m0512_two
        
        # Write updated file
        with open(m4096_path, 'w') as f:
            f.write('\n'.join(output_lines) + '\n')
        
        print(f"Updated {m4096_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {dir_path}: {e}")
        return False

def process_directory(directory: str, max_parallel: int = None):
    """Process all directories to update 4096_m files with entropy data from 0512_m."""
    base_dir = Path(directory)
    
    # First, backup all 4096_m files
    print("Creating backups of existing 4096_m files...")
    backups = []
    for root, _, files in os.walk(base_dir):
        if "4096_m" in files:
            file_path = Path(root) / "4096_m"
            backup_path = backup_file(file_path)
            backups.append(backup_path)
    print(f"Created {len(backups)} backups")
    
    # Collect all directories with 4096_m files
    dirs_to_process = []
    for root, _, files in os.walk(base_dir):
        if "4096_m" in files and "0512_m" in files:
            dirs_to_process.append(Path(root))
    
    if not dirs_to_process:
        print("No directories found with both 4096_m and 0512_m files")
        return
    
    print(f"\nFound {len(dirs_to_process)} directories to process")
    
    # Process directories in parallel
    if max_parallel is None:
        max_parallel = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {max_parallel} parallel processes")
    
    with multiprocessing.Pool(max_parallel) as pool:
        results = pool.map(process_single_dir, dirs_to_process)
    
    # Report results
    successful = sum(1 for r in results if r)
    print(f"\nSuccessfully processed {successful}/{len(dirs_to_process)} directories")

def main():
    parser = argparse.ArgumentParser(description="Update 4096_m files with entropy data from 0512_m files")
    parser.add_argument("directory", help="Directory containing DMRG data")
    parser.add_argument("--max-parallel", type=int, help="Maximum number of parallel processes")
    args = parser.parse_args()
    
    process_directory(args.directory, args.max_parallel)

if __name__ == "__main__":
    main() 