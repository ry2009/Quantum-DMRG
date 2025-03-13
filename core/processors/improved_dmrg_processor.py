import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import re
import json
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

class ImprovedDMRGProcessor:
    """
    Enhanced processor for DMRG data files with better graph construction.
    Implements the approach described in the paper for creating quantum-aware orbital graphs.
    """
    
    def __init__(self, file_path, mi_threshold=0.004, system_name=None):
        """
        Initialize the DMRG file processor.
        
        Args:
            file_path: Path to the DMRG data file
            mi_threshold: Threshold for mutual information to create edges
            system_name: Optional name for the molecular system
        """
        self.file_path = file_path
        self.mi_threshold = mi_threshold
        
        # Extract system name and bond dimension from file path if not provided
        if system_name is None:
            # Extract from directory name (assumes path structure like */system_name/bondDim_m)
            self.system_name = os.path.basename(os.path.dirname(file_path))
        else:
            self.system_name = system_name
            
        # Extract bond dimension from filename
        match = re.search(r'(\d+)_m', os.path.basename(file_path))
        if match:
            self.bond_dim = int(match.group(1))
        else:
            self.bond_dim = 0
            print(f"Warning: Could not extract bond dimension from {file_path}")
        
        # Parse the file
        self._parse_file()
        
    def _parse_file(self):
        """Parse the DMRG data file and extract relevant information."""
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
            
            # The files seem to have the energies at the beginning without clear labels
            # Line 1: HF energy, Line 2: DMRG energy
            energy_lines = content.strip().split('\n')[:3]
            
            # Extract HF and DMRG energies - check for format with or without labels
            if len(energy_lines) >= 2:
                # Try to extract labeled energies first
                hf_match = re.search(r'HF energy\s+=\s+(-?\d+\.\d+)', content)
                dmrg_match = re.search(r'DMRG energy\s+=\s+(-?\d+\.\d+)', content)
                
                if hf_match and dmrg_match:
                    # Case 1: Files have labeled energies
                    self.hf_energy = float(hf_match.group(1))
                    self.dmrg_energy = float(dmrg_match.group(1))
                else:
                    # Case 2: Files have energies as first two lines
                    try:
                        # Try to extract the first two numbers as energies
                        energy_values = []
                        for line in energy_lines:
                            line = line.strip()
                            if re.match(r'^-?\d+\.\d+$', line):
                                energy_values.append(float(line))
                            else:
                                # Try to extract number from line with potential other content
                                num_match = re.search(r'(-?\d+\.\d+)', line)
                                if num_match:
                                    energy_values.append(float(num_match.group(1)))
                        
                        if len(energy_values) >= 2:
                            self.hf_energy = energy_values[0]
                            self.dmrg_energy = energy_values[1]
                        else:
                            raise ValueError(f"Could not extract enough energy values from beginning of file")
                    except Exception as e:
                        raise ValueError(f"Could not parse energy values: {e}")
                
                self.correlation_energy = self.dmrg_energy - self.hf_energy
            else:
                raise ValueError(f"Not enough lines in file to extract energies")
            
            # Extract truncation error - check if it's on the third line
            if len(energy_lines) >= 3 and re.match(r'^-?\d+\.\d+e?[+-]?\d*$', energy_lines[2].strip()):
                # Line 3 might be the truncation error
                self.truncation_error = float(energy_lines[2].strip())
            else:
                # Try regular expression search
                tre_match = re.search(r'Truncation error\s+=\s+(-?\d+\.\d+(?:e[+-]\d+)?)', content)
                if tre_match:
                    self.truncation_error = float(tre_match.group(1))
                else:
                    # Last attempt - look for a standalone number after the energies
                    general_tre_match = re.search(r'\n\s*(-?\d+\.\d+e?[+-]?\d*)\s*\n', content)
                    if general_tre_match:
                        self.truncation_error = float(general_tre_match.group(1))
                    else:
                        self.truncation_error = 0.0
                        print(f"Warning: Could not extract truncation error from {self.file_path}")
            
            # Extract orbital occupations - search for a line with many integers or floats
            lines = content.strip().split('\n')
            occ_line = None
            
            for i, line in enumerate(lines):
                # Looking for a line with many numbers separated by spaces
                if re.match(r'^\s*(\d+\.?\d*\s+)+\d+\.?\d*\s*$', line) or re.match(r'^\s*(\d+\s+)+\d+\s*$', line):
                    values = re.findall(r'\d+\.?\d*', line)
                    # If line has many numbers, it's likely the occupations
                    if len(values) > 10:  # Arbitrary threshold for "many"
                        occ_line = line
                        break
            
            if occ_line:
                self.occupations = [float(x) for x in re.findall(r'\d+\.?\d*', occ_line)]
                self.num_orbitals = len(self.occupations)
            else:
                # Try the regular expression pattern
                occ_section = re.search(r'Orbital occupations:\s+([\d\.\s]+)', content)
                if occ_section:
                    occ_str = occ_section.group(1).strip()
                    self.occupations = [float(x) for x in occ_str.split()]
                    self.num_orbitals = len(self.occupations)
                else:
                    raise ValueError(f"Could not extract orbital occupations from {self.file_path}")
            
            # Extract single-site entropy
            # First, try the labeled format
            s1_matches = re.findall(r'S\(1\)\[\s*(\d+)\]\s+=\s+(-?\d+\.\d+(?:e[+-]\d+)?)', content)
            
            if s1_matches:
                # Case 1: Files have labeled entropies
                self.s1 = [0.0] * self.num_orbitals
                for idx_str, val_str in s1_matches:
                    idx = int(idx_str)
                    if idx < self.num_orbitals:
                        self.s1[idx] = float(val_str)
            else:
                # Case 2: Files list entropies by index
                self.s1 = [0.0] * self.num_orbitals
                s1_simple_matches = []
                
                # Look for lines with format "number whitespace number"
                for line in lines:
                    match = re.match(r'^\s*(\d+)\s+(-?\d+\.\d+(?:e[+-]\d+)?)\s*$', line.strip())
                    if match:
                        idx = int(match.group(1)) - 1  # Convert to 0-indexed
                        val = float(match.group(2))
                        if 0 <= idx < self.num_orbitals:
                            self.s1[idx] = val
                            s1_simple_matches.append((idx, val))
                
                if not s1_simple_matches:
                    raise ValueError(f"Could not extract single-site entropy from {self.file_path}")
            
            # Initialize two-site entropy matrix
            self.s2 = np.zeros((self.num_orbitals, self.num_orbitals))
            
            # Try the labeled format first
            s2_matches = re.findall(r'S\(2\)\[\s*(\d+),\s*(\d+)\]\s+=\s+(-?\d+\.\d+(?:e[+-]\d+)?)', content)
            
            if s2_matches:
                # Case 1: Files have labeled two-site entropies
                for i_str, j_str, val_str in s2_matches:
                    i, j = int(i_str), int(j_str)
                    if i < self.num_orbitals and j < self.num_orbitals:
                        self.s2[i, j] = float(val_str)
                        self.s2[j, i] = float(val_str)  # Symmetric matrix
            else:
                # Case 2: Files list two-site entropies in some other format
                # This is tricky without knowing the exact format, so we'll
                # try to infer S2 from context or use defaults
                
                # Two site entropies might be listed as "i,j val" format
                s2_simple_matches = []
                
                for line in lines:
                    match = re.match(r'^\s*(\d+),\s*(\d+)\s+(-?\d+\.\d+(?:e[+-]\d+)?)\s*$', line.strip())
                    if match:
                        i = int(match.group(1)) - 1  # Convert to 0-indexed
                        j = int(match.group(2)) - 1
                        val = float(match.group(3))
                        if 0 <= i < self.num_orbitals and 0 <= j < self.num_orbitals:
                            self.s2[i, j] = val
                            self.s2[j, i] = val  # Symmetric
                            s2_simple_matches.append((i, j, val))
                
                # If we don't find any S2 values, we'll compute a simplistic approximation
                # based on orbital entropies (s1)
                if not s2_simple_matches:
                    print(f"Warning: No two-site entropy found in {self.file_path}, approximating")
                    # Simple approximation: S2[i,j] ~ S1[i] + S1[j] - k*S1[i]*S1[j]
                    # where k is a parameter to ensure mutual information is positive
                    for i in range(self.num_orbitals):
                        for j in range(i+1, self.num_orbitals):
                            # Approximation to ensure mutual info is small but positive
                            self.s2[i, j] = self.s1[i] + self.s1[j] - 0.1 * self.s1[i] * self.s1[j]
                            self.s2[j, i] = self.s2[i, j]
            
            # Compute mutual information: I(i,j) = S(1)[i] + S(1)[j] - S(2)[i,j]
            self.mutual_info = np.zeros((self.num_orbitals, self.num_orbitals))
            for i in range(self.num_orbitals):
                for j in range(self.num_orbitals):
                    if i != j:
                        self.mutual_info[i, j] = max(0, self.s1[i] + self.s1[j] - self.s2[i, j])
            
            # Set diagonal to 0
            np.fill_diagonal(self.mutual_info, 0.0)
            
        except Exception as e:
            print(f"Error parsing {self.file_path}: {e}")
            raise
    
    def to_pyg_data(self):
        """
        Convert parsed DMRG data to a PyTorch Geometric Data object.
        
        Returns:
            data: PyTorch Geometric Data object with node features, edge features, 
                 and the correlation energy as target.
        """
        # Build node features: [single_site_entropy, normalized_occupation]
        normalized_occupations = np.array(self.occupations, dtype=np.float32) / 2.0  # Normalize to [0, 1]
        s1_array = np.array(self.s1, dtype=np.float32)
        
        # Stack features as columns [entropy, occupation]
        x = torch.tensor(np.stack([s1_array, normalized_occupations], axis=1), dtype=torch.float)
        
        # Build edge list using mutual information matrix
        edge_list = []
        edge_features = []
        
        for i in range(self.num_orbitals):
            for j in range(i+1, self.num_orbitals):
                if self.mutual_info[i, j] > self.mi_threshold:
                    # Add both directions for an undirected graph
                    edge_list.append([i, j])
                    edge_features.append([self.mutual_info[i, j]])
                    edge_list.append([j, i])
                    edge_features.append([self.mutual_info[i, j]])
        
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Empty edge case
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        
        # Global features: truncation error (log scale often works better)
        log_tre = np.log10(max(self.truncation_error, 1e-15))  # Avoid log(0)
        global_feature = torch.tensor([[log_tre]], dtype=torch.float)
        
        # Target: correlation energy (to be replaced with delta energy for Δ-ML)
        y = torch.tensor([self.correlation_energy], dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            global_feature=global_feature
        )
        
        # Add additional attributes
        data.bond_dim = torch.tensor([self.bond_dim], dtype=torch.long)
        data.system_name = self.system_name
        data.hf_energy = torch.tensor([self.hf_energy], dtype=torch.float)
        data.dmrg_energy = torch.tensor([self.dmrg_energy], dtype=torch.float)
        
        return data
    
    def visualize_orbital_graph(self, output_file=None):
        """
        Visualize the orbital graph with mutual information edges.
        
        Args:
            output_file: Path to save the visualization image.
                         If None, the image will be displayed.
        """
        G = nx.Graph()
        
        # Add nodes
        for i in range(self.num_orbitals):
            G.add_node(i, entropy=self.s1[i], occupation=self.occupations[i])
        
        # Add edges based on mutual information threshold
        for i in range(self.num_orbitals):
            for j in range(i+1, self.num_orbitals):
                if self.mutual_info[i, j] > self.mi_threshold:
                    G.add_edge(i, j, weight=self.mutual_info[i, j])
        
        # Set up the plot
        plt.figure(figsize=(10, 8))
        
        # Node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on entropy
        node_colors = [G.nodes[i]['entropy'] for i in G.nodes]
        
        # Node sizes based on occupation (scaled for visibility)
        node_sizes = [G.nodes[i]['occupation'] * 100 + 50 for i in G.nodes]
        
        # Edge weights for line thickness
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges]
        
        # Draw the graph
        nx.draw_networkx(
            G, pos, 
            node_color=node_colors, 
            node_size=node_sizes,
            width=edge_weights,
            with_labels=True,
            cmap=plt.cm.viridis
        )
        
        plt.title(f"Orbital Graph: {self.system_name}, Bond Dim = {self.bond_dim}")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis))
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()


class DeltaMLDataset(Dataset):
    """
    Dataset for Δ-ML approach, creating pairs of low and high bond dimension calculations.
    Now supports multiple bond dimension pairs for maximum dataset utilization.
    """
    
    def __init__(self, data_dir, bond_dim_pairs=None, mi_threshold=0.004, transform=None):
        """
        Initialize the Δ-ML dataset.
        
        Args:
            data_dir: Directory containing DMRG data files
            bond_dim_pairs: List of (low_bond_dim, high_bond_dim) tuples to include.
                If None, will automatically find all valid pairs with high > low.
            mi_threshold: Threshold for mutual information to create edges
            transform: PyTorch Geometric transform to apply
        """
        super(DeltaMLDataset, self).__init__(root=None, transform=transform)
        self.data_dir = data_dir
        self.mi_threshold = mi_threshold
        
        # Dictionary to store all available bond dimensions for each system
        system_files = {}
        
        # Scan all files and organize by system and bond dimension
        print("Scanning data directory for DMRG files...")
        for root, dirs, files in os.walk(data_dir):
            # Check if this directory contains DMRG files
            dmrg_files = [f for f in files if f.endswith('_m')]
            if dmrg_files:
                system_name = os.path.basename(root)
                if system_name not in system_files:
                    system_files[system_name] = {}
                
                for file in dmrg_files:
                    try:
                        # Extract bond dimension from filename
                        bond_dim = int(file.split('_')[0])
                        system_files[system_name][bond_dim] = os.path.join(root, file)
                    except (ValueError, IndexError):
                        print(f"Warning: Could not extract bond dimension from {file}")
        
        # Find valid bond dimension pairs for each system
        self.delta_pairs = []
        
        if bond_dim_pairs is None:
            # Auto-detect pairs: for each system, create all valid low→high pairs
            for system, bond_dims in system_files.items():
                bond_dim_list = sorted(bond_dims.keys())
                
                # Create pairs where high > low
                for i, low_dim in enumerate(bond_dim_list[:-1]):
                    for high_dim in bond_dim_list[i+1:]:
                        self.delta_pairs.append((
                            system,
                            low_dim, 
                            high_dim,
                            system_files[system][low_dim],
                            system_files[system][high_dim]
                        ))
        else:
            # Use user-provided bond dimension pairs
            for system, bond_dims in system_files.items():
                for low_dim, high_dim in bond_dim_pairs:
                    if low_dim in bond_dims and high_dim in bond_dims:
                        self.delta_pairs.append((
                            system,
                            low_dim, 
                            high_dim,
                            system_files[system][low_dim],
                            system_files[system][high_dim]
                        ))
        
        print(f"Found {len(self.delta_pairs)} delta pairs across {len(system_files)} systems")
        
        # Process all delta pairs
        self.data_list = []
        for system_name, low_dim, high_dim, low_file, high_file in tqdm(self.delta_pairs, desc="Processing DMRG files"):
            try:
                # Parse low and high bond dimension files
                low_processor = ImprovedDMRGProcessor(low_file, mi_threshold, system_name)
                high_processor = ImprovedDMRGProcessor(high_file, mi_threshold, system_name)
                
                # Create the PyG data object from the low-level calculation
                data = low_processor.to_pyg_data()
                
                # Update the target to be the delta energy (high - low)
                delta_energy = high_processor.dmrg_energy - low_processor.dmrg_energy
                data.y = torch.tensor([delta_energy], dtype=torch.float)
                
                # Store useful metadata
                data.low_bond_dim = torch.tensor([low_dim], dtype=torch.int)
                data.high_bond_dim = torch.tensor([high_dim], dtype=torch.int)
                data.high_dmrg_energy = torch.tensor([high_processor.dmrg_energy], dtype=torch.float)
                data.low_dmrg_energy = torch.tensor([low_processor.dmrg_energy], dtype=torch.float)
                data.high_tre = torch.tensor([high_processor.truncation_error], dtype=torch.float)
                
                # Add the processed data to our list
                self.data_list.append(data)
                
                if len(self.data_list) % 50 == 0:
                    print(f"Processed {len(self.data_list)} delta pairs so far...")
                
            except Exception as e:
                print(f"Error processing {system_name} ({low_dim}→{high_dim}): {e}")
        
        print(f"Dataset contains {len(self.data_list)} examples")
    
    def len(self):
        """Return the number of examples in the dataset."""
        return len(self.data_list)
    
    def get(self, idx):
        """Get a single data example by index."""
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        return data


def process_all_systems(data_dir, output_dir, mi_threshold=0.004):
    """
    Process all DMRG files and save system statistics and visualizations.
    
    Args:
        data_dir: Directory containing DMRG data files
        output_dir: Directory to save outputs
        mi_threshold: Threshold for mutual information
    """
    os.makedirs(output_dir, exist_ok=True)
    graph_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    
    system_stats = []
    
    # Process each DMRG file
    for root, dirs, files in os.walk(data_dir):
        dmrg_files = [f for f in files if f.endswith('_m')]
        if not dmrg_files:
            continue
        
        system_name = os.path.basename(root)
        print(f"\nProcessing system: {system_name}")
        
        for dmrg_file in dmrg_files:
            file_path = os.path.join(root, dmrg_file)
            try:
                processor = ImprovedDMRGProcessor(file_path, mi_threshold)
                
                # Collect system stats
                stats = {
                    "system_name": system_name,
                    "bond_dim": processor.bond_dim,
                    "num_orbitals": processor.num_orbitals,
                    "hf_energy": processor.hf_energy,
                    "dmrg_energy": processor.dmrg_energy,
                    "correlation_energy": processor.correlation_energy,
                    "truncation_error": processor.truncation_error,
                    "avg_entropy": np.mean(processor.s1),
                    "max_entropy": np.max(processor.s1),
                    "avg_occupation": np.mean(processor.occupations)
                }
                
                # Count edges above threshold
                edge_count = np.sum(processor.mutual_info > mi_threshold) // 2  # Divide by 2 for undirected graph
                stats["edge_count"] = edge_count
                stats["edge_density"] = edge_count / (processor.num_orbitals * (processor.num_orbitals - 1) / 2)
                
                system_stats.append(stats)
                
                # Create visualization
                graph_path = os.path.join(graph_dir, f"{system_name}_M{processor.bond_dim}.png")
                processor.visualize_orbital_graph(graph_path)
                
                print(f"  Processed M={processor.bond_dim}: "
                      f"E_corr={processor.correlation_energy:.6f} Ha, "
                      f"TRE={processor.truncation_error:.2e}, "
                      f"Edges={edge_count}")
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
    
    # Save system stats
    stats_file = os.path.join(output_dir, "system_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(system_stats, f, indent=2)
    
    print(f"\nProcessed {len(system_stats)} DMRG files across {len(set(s['system_name'] for s in system_stats))} systems")
    print(f"Results saved to {output_dir}")
    
    return system_stats


if __name__ == "__main__":
    # Example usage
    test_dir = "test"
    train_dir = "train"
    output_dir = "improved_dmrg_analysis"
    
    # Process all systems in test and train directories
    print("Processing test systems...")
    test_stats = process_all_systems(test_dir, os.path.join(output_dir, "test"))
    
    print("\nProcessing train systems...")
    train_stats = process_all_systems(train_dir, os.path.join(output_dir, "train"))
    
    # Example of creating a delta-ML dataset
    print("\nCreating Delta-ML dataset...")
    dataset = DeltaMLDataset(test_dir, bond_dim_pairs=[(256, 3072)], mi_threshold=0.004)
    print(f"Dataset size: {len(dataset)}")
    
    # Example of accessing a data point
    if len(dataset) > 0:
        data = dataset[0]
        print(f"Example data point:")
        print(f"  System: {data.system_name}")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges: {data.edge_index.shape[1] // 2}")  # Divide by 2 for undirected
        print(f"  Delta energy: {data.y.item():.8f} Ha")
        print(f"  Log truncation error: {data.global_feature.item():.4f}") 