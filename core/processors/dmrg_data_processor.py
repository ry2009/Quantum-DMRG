import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import json
import matplotlib.pyplot as plt
import re
import argparse

class DMRGDataset(Dataset):
    """
    Dataset for DMRG data.
    """
    def __init__(self, root_dir, bond_dims=[512], system_type='pah', max_orbitals=100, mi_threshold=0.01, transform=None, pre_transform=None, predict_total_energy=True):
        """
        Args:
            root_dir (str): Directory with all the DMRG data.
            bond_dims (list): List of bond dimensions to include.
            system_type (str): Type of molecular system.
            max_orbitals (int): Maximum number of orbitals to consider.
            mi_threshold (float): Threshold for mutual information to create edges.
            transform (callable, optional): Optional transform to be applied on a sample.
            pre_transform (callable, optional): Optional pre-transform to be applied on a sample.
            predict_total_energy (bool): Flag to choose prediction target.
        """
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.bond_dims = bond_dims
        self.system_type = system_type
        self.max_orbitals = max_orbitals
        self.mi_threshold = mi_threshold
        self.predict_total_energy = predict_total_energy
        self._indices = None
        
        # Process data
        self.data_list = []
        self._process_data()
        
    def _process_data(self):
        """Process DMRG data files."""
        print(f"Processing DMRG data from {self.root_dir}...")
        
        # Walk through the directory structure
        for root, dirs, files in os.walk(self.root_dir):
            for bond_dim in self.bond_dims:
                # Look for files with the specified bond dimension
                bond_dim_str = f"{bond_dim:04d}_m"
                matching_files = [f for f in files if bond_dim_str in f]
                
                for file_name in matching_files:
                    file_path = os.path.join(root, file_name)
                    
                    try:
                        # Read DMRG data
                        dmrg_data = self.read_dmrg_file(file_path)
                        
                        # Skip if number of orbitals exceeds max_orbitals
                        if dmrg_data['norbs'] > self.max_orbitals:
                            continue
                        
                        # Create graph data
                        graph_data = self.create_graph_data(dmrg_data)
                        
                        # Add to data list
                        self.data_list.append(graph_data)
            
        except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
        
        print(f"Processed {len(self.data_list)} DMRG data files.")
        
    @property
    def raw_file_names(self):
        # Not using the raw_file_names functionality of PyG
        return []
        
    @property
    def processed_file_names(self):
        # Not using the processed_file_names functionality of PyG
        return []
        
    def process(self):
        # We handle processing in _process_data
        pass
        
    def download(self):
        # No download needed
        pass
    
    def read_dmrg_file(self, file_path):
        """Read DMRG data from file."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Extract basic information
        hf_energy = float(lines[0].strip())
        dmrg_energy = float(lines[1].strip())
        truncation_error = float(lines[2].strip())
        
        # Extract occupation numbers
        occupations = list(map(int, lines[3].strip().split()))
        norbs = len(occupations)
        
        # Extract single-site entropies
        entrop1 = np.zeros(norbs)
        for i in range(norbs):
            parts = lines[4 + i].strip().split()
            orbital_idx = int(parts[0]) - 1  # Convert to 0-indexed
            entropy = float(parts[1])
            entrop1[orbital_idx] = entropy
        
        # Extract two-site entropies
        entrop2 = np.zeros((norbs, norbs))
        line_idx = 4 + norbs
        
        while line_idx < len(lines):
            parts = lines[line_idx].strip().split()
            if len(parts) >= 3:
                i = int(parts[0]) - 1  # Convert to 0-indexed
                j = int(parts[1]) - 1  # Convert to 0-indexed
                entropy = float(parts[2])
                entrop2[i, j] = entropy
                entrop2[j, i] = entropy  # Symmetric
            line_idx += 1
        
        # Calculate mutual information
        mutual_info = np.zeros((norbs, norbs))
        for i in range(norbs):
            for j in range(i+1, norbs):
                mutual_info[i, j] = 0.5 * (entrop1[i] + entrop1[j] - entrop2[i, j])
                mutual_info[j, i] = mutual_info[i, j]  # Symmetric
        
        return {
            'norbs': norbs,
            'hf_energy': hf_energy,
            'dmrg_energy': dmrg_energy,
            'truncation_error': truncation_error,
            'occupations': occupations,
            'entrop1': entrop1,
            'entrop2': entrop2,
            'mutual_info': mutual_info,
            'file_path': file_path
        }
    
    def create_graph_data(self, dmrg_data):
        """Create graph data from DMRG data."""
        norbs = dmrg_data['norbs']
        
        # Create node features: occupation and single-site entropy
        x = torch.zeros((norbs, 2))
        for i in range(norbs):
            x[i, 0] = dmrg_data['occupations'][i]
            x[i, 1] = dmrg_data['entrop1'][i]
        
        # Create edges based on mutual information threshold
        edge_list = []
        edge_attr_list = []
        
        for i in range(norbs):
            for j in range(i+1, norbs):
                mi = dmrg_data['mutual_info'][i, j]
                if mi > self.mi_threshold:
                    # Add edge i -> j
                    edge_list.append([i, j])
                    edge_attr_list.append([mi])
                    
                    # Add edge j -> i (undirected graph)
                    edge_list.append([j, i])
                    edge_attr_list.append([mi])
        
        # If no edges meet the threshold, add minimal connectivity
        if len(edge_list) == 0:
            print(f"Warning: No edges meet the threshold for file {dmrg_data['file_path']}. Adding minimal connectivity.")
            for i in range(norbs - 1):
                # Add edge i -> i+1
                edge_list.append([i, i+1])
                edge_attr_list.append([0.0])
                
                # Add edge i+1 -> i
                edge_list.append([i+1, i])
                edge_attr_list.append([0.0])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        
        # Target: correlation energy or total energy
        if self.predict_total_energy:
            y = torch.tensor([[dmrg_data['dmrg_energy']]], dtype=torch.float)
        else:
            y = torch.tensor([[dmrg_data['dmrg_energy'] - dmrg_data['hf_energy']]], dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            norbs=norbs,
            hf_energy=dmrg_data['hf_energy'],
            dmrg_energy=dmrg_data['dmrg_energy'],
            truncation_error=dmrg_data['truncation_error'],
            file_path=dmrg_data['file_path']
        )
        
        return data
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    def indices(self):
        if self._indices is None:
            return range(len(self.data_list))
        else:
            return self._indices
    
    def visualize(self, idx):
        """Visualize the graph for a specific data point."""
        data = self.data_list[idx]
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(data.x.size(0)):
            G.add_node(i, occupation=data.x[i, 0].item(), entropy=data.x[i, 1].item())
        
        # Add edges
        for j in range(data.edge_index.size(1)):
            src, dst = data.edge_index[0, j].item(), data.edge_index[1, j].item()
            if src < dst:  # Only add one direction for visualization
                G.add_edge(src, dst, coupling=data.edge_attr[j, 0].item())
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on occupation
        node_colors = [G.nodes[i]['occupation'] for i in G.nodes]
        
        # Node sizes based on entropy
        node_sizes = [5000 * G.nodes[i]['entropy'] + 100 for i in G.nodes]
        
        # Edge widths based on mutual information
        edge_widths = [5 * G.edges[e]['coupling'] for e in G.edges]
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes,
                width=edge_widths, edge_color='gray', cmap=plt.cm.viridis)
        
        # Add title with energy information
        if self.predict_total_energy:
            plt.title(f"DMRG Energy: {data.y.item():.6f} Ha")
                else:
            plt.title(f"Correlation Energy: {data.y.item():.6f} Ha")
        
        # Add colorbar for occupation
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Orbital Occupation')
        
        plt.tight_layout()
        plt.show()
        
        # Print additional information
        print(f"File: {data.file_path}")
        print(f"Number of orbitals: {data.norbs}")
        print(f"HF Energy: {data.hf_energy:.8f} Ha")
        print(f"DMRG Energy: {data.dmrg_energy:.8f} Ha")
        print(f"Truncation Error: {data.truncation_error:.8e}")
        
        return G

def main():
    # Process training data
    print("Processing training data...")
    train_processor = DMRGDataset('train', system_type='nitrogen', predict_total_energy=True)
    train_data, train_metadata = train_processor.process_all()
    train_processor.save_metadata('train_metadata.json')
    print(f"Processed {len(train_data)} training datapoints")
    print(f"Found {train_metadata['total_systems']} training systems")
    
    # Process test data
    print("\nProcessing test data...")
    test_processor = DMRGDataset('test', system_type='nitrogen', predict_total_energy=True)
    test_data, test_metadata = test_processor.process_all()
    test_processor.save_metadata('test_metadata.json')
    print(f"Processed {len(test_data)} test datapoints")
    print(f"Found {test_metadata['total_systems']} test systems")
    
    # Print some statistics
    print("\nTraining data statistics:")
    print(f"Bond dimensions found: {train_metadata['bond_dimensions']}")
    print("\nTest data statistics:")
    print(f"Bond dimensions found: {test_metadata['bond_dimensions']}")
    
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = main() 