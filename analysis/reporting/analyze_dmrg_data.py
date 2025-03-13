import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from glob import glob
import pandas as pd
import torch
from torch_geometric.data import Data

class DMRGFileParser:
    """Parser for DMRG data files containing energy, entropy, and orbital information."""
    
    def __init__(self, file_path, mi_threshold=0.004):
        """
        Initialize the DMRG file parser.
        
        Args:
            file_path (str): Path to the DMRG data file
            mi_threshold (float): Threshold for mutual information to create edges in the graph
        """
        self.file_path = file_path
        self.mi_threshold = mi_threshold
        self.hf_energy = None
        self.dmrg_energy = None
        self.truncation_error = None
        self.orbital_occupations = None
        self.single_site_entropies = {}
        self.two_site_entropies = {}
        self.mutual_information = {}
        self.num_orbitals = 0
        
        # Extract system and bond dimension from filename
        self.system_name = os.path.basename(os.path.dirname(file_path))
        self.bond_dim = int(os.path.basename(file_path).split('_')[0])
        
        # Parse the file
        self._parse_file()
        
    def _parse_file(self):
        """Parse the DMRG data file to extract all relevant information."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            
        # Extract energies and truncation error
        self.hf_energy = float(lines[0].strip())
        self.dmrg_energy = float(lines[1].strip())
        self.truncation_error = float(lines[2].strip())
        
        # Extract orbital occupations
        self.orbital_occupations = [int(x) for x in lines[3].strip().split()]
        self.num_orbitals = len(self.orbital_occupations)
        
        # Extract single-site entropies
        line_idx = 4
        for i in range(self.num_orbitals):
            orbital_idx, entropy = lines[line_idx].strip().split()
            orbital_idx = int(orbital_idx)
            entropy = float(entropy)
            self.single_site_entropies[orbital_idx] = entropy
            line_idx += 1
        
        # Extract two-site entropies and compute mutual information
        while line_idx < len(lines) and lines[line_idx].strip():
            parts = lines[line_idx].strip().split()
            if len(parts) == 3:
                i, j, entropy = parts
                i = int(i)
                j = int(j)
                entropy = float(entropy)
                self.two_site_entropies[(i, j)] = entropy
                
                # Compute mutual information
                mi = self.single_site_entropies[i] + self.single_site_entropies[j] - entropy
                self.mutual_information[(i, j)] = mi
            line_idx += 1
    
    def get_correlation_energy(self):
        """Calculate the correlation energy (difference between DMRG and HF energies)."""
        return self.dmrg_energy - self.hf_energy
    
    def build_orbital_graph(self):
        """
        Build a NetworkX graph from orbital data where:
        - Nodes are orbitals with single-site entropy as node attribute
        - Edges are connections between orbitals with mutual info above threshold
        """
        G = nx.Graph()
        
        # Add nodes with attributes
        for i in range(1, self.num_orbitals + 1):
            G.add_node(i, 
                       entropy=self.single_site_entropies[i],
                       occupation=self.orbital_occupations[i-1])
        
        # Add edges with mutual information above threshold
        for (i, j), mi in self.mutual_information.items():
            if mi > self.mi_threshold:
                G.add_edge(i, j, mutual_info=mi)
        
        return G
    
    def to_pyg_data(self):
        """
        Convert the orbital data to a PyTorch Geometric Data object for GNN processing.
        
        Returns:
            Data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, num_node_features]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, num_edge_features]
                - y: Target value (correlation energy or energy difference)
        """
        # Node features: [single-site entropy, orbital occupation]
        node_features = []
        for i in range(1, self.num_orbitals + 1):
            node_features.append([
                self.single_site_entropies[i],
                self.orbital_occupations[i-1]
            ])
        
        # Create edge list and edge features from mutual information pairs above threshold
        edge_index = []
        edge_attr = []
        
        for (i, j), mi in self.mutual_information.items():
            if mi > self.mi_threshold:
                # Convert to 0-indexed for PyG
                edge_index.append([i-1, j-1])
                edge_index.append([j-1, i-1])  # Add both directions for undirected graph
                
                # Edge features: mutual information
                edge_attr.append([mi])
                edge_attr.append([mi])  # Same feature for both directions
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Target value: correlation energy
        y = torch.tensor([[self.get_correlation_energy()]], dtype=torch.float)
        
        # Also include truncation error as a global feature
        global_feature = torch.tensor([[self.truncation_error]], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.global_feature = global_feature
        
        return data


def analyze_dmrg_files(directory_path, mi_threshold=0.004, output_dir="plots"):
    """
    Analyze all DMRG files in the given directory and generate summary visualizations.
    
    Args:
        directory_path (str): Path to directory containing DMRG files
        mi_threshold (float): Threshold for mutual information
        output_dir (str): Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all DMRG files
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('_m'):
                file_paths.append(os.path.join(root, file))
    
    print(f"Found {len(file_paths)} DMRG data files")
    
    # Parse all files
    parsed_files = []
    for file_path in file_paths:
        try:
            parser = DMRGFileParser(file_path, mi_threshold)
            parsed_files.append(parser)
            print(f"Processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Analyze energies across bond dimensions
    systems = {}
    for parser in parsed_files:
        system_name = parser.system_name
        if system_name not in systems:
            systems[system_name] = []
        systems[system_name].append((parser.bond_dim, parser.dmrg_energy, parser.hf_energy, parser.get_correlation_energy()))
    
    # For each system, plot energy vs bond dimension
    plt.figure(figsize=(12, 10))
    for system, values in systems.items():
        values.sort()  # Sort by bond dimension
        bond_dims, dmrg_energies, _, _ = zip(*values)
        plt.plot(bond_dims, dmrg_energies, 'o-', label=system)
    
    plt.xlabel('Bond Dimension (M)')
    plt.ylabel('DMRG Energy (Ha)')
    plt.title('DMRG Energy vs Bond Dimension')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_vs_bond_dim.png'))
    plt.close()
    
    # Analyze graph properties
    graph_stats = []
    for parser in parsed_files:
        G = parser.build_orbital_graph()
        stats = {
            'system': parser.system_name,
            'bond_dim': parser.bond_dim,
            'num_orbitals': parser.num_orbitals,
            'num_edges': G.number_of_edges(),
            'avg_degree': np.mean([d for _, d in G.degree()]),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'avg_mi': np.mean([data['mutual_info'] for _, _, data in G.edges(data=True)]),
            'corr_energy': parser.get_correlation_energy(),
            'truncation_error': parser.truncation_error
        }
        graph_stats.append(stats)
    
    # Convert to pandas DataFrame for analysis
    df = pd.DataFrame(graph_stats)
    print("\nSummary statistics:")
    print(df.describe())
    
    # Save statistics to CSV
    df.to_csv(os.path.join(output_dir, 'graph_statistics.csv'), index=False)
    
    # Plot graph properties
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(df['bond_dim'], df['num_edges'], alpha=0.7)
    plt.xlabel('Bond Dimension (M)')
    plt.ylabel('Number of Edges')
    plt.title('Number of Edges vs Bond Dimension')
    
    plt.subplot(2, 2, 2)
    plt.scatter(df['bond_dim'], df['avg_mi'], alpha=0.7)
    plt.xlabel('Bond Dimension (M)')
    plt.ylabel('Average Mutual Information')
    plt.title('Average MI vs Bond Dimension')
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['truncation_error'], df['corr_energy'], alpha=0.7)
    plt.xlabel('Truncation Error')
    plt.ylabel('Correlation Energy (Ha)')
    plt.title('Correlation Energy vs Truncation Error')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['num_edges'], df['corr_energy'], alpha=0.7)
    plt.xlabel('Number of Edges')
    plt.ylabel('Correlation Energy (Ha)')
    plt.title('Correlation Energy vs Number of Edges')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_property_analysis.png'))
    plt.close()
    
    return parsed_files, df


def visualize_orbital_graph(parser, output_dir="plots"):
    """
    Visualize the orbital graph from a DMRG file parser.
    
    Args:
        parser (DMRGFileParser): Parser containing orbital graph data
        output_dir (str): Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    G = parser.build_orbital_graph()
    
    # Node colors based on orbital occupation
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['occupation'] == 2:
            node_colors.append('red')  # Doubly occupied
        elif G.nodes[node]['occupation'] == 1:
            node_colors.append('blue')  # Singly occupied
        else:
            node_colors.append('lightgray')  # Unoccupied
    
    # Edge widths based on mutual information
    edge_widths = [G[u][v]['mutual_info'] * 20 for u, v in G.edges()]
    
    # Node sizes based on single-site entropy
    node_sizes = [G.nodes[node]['entropy'] * 500 for node in G.nodes()]
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 10))
    nx.draw_networkx(
        G, pos=pos, 
        node_color=node_colors, 
        node_size=node_sizes,
        width=edge_widths,
        with_labels=True,
        font_size=10,
        font_weight='bold',
        alpha=0.8
    )
    
    plt.title(f"Orbital Graph for {parser.system_name} (M = {parser.bond_dim})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{parser.system_name}_M{parser.bond_dim}_graph.png"))
    plt.close()


def delta_ml_analysis(files, output_dir="plots"):
    """
    Analyze the potential for Delta-ML by comparing different bond dimensions.
    
    Args:
        files (list): List of DMRGFileParser objects
        output_dir (str): Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group files by system
    systems = {}
    for parser in files:
        system_name = parser.system_name
        if system_name not in systems:
            systems[system_name] = []
        systems[system_name].append(parser)
    
    # For each system, analyze energy differences between bond dimensions
    delta_data = []
    for system_name, parsers in systems.items():
        # Sort parsers by bond dimension
        parsers.sort(key=lambda x: x.bond_dim)
        
        # If we have multiple bond dimensions, compute energy differences
        if len(parsers) > 1:
            for i in range(len(parsers) - 1):
                low_parser = parsers[i]
                high_parser = parsers[i+1]
                
                delta = high_parser.dmrg_energy - low_parser.dmrg_energy
                
                delta_data.append({
                    'system': system_name,
                    'low_bond_dim': low_parser.bond_dim,
                    'high_bond_dim': high_parser.bond_dim,
                    'low_energy': low_parser.dmrg_energy,
                    'high_energy': high_parser.dmrg_energy,
                    'delta_energy': delta,
                    'low_tre': low_parser.truncation_error,
                    'high_tre': high_parser.truncation_error,
                    'ratio_tre': high_parser.truncation_error / low_parser.truncation_error
                })
    
    if not delta_data:
        print("No systems with multiple bond dimensions found for Delta-ML analysis")
        return
    
    # Convert to DataFrame and analyze
    df = pd.DataFrame(delta_data)
    print("\nDelta-ML Analysis:")
    print(df.describe())
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'delta_ml_analysis.csv'), index=False)
    
    # Plot delta energy vs truncation error ratio
    plt.figure(figsize=(10, 8))
    plt.scatter(df['ratio_tre'], df['delta_energy'], alpha=0.7)
    plt.xlabel('Truncation Error Ratio (high/low)')
    plt.ylabel('Energy Difference (Ha)')
    plt.title('Energy Difference vs Truncation Error Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_vs_tre_ratio.png'))
    plt.close()
    
    # Plot delta energy vs low bond dimension
    plt.figure(figsize=(10, 8))
    plt.scatter(df['low_bond_dim'], df['delta_energy'], alpha=0.7)
    plt.xlabel('Low Bond Dimension')
    plt.ylabel('Energy Difference (Ha)')
    plt.title('Energy Difference vs Low Bond Dimension')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_vs_low_bond_dim.png'))
    plt.close()


if __name__ == "__main__":
    # Set mutual information threshold as used in the paper
    MI_THRESHOLD = 0.004
    
    # Create output directory for plots and analysis
    OUTPUT_DIR = "dmrg_analysis_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Analyze test directory
    print("\nAnalyzing test directory:")
    test_files, test_stats = analyze_dmrg_files("test", mi_threshold=MI_THRESHOLD, 
                                              output_dir=os.path.join(OUTPUT_DIR, "test"))
    
    # Analyze train directory
    print("\nAnalyzing train directory:")
    train_files, train_stats = analyze_dmrg_files("train", mi_threshold=MI_THRESHOLD,
                                                output_dir=os.path.join(OUTPUT_DIR, "train"))
    
    # Visualize a few example orbital graphs
    if test_files:
        visualize_orbital_graph(test_files[0], output_dir=os.path.join(OUTPUT_DIR, "graphs"))
    
    if train_files:
        # Pick a few different systems to visualize
        systems = set(parser.system_name for parser in train_files)
        for system in list(systems)[:3]:
            system_files = [p for p in train_files if p.system_name == system]
            if system_files:
                visualize_orbital_graph(system_files[0], output_dir=os.path.join(OUTPUT_DIR, "graphs"))
    
    # Perform Delta-ML analysis
    print("\nPerforming Delta-ML analysis:")
    delta_ml_analysis(test_files + train_files, output_dir=os.path.join(OUTPUT_DIR, "delta_ml"))
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}") 