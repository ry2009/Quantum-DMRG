import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from analyze_dmrg_data import DMRGFileParser

def analyze_sample_file(file_path, mi_threshold=0.004):
    """
    Analyze a single DMRG file in detail, printing all extracted features.
    
    Args:
        file_path (str): Path to the DMRG data file
        mi_threshold (float): Threshold for mutual information
    """
    print(f"\nAnalyzing file: {file_path}")
    parser = DMRGFileParser(file_path, mi_threshold)
    
    # Basic information
    print(f"System: {parser.system_name}")
    print(f"Bond Dimension (M): {parser.bond_dim}")
    print(f"Number of orbitals: {parser.num_orbitals}")
    
    # Energy information
    print(f"\nHartree-Fock Energy: {parser.hf_energy:.8f} Ha")
    print(f"DMRG Energy: {parser.dmrg_energy:.8f} Ha")
    print(f"Correlation Energy: {parser.get_correlation_energy():.8f} Ha")
    print(f"Truncation Error: {parser.truncation_error:.10f}")
    
    # Orbital occupations
    print("\nOrbital Occupations:")
    for i, occ in enumerate(parser.orbital_occupations, 1):
        print(f"  Orbital {i}: {occ}")
    
    # Single-site entropies
    print("\nSingle-site Entropies (first 10):")
    for i, entropy in list(parser.single_site_entropies.items())[:10]:
        print(f"  Orbital {i}: {entropy:.6f}")
    
    # Mutual information
    print("\nMutual Information (first 10 pairs):")
    for (i, j), mi in list(parser.mutual_information.items())[:10]:
        print(f"  Orbitals {i}-{j}: {mi:.6f}")
    
    # Graph properties
    G = parser.build_orbital_graph()
    print(f"\nGraph Properties (MI threshold = {mi_threshold}):")
    print(f"  Number of nodes: {G.number_of_nodes()}")
    print(f"  Number of edges: {G.number_of_edges()}")
    print(f"  Average node degree: {np.mean([d for _, d in G.degree()]):.2f}")
    print(f"  Graph density: {nx.density(G):.4f}")
    
    # Convert to PyG data
    data = parser.to_pyg_data()
    print("\nPyTorch Geometric Data:")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge index shape: {data.edge_index.shape}")
    print(f"  Edge features shape: {data.edge_attr.shape}")
    print(f"  Target value: {data.y.item():.8f}")
    
    return parser


def compare_bond_dimensions(system_path, mi_threshold=0.004, output_dir="delta_ml_results"):
    """
    Compare different bond dimensions for the same system to analyze Δ-ML potential.
    
    Args:
        system_path (str): Path to directory containing DMRG files for a system
        mi_threshold (float): Threshold for mutual information
        output_dir (str): Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all DMRG files in the system directory
    file_paths = []
    for file in os.listdir(system_path):
        if file.endswith('_m'):
            file_paths.append(os.path.join(system_path, file))
    
    # Parse all files
    parsers = []
    for file_path in file_paths:
        try:
            parser = DMRGFileParser(file_path, mi_threshold)
            parsers.append(parser)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Sort by bond dimension
    parsers.sort(key=lambda x: x.bond_dim)
    
    if len(parsers) < 2:
        print("Need at least two bond dimensions for comparison")
        return
    
    # Get system name
    system_name = parsers[0].system_name
    
    # Extract energies
    bond_dims = [p.bond_dim for p in parsers]
    dmrg_energies = [p.dmrg_energy for p in parsers]
    hf_energies = [p.hf_energy for p in parsers]
    tre_values = [p.truncation_error for p in parsers]
    
    # Compute energy differences (deltas) between consecutive bond dimensions
    deltas = []
    for i in range(len(parsers) - 1):
        deltas.append(dmrg_energies[i+1] - dmrg_energies[i])
    
    # Plot energies vs bond dimension
    plt.figure(figsize=(10, 6))
    plt.plot(bond_dims, dmrg_energies, 'o-', label="DMRG Energy")
    plt.axhline(y=hf_energies[0], color='r', linestyle='--', label="HF Energy")
    plt.xlabel("Bond Dimension (M)")
    plt.ylabel("Energy (Ha)")
    plt.title(f"Energy vs Bond Dimension for {system_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{system_name}_energy_vs_bond_dim.png"))
    
    # Plot energy differences (deltas)
    plt.figure(figsize=(10, 6))
    plt.plot(bond_dims[:-1], deltas, 'o-')
    plt.xlabel("Bond Dimension (M)")
    plt.ylabel("Energy Difference, Δ (Ha)")
    plt.title(f"Energy Difference (Δ) vs Bond Dimension for {system_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{system_name}_delta_vs_bond_dim.png"))
    
    # Plot truncation error vs bond dimension
    plt.figure(figsize=(10, 6))
    plt.plot(bond_dims, tre_values, 'o-')
    plt.xlabel("Bond Dimension (M)")
    plt.ylabel("Truncation Error")
    plt.yscale('log')
    plt.title(f"Truncation Error vs Bond Dimension for {system_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{system_name}_tre_vs_bond_dim.png"))
    
    # Create data summary
    summary = pd.DataFrame({
        "bond_dim": bond_dims,
        "hf_energy": hf_energies,
        "dmrg_energy": dmrg_energies,
        "truncation_error": tre_values
    })
    
    # Add deltas to summary
    delta_df = pd.DataFrame({
        "low_bond_dim": bond_dims[:-1],
        "high_bond_dim": bond_dims[1:],
        "delta_energy": deltas
    })
    
    # Print summary statistics
    print(f"\nSummary for {system_name}:")
    print(summary)
    print("\nEnergy Differences (Deltas):")
    print(delta_df)
    
    # Save to CSV
    summary.to_csv(os.path.join(output_dir, f"{system_name}_summary.csv"), index=False)
    delta_df.to_csv(os.path.join(output_dir, f"{system_name}_deltas.csv"), index=False)
    
    # Analyze correlation between truncation error and energy
    print("\nCorrelation between TRE and DMRG Energy:", 
          np.corrcoef(tre_values, dmrg_energies)[0, 1])
    
    # If we have at least 3 bond dimensions, can we predict the delta from TRE?
    if len(parsers) >= 3:
        # Try a simple linear relationship between log(tre) and delta
        log_tre = np.log10(tre_values[:-1])
        log_tre_ratio = np.log10([tre_values[i]/tre_values[i+1] for i in range(len(tre_values)-1)])
        
        print("\nCorrelation between log(TRE) and Delta:", 
              np.corrcoef(log_tre, deltas)[0, 1])
        print("Correlation between log(TRE Ratio) and Delta:", 
              np.corrcoef(log_tre_ratio, deltas)[0, 1])
        
        # Plot delta vs log(TRE)
        plt.figure(figsize=(10, 6))
        plt.scatter(log_tre, deltas)
        plt.xlabel("Log10(Truncation Error)")
        plt.ylabel("Energy Difference, Δ (Ha)")
        plt.title(f"Energy Difference (Δ) vs Log(Truncation Error) for {system_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{system_name}_delta_vs_log_tre.png"))


def graph_feature_analysis(parsers, mi_threshold=0.004, output_dir="delta_ml_results"):
    """
    Analyze how graph features (node and edge properties) change with bond dimension.
    
    Args:
        parsers (list): List of DMRGFileParser objects for the same system
        mi_threshold (float): Threshold for mutual information
        output_dir (str): Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by bond dimension
    parsers.sort(key=lambda x: x.bond_dim)
    system_name = parsers[0].system_name
    
    # Extract bond dimensions
    bond_dims = [p.bond_dim for p in parsers]
    
    # Compute graph statistics
    graph_stats = []
    for parser in parsers:
        G = parser.build_orbital_graph()
        
        # Node entropy statistics
        node_entropies = list(parser.single_site_entropies.values())
        avg_node_entropy = np.mean(node_entropies)
        max_node_entropy = np.max(node_entropies)
        
        # Edge mutual information statistics
        edge_mis = [data['mutual_info'] for _, _, data in G.edges(data=True)]
        avg_mi = np.mean(edge_mis) if edge_mis else 0
        max_mi = np.max(edge_mis) if edge_mis else 0
        
        stats = {
            'bond_dim': parser.bond_dim,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'avg_degree': np.mean([d for _, d in G.degree()]),
            'density': nx.density(G),
            'avg_node_entropy': avg_node_entropy,
            'max_node_entropy': max_node_entropy,
            'avg_mi': avg_mi,
            'max_mi': max_mi,
            'dmrg_energy': parser.dmrg_energy,
            'truncation_error': parser.truncation_error
        }
        graph_stats.append(stats)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(graph_stats)
    
    # Plot graph statistics vs bond dimension
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Number of edges vs bond dimension
    axs[0, 0].plot(bond_dims, stats_df['num_edges'], 'o-')
    axs[0, 0].set_xlabel('Bond Dimension (M)')
    axs[0, 0].set_ylabel('Number of Edges')
    axs[0, 0].set_title('Number of Edges vs Bond Dimension')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average node entropy vs bond dimension
    axs[0, 1].plot(bond_dims, stats_df['avg_node_entropy'], 'o-')
    axs[0, 1].set_xlabel('Bond Dimension (M)')
    axs[0, 1].set_ylabel('Average Node Entropy')
    axs[0, 1].set_title('Average Node Entropy vs Bond Dimension')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average mutual information vs bond dimension
    axs[1, 0].plot(bond_dims, stats_df['avg_mi'], 'o-')
    axs[1, 0].set_xlabel('Bond Dimension (M)')
    axs[1, 0].set_ylabel('Average Mutual Information')
    axs[1, 0].set_title('Average Mutual Information vs Bond Dimension')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Graph density vs bond dimension
    axs[1, 1].plot(bond_dims, stats_df['density'], 'o-')
    axs[1, 1].set_xlabel('Bond Dimension (M)')
    axs[1, 1].set_ylabel('Graph Density')
    axs[1, 1].set_title('Graph Density vs Bond Dimension')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{system_name}_graph_statistics.png"))
    
    # Compute correlations with DMRG energy
    print(f"\nFeature correlations with DMRG energy for {system_name}:")
    for col in stats_df.columns:
        if col != 'dmrg_energy' and col != 'bond_dim':
            corr = np.corrcoef(stats_df[col], stats_df['dmrg_energy'])[0, 1]
            print(f"  {col}: {corr:.4f}")
    
    # Save to CSV
    stats_df.to_csv(os.path.join(output_dir, f"{system_name}_graph_statistics.csv"), index=False)


def analyze_dataset_statistics(train_dir, test_dir, output_dir="delta_ml_results"):
    """
    Analyze statistics across the entire dataset.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test data directory
        output_dir (str): Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all DMRG files
    train_files = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith('_m'):
                train_files.append(os.path.join(root, file))
    
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('_m'):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(train_files)} training files and {len(test_files)} test files")
    
    # Parse a subset of files to analyze statistics
    # (parsing all might be too time-consuming)
    max_files = 100
    train_sample = train_files[:min(len(train_files), max_files)]
    test_sample = test_files[:min(len(test_files), max_files)]
    
    # Parse files
    train_parsers = []
    for file_path in train_sample:
        try:
            parser = DMRGFileParser(file_path)
            train_parsers.append(parser)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    test_parsers = []
    for file_path in test_sample:
        try:
            parser = DMRGFileParser(file_path)
            test_parsers.append(parser)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Analyze dataset statistics
    train_stats = analyze_dataset_parsers(train_parsers, "Training")
    test_stats = analyze_dataset_parsers(test_parsers, "Test")
    
    # Save statistics to CSV
    train_stats.to_csv(os.path.join(output_dir, "train_dataset_statistics.csv"), index=False)
    test_stats.to_csv(os.path.join(output_dir, "test_dataset_statistics.csv"), index=False)
    
    # Compare train and test distributions
    compare_datasets(train_stats, test_stats, output_dir)


def analyze_dataset_parsers(parsers, dataset_name):
    """
    Analyze a set of parsers and extract statistics.
    
    Args:
        parsers (list): List of DMRGFileParser objects
        dataset_name (str): Name of the dataset for logging
    
    Returns:
        DataFrame: Statistics for all systems
    """
    # Extract system statistics
    stats = []
    for parser in parsers:
        # Build graph to get graph properties
        G = parser.build_orbital_graph()
        
        # Get node and edge statistics
        if G.number_of_edges() > 0:
            avg_mi = np.mean([data['mutual_info'] for _, _, data in G.edges(data=True)])
        else:
            avg_mi = 0
        
        # Statistics
        stat = {
            'system': parser.system_name,
            'bond_dim': parser.bond_dim,
            'num_orbitals': parser.num_orbitals,
            'num_edges': G.number_of_edges(),
            'graph_density': nx.density(G),
            'avg_node_entropy': np.mean(list(parser.single_site_entropies.values())),
            'avg_mi': avg_mi,
            'hf_energy': parser.hf_energy,
            'dmrg_energy': parser.dmrg_energy,
            'corr_energy': parser.get_correlation_energy(),
            'truncation_error': parser.truncation_error
        }
        stats.append(stat)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats)
    
    # Print basic statistics
    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"  Number of samples: {len(stats_df)}")
    print(f"  Number of unique systems: {len(stats_df['system'].unique())}")
    print(f"  Bond dimensions present: {sorted(stats_df['bond_dim'].unique())}")
    print(f"  Average number of orbitals: {stats_df['num_orbitals'].mean():.2f}")
    print(f"  Average graph density: {stats_df['graph_density'].mean():.4f}")
    print(f"  Average correlation energy: {stats_df['corr_energy'].mean():.8f} Ha")
    
    return stats_df


def compare_datasets(train_stats, test_stats, output_dir):
    """
    Compare training and test dataset distributions.
    
    Args:
        train_stats (DataFrame): Training set statistics
        test_stats (DataFrame): Test set statistics
        output_dir (str): Directory to save output plots
    """
    # Compare distributions of key features
    features = ['num_orbitals', 'graph_density', 'avg_node_entropy', 'corr_energy', 'truncation_error']
    
    fig, axs = plt.subplots(len(features), 1, figsize=(10, 4*len(features)))
    
    for i, feature in enumerate(features):
        # Calculate range for histogram
        all_values = np.concatenate([train_stats[feature].values, test_stats[feature].values])
        min_val, max_val = np.min(all_values), np.max(all_values)
        bins = np.linspace(min_val, max_val, 30)
        
        # Plot histograms
        axs[i].hist(train_stats[feature], bins=bins, alpha=0.5, label='Train')
        axs[i].hist(test_stats[feature], bins=bins, alpha=0.5, label='Test')
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel('Count')
        axs[i].set_title(f'{feature} Distribution')
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_test_distribution_comparison.png"))
    plt.close()


if __name__ == "__main__":
    import networkx as nx
    
    # Create output directory
    OUTPUT_DIR = "delta_ml_exploration"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Analyze a sample file
    print("\n===== Analyzing Sample DMRG File =====")
    sample_file = "/Users/ryanmathieu/Documents/GitHub/QC_DMRG_pred/test/1.C22H12_triangulene_s0/0256_m"
    sample_parser = analyze_sample_file(sample_file)
    
    # 2. Compare bond dimensions for a specific system
    print("\n===== Comparing Bond Dimensions =====")
    compare_bond_dimensions("/Users/ryanmathieu/Documents/GitHub/QC_DMRG_pred/test/1.C22H12_triangulene_s0", 
                           output_dir=OUTPUT_DIR)
    
    # 3. Perform graph feature analysis for a sample system
    print("\n===== Graph Feature Analysis =====")
    system_path = "/Users/ryanmathieu/Documents/GitHub/QC_DMRG_pred/test/1.C22H12_triangulene_s0"
    parsers = []
    for file in os.listdir(system_path):
        if file.endswith('_m'):
            parser = DMRGFileParser(os.path.join(system_path, file))
            parsers.append(parser)
    
    if parsers:
        graph_feature_analysis(parsers, output_dir=OUTPUT_DIR)
    
    # 4. Analyze dataset statistics
    print("\n===== Analyzing Dataset Statistics =====")
    analyze_dataset_statistics("train", "test", output_dir=OUTPUT_DIR)
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}") 