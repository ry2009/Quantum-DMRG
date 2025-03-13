import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import json
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import itertools

class SpinDataset(Dataset):
    """
    Dataset for quantum spin systems.
    """
    def __init__(self, root_dir, system_type='heisenberg', dim=1, size=10, transform=None, pre_transform=None):
        """
        Args:
            root_dir (str): Directory with all the spin system data.
            system_type (str): Type of spin system ('heisenberg', 'ising', 'hubbard').
            dim (int): Dimension of the system (1 for chain, 2 for lattice).
            size (int): Size of the system (number of sites).
            transform (callable, optional): Optional transform to be applied on a sample.
            pre_transform (callable, optional): Optional pre-transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.system_type = system_type
        self.dim = dim
        self.size = size
        self.transform = transform
        self.pre_transform = pre_transform
        
        # Create directory if it doesn't exist
        os.makedirs(root_dir, exist_ok=True)
        
        # Generate data if it doesn't exist
        self.data_list = []
        self._generate_data()
        
    def _generate_data(self):
        """Generate or load data for the spin system."""
        # Check if data already exists
        data_file = os.path.join(self.root_dir, f"{self.system_type}_{self.dim}d_{self.size}.pt")
        
        if os.path.exists(data_file):
            # Load existing data
            self.data_list = torch.load(data_file)
            print(f"Loaded {len(self.data_list)} samples from {data_file}")
        else:
            # Generate new data
            print(f"Generating new data for {self.system_type} {self.dim}D system of size {self.size}...")
            
            # Generate different parameter configurations
            if self.system_type == 'heisenberg':
                # For Heisenberg model: J (exchange), h (magnetic field)
                J_values = np.linspace(0.1, 2.0, 10)
                h_values = np.linspace(0.0, 2.0, 10)
                
                for J, h in itertools.product(J_values, h_values):
                    # Create graph data for this configuration
                    graph_data = self._create_heisenberg_data(J, h)
                    self.data_list.append(graph_data)
                    
            elif self.system_type == 'ising':
                # For Ising model: J (coupling), h (transverse field)
                J_values = np.linspace(0.1, 2.0, 10)
                h_values = np.linspace(0.1, 2.0, 10)
                
                for J, h in itertools.product(J_values, h_values):
                    # Create graph data for this configuration
                    graph_data = self._create_ising_data(J, h)
                    self.data_list.append(graph_data)
                    
            elif self.system_type == 'hubbard':
                # For Hubbard model: t (hopping), U (on-site interaction)
                t_values = np.linspace(0.1, 2.0, 10)
                U_values = np.linspace(0.1, 10.0, 10)
                
                for t, U in itertools.product(t_values, U_values):
                    # Create graph data for this configuration
                    graph_data = self._create_hubbard_data(t, U)
                    self.data_list.append(graph_data)
            
            # Save generated data
            torch.save(self.data_list, data_file)
            print(f"Generated and saved {len(self.data_list)} samples to {data_file}")
    
    def _create_heisenberg_data(self, J, h):
        """Create graph data for Heisenberg model with parameters J and h."""
        # Create graph structure based on dimension
        if self.dim == 1:
            # 1D chain
            G = nx.path_graph(self.size)
        else:
            # 2D lattice
            side_length = int(np.sqrt(self.size))
            G = nx.grid_2d_graph(side_length, side_length)
            # Convert 2D coordinates to 1D indices
            G = nx.convert_node_labels_to_integers(G)
        
        # Calculate ground state energy using exact diagonalization (for small systems)
        # or DMRG (for larger systems)
        if self.size <= 12:
            energy = self._exact_diagonalization_heisenberg(J, h)
        else:
            # For larger systems, we would use DMRG
            # This is a placeholder - in practice, you would call a DMRG solver
            energy = -J * self.size * 0.4 - h * self.size * 0.5  # Approximate
        
        # Create node features: local magnetic field and position
        x = torch.zeros((self.size, 2))
        for i in range(self.size):
            x[i, 0] = h  # Local field
            x[i, 1] = i / self.size  # Normalized position
        
        # Create edge features: coupling strength
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        edge_attr = torch.ones((edge_index.size(1), 1)) * J
        
        # Add reverse edges for undirected graph
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([[energy]]),
            params=torch.tensor([J, h])
        )
        
        return data
    
    def _create_ising_data(self, J, h):
        """Create graph data for Transverse Field Ising model with parameters J and h."""
        # Similar to Heisenberg but with different Hamiltonian
        if self.dim == 1:
            G = nx.path_graph(self.size)
        else:
            side_length = int(np.sqrt(self.size))
            G = nx.grid_2d_graph(side_length, side_length)
            G = nx.convert_node_labels_to_integers(G)
        
        # Calculate energy (placeholder)
        if self.size <= 12:
            energy = self._exact_diagonalization_ising(J, h)
        else:
            energy = -J * self.size * 0.3 - h * self.size * 0.7  # Approximate
        
        # Create node features
        x = torch.zeros((self.size, 2))
        for i in range(self.size):
            x[i, 0] = h  # Transverse field
            x[i, 1] = i / self.size  # Normalized position
        
        # Create edge features
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        edge_attr = torch.ones((edge_index.size(1), 1)) * J
        
        # Add reverse edges
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Create graph data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([[energy]]),
            params=torch.tensor([J, h])
        )
        
        return data
    
    def _create_hubbard_data(self, t, U):
        """Create graph data for Hubbard model with parameters t and U."""
        # For Hubbard model, we need to consider electron hopping and on-site interaction
        if self.dim == 1:
            G = nx.path_graph(self.size)
        else:
            side_length = int(np.sqrt(self.size))
            G = nx.grid_2d_graph(side_length, side_length)
            G = nx.convert_node_labels_to_integers(G)
        
        # Calculate energy (placeholder)
        if self.size <= 6:  # Hubbard models get very large quickly
            energy = self._exact_diagonalization_hubbard(t, U)
        else:
            energy = -t * self.size * 0.5 + U * self.size * 0.25  # Approximate
        
        # Create node features: on-site interaction and position
        x = torch.zeros((self.size, 2))
        for i in range(self.size):
            x[i, 0] = U  # On-site interaction
            x[i, 1] = i / self.size  # Normalized position
        
        # Create edge features: hopping parameter
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        edge_attr = torch.ones((edge_index.size(1), 1)) * t
        
        # Add reverse edges
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Create graph data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([[energy]]),
            params=torch.tensor([t, U])
        )
        
        return data
    
    def _exact_diagonalization_heisenberg(self, J, h):
        """
        Perform exact diagonalization for small Heisenberg systems.
        This is a simplified implementation for demonstration.
        """
        # For a real implementation, you would use a proper ED library
        # This is just a placeholder that returns approximate values
        return -J * self.size * 0.4 - h * self.size * 0.5
    
    def _exact_diagonalization_ising(self, J, h):
        """Placeholder for exact diagonalization of Ising model."""
        return -J * self.size * 0.3 - h * self.size * 0.7
    
    def _exact_diagonalization_hubbard(self, t, U):
        """Placeholder for exact diagonalization of Hubbard model."""
        return -t * self.size * 0.5 + U * self.size * 0.25
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    def visualize(self, idx):
        """Visualize the spin system graph."""
        data = self.data_list[idx]
        
        # Convert to networkx for visualization
        G = nx.Graph()
        for i in range(data.x.size(0)):
            G.add_node(i, field=data.x[i, 0].item())
        
        for j in range(data.edge_index.size(1) // 2):  # Only use half the edges (undirected)
            src, dst = data.edge_index[0, j].item(), data.edge_index[1, j].item()
            G.add_edge(src, dst, coupling=data.edge_attr[j, 0].item())
        
        # Plot
        plt.figure(figsize=(8, 6))
        
        if self.dim == 1:
            # 1D layout
            pos = {i: (i, 0) for i in range(self.size)}
        else:
            # 2D grid layout
            side_length = int(np.sqrt(self.size))
            pos = {i: (i % side_length, i // side_length) for i in range(self.size)}
        
        # Node colors based on field strength
        node_colors = [G.nodes[i]['field'] for i in G.nodes]
        
        # Edge colors based on coupling strength
        edge_colors = [G.edges[e]['coupling'] for e in G.edges]
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, 
                node_size=500, cmap=plt.cm.viridis, edge_cmap=plt.cm.plasma)
        
        # Add title with energy
        energy = data.y.item()
        params = data.params.numpy()
        if self.system_type == 'heisenberg':
            title = f"Heisenberg Model: J={params[0]:.2f}, h={params[1]:.2f}, E={energy:.4f}"
        elif self.system_type == 'ising':
            title = f"Ising Model: J={params[0]:.2f}, h={params[1]:.2f}, E={energy:.4f}"
        else:
            title = f"Hubbard Model: t={params[0]:.2f}, U={params[1]:.2f}, E={energy:.4f}"
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
        return G 