'''
import networkx as nx
import numpy as np
import torch
# ==== BEGIN PATCH: decoder pairs dataset for external MoleMCL latents ====
import torch
from torch.utils.data import Dataset

class GraphAdjSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_nodes, features='id'):
        self.max_num_nodes = max_num_nodes
        self.adj_all = []
        self.len_all = []
        self.feature_all = []

        for G in G_list:
            adj = nx.to_numpy_matrix(G)
            # the diagonal entries are 1 since they denote node probability
            self.adj_all.append(
                    np.asarray(adj) + np.identity(G.number_of_nodes()))
            self.len_all.append(G.number_of_nodes())
            if features == 'id':
                self.feature_all.append(np.identity(max_num_nodes))
            elif features == 'deg':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'struct':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()],
                                             'constant'),
                                      axis=1)
                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                self.feature_all.append(np.hstack([degs, clusterings]))

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0
        
        adj_vectorized = adj_padded[np.triu(np.ones((self.max_num_nodes,self.max_num_nodes)) ) == 1]
        # the following 2 lines recover the upper triangle of the adj matrix
        #recovered = np.zeros((self.max_num_nodes, self.max_num_nodes))
        #recovered[np.triu(np.ones((self.max_num_nodes, self.max_num_nodes)) ) == 1] = adj_vectorized
        #print(recovered)
        
        return {'adj':adj_padded,
                'adj_decoded':adj_vectorized, 
                'features':self.feature_all[idx].copy()}

# ==== BEGIN: DecoderPairs dataset for external latents (append to end of data.py) ====
import torch
from torch.utils.data import Dataset

class DecoderPairsDataset(Dataset):
    """
    Expects a torch file with keys:
      - H: [M, D]                MoleMCL latents
      - A: [M, N, N]             adjacency (0/1), padded
      - node_mask: [M, N] (optional) 1=real node, 0=pad
    If node_mask is missing, we infer it from A (row sum > 0 => real node).
    """
    def __init__(self, path_pt: str):
        blob = torch.load(path_pt, map_location="cpu")

        self.H = blob["H"].float()           # [M, D]
        self.A = blob["A"].float()           # [M, N, N]

        if "node_mask" in blob:
            self.mask = blob["node_mask"].float()  # [M, N]
        else:
            # Fallback: any node with non-zero row degree is real
            row_deg = self.A.sum(dim=-1)           # [M, N]
            self.mask = (row_deg > 0).float()

        self.max_nodes  = self.A.size(1)
        self.latent_dim = self.H.size(1)

        assert self.H.size(0) == self.A.size(0) == self.mask.size(0), \
            "Mismatched sample counts in H/A/mask."

    def __len__(self):
        return self.H.size(0)

    def __getitem__(self, i: int):
        return {
            "H": self.H[i],          # [D]
            "A": self.A[i],          # [N, N]
            "mask": self.mask[i],    # [N]
            "latent_dim": self.latent_dim,
            "max_nodes": self.max_nodes,
        }


def decpairs_collate(batch):
    return {
        "H": torch.stack([b["H"] for b in batch], dim=0),       # [B, D]
        "A": torch.stack([b["A"] for b in batch], dim=0),       # [B, N, N]
        "mask": torch.stack([b["mask"] for b in batch], dim=0), # [B, N]
        "latent_dim": batch[0]["latent_dim"],
        "max_nodes": batch[0]["max_nodes"],
    }
# ==== END: DecoderPairs dataset ====
'''
import networkx as nx
import numpy as np
import torch

class GraphAdjSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_nodes, features='id'):
        self.max_num_nodes = max_num_nodes
        self.adj_all = []
        self.len_all = []
        self.feature_all = []

        for G in G_list:
            adj = nx.to_numpy_matrix(G)
            # the diagonal entries are 1 since they denote node probability
            self.adj_all.append(
                    np.asarray(adj) + np.identity(G.number_of_nodes()))
            self.len_all.append(G.number_of_nodes())
            if features == 'id':
                self.feature_all.append(np.identity(max_num_nodes))
            elif features == 'deg':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'struct':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()],
                                             'constant'),
                                      axis=1)
                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                self.feature_all.append(np.hstack([degs, clusterings]))

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0
        
        adj_vectorized = adj_padded[np.triu(np.ones((self.max_num_nodes,self.max_num_nodes)) ) == 1]
        # the following 2 lines recover the upper triangle of the adj matrix
        #recovered = np.zeros((self.max_num_nodes, self.max_num_nodes))
        #recovered[np.triu(np.ones((self.max_num_nodes, self.max_num_nodes)) ) == 1] = adj_vectorized
        #print(recovered)
        
        return {'adj':adj_padded,
                'adj_decoded':adj_vectorized, 
                'features':self.feature_all[idx].copy()}


import torch

class ExternalLatentGraphDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_path: str):
        pack = torch.load(pairs_path, map_location="cpu")
        self.H = pack["H"].float()                  # [M, d]
        self.X = pack["X"].float()                  # [M, N, C_node]
        self.A = pack["A"].float()                  # [M, N, N, C_bond]
        self.node_mask = pack["node_mask"].bool()   # [M, N]
        self.N_MAX = int(pack["N_MAX"])
        self.C_NODE = int(pack["C_NODE"])
        self.C_BOND = int(pack["C_BOND"])
        self.NODE_SPECS = pack.get("NODE_SPECS", [])
        self.BOND_SPECS = pack.get("BOND_SPECS", [])
        self.latent_dim = int(self.H.size(1))

    def __len__(self):
        return self.H.size(0)

    def __getitem__(self, idx):
        return {
            "H": self.H[idx],
            "X": self.X[idx],
            "A": self.A[idx],
            "node_mask": self.node_mask[idx],
        }
