# updated_dataset/qac/qac_dataset.py
import os
import pandas as pd
from typing import Optional, List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem

# --- MoleMCL-compatible vocabularies ---
POSSIBLE_ATOMIC_NUMS = list(range(1, 119))  # 1..118
POSSIBLE_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
POSSIBLE_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
POSSIBLE_BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

def _mol_to_pyg(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Keep implicit H to match typical MoleMCL preprocessing
    Chem.SanitizeMol(mol)

    # --- nodes: two integer features [atom_type_idx, chirality_idx] ---
    x_list: List[List[int]] = []
    for a in mol.GetAtoms():
        Z = a.GetAtomicNum()
        try:
            atom_idx = POSSIBLE_ATOMIC_NUMS.index(Z)
        except ValueError:
            # map unknown atomic number to last valid index bucket
            atom_idx = len(POSSIBLE_ATOMIC_NUMS) - 1
        ch = a.GetChiralTag()
        try:
            ch_idx = POSSIBLE_CHIRALITY.index(ch)
        except ValueError:
            ch_idx = 0  # unspecified
        x_list.append([atom_idx, ch_idx])
    x = torch.tensor(x_list, dtype=torch.long)  # [n, 2]

    # --- edges: two integer features [bond_type_idx, bond_dir_idx] ---
    ei0, ei1, eattr = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = b.GetBondType()
        bd = b.GetBondDir()
        try:
            bt_idx = POSSIBLE_BOND_TYPES.index(bt)
        except ValueError:
            bt_idx = 0
        try:
            bd_idx = POSSIBLE_BOND_DIRS.index(bd)
        except ValueError:
            bd_idx = 0
        # undirected → add both directions
        ei0 += [u, v]
        ei1 += [v, u]
        eattr += [[bt_idx, bd_idx], [bt_idx, bd_idx]]

    if len(eattr) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor([ei0, ei1], dtype=torch.long)     # [2, E]
        edge_attr  = torch.tensor(eattr, dtype=torch.long)          # [E, 2]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class QACGraphDataset(Dataset):
    """
    Reads a CSV with columns: smiles,label  (1=QAC, 0=non-QAC)
    Emits MoleMCL-compatible PyG Data:
      x.long() with shape [n,2], edge_attr.long() with shape [E,2], y.long()
    """
    def __init__(self, csv_path: str):
        assert os.path.exists(csv_path), f"Not found: {csv_path}"
        df = pd.read_csv(csv_path)
        self.smiles = df["smiles"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

        # Pre-validate to avoid runtime RDKit errors
        self._valid_idx = []
        for i, s in enumerate(self.smiles):
            try:
                d = _mol_to_pyg(s)
                if d is not None:
                    self._valid_idx.append(i)
            except Exception:
                pass

    def __len__(self):
        return len(self._valid_idx)

    def __getitem__(self, idx):
        real_i = self._valid_idx[idx]
        s = self.smiles[real_i]
        y = self.labels[real_i]
        d = _mol_to_pyg(s)
        # ensure correct dtypes
        d.x = d.x.long()
        d.edge_index = d.edge_index.long()
        d.edge_attr = d.edge_attr.long()
        d.y = torch.tensor([y], dtype=torch.long)
        d.smiles = s
        return d
