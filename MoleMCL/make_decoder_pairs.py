# scripts/make_decoder_pairs.py
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from updated_dataset.qac_dataset import QACGraphDataset

# ---- config ----
BATCH_SIZE = 1
N_MAX = 50
CKPT = "pretrained_models/qac_molemcl.pt"
OUT_PATH = "results/decoder_pairs.pt"

# Node vocab: (atomic_number, formal_charge)
# Ensure a class for quaternary nitrogen (7, +1)
NODE_SPECS = [
    (6, 0),   # C
    (7, 0),   # N
    (7, 1),   # N+
    (8, 0),   # O
    (16, 0),  # S
    (9, 0),   # F
    (15, 0),  # P
    (17, 0),  # Cl
    (35, 0),  # Br
    (53, 0),  # I
]
SPEC_TO_IDX = {spec: i for i, spec in enumerate(NODE_SPECS)}
C_NODE = len(NODE_SPECS)

# Bond classes (include "none" as class 0)
BOND_SPECS = ["none", "single", "double", "triple", "aromatic"]
C_BOND = len(BOND_SPECS)
BOND_IDX = {name: i for i, name in enumerate(BOND_SPECS)}

# ---- repo imports ----
from model import GNN_graphpred

def infer_formal_charge(data):
    # If available in data.x[:,1], use it; else default to 0
    if hasattr(data, "x") and data.x is not None and data.x.size(1) >= 2:
        ch = data.x[:, 1].long().clamp_(-3, 3)
        return ch
    return torch.zeros(data.x.size(0), dtype=torch.long)

def atom_to_class(atomic_num, formal_charge):
    key = (int(atomic_num), int(formal_charge))
    if key in SPEC_TO_IDX:
        return SPEC_TO_IDX[key]
    # fallback to neutral element if known
    key0 = (int(atomic_num), 0)
    if key0 in SPEC_TO_IDX:
        return SPEC_TO_IDX[key0]
    # default to carbon
    return SPEC_TO_IDX[(6, 0)]

def edge_attr_to_bond_class(edge_attr):
    # Try to infer bond type from common encodings
    # Returns integer in {1:single, 2:double, 3:triple, 4:aromatic}
    if edge_attr is None:
        return 1
    if edge_attr.dim() == 0:
        v = int(edge_attr.item())
        return int(max(1, min(4, v)))
    if edge_attr.dim() == 1:
        v = int(edge_attr[0].item())
        return int(max(1, min(4, v)))
    # one-hot or multi-dim: take argmax and map to 1..4
    idx = int(edge_attr.argmax().item()) + 1
    return int(max(1, min(4, idx)))

@torch.no_grad()
def graph_to_targets(data, n_max=N_MAX):
    # Nodes
    n = data.x.size(0)
    n_use = min(n, n_max)
    atomic_nums = data.x[:, 0].long()
    formal_charges = infer_formal_charge(data)

    node_classes = torch.full((n_max,), fill_value=-1, dtype=torch.long)
    for i in range(n_use):
        node_classes[i] = atom_to_class(atomic_nums[i].item(), formal_charges[i].item())

    # One-hot nodes
    X = torch.zeros(n_max, C_NODE)
    if n_use > 0:
        valid = node_classes[:n_use].clamp_min(0)
        X[:n_use] = F.one_hot(valid, num_classes=C_NODE).float()

    # Edges with bond types
    A = torch.zeros(n_max, n_max, C_BOND)  # one-hot over bond classes (none/single/double/triple/aromatic)
    if n_use > 0:
        # Build dense typed adjacency from edge_index (+ edge_attr if present)
        n_all = data.x.size(0)
        dense = torch.zeros(n_all, n_all, C_BOND)
        if hasattr(data, "edge_index") and data.edge_index is not None:
            ei = data.edge_index
            ea = getattr(data, "edge_attr", None)
            for e_idx in range(ei.size(1)):
                u = int(ei[0, e_idx].item())
                v = int(ei[1, e_idx].item())
                bcls = 1  # default single
                if ea is not None:
                    ea_e = ea[e_idx]
                    bcls = edge_attr_to_bond_class(ea_e)
                dense[u, v, bcls] = 1.0
        # Symmetrize and crop
        dense = torch.maximum(dense, dense.transpose(0, 1))
        A[:n_use, :n_use, :] = dense[:n_use, :n_use, :]
    # Set diagonal to "none"
    idx = torch.arange(n_max)
    A[idx, idx, :] = 0.0
    A[idx, idx, BOND_IDX["none"]] = 1.0

    # Node mask
    node_mask = torch.zeros(n_max, dtype=torch.bool)
    node_mask[:n_use] = True

    return X, A, node_mask

@torch.no_grad()
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = QACGraphDataset("updated_dataset/qac_data.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GNN_graphpred(
        num_layer=5, emb_dim=300, num_tasks=1,
        JK="last", drop_ratio=0.5, graph_pooling="mean", gnn_type="gin"
    ).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {CKPT}")
    model.eval()

    H_list, X_list, A_list, y_list, node_mask_list = [], [], [], [], []

    for i, data in enumerate(loader, 1):
        data = data.to(device)
        _, H = model(data, return_h=True)                  # [1, d]
        X, A, node_mask = graph_to_targets(data.cpu())     # CPU tensors

        H_list.append(H.squeeze(0).cpu())                  # [d]
        X_list.append(X)                                    # [N_MAX, C_NODE]
        A_list.append(A)                                    # [N_MAX, N_MAX, C_BOND]
        node_mask_list.append(node_mask)                    # [N_MAX]
        y_list.append(data.y.view(-1).cpu()[0])            # label

        if i % 200 == 0:
            print(f"Processed {i} molecules...")

    H = torch.stack(H_list)
    X = torch.stack(X_list)
    A = torch.stack(A_list)
    node_mask = torch.stack(node_mask_list)
    y = torch.stack(y_list)

    torch.save(
        {
            "H": H, "X": X, "A": A, "y": y, "node_mask": node_mask,
            "N_MAX": N_MAX, "C_NODE": C_NODE, "C_BOND": C_BOND,
            "NODE_SPECS": NODE_SPECS, "BOND_SPECS": BOND_SPECS
        },
        OUT_PATH
    )

    print(f"\nSaved {OUT_PATH}")
    print(f"H: {tuple(H.shape)} | X: {tuple(X.shape)} | A: {tuple(A.shape)} | "
          f"node_mask: {tuple(node_mask.shape)} | y: {tuple(y.shape)}")

if __name__ == "__main__":
    main()