import os
import argparse
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from model import GraphDecoderOnly

def load_pairs_meta(pairs_path):
    pack = torch.load(pairs_path, map_location="cpu")
    node_specs = pack.get("NODE_SPECS", [])
    bond_specs = pack.get("BOND_SPECS", ["none","single","double","triple","aromatic"])
    N = int(pack["N_MAX"])
    C_NODE = int(pack["C_NODE"])
    C_BOND = int(pack["C_BOND"])
    node_mask = pack["node_mask"].bool()
    return node_specs, bond_specs, N, C_NODE, C_BOND, node_mask

def nodeclass_to_atom(node_class, node_specs):
    z, charge = node_specs[int(node_class)]
    atom = rdchem.Atom(int(z))
    atom.SetFormalCharge(int(charge))
    return atom

def bondclass_to_rdkit(bond_class, bond_specs):
    name = bond_specs[int(bond_class)]
    return {
        "single": rdchem.BondType.SINGLE,
        "double": rdchem.BondType.DOUBLE,
        "triple": rdchem.BondType.TRIPLE,
        "aromatic": rdchem.BondType.AROMATIC,
        "none": None,
    }.get(name, rdchem.BondType.SINGLE)

def choose_node_counts(node_mask, num):
    lengths = node_mask.sum(dim=1).cpu().numpy().astype(int)
    idx = np.random.choice(len(lengths), size=num, replace=True)
    return lengths[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder_ckpt", required=True)
    ap.add_argument("--gaussian", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="results/samples.csv")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load decoder and config
    pack = torch.load(args.decoder_ckpt, map_location=device)
    cfg = pack["config"]
    decoder = GraphDecoderOnly(
        latent_dim=cfg["latent_dim"],
        max_num_nodes=cfg["N_MAX"],
        node_classes=cfg["C_NODE"],
        bond_classes=cfg["C_BOND"],
        hidden_dim=cfg.get("hidden_dim", 512),
        n_layers=cfg.get("n_layers", 3),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    decoder.load_state_dict(pack["model"])
    decoder.eval()

    # Pairs meta (vocab and node length distribution)
    node_specs, bond_specs, N, C_NODE, C_BOND, all_node_mask = load_pairs_meta(args.pairs)

    # Gaussian prior
    g = torch.load(args.gaussian, map_location="cpu")
    mu, std = g["mu"].float().to(device), g["std"].float().to(device)
    eps = torch.randn(args.num, mu.numel(), device=device)
    H = mu.unsqueeze(0) + eps * std.unsqueeze(0)

    with torch.no_grad():
        node_logits, edge_logits = decoder(H)  # [B,N,C_node], [B,N,N,C_bond]

    Ks = choose_node_counts(all_node_mask, args.num)

    smiles = []
    for b in range(args.num):
        K = int(max(1, min(N, Ks[b])))
        node_pred = node_logits[b, :K, :].argmax(dim=-1).cpu().tolist()
        edge_prob = torch.softmax(edge_logits[b, :K, :K, :], dim=-1)  # class probs per pair
        # 1) Compute non-none prob and candidate list (upper triangle)
        p_none = edge_prob[:, :, 0]                # [K,K]
        p_non = 1.0 - p_none
        candidates = []
        for i in range(K):
            for j in range(i+1, K):
                candidates.append((float(p_non[i, j].item()), i, j))
        candidates.sort(reverse=True)              # high -> low non-none prob

        # 2) Simple per-atom max valence table (heavy-atom valence; conservative)
        #    Adjust if you wish (N+ can be 4; C up to 4, etc.)
        from collections import defaultdict
        order_of = {"none": 0, "single": 1, "double": 2, "triple": 3, "aromatic": 1}  # treat aromatic ~1 for simplicity
        max_valence = defaultdict(lambda: 4)  # default
        # override common elements if desired:
        # max_valence[6] = 4   # C
        # max_valence[7] = 3   # N (neutral); if you predict N+ node class, allow 4:
        # if node_specs[node_pred[i]] == (7,1): max_valence_this_atom = 4

        # Precompute per-atom current valence
        curr_val = [0.0 for _ in range(K)]

        def nodeclass_to_z_charge(cls):
            z, charge = node_specs[int(cls)]
            return z, charge

        def best_non_none_class(i, j):
            # pick the highest-prob non-none class
            probs = edge_prob[i, j, 1:]  # exclude none
            c_rel = int(probs.argmax().item()) + 1
            return c_rel

        # 3) Greedy add edges while respecting valence
        edges = []
        for score, i, j in candidates:
            # gate by non-none probability threshold (tau)
            if score < args.tau:
                continue
            c = best_non_none_class(i, j)
            # compute bond order
            bond_name = bond_specs[c]
            order = order_of.get(bond_name, 1)
            # get per-atom max valence (allow 4 if predicted N+)
            zi, chi = nodeclass_to_z_charge(node_pred[i])
            zj, chj = nodeclass_to_z_charge(node_pred[j])
            max_i = 4
            max_j = 4
            if zi == 7 and chi == 1:  # quaternary nitrogen
                max_i = 4
            if zj == 7 and chj == 1:
                max_j = 4
            # conservative extra guard: disallow triples on degree>1 carbons
            if order == 3 and (curr_val[i] > 1 or curr_val[j] > 1):
                continue
            # check valence feasibility
            if curr_val[i] + order <= max_i and curr_val[j] + order <= max_j:
                edges.append((i, j, c))
                curr_val[i] += order
                curr_val[j] += order

        # Build RDKit mol with chosen edges only
        mol = rdchem.RWMol()
        idx_map = []
        for cls in node_pred:
            z, charge = nodeclass_to_z_charge(cls)
            atom = rdchem.Atom(int(z))
            atom.SetFormalCharge(int(charge))
            idx_map.append(mol.AddAtom(atom))
        for (i, j, c) in edges:
            bt = bondclass_to_rdkit(c, bond_specs)
            if bt is None: 
                continue
            try:
                mol.AddBond(i, j, bt)
            except Exception:
                pass
        try:
            m = mol.GetMol()
            Chem.SanitizeMol(m)
            smi = Chem.MolToSmiles(m, canonical=True)
        except Exception:
            smi = ""
        smiles.append(smi)

if __name__ == "__main__":
    main()