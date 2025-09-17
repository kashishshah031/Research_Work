# scripts/dump_h_bank.py
import os
import torch
from torch_geometric.loader import DataLoader

# === EDIT THESE LINES IF NEEDED ===
CKPT = "pretrained_models/qac_molemcl.pt"   # optional; leave if you don't have it
DATASET_NAME = "qac"                        # your dataset name
BATCH_SIZE = 128
OUT_PATH = "results/H_bank.pt"
from updated_dataset.qac_dataset import QACGraphDataset

# ---- imports from your repo ----
from model import GNN_graphpred
from loader import MoleculeDataset

@torch.no_grad()
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) dataset & loader
    #dataset = MoleculeDataset("./dataset/" + DATASET_NAME, dataset=DATASET_NAME)
    #loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    dataset = QACGraphDataset("updated_dataset/qac_data.csv")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2) model
    model = GNN_graphpred(
        num_layer=5, emb_dim=300, num_tasks=1,
        JK="last", drop_ratio=0.5, graph_pooling="mean", gnn_type="gin"
    ).to(device)

    # (optional) load weights
    try:
        state = torch.load(CKPT, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {CKPT}")
    except Exception as e:
        print(f"[warning] no checkpoint loaded ({e}) — continuing with current weights")

    model.eval()

    H_all, y_all = [], []
    total = 0
    for i, batch in enumerate(loader, 1):
        batch = batch.to(device)
        _, H = model(batch, return_h=True)
        H_all.append(H.cpu())
        # labels
        if hasattr(batch, "y") and batch.y is not None:
            y_all.append(batch.y.view(-1).cpu())
        else:
            raise RuntimeError("Batch has no .y labels — needed for QAC/non-QAC split.")
        total += H.size(0)
        if i % 20 == 0:
            print(f"processed {total} molecules...")

    H_all = torch.cat(H_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    torch.save({"H": H_all, "y": y_all}, OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")
    print(f"H shape: {tuple(H_all.shape)} | y shape: {tuple(y_all.shape)}")

if __name__ == "__main__":
    main()
