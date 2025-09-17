# scripts/check_h.py
import torch
from torch_geometric.loader import DataLoader
from updated_dataset.qac_dataset import QACGraphDataset

# === EDIT THESE THREE LINES IF NEEDED ===
CKPT = "pretrained_models/qac_molemcl.pt"   # your trained MoleMCL weights (if you have them)
DATASET_NAME = "qac"                        # whatever your loader expects (e.g., "qac" or "chembl")
BATCH_SIZE = 32

# ---- imports from your repo ----
from model import GNN_graphpred
from loader import MoleculeDataset  # if your dataset is different, adjust import

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) dataset & loader
    #dataset = MoleculeDataset("./dataset/" + DATASET_NAME, dataset=DATASET_NAME)
    dataset = QACGraphDataset("updated_dataset/qac_data.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2) model
    model = GNN_graphpred(
        num_layer=5,            # <-- use your training config
        emb_dim=300,            # <-- use your training config
        num_tasks=1,            # binary (QAC vs non-QAC)
        JK="last",
        drop_ratio=0.5,
        graph_pooling="mean",
        gnn_type="gin",
    ).to(device)

    # (optional) load pretrained weights if you have them
    try:
        state = torch.load(CKPT, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {CKPT}")
    except Exception as e:
        print(f"[warning] could not load checkpoint ({e}); continuing with random weights")

    model.eval()

    # 3) one batch → get H
    batch = next(iter(loader)).to(device)
    logits, H = model(batch, return_h=True)

    print("logits shape:", tuple(logits.shape))
    print("H shape:", tuple(H.shape))  # EXPECT: (BATCH_SIZE, emb_dim)

if __name__ == "__main__":
    main()
