# extract_embeddings.py

import os
import torch
from torch_geometric.loader import DataLoader
from model import GNN_graphpred
from updated_dataset.qac_dataset import QACGraphDataset  # ✅ use your dataset class

@torch.no_grad()
def main():
    # ---- paths ----
    csv_path   = "updated_dataset/qac_data.csv"     # merged CSV
    ckpt_path  = "pretrained_models/qac_molemcl.pt"     # from pretrain_dcm.py
    out_path   = "results/qac_embeddings.pt"
    os.makedirs("results", exist_ok=True)

    # ---- data ----
    dataset = QACGraphDataset(csv_path)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN_graphpred(
        num_layer=5, emb_dim=300, num_tasks=1,
        JK="last", drop_ratio=0.5,
        graph_pooling="mean", gnn_type="gin"
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ---- extract ----
    all_embeds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        graph_emb = model.forward_emb(batch)
        all_embeds.append(graph_emb.cpu())
        all_labels.append(batch.y.view(-1).cpu())

    embeddings = torch.cat(all_embeds, dim=0)
    labels     = torch.cat(all_labels, dim=0)

    torch.save({"embeddings": embeddings, "labels": labels}, out_path)
    print(f"Saved embeddings: {out_path}")
    print(f"embeddings={embeddings.shape}, labels={labels.shape}")

if __name__ == "__main__":
    main()
