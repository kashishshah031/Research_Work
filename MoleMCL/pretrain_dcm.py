# Root directory: updated_pretrain_dcm.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from model import GNN_graphpred
from updated_dataset.qac_dataset import QACGraphDataset
import os


#  DCM loss function

def loss_cl(x1, x2, y=None, temperature=0.1, alpha_dcm=0.5):
    T = temperature
    alpha = alpha_dcm

    # cosine similarity
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    cos_sim = torch.matmul(x1, x2.t())  # [B, B]

    B = cos_sim.size(0)
    device = cos_sim.device

    if y is not None:
        y = y.view(-1)
        same = (y[:, None] == y[None, :]).to(cos_sim.dtype)  # [B, B]
    else:
        # If no labels, treat only the diagonal as "same" (like identity pairs)
        same = torch.eye(B, device=device, dtype=cos_sim.dtype)

    # DCM weights for denominator: same-label -> alpha ; diff-label -> 1/T
    dcm = same * alpha + (1.0 - same) * (1.0 / T)

    # numerator: positive (diagonal) / T
    pos = cos_sim.diag() / T  # [B]

    # denominator: include positive (diagonal) and all others
    den = torch.logsumexp(cos_sim * dcm, dim=1)  # [B]

    loss = -(pos - den).mean()
    return loss

'''
import torch
import torch.nn.functional as F

def loss_cl(x1, x2, y=None, temperature=0.1, alpha_dcm=0.5):
    T = temperature
    alpha = alpha_dcm

    # cosine similarity
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    cos_sim = torch.matmul(x1, x2.t())  # [B, B]

    B = cos_sim.size(0)
    device = cos_sim.device

    if y is not None:
        y = y.view(-1)
        same = (y[:, None] == y[None, :]).to(cos_sim.dtype)  # [B, B]
    else:
        # If no labels, treat only the diagonal as "same" (identity pairs)
        same = torch.eye(B, device=device, dtype=cos_sim.dtype)

    # DCM weights for denominator: same-label -> alpha ; diff-label -> 1/T
    dcm = same * alpha + (1.0 - same) * (1.0 / T)

    # numerator: positive (diagonal) MUST be scaled by alpha to match the denominator
    # This is the corrected line:
    pos = cos_sim.diag() * alpha # [B]

    # denominator: include positive (diagonal) and all others with DCM weighting
    den = torch.logsumexp(cos_sim * dcm, dim=1)  # [B]

    # The formula is now mathematically consistent
    loss = -(pos - den).mean()
    return loss
'''
def train():
    dataset = QACGraphDataset("updated_dataset/qac_data.csv")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GNN_graphpred(
        5, 300, num_tasks=1,
        JK="last", drop_ratio=0.5,
        graph_pooling="mean", gnn_type="gin"
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, 101):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)

            # 🔧 make sure index tensors are integer types for embeddings
            batch.x = batch.x.long()
            batch.edge_attr = batch.edge_attr.long()
            batch.y = batch.y.view(-1).long()

            if batch.edge_attr.dim() == 1:
                batch.edge_attr = batch.edge_attr.view(-1, 1)
            if batch.edge_attr.size(1) == 1:
                zeros = torch.zeros(batch.edge_attr.size(0), 1, dtype=batch.edge_attr.dtype, device=batch.edge_attr.device)
                batch.edge_attr = torch.cat([batch.edge_attr, zeros], dim=1)

            # two stochastic forward passes (views)
            node_emb1 = model.gnn(batch.x, batch.edge_index, batch.edge_attr)
            graph_emb1 = global_mean_pool(node_emb1, batch.batch)

            node_emb2 = model.gnn(batch.x, batch.edge_index, batch.edge_attr)
            graph_emb2 = global_mean_pool(node_emb2, batch.batch)

            loss = loss_cl(graph_emb1, graph_emb2, batch.y,
                           temperature=0.1, alpha_dcm=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.4f}")

    os.makedirs("pretrained_models", exist_ok=True)
    torch.save(model.state_dict(), "pretrained_models/qac_molemcl.pt")
    print("✅ Saved model at pretrained_models/qac_molemcl.pt")


if __name__ == '__main__':
    train()
