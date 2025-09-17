import torch
from torch_geometric.loader import DataLoader
from qac_dataset import QACGraphDataset

CSV = "updated_dataset/qac_data.csv"  # full merged file (shuffled)

def main():
    ds = QACGraphDataset(CSV)
    print(f"Dataset size (valid molecules): {len(ds)}")
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    print("Batch.x        :", batch.x.shape)          # [total_nodes, feat_dim]
    print("Batch.edge_idx :", batch.edge_index.shape) # [2, total_edges]
    print("Batch.edge_attr:", batch.edge_attr.shape)  # [total_edges, 1]
    print("Batch.y        :", batch.y.shape)          # [batch_size, 1]
    print("Batch.ptr      :", batch.ptr.shape)        # [batch_size+1] node pointers

if __name__ == "__main__":
    main()
