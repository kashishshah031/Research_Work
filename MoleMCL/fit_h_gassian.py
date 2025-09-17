# scripts/fit_h_gaussian.py
import os
import torch

IN_PATH  = "results/H_bank.pt"
OUT_PATH = "results/H_gaussian.pt"

def main():
    assert os.path.exists(IN_PATH), f"Missing {IN_PATH}. Run dump_h_bank.py first."

    pack = torch.load(IN_PATH, map_location="cpu")
    H, y = pack["H"], pack["y"]

    # keep only QAC (label==1)
    mask_qac = (y == 1)
    H_qac = H[mask_qac]
    n_qac = H_qac.size(0)

    if n_qac == 0:
        raise RuntimeError("No QAC rows found (y==1). Check labels in H_bank.pt")

    # per-dimension mean & std (avoid zeros)
    mu  = H_qac.mean(dim=0)
    std = H_qac.std(dim=0).clamp_min(1e-6)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save({"mu": mu, "std": std, "n_qac": int(n_qac)}, OUT_PATH)

    print(f"Saved {OUT_PATH}")
    print(f"QAC count: {n_qac}")
    print(f"mu shape:  {tuple(mu.shape)} | std shape: {tuple(std.shape)}")

if __name__ == "__main__":
    main()
