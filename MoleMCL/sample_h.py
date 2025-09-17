# scripts/sample_h.py
import torch

GAUSS_PATH = "results/H_gaussian.pt"

def sample_H(n, path=GAUSS_PATH, device="cpu"):
    g = torch.load(path, map_location=device)
    mu, std = g["mu"], g["std"]
    return mu + torch.randn(n, mu.numel(), device=device) * std

def main():
    H = sample_H(5)  # sample 5 new latents
    print("Sampled H shape:", tuple(H.shape))  # expect (5, 300)
    print("First row (first 5 dims):", H[0, :5].tolist())
    # quick sanity: mean/std of this tiny sample (won't match perfectly, just a check)
    print("Mini-sample mean (approx):", H.mean(0)[:5].tolist())
    print("Mini-sample std  (approx):", H.std(0)[:5].tolist())

if __name__ == "__main__":
    main()
