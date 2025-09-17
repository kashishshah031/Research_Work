import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import os

# ---- Load embeddings ----
data = torch.load("results/qac_embeddings.pt")
embeddings = data["embeddings"].numpy()
labels = data["labels"].numpy()

# normalize vectors (for cosine distance)
embeddings = normalize(embeddings, axis=1)

# split into QAC and non-QAC
qac = embeddings[labels == 1]
non_qac = embeddings[labels == 0]

# centroid of QAC embeddings
qac_centroid = qac.mean(axis=0, keepdims=True)

# cosine distance = 1 - cosine similarity
def cosine_distance(a, b):
    sim = np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))
    return 1 - sim.squeeze()

qac_dist = cosine_distance(qac, qac_centroid)
nonqac_dist = cosine_distance(non_qac, qac_centroid)

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(qac_dist, bins=30, color="orange", alpha=0.7)
axes[0].set_title("QAC Cosine Distance")
axes[0].set_xlabel("Cosine Distance")
axes[0].set_ylabel("Count")

axes[1].hist(nonqac_dist, bins=30, color="royalblue", alpha=0.7)
axes[1].set_title("Non-QAC Cosine Distance")
axes[1].set_xlabel("Cosine Distance")
axes[1].set_ylabel("Count")

fig.suptitle("QAC vs Non-QAC Distance to QAC Centroid")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/qac_histogram.png", dpi=300)
plt.show()
