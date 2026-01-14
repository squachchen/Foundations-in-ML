# -*- coding: utf-8 -*-
"""
Trains 3 linear classifiers simultaneously:
  Bream vs Roach
  Bream vs Pike
  Roach vs Pike
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# ============================================================
# Part 0. Load data
# ============================================================

df_load = load_dataset("scikit-learn/Fish")
df = df_load["train"].to_pandas()

df["Volume"] = df["Height"] * (df["Length2"] ** 2) * df["Width"]

# Keep only 3 species
df3 = df[df["Species"].isin(["Bream", "Roach", "Pike"])].copy()

X = df3[["Volume", "Weight"]].values

# Species → class index
species_map = {"Bream": 0, "Roach": 1, "Pike": 2}
y_class = df3["Species"].map(species_map).values

# ============================================================
# Part 1. Build pairwise labels
# ============================================================

# y[:,0] = Bream vs Roach
# y[:,1] = Bream vs Pike
# y[:,2] = Roach vs Pike

y = np.zeros((len(y_class), 3))

# Bream vs Roach
mask = np.isin(y_class, [0, 1])
y[mask, 0] = np.where(y_class[mask] == 0, +1, -1)

# Bream vs Pike
mask = np.isin(y_class, [0, 2])
y[mask, 1] = np.where(y_class[mask] == 0, +1, -1)

# Roach vs Pike
mask = np.isin(y_class, [1, 2])
y[mask, 2] = np.where(y_class[mask] == 1, +1, -1)

# ============================================================
# Part 2. Standardize features
# ============================================================

mu = X.mean(axis=0)
sigma = X.std(axis=0)
Xs = (X - mu) / sigma

# ============================================================
# Part 3. Train linear network (2 → 3)
# ============================================================

def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

W = np.random.randn(2, 3) * 0.01
b = np.zeros(3)

lr = 0.15
epochs = 50

for _ in range(epochs):
    scores = Xs @ W + b          # (N,3)
    margins = y * scores
    mask = (y != 0)

    p = sigmoid(-margins)

    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)

    for k in range(3):
        mk = mask[:, k]
        grad_W[:, k] = np.mean(
            (-y[mk, k][:, None] * Xs[mk]) * p[mk, k][:, None],
            axis=0
        )
        grad_b[k] = np.mean(-y[mk, k] * p[mk, k])

    W -= lr * grad_W
    b -= lr * grad_b

# ============================================================
# Part 4. Convert boundaries back to original scale
# ============================================================

W_orig = W / sigma[:, None]
b_orig = b - (mu @ W_orig)

# ============================================================
# Part 5. Final plot (ONLY final result)
# ============================================================

plt.figure(figsize=(9, 6))

for species, idx in species_map.items():
    pts = df3[y_class == idx]
    plt.scatter(
        pts["Volume"],
        pts["Weight"],
        label=species
    )

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx = np.linspace(x_min, x_max, 300)

labels = [
    "Bream vs Roach",
    "Bream vs Pike",
    "Roach vs Pike"
]

for k in range(3):
    w1, w2 = W_orig[:, k]
    bb = b_orig[k]

    if abs(w2) > 1e-8:
        yy = -(w1 * xx + bb) / w2
        plt.plot(xx, yy, linestyle="--", label=labels[k])

plt.xlabel("Volume")
plt.ylabel("Weight")
plt.title("Final Pairwise Linear Decision Boundaries")
plt.legend()
plt.show()
