"""Example runner for the NumPy K-Means implementation.

Generates synthetic 2D blobs, fits KMeans, prints inertia, and saves a plot.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.kmeans import KMeans


def make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=None):
    rng = np.random.default_rng(random_state)
    angles = rng.random(centers) * 2 * np.pi
    base = np.column_stack([np.cos(angles), np.sin(angles)]) * 5.0
    X = []
    y = []
    per = n_samples // centers
    for i, c in enumerate(base):
        points = rng.normal(loc=c, scale=cluster_std, size=(per, 2))
        X.append(points)
        y.extend([i] * per)
    X = np.vstack(X)
    return X, np.array(y)


def main():
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.9, random_state=42)

    km = KMeans(n_clusters=3, init='kmeans++', n_init=5, random_state=0)
    km.fit(X)

    print(f"Inertia: {km.inertia_:.4f}")

    labels = km.labels_
    centers = km.cluster_centers_

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30, alpha=0.8)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, linewidths=2)
    plt.title('K-Means clustering result')
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'kmeans_result.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
