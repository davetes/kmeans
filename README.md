## Group Members

Contributors:

- Salhadin Assefa — ID No: UGR/1616/15
- Tesfahun Kere — ID No: UGR/3338/15
- Mekunint Werkie — ID No: UGR/2270/15
- Nebyu Minuyelet — ID No: UGR/0407/15

# K-Means Clustering (NumPy implementation)

This project implements the K-Means clustering algorithm using basic Python and NumPy (no high-level ML libraries).

**Files:**
- `src/kmeans.py`: `KMeans` class (fit, predict) with `kmeans++` and `random` init.
- `examples/run_kmeans.py`: small example that generates 2D blobs, fits K-Means, and saves a plot.
- `requirements.txt`: minimal dependencies.

**Quick start**

Install requirements:

```bash
pip install -r requirements.txt
```

Run the example:

```bash
python examples/run_kmeans.py
```

Theory / Math
--------------

K-Means partitions n samples x_i into k clusters by minimizing the within-cluster sum of squared distances (WCSS):

$$\text{inertia} = \sum_{i=1}^{n} \min_{1\le j\le k} \|x_i - \mu_j\|^2$$

where $\mu_j$ are cluster centroids. Lloyd's algorithm (commonly called "K-Means") iterates two steps:

1. Assignment: assign each point to the nearest centroid.
2. Update: recompute centroids as the mean of assigned points.

This converges to a local minimum of the inertia (not necessarily the global minimum).

Initialization: `kmeans++` initializes the first centroid uniformly at random, then samples subsequent centroids with probability proportional to squared distance to the nearest existing centroid. This reduces bad initializations and usually improves convergence and results.

Implementation choices
----------------------
- Initialization: supports `kmeans++` (default) and `random`.
- Multiple restarts: `n_init` independent initializations; we keep the run with smallest inertia.
- Stopping criteria: maximum iterations `max_iter` and centroid movement threshold `tol` (stop if max centroid shift <= tol) and also stop if labels stop changing.
- Empty clusters: when a cluster gets no points during an update, the implementation preserves the previous centroid for that cluster (simple and stable fallback).

Limitations
-----------
- K-Means finds a local minima — results depend on initialization.
- Assumes spherical clusters of similar size; performs poorly on elongated or very different-sized clusters.
- Sensitive to feature scaling — always scale features if magnitudes differ.
- Uses mean as centroid — not robust to outliers.

Analysis and hyperparameter impact
---------------------------------
- `n_clusters (k)`: increasing `k` always reduces inertia but risks overfitting; elbow method or silhouette analysis help pick `k`.
- `init`: `kmeans++` typically yields better, more consistent starting points than `random`.
- `n_init`: more restarts reduce chance of landing in poor local minima at cost of runtime.
- `tol` and `max_iter`: small `tol` yields more precise centers but may increase runtime; `max_iter` is a safeguard.

What typically works well
------------------------
- `kmeans++` + `n_init` in [5, 20] gives robust, repeatable clustering on well-separated spherical clusters.
- Scaling input features (standardization) leads to meaningful clusters when features have different units.

Common failures
---------------
- Non-spherical clusters (e.g., moons or concentric rings) where K-Means splits a single natural cluster.
- Large numbers of outliers shifting centroids.

Extensions and improvements
--------------------------
- Use `k-medoids` or Gaussian Mixture Models for robustness or non-spherical clusters.
- Use spectral clustering or DBSCAN for complex shapes.
- Implement centroid re-seeding strategies for empty clusters.

If you want, I can add clustering metrics (silhouette score), unit tests, or a small CLI wrapper next.
