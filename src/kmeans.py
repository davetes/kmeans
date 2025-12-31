"""Simple K-Means clustering implemented with NumPy.

Implements Lloyd's algorithm with support for `kmeans++` and `random` initialization,
multiple `n_init` restarts, `tol`-based stopping criterion and `max_iter`.

Usage:
    from src.kmeans import KMeans
    km = KMeans(n_clusters=3, init='kmeans++', random_state=0)
    km.fit(X)
    labels = km.predict(X)

"""
from typing import Optional
import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "kmeans++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        if init not in ("kmeans++", "random"):
            raise ValueError("init must be 'kmeans++' or 'random'")

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = max(1, int(n_init))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        # Attributes populated after fit
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _check_array(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input array must be 2D")
        return X

    def _kmeans_pp_init(self, X, k, rand):
        n_samples, _ = X.shape
        centers = np.empty((k, X.shape[1]), dtype=float)
        # Choose first center uniformly at random
        idx = rand.integers(0, n_samples)
        centers[0] = X[idx]
        # Distances squared to nearest center
        closest_d2 = np.sum((X - centers[0]) ** 2, axis=1)

        for c in range(1, k):
            # Choose next center with probability proportional to squared distance
            probs = closest_d2 / closest_d2.sum()
            choice = rand.choice(n_samples, p=probs)
            centers[c] = X[choice]
            d2 = np.sum((X - centers[c]) ** 2, axis=1)
            closest_d2 = np.minimum(closest_d2, d2)

        return centers

    def _random_init(self, X, k, rand):
        n_samples = X.shape[0]
        idx = rand.choice(n_samples, size=k, replace=False)
        return X[idx].copy()

    def _assign_labels(self, X, centers):
        # Compute squared distances, assign to nearest center
        # Shape: (n_samples, k)
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = dists[np.arange(dists.shape[0]), labels].sum()
        return labels, inertia

    def _compute_centers(self, X, labels, k, centers_prev):
        n_features = X.shape[1]
        centers = np.zeros((k, n_features), dtype=float)
        for i in range(k):
            members = X[labels == i]
            if members.size == 0:
                # Empty cluster: reinitialize to a random previous center to avoid collapse
                centers[i] = centers_prev[i]
            else:
                centers[i] = members.mean(axis=0)
        return centers

    def fit(self, X):
        X = self._check_array(X)
        n_samples = X.shape[0]
        k = self.n_clusters

        best_inertia = np.inf
        best_centers = None
        best_labels = None

        rand = np.random.default_rng(self.random_state)

        for init_no in range(self.n_init):
            # Initialize centers
            if self.init == "kmeans++":
                centers = self._kmeans_pp_init(X, k, rand)
            else:
                centers = self._random_init(X, k, rand)

            labels = np.full(n_samples, -1, dtype=int)

            for it in range(self.max_iter):
                labels_new, inertia = self._assign_labels(X, centers)

                # Recompute centers
                centers_new = self._compute_centers(X, labels_new, k, centers)

                # Check movement
                center_shift = np.sqrt(((centers_new - centers) ** 2).sum(axis=1))
                centers = centers_new

                # Stopping conditions: no label changes or small center movement
                if np.array_equal(labels, labels_new):
                    break
                if center_shift.max() <= self.tol:
                    labels = labels_new
                    break

                labels = labels_new

            # Final assignment & inertia for this init
            labels_final, inertia_final = self._assign_labels(X, centers)

            if inertia_final < best_inertia:
                best_inertia = inertia_final
                best_centers = centers.copy()
                best_labels = labels_final.copy()

        # Save best run
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self

    def predict(self, X):
        X = self._check_array(X)
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")
        labels, _ = self._assign_labels(X, self.cluster_centers_)
        return labels


__all__ = ["KMeans"]
