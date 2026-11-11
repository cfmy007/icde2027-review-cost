"""Clustering algorithms and evaluation metrics.

ACC is accuracy after optimal cluster-to-class matching; NMI is normalized
mutual information; ARI is the adjusted Rand index; and purity is the fraction
of records in their cluster's majority class. Purity is sensitive to the number
of clusters. Writer NMI (wNMI) measures the association between cluster and
writer identities; its direction depends on the analysis goal.

ACC, NMI, purity, and wNMI lie in [0, 1]; ARI can be negative. ACC, NMI, and
ARI are invariant to permutations of cluster labels. The returned n_clusters
is a count rather than a normalized metric.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Metrics
def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return accuracy after optimal cluster-to-class matching."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = np.unique(y_pred)
    cols = np.unique(y_true)
    table = np.zeros((len(rows), len(cols)), dtype=np.int64)
    r_index = {r: i for i, r in enumerate(rows)}
    c_index = {c: j for j, c in enumerate(cols)}
    for t, p in zip(y_true, y_pred):
        table[r_index[p], c_index[t]] += 1
    r, c = linear_sum_assignment(-table)
    return float(table[r, c].sum() / len(y_true))


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of records in their cluster's majority class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for p in np.unique(y_pred):
        labels = y_true[y_pred == p]
        if len(labels):
            total += np.bincount(labels).max()
    return float(total / len(y_true))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, writer: np.ndarray | None = None) -> dict:
    out = {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "purity": purity(y_true, y_pred),
        "n_clusters": int(len(np.unique(y_pred))),
    }
    if writer is not None:
        out["wNMI"] = float(normalized_mutual_info_score(np.asarray(writer), y_pred))
    return out


# Algorithms
def _kmeans(F: np.ndarray, k: int, seed: int, n_init: int = 10):
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
    return model.fit_predict(F)


def kmeans(F: np.ndarray, k: int, seed: int = 0, **kw) -> np.ndarray:
    return _kmeans(F, k, seed, **kw)


def spherical_kmeans(F: np.ndarray, k: int, seed: int = 0, **kw) -> np.ndarray:
    """Normalize samples to unit length and apply k-means."""
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
    return _kmeans(Fn, k, seed, **kw)


def agglomerative(F: np.ndarray, k: int, linkage: str = "ward") -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return model.fit_predict(F)


def gmm(F: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=k, covariance_type="diag", random_state=seed, max_iter=200, n_init=1
    )
    return model.fit_predict(F)


ALGORITHMS = {
    "kmeans": kmeans,
    "spherical": spherical_kmeans,
    "ward": agglomerative,
    "gmm": gmm,
}


def run_suite(
    F: np.ndarray,
    y: np.ndarray,
    writer: np.ndarray | None = None,
    methods: list[str] | None = None,
    k: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Run clustering methods and return results sorted by ACC."""
    methods = methods or ["kmeans", "spherical"]
    k = k or int(len(np.unique(y)))
    rows = []
    for name in methods:
        pred = ALGORITHMS[name](F, k, seed=seed) if name != "ward" else ALGORITHMS[name](F, k)
        m = evaluate(y, pred, writer)
        m.update({"method": name, "k": k})
        rows.append(m)
    return sorted(rows, key=lambda r: -r["ACC"])


def trivial_baselines(y: np.ndarray, writer: np.ndarray | None = None, seed: int = 0) -> list[dict]:
    """Evaluate single-cluster and random-assignment baselines."""
    rng = np.random.default_rng(seed)
    k = int(len(np.unique(y)))
    rows = []
    for name, pred in [
        ("single-cluster", np.zeros_like(y)),
        ("random", rng.integers(0, k, size=len(y))),
    ]:
        m = evaluate(y, pred, writer)
        m.update({"method": name, "k": k})
        rows.append(m)
    return rows
