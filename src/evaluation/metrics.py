"""Ranking and rating metrics."""
import numpy as np


def precision_recall_ndcg(recommended, relevant, k):
    """Binary-relevance Precision@K, Recall@K and NDCG@K for one user."""
    recs = recommended[:k]
    hits = np.array([1.0 if item in relevant else 0.0 for item in recs])
    precision = hits.sum() / k
    recall = hits.sum() / len(relevant)
    discounts = 1.0 / np.log2(np.arange(2, len(recs) + 2))
    dcg = (hits * discounts).sum()
    ideal = discounts[: min(len(relevant), len(recs))].sum()
    return precision, recall, (dcg / ideal if ideal > 0 else 0.0)


def catalog_coverage(rec_lists, n_catalog_items):
    """Share of the catalogue that appears in at least one user's top-K list."""
    return len({i for recs in rec_lists for i in recs}) / n_catalog_items


def intra_list_diversity(recs, genre_vectors):
    """Mean pairwise (1 - cosine similarity) of genre vectors inside one recommendation list."""
    m = np.array([genre_vectors[i] for i in recs], dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    unit = np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)
    dist = 1.0 - unit @ unit.T
    n = len(recs)
    return float(dist[~np.eye(n, dtype=bool)].mean())


def rmse_mae(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt((err ** 2).mean())), float(np.abs(err).mean())
