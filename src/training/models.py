"""Recommender models: a popularity baseline and a personalised SVD (matrix factorisation) model."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class PopularityRecommender:
    """Baseline: recommends the most popular (most rated) items the user has not rated yet.

    Popularity comes from the feature store (item_popularity_count). The same list is
    used for every user, so the model is not personalised.
    """
    name = "popularity_baseline"

    def fit(self, feature_store, train):
        items = feature_store.item_features()
        items = items[~items["item_is_cold"]]
        order = np.argsort(-items["item_popularity_count"].to_numpy(), kind="stable")
        self.ranked_items = items["item_id"].to_numpy()[order]
        self.item_avg = dict(zip(items["item_id"], items["item_avg_rating"]))
        self.global_mean = float(train["rating"].mean())
        return self

    def recommend(self, user_id, k, exclude=()):
        exclude, out = set(exclude), []
        for item in self.ranked_items:
            if item not in exclude:
                out.append(int(item))
                if len(out) == k:
                    break
        return out

    def predict_rating(self, user_ids, item_ids):
        """Rating baseline: the item's average rating (global mean for unknown items)."""
        return np.array([self.item_avg.get(i, self.global_mean) for i in item_ids])


class SVDRecommender:
    """Personalised model: truncated SVD (matrix factorisation) of the mean-centred
    user-item rating matrix. Predicted rating = user average + latent-factor score."""
    name = "svd_matrix_factorization"

    def __init__(self, n_components=20, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, feature_store, matrix, user_ids, item_ids):
        self.user_ids, self.item_ids = np.asarray(user_ids), np.asarray(item_ids)
        self.user_pos = {u: i for i, u in enumerate(self.user_ids)}
        self.item_pos = {it: i for i, it in enumerate(self.item_ids)}
        # user_avg_rating comes from the feature store (same feature used at prediction time)
        users = feature_store.users.set_index("user_id")
        self.user_avg = users.loc[self.user_ids, "user_avg_rating"].to_numpy()

        coo = matrix.tocoo()
        centred = sparse.csr_matrix(
            (coo.data - self.user_avg[coo.row], (coo.row, coo.col)), shape=matrix.shape)
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.user_factors = svd.fit_transform(centred)     # U * Sigma
        self.item_factors = svd.components_.T              # V
        self.explained_variance = float(svd.explained_variance_ratio_.sum())
        self.seen = matrix.tocsr()
        return self

    def recommend(self, user_id, k, exclude=()):
        u = self.user_pos[user_id]
        scores = self.user_avg[u] + self.user_factors[u] @ self.item_factors.T
        scores = scores.copy()
        scores[self.seen[u].indices] = -np.inf               # never re-recommend rated items
        for item in exclude:
            if item in self.item_pos:
                scores[self.item_pos[item]] = -np.inf
        top = np.argsort(-scores, kind="stable")[:k]
        return [int(self.item_ids[i]) for i in top]

    def predict_rating(self, user_ids, item_ids):
        u = np.array([self.user_pos[x] for x in user_ids])
        i = np.array([self.item_pos[x] for x in item_ids])
        pred = self.user_avg[u] + (self.user_factors[u] * self.item_factors[i]).sum(axis=1)
        return np.clip(pred, 1, 5)
