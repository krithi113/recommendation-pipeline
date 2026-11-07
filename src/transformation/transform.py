"""Stage 3b - Data transformation.

* Time-based train/test split (test = the most recent interactions).
* Derived datasets built from TRAIN ONLY, so no test information leaks in:
  user activity summary, item popularity and the user-item rating matrix.
"""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils import ROOT, get_logger, load_config

log = get_logger(__name__)


def time_split(inter: pd.DataFrame, test_fraction: float):
    cutoff = int(inter["timestamp"].quantile(1 - test_fraction))
    train = inter[inter["timestamp"] <= cutoff].copy()
    test = inter[inter["timestamp"] > cutoff].copy()
    assert train["timestamp"].max() <= cutoff < test["timestamp"].min(), "time leakage!"
    info = {
        "cutoff_timestamp": cutoff,
        "cutoff_time_utc": str(pd.to_datetime(cutoff, unit="s", utc=True)),
        "train_rows": len(train), "test_rows": len(test),
        "test_users": int(test["user_id"].nunique()),
        "test_users_not_in_train": int((~test["user_id"].drop_duplicates()
                                        .isin(train["user_id"])).sum()),
        "test_items_not_in_train": int((~test["item_id"].drop_duplicates()
                                        .isin(train["item_id"])).sum()),
    }
    return train, test, info


def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(n_interactions=("rating", "size"),
                avg_rating=("rating", "mean"),
                rating_std=("rating", "std"),
                first_interaction=("interaction_time", "min"),
                last_interaction=("interaction_time", "max")).reset_index()
    out["rating_std"] = out["rating_std"].fillna(0.0)
    out["active_span_days"] = (out["last_interaction"] - out["first_interaction"]).dt.days
    return out


def item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = g.agg(n_ratings=("rating", "size"), avg_rating=("rating", "mean")).reset_index()
    out["popularity_rank"] = out["n_ratings"].rank(method="min", ascending=False).astype(int)
    out["popularity_score"] = out["n_ratings"] / out["n_ratings"].max()
    return out.sort_values("popularity_rank").reset_index(drop=True)


def user_item_matrix(train: pd.DataFrame):
    user_ids = np.sort(train["user_id"].unique())
    item_ids = np.sort(train["item_id"].unique())
    rows = pd.Index(user_ids).get_indexer(train["user_id"])
    cols = pd.Index(item_ids).get_indexer(train["item_id"])
    mat = sparse.csr_matrix((train["rating"].to_numpy(dtype=float), (rows, cols)),
                            shape=(len(user_ids), len(item_ids)))
    return mat, user_ids, item_ids


def run() -> dict:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed"]
    rep = ROOT / cfg["paths"]["reports"]
    rep.mkdir(exist_ok=True)
    inter = pd.read_parquet(proc / "clean_interactions.parquet")

    train, test, info = time_split(inter, cfg["split"]["test_fraction"])
    train.to_parquet(proc / "train.parquet", index=False)
    test.to_parquet(proc / "test.parquet", index=False)

    user_activity_summary(train).to_parquet(proc / "user_activity_summary.parquet", index=False)
    item_popularity(train).to_parquet(proc / "item_popularity.parquet", index=False)

    mat, user_ids, item_ids = user_item_matrix(train)
    sparse.save_npz(proc / "user_item_matrix.npz", mat)
    (proc / "matrix_index.json").write_text(
        json.dumps({"user_ids": user_ids.tolist(), "item_ids": item_ids.tolist()}))
    info["matrix_shape"] = list(mat.shape)
    info["matrix_sparsity"] = round(1 - mat.nnz / (mat.shape[0] * mat.shape[1]), 4)
    (rep / "split_info.json").write_text(json.dumps(info, indent=2))

    log.info("Split at %s: train=%d test=%d", info["cutoff_time_utc"],
             info["train_rows"], info["test_rows"])
    log.info("Cold-start in test: %d users and %d items not seen in train",
             info["test_users_not_in_train"], info["test_items_not_in_train"])
    log.info("User-item matrix %s, sparsity=%.4f", mat.shape, info["matrix_sparsity"])
    return info


if __name__ == "__main__":
    run()
