"""Stage 4 - Feature management.

Builds user, item and interaction features from the TRAINING data only (as of the
split cutoff), stores them in the offline feature store (data/features/) and writes
the feature catalog. The same FeatureStore class is used for training and prediction.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.features.feature_definitions import FEATURES, GENRES
from src.utils import ROOT, get_logger, load_config

log = get_logger(__name__)


# ------------------------------------------------------------------ builders
def build_user_features(users, train, items, as_of) -> pd.DataFrame:
    g = train.groupby("user_id")
    stats = g.agg(user_activity_count=("rating", "size"),
                  user_avg_rating=("rating", "mean"),
                  user_rating_std=("rating", "std"),
                  last=("interaction_time", "max")).reset_index()
    out = users.rename(columns={"age": "user_age", "age_group": "user_age_group",
                                "gender": "user_gender", "occupation": "user_occupation"})
    out = out[["user_id", "user_age", "user_age_group", "user_gender", "user_occupation"]]
    out = out.merge(stats, on="user_id", how="left")

    out["user_activity_count"] = out["user_activity_count"].fillna(0).astype(int)
    out["user_is_cold"] = out["user_activity_count"] == 0
    out["user_avg_rating"] = out["user_avg_rating"].fillna(train["rating"].mean())
    out["user_rating_std"] = out["user_rating_std"].fillna(0.0)
    out["user_days_since_last_interaction"] = (as_of - out["last"]).dt.days.astype(float)
    out = out.drop(columns="last")

    # preferred genre: most frequent genre among items rated 4+ (all items if none)
    def genre_counts(df):
        m = df[["user_id", "item_id"]].merge(items[["item_id"] + GENRES], on="item_id")
        return m.groupby("user_id")[GENRES].sum()
    any_counts = genre_counts(train)
    counts = genre_counts(train[train["rating"] >= 4]).reindex(any_counts.index).fillna(0)
    empty = counts.sum(axis=1) == 0
    counts.loc[empty] = any_counts.loc[empty]
    known = counts.drop(columns="unknown")
    pref = known.idxmax(axis=1).where(known.sum(axis=1) > 0, "unknown")
    out["user_preferred_genre"] = out["user_id"].map(pref).fillna("unknown")
    return out


def build_item_features(items, train, as_of) -> pd.DataFrame:
    stats = train.groupby("item_id").agg(item_popularity_count=("rating", "size"),
                                         item_avg_rating=("rating", "mean")).reset_index()
    out = items[["item_id", "release_year", "n_genres"] + GENRES].rename(
        columns={"release_year": "item_release_year", "n_genres": "item_n_genres",
                 **{g: f"genre_{g}" for g in GENRES}})
    out = out.merge(stats, on="item_id", how="left")
    out["item_popularity_count"] = out["item_popularity_count"].fillna(0).astype(int)
    out["item_is_cold"] = out["item_popularity_count"] == 0
    out["item_avg_rating"] = out["item_avg_rating"].fillna(train["rating"].mean())
    out["item_popularity_score"] = out["item_popularity_count"] / out["item_popularity_count"].max()
    out["item_popularity_rank"] = out["item_popularity_count"].rank(
        method="min", ascending=False).astype(int)
    out["item_age_years"] = (as_of.year - out["item_release_year"]).clip(lower=0).astype(int)
    return out


def build_interaction_features(train, user_feats, item_feats, as_of) -> pd.DataFrame:
    df = train[["user_id", "item_id", "rating", "interaction_time"]].merge(
        user_feats[["user_id", "user_avg_rating", "user_preferred_genre"]], on="user_id").merge(
        item_feats[["item_id", "item_avg_rating"] + [f"genre_{g}" for g in GENRES]], on="item_id")
    out = df[["user_id", "item_id", "rating"]].copy()
    out["interaction_recency_days"] = (as_of - df["interaction_time"]).dt.total_seconds() / 86400
    out["rating_minus_user_avg"] = df["rating"] - df["user_avg_rating"]
    out["rating_minus_item_avg"] = df["rating"] - df["item_avg_rating"]
    flags = df[[f"genre_{g}" for g in GENRES]].to_numpy()
    col = pd.Index(GENRES).get_indexer(df["user_preferred_genre"])
    out["item_in_preferred_genre"] = flags[np.arange(len(df)), col]
    return out


# ------------------------------------------------------------------ feature store
class FeatureStore:
    """Offline feature store - the SAME lookups are used by training and by prediction."""

    def __init__(self, features_dir=None):
        cfg = load_config()
        self.dir = features_dir or ROOT / cfg["paths"]["features"]
        self.users = pd.read_parquet(self.dir / "user_features.parquet")
        self.items = pd.read_parquet(self.dir / "item_features.parquet")
        self.metadata = json.loads((self.dir / "feature_metadata.json").read_text())

    def user_features(self, user_ids=None):
        return self.users if user_ids is None else self.users[self.users["user_id"].isin(user_ids)]

    def item_features(self, item_ids=None):
        return self.items if item_ids is None else self.items[self.items["item_id"].isin(item_ids)]

    def pair_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Features for (user_id, item_id) pairs at prediction time (no rating needed)."""
        out = pairs[["user_id", "item_id"]].merge(self.users, on="user_id", how="left").merge(
            self.items, on="item_id", how="left")
        flags = out[[f"genre_{g}" for g in GENRES]].to_numpy()
        col = pd.Index(GENRES).get_indexer(out["user_preferred_genre"])
        out["item_in_preferred_genre"] = flags[np.arange(len(out)), col]
        return out


# ------------------------------------------------------------------ catalog
def write_catalog(tables: dict, rep_dir, feat_dir) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        entity_table = {"user": "user_features", "item": "item_features",
                        "user-item": "interaction_features"}[f["entity"]]
        missing = [c for c in f["columns"] if c not in tables[entity_table].columns]
        assert not missing, f"Catalog out of sync with data, missing columns: {missing}"
        rows.append({k: f[k] for k in ["feature", "entity", "dtype", "description",
                                      "computation", "source", "used_by"]})
    cat = pd.DataFrame(rows)
    cat.to_csv(feat_dir / "feature_catalog.csv", index=False)
    lines = ["# Feature Catalog", "",
             "All features are computed from the training period only (as of the split cutoff).", "",
             "| Feature | Entity | Type | Description | Computation | Source | Used by |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in
                     ["feature", "entity", "dtype", "description", "computation",
                      "source", "used_by"]) + " |")
    (rep_dir / "feature_catalog.md").write_text("\n".join(lines))
    return cat


def run() -> dict:
    cfg = load_config()
    proc, feat = ROOT / cfg["paths"]["processed"], ROOT / cfg["paths"]["features"]
    rep = ROOT / cfg["paths"]["reports"]
    for d in (feat, rep):
        d.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(proc / "clean_users.parquet")
    items = pd.read_parquet(proc / "clean_items.parquet")
    train = pd.read_parquet(proc / "train.parquet")
    split = json.loads((rep / "split_info.json").read_text())
    as_of = pd.to_datetime(split["cutoff_timestamp"], unit="s", utc=True)

    user_f = build_user_features(users, train, items, as_of)
    item_f = build_item_features(items, train, as_of)
    inter_f = build_interaction_features(train, user_f, item_f, as_of)

    user_f.to_parquet(feat / "user_features.parquet", index=False)
    item_f.to_parquet(feat / "item_features.parquet", index=False)
    inter_f.to_parquet(feat / "interaction_features.parquet", index=False)
    write_catalog({"user_features": user_f, "item_features": item_f,
                   "interaction_features": inter_f}, rep, feat)

    meta = {"feature_version": cfg["features"]["version"], "as_of": str(as_of),
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "rows": {"user_features": len(user_f), "item_features": len(item_f),
                     "interaction_features": len(inter_f)},
            "source": "train.parquet (training period only)"}
    (feat / "feature_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("Features built (v%s, as of %s): users=%d items=%d interactions=%d",
             meta["feature_version"], as_of, len(user_f), len(item_f), len(inter_f))
    log.info("Cold users=%d, cold items=%d, catalog written to reports/feature_catalog.md",
             int(user_f["user_is_cold"].sum()), int(item_f["item_is_cold"].sum()))
    return meta


if __name__ == "__main__":
    run()
