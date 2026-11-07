"""Stage 6 - Model evaluation.

Protocol: for every test user who also has training history, each model recommends the
top-K items the user has not rated in training. Relevant items are the user's test items
rated >= relevance_threshold. Only items known from training can be recommended, so test
interactions with cold users/items are excluded (and counted) - a documented limitation.
"""
import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.evaluation.metrics import (catalog_coverage, intra_list_diversity,
                                    precision_recall_ndcg, rmse_mae)
from src.features.build_features import FeatureStore
from src.training.train import setup_mlflow
from src.utils import ROOT, get_logger, load_config

log = get_logger(__name__)


def run() -> pd.DataFrame:
    cfg = load_config()
    setup_mlflow(cfg)
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    proc, models_dir = ROOT / cfg["paths"]["processed"], ROOT / cfg["paths"]["models"]
    rep = ROOT / cfg["paths"]["reports"]

    fs = FeatureStore()
    train = pd.read_parquet(proc / "train.parquet")
    test = pd.read_parquet(proc / "test.parquet")
    run_ids = json.loads((models_dir / "run_info.json").read_text())

    warm_users, warm_items = set(train["user_id"]), set(train["item_id"])
    test_warm = test[test["user_id"].isin(warm_users) & test["item_id"].isin(warm_items)]
    relevant = (test_warm[test_warm["rating"] >= thr].groupby("user_id")["item_id"].apply(set))
    seen = train.groupby("user_id")["item_id"].apply(set)
    genre_cols = [c for c in fs.items.columns if c.startswith("genre_")]
    genre_vec = {r.item_id: r[genre_cols].to_numpy() for _, r in fs.items.iterrows()}

    info = {"k": k, "relevance_threshold": thr, "test_rows": len(test),
            "test_rows_evaluated_warm": len(test_warm),
            "users_evaluated": int(len(relevant)),
            "test_users_cold_skipped": int(test["user_id"].nunique() - test_warm["user_id"].nunique())}
    log.info("Evaluating on %d users, %d warm test rows (K=%d, relevant = rating >= %d)",
             info["users_evaluated"], info["test_rows_evaluated_warm"], k, thr)

    rows = []
    for name in ["popularity_baseline", "svd_matrix_factorization"]:
        model = joblib.load(models_dir / f"{name}.joblib")
        recs = {u: model.recommend(u, k, exclude=seen.get(u, ())) for u in relevant.index}
        per_user = np.array([precision_recall_ndcg(recs[u], relevant[u], k) for u in relevant.index])
        div = [intra_list_diversity(r, genre_vec) for r in recs.values()]
        preds = model.predict_rating(test_warm["user_id"], test_warm["item_id"])
        rmse, mae = rmse_mae(test_warm["rating"], preds)
        se = per_user.std(axis=0, ddof=1) / np.sqrt(len(per_user))
        res = {"model": name,
               f"precision@{k}": per_user[:, 0].mean(), f"recall@{k}": per_user[:, 1].mean(),
               f"ndcg@{k}": per_user[:, 2].mean(),
               "coverage": catalog_coverage(recs.values(), len(warm_items)),
               "diversity": float(np.mean(div)), "rmse": rmse, "mae": mae,
               f"precision@{k}_se": se[0], f"recall@{k}_se": se[1], f"ndcg@{k}_se": se[2]}
        rows.append(res)
        with mlflow.start_run(run_id=run_ids[name]):          # add metrics to the training run
            mlflow.log_params({"eval_k": k, "eval_relevance_threshold": thr})
            mlflow.log_metrics({f"precision_at_{k}": res[f"precision@{k}"],
                                f"recall_at_{k}": res[f"recall@{k}"],
                                f"ndcg_at_{k}": res[f"ndcg@{k}"], "coverage": res["coverage"],
                                "diversity": res["diversity"], "rmse": rmse, "mae": mae,
                                "users_evaluated": info["users_evaluated"]})
        log.info("%s: P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f coverage=%.3f diversity=%.3f RMSE=%.3f MAE=%.3f",
                 name, k, res[f"precision@{k}"], k, res[f"recall@{k}"], k, res[f"ndcg@{k}"],
                 res["coverage"], res["diversity"], rmse, mae)

    results = pd.DataFrame(rows)
    results.round(4).to_csv(rep / "model_results.csv", index=False)
    (rep / "evaluation_info.json").write_text(json.dumps(info, indent=2))
    _write_markdown(results, info, rep / "model_results.md")
    for name in run_ids:
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_artifact(str(rep / "model_results.md"))
    return results


def _write_markdown(results, info, path):
    k = info["k"]
    pretty = {"popularity_baseline": "Popularity baseline",
              "svd_matrix_factorization": "SVD matrix factorisation"}
    lines = ["# Model Evaluation Results", "",
             f"K = {k}; a test item is relevant if its rating >= {info['relevance_threshold']}. "
             f"Evaluated on {info['users_evaluated']} users with training history "
             f"({info['test_rows_evaluated_warm']} test ratings of known users and items; "
             f"{info['test_users_cold_skipped']} cold-start test users skipped).", "",
             f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | Coverage | Diversity | RMSE | MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in results.iterrows():
        def pm(m):
            return f"{r[f'{m}@{k}']:.4f} (±{r[f'{m}@{k}_se']:.4f})"
        lines.append(f"| {pretty[r['model']]} | {pm('precision')} | {pm('recall')} | {pm('ndcg')} | "
                     f"{r['coverage']:.4f} | {r['diversity']:.4f} | {r['rmse']:.4f} | {r['mae']:.4f} |")
    lines += ["", "Values in brackets are standard errors across users.",
              "Coverage = share of the item catalogue appearing in any user's top-K list. "
              "Diversity = mean pairwise genre dissimilarity (1 - cosine) inside a top-K list."]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    run()
