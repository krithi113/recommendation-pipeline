"""Stage 5 - Model training. Trains both models and tracks them with MLflow."""
import json
import time

import joblib
import mlflow
import pandas as pd
from scipy import sparse

from src.features.build_features import FeatureStore
from src.training.models import PopularityRecommender, SVDRecommender
from src.utils import ROOT, get_logger, load_config

log = get_logger(__name__)


def setup_mlflow(cfg):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def run() -> dict:
    cfg = load_config()
    setup_mlflow(cfg)
    proc, models_dir = ROOT / cfg["paths"]["processed"], ROOT / cfg["paths"]["models"]
    rep = ROOT / cfg["paths"]["reports"]
    models_dir.mkdir(exist_ok=True)

    fs = FeatureStore()
    train = pd.read_parquet(proc / "train.parquet")
    matrix = sparse.load_npz(proc / "user_item_matrix.npz")
    index = json.loads((proc / "matrix_index.json").read_text())
    split = json.loads((rep / "split_info.json").read_text())
    common = {"feature_version": fs.metadata["feature_version"],
              "split_cutoff_utc": split["cutoff_time_utc"], "train_rows": len(train),
              "matrix_shape": str(matrix.shape)}
    run_ids = {}

    # ---------------- baseline ----------------
    with mlflow.start_run(run_name="popularity_baseline") as run:
        mlflow.log_params({**common, "model": "popularity",
                           "ranking_feature": "item_popularity_count"})
        t = time.time()
        model = PopularityRecommender().fit(fs, train)
        mlflow.log_metric("train_seconds", time.time() - t)
        path = models_dir / "popularity_baseline.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(rep / "feature_catalog.md"))
        run_ids["popularity_baseline"] = run.info.run_id
        log.info("Trained popularity baseline (run %s)", run.info.run_id)

    # ---------------- personalised SVD ----------------
    p = cfg["models"]["svd"]
    with mlflow.start_run(run_name="svd_matrix_factorization") as run:
        mlflow.log_params({**common, "model": "truncated_svd", "n_components": p["n_components"],
                           "random_state": p["random_state"], "centering_feature": "user_avg_rating"})
        t = time.time()
        model = SVDRecommender(p["n_components"], p["random_state"]).fit(
            fs, matrix, index["user_ids"], index["item_ids"])
        mlflow.log_metric("train_seconds", time.time() - t)
        mlflow.log_metric("explained_variance_ratio", model.explained_variance)
        path = models_dir / "svd_matrix_factorization.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(rep / "feature_catalog.md"))
        run_ids["svd_matrix_factorization"] = run.info.run_id
        log.info("Trained SVD (%d factors, explained variance %.3f, run %s)",
                 p["n_components"], model.explained_variance, run.info.run_id)

    (models_dir / "run_info.json").write_text(json.dumps(run_ids, indent=2))
    return run_ids


if __name__ == "__main__":
    run()
