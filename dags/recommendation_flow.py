"""Orchestration - Prefect flow that runs the complete recommendation pipeline.

    ingest -> inject faults -> validate -> prepare -> transform -> features -> train -> evaluate

Each task receives the result of the previous one, so Prefect builds the dependency chain
(a task never starts before its upstream task has finished, and a failure stops the chain).

Run once :  python dags/recommendation_flow.py
Scheduled:  python dags/recommendation_flow.py --serve      (daily at 02:00)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from prefect import flow, get_run_logger, task  # noqa: E402

from src.evaluation import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import ingest, inject_faults  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.validation import validate  # noqa: E402


@task(name="1-ingest-data", retries=2, retry_delay_seconds=10)
def ingest_task():
    result = ingest.run()
    get_run_logger().info("Ingested %s", result)
    return result


@task(name="2-inject-faults")
def inject_faults_task(upstream):
    if not load_config()["validation"]["use_faulty_data"]:
        get_run_logger().info("Fault injection disabled in config - skipped")
        return {}
    result = inject_faults.run()
    get_run_logger().info("Injected faults: %s", result)
    return result


@task(name="3-validate-data")
def validate_task(upstream):
    summary = validate.run()
    get_run_logger().info("Validation summary: %s", {t: (s["valid"], s["rejected"])
                                                      for t, s in summary.items()})
    return summary


@task(name="4-prepare-data")
def prepare_task(upstream):
    result = prepare.run()
    get_run_logger().info("Prepared %s", result)
    return result


@task(name="5-transform-and-split")
def transform_task(upstream):
    info = transform.run()
    get_run_logger().info("Time split at %s: train=%d test=%d", info["cutoff_time_utc"],
                          info["train_rows"], info["test_rows"])
    return info


@task(name="6-build-features")
def features_task(upstream):
    meta = build_features.run()
    get_run_logger().info("Feature store version %s built: %s", meta["feature_version"], meta["rows"])
    return meta


@task(name="7-train-models")
def train_task(upstream):
    run_ids = train.run()
    get_run_logger().info("MLflow runs: %s", run_ids)
    return run_ids


@task(name="8-evaluate-models")
def evaluate_task(upstream):
    results = evaluate.run()
    get_run_logger().info("Results:\n%s", results[["model"] + [c for c in results.columns
                          if c.startswith(("precision@", "recall@", "ndcg@"))
                          and not c.endswith("_se")] + ["coverage", "diversity", "rmse", "mae"]]
                          .round(4).to_string(index=False))
    return results.round(4).to_dict("records")


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline():
    ingested = ingest_task()
    faults = inject_faults_task(ingested)
    validated = validate_task(faults)
    prepared = prepare_task(validated)
    split = transform_task(prepared)
    features = features_task(split)
    runs = train_task(features)
    return evaluate_task(runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true",
                        help="register a deployment that runs the pipeline daily at 02:00")
    args = parser.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="recommendation-pipeline-daily", cron="0 2 * * *")
    else:
        recommendation_pipeline()
