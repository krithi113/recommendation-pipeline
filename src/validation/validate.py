"""Stage 2 - Data validation.

Every table is checked automatically (types, missing values, ranges, duplicates and
referential integrity). Pandera performs the column checks; duplicates and
cross-table checks use pandas. Valid rows go to data/processed/, rejected rows go to
data/errors/ with an `error_reason`, and a report is written to reports/.
"""
import json
from datetime import datetime, timezone

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # older pandera versions
    import pandera as pa

from src.ingestion.ingest import GENRES
from src.utils import ROOT, get_logger, load_config

log = get_logger(__name__)


# ------------------------------------------------------------------ helpers
def _flag(reason: pd.Series, mask, label: str) -> None:
    """Give a row its FIRST failing reason only, so each bad row is counted once."""
    reason.loc[mask & reason.isna()] = label


def _coerce_numeric(df: pd.DataFrame, cols, reason: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        num = pd.to_numeric(df[c], errors="coerce")
        _flag(reason, df[c].notna() & num.isna(), "bad_dtype")   # text where a number belongs
        out[c] = num
    return out


def _label(column: str, check: str) -> str:
    if check == "not_nullable":
        return f"missing_{column}"
    return {"rating": "invalid_rating", "timestamp": "invalid_timestamp",
            "age": "invalid_age", "gender": "invalid_gender"}.get(column, f"invalid_{column}")


def _pandera_flags(df: pd.DataFrame, schema, reason: pd.Series) -> None:
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        fc = exc.failure_cases.dropna(subset=["index"])
        for _, r in fc.iterrows():
            _flag(reason, df.index == r["index"], _label(r["column"], r["check"]))


def _split(raw: pd.DataFrame, reason: pd.Series):
    errors = raw[reason.notna()].copy()
    errors["error_reason"] = reason[reason.notna()]
    valid_mask = reason.isna()
    return valid_mask, errors


# ------------------------------------------------------------------ tables
def validate_users(raw: pd.DataFrame, cfg: dict):
    v = cfg["validation"]
    reason = pd.Series(pd.NA, index=raw.index, dtype="object")
    df = _coerce_numeric(raw, ["user_id", "age"], reason)
    schema = pa.DataFrameSchema({
        "user_id": pa.Column(float, nullable=False),
        "age": pa.Column(float, pa.Check.in_range(v["age_min"], v["age_max"]), nullable=False),
        "gender": pa.Column(object, pa.Check.isin(["M", "F"]), nullable=False),
        "occupation": pa.Column(object, nullable=False),
    })
    _pandera_flags(df, schema, reason)
    dup = df[reason.isna()].duplicated(subset=["user_id"], keep="first")
    _flag(reason, df.index.isin(dup[dup].index), "duplicate")
    valid_mask, errors = _split(raw, reason)
    valid = df[valid_mask].astype({"user_id": int, "age": int})
    return valid, errors


def validate_items(raw: pd.DataFrame, cfg: dict):
    reason = pd.Series(pd.NA, index=raw.index, dtype="object")
    df = _coerce_numeric(raw, ["item_id"] + GENRES, reason)
    schema = pa.DataFrameSchema({
        "item_id": pa.Column(float, nullable=False),
        "title": pa.Column(object, nullable=False),
    })
    _pandera_flags(df, schema, reason)
    dup = df[reason.isna()].duplicated(subset=["item_id"], keep="first")
    _flag(reason, df.index.isin(dup[dup].index), "duplicate")
    valid_mask, errors = _split(raw, reason)
    valid = df[valid_mask].astype({c: int for c in ["item_id"] + GENRES})
    return valid, errors


def validate_interactions(raw: pd.DataFrame, valid_users, valid_items, cfg: dict):
    v = cfg["validation"]
    now_ts = int(datetime.now(timezone.utc).timestamp())
    reason = pd.Series(pd.NA, index=raw.index, dtype="object")
    df = _coerce_numeric(raw, ["user_id", "item_id", "rating", "timestamp"], reason)
    schema = pa.DataFrameSchema({
        "user_id": pa.Column(float, nullable=False),
        "item_id": pa.Column(float, nullable=False),
        "rating": pa.Column(float, pa.Check.in_range(v["rating_min"], v["rating_max"]),
                            nullable=False),
        "timestamp": pa.Column(float, pa.Check.in_range(v["min_valid_timestamp"], now_ts),
                               nullable=False),
    })
    _pandera_flags(df, schema, reason)
    # referential integrity: every interaction must point to a valid user and item
    _flag(reason, ~df["user_id"].isin(valid_users["user_id"]), "orphan_user")
    _flag(reason, ~df["item_id"].isin(valid_items["item_id"]), "orphan_item")
    dup = df[reason.isna()].duplicated(subset=["user_id", "item_id"], keep="first")
    _flag(reason, df.index.isin(dup[dup].index), "duplicate")
    valid_mask, errors = _split(raw, reason)
    valid = df[valid_mask].astype(
        {"user_id": int, "item_id": int, "rating": int, "timestamp": int})
    return valid, errors


# ------------------------------------------------------------------ report
def _write_report(summary: dict, injected: dict, path) -> None:
    lines = ["# Data Validation Report", "",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             "## Summary", "",
             "| Table | Rows in | Valid | Rejected |", "|---|---:|---:|---:|"]
    for t, s in summary.items():
        lines.append(f"| {t} | {s['rows_in']} | {s['valid']} | {s['rejected']} |")
    lines += ["", "## Rejected rows by reason (detected vs injected)", "",
              "| Table | Reason | Detected | Injected | Match |", "|---|---|---:|---:|:--:|"]
    for t, s in summary.items():
        for reason, n in s["reasons"].items():
            inj = injected.get(t, {}).get(reason)
            match = "" if inj is None else ("yes" if inj == n else "NO")
            lines.append(f"| {t} | {reason} | {n} | {'' if inj is None else inj} | {match} |")
    lines += ["", "## Checks implemented", "",
              "- Missing values in required columns",
              "- Incorrect data types (text where a number is expected)",
              "- Invalid ratings (outside 1-5)",
              "- Invalid timestamps (<= 1990 or in the future)",
              "- Invalid user age and gender values",
              "- Duplicate users, items and (user, item) interactions",
              "- Referential integrity (interactions must reference a valid user and item)",
              "", "Rejected rows are stored in `data/errors/`, valid rows in `data/processed/`."]
    path.write_text("\n".join(lines))


def run() -> dict:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    src = raw_dir / "with_faults" if cfg["validation"]["use_faulty_data"] else raw_dir
    out_dir = ROOT / cfg["paths"]["processed"]
    err_dir = ROOT / cfg["paths"]["errors"]
    rep_dir = ROOT / cfg["paths"]["reports"]
    for d in (out_dir, err_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)
    log.info("Validating tables from %s", src)

    raw = {t: pd.read_csv(src / f"{t}.csv", dtype=str) for t in ("users", "items", "interactions")}
    users, users_err = validate_users(raw["users"], cfg)
    items, items_err = validate_items(raw["items"], cfg)
    inter, inter_err = validate_interactions(raw["interactions"], users, items, cfg)

    results = {"users": (users, users_err), "items": (items, items_err),
               "interactions": (inter, inter_err)}
    summary = {}
    for t, (valid, err) in results.items():
        valid.reset_index(drop=True).to_parquet(out_dir / f"valid_{t}.parquet", index=False)
        err.to_csv(err_dir / f"{t}_errors.csv", index=False)
        reasons = err["error_reason"].value_counts().to_dict()
        summary[t] = {"rows_in": len(raw[t]), "valid": len(valid),
                      "rejected": len(err), "reasons": reasons}
        log.info("%s: in=%d valid=%d rejected=%d reasons=%s",
                 t, len(raw[t]), len(valid), len(err), reasons)

    fault_log = rep_dir / "fault_injection_log.json"
    injected = json.loads(fault_log.read_text()) if fault_log.exists() else {}
    (rep_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(summary, injected, rep_dir / "validation_report.md")
    log.info("Validation report written to %s", rep_dir / "validation_report.md")
    return summary


if __name__ == "__main__":
    run()
