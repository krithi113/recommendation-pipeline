"""Stage 1b - Controlled fault injection.

MovieLens is almost perfectly clean, so the validation stage would find nothing.
This script copies the raw tables into data/raw/with_faults/ and adds a known number
of data-quality problems (fixed random seed). The exact counts are written to
reports/fault_injection_log.json so the validation report can compare
"injected" against "detected".
"""
import json

import numpy as np
import pandas as pd

from src.utils import ROOT, get_logger, load_config

log = get_logger(__name__)

FUTURE_TS = "4102444800"  # 2100-01-01


def run() -> dict:
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw"]
    out = raw / "with_faults"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["data"]["seed"])

    users = pd.read_csv(raw / "users.csv", dtype=str)
    items = pd.read_csv(raw / "items.csv", dtype=str)
    inter = pd.read_csv(raw / "interactions.csv", dtype=str)

    # ---------------- interactions ----------------
    spec = [("invalid_rating", 30), ("missing_rating", 20), ("bad_dtype", 10),
            ("ts_negative", 15), ("ts_future", 15),
            ("orphan_user", 20), ("orphan_item", 20), ("duplicate", 50)]
    perm = rng.permutation(len(inter))  # disjoint rows for every fault type
    idx, start = {}, 0
    for name, k in spec:
        idx[name] = perm[start:start + k]
        start += k

    inter.loc[idx["invalid_rating"], "rating"] = rng.choice(["0", "6", "7", "-1"], size=30)
    inter.loc[idx["missing_rating"], "rating"] = np.nan
    inter.loc[idx["bad_dtype"], "rating"] = rng.choice(["five", "great", "n.a"], size=10)
    inter.loc[idx["ts_negative"], "timestamp"] = "-100"
    inter.loc[idx["ts_future"], "timestamp"] = FUTURE_TS
    inter.loc[idx["orphan_user"], "user_id"] = (9000 + rng.integers(0, 500, size=20)).astype(str)
    inter.loc[idx["orphan_item"], "item_id"] = (9000 + rng.integers(0, 500, size=20)).astype(str)
    dup_rows = inter.iloc[idx["duplicate"]].copy()        # untouched rows, copied
    inter = pd.concat([inter, dup_rows], ignore_index=True)

    # ---------------- users (new bad rows have no interactions) ----------------
    bad_users = pd.DataFrame({
        "user_id": [str(i) for i in range(2001, 2011)],
        "age": ["-5", "150", "0", "999", np.nan, np.nan, np.nan, "30", "30", "30"],
        "gender": ["M"] * 7 + ["X"] * 3,
        "occupation": ["student"] * 10,
        "zip_code": ["00000"] * 10})
    dup_users = users.sample(3, random_state=cfg["data"]["seed"])
    users = pd.concat([users, bad_users, dup_users], ignore_index=True)

    # ---------------- items ----------------
    bad_items = items.iloc[:4].copy()
    bad_items["item_id"] = ["3001", "3002", "3003", "3004"]
    bad_items["title"] = np.nan
    dup_items = items.sample(3, random_state=cfg["data"]["seed"])
    items = pd.concat([items, bad_items, dup_items], ignore_index=True)

    users.to_csv(out / "users.csv", index=False)
    items.to_csv(out / "items.csv", index=False)
    inter.to_csv(out / "interactions.csv", index=False)

    injected = {
        "interactions": {"bad_dtype": 10, "missing_rating": 20, "invalid_rating": 30,
                         "invalid_timestamp": 30, "orphan_user": 20, "orphan_item": 20,
                         "duplicate": 50},
        "users": {"invalid_age": 4, "missing_age": 3, "invalid_gender": 3, "duplicate": 3},
        "items": {"missing_title": 4, "duplicate": 3},
    }
    (ROOT / cfg["paths"]["reports"]).mkdir(exist_ok=True)
    with open(ROOT / cfg["paths"]["reports"] / "fault_injection_log.json", "w") as f:
        json.dump(injected, f, indent=2)
    log.info("Fault injection done: users=%d items=%d interactions=%d (written to %s)",
             len(users), len(items), len(inter), out)
    return injected


if __name__ == "__main__":
    run()
