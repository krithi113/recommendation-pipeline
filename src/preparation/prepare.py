"""Stage 3a - Data preparation: clean and standardise the validated tables."""
import pandas as pd

from src.utils import ROOT, get_logger, load_config

log = get_logger(__name__)

GENRE_COLS = ["unknown", "action", "adventure", "animation", "childrens", "comedy",
              "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
              "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western"]


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gender"] = out["gender"].str.strip().str.upper()
    out["occupation"] = out["occupation"].str.strip().str.lower()
    out["zip_code"] = out["zip_code"].astype(str).str.strip()
    out["age_group"] = pd.cut(
        out["age"], bins=[0, 17, 24, 34, 44, 54, 64, 120],
        labels=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)
    out = out.drop_duplicates(subset="user_id")
    return out[["user_id", "age", "age_group", "gender", "occupation", "zip_code"]]


def clean_items(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title"] = out["title"].str.strip()
    # standardise date format and handle missing release dates (impute with median year)
    out["release_date"] = pd.to_datetime(out["release_date"], format="%d-%b-%Y", errors="coerce")
    out["release_year"] = out["release_date"].dt.year
    out["release_year_imputed"] = out["release_year"].isna()
    n_missing = int(out["release_year_imputed"].sum())
    out["release_year"] = out["release_year"].fillna(out["release_year"].median()).astype(int)
    if n_missing:
        log.info("Imputed release_year (median) for %d item(s) with missing release date", n_missing)
    out["n_genres"] = out[GENRE_COLS].sum(axis=1)
    out = out.drop(columns=["release_date", "video_release_date", "imdb_url"], errors="ignore")
    return out.drop_duplicates(subset="item_id")


def clean_interactions(df: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset=["user_id", "item_id"]).copy()
    before = len(out)
    out = out[out["user_id"].isin(users["user_id"]) & out["item_id"].isin(items["item_id"])]
    if len(out) != before:
        log.warning("Dropped %d interactions with no matching user/item", before - len(out))
    out["interaction_time"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def run() -> dict:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed"]
    users = clean_users(pd.read_parquet(proc / "valid_users.parquet"))
    items = clean_items(pd.read_parquet(proc / "valid_items.parquet"))
    inter = clean_interactions(pd.read_parquet(proc / "valid_interactions.parquet"), users, items)

    users.to_parquet(proc / "clean_users.parquet", index=False)
    items.to_parquet(proc / "clean_items.parquet", index=False)
    inter.to_parquet(proc / "clean_interactions.parquet", index=False)
    log.info("Prepared: users=%d items=%d interactions=%d", len(users), len(items), len(inter))
    return {"users": len(users), "items": len(items), "interactions": len(inter)}


if __name__ == "__main__":
    run()
