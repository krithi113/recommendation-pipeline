"""Stage 1 - Data ingestion: download MovieLens 100K and write raw CSV tables."""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import ROOT, get_logger, load_config

log = get_logger(__name__)

GENRES = ["unknown", "action", "adventure", "animation", "childrens", "comedy",
          "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
          "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western"]


def download(url: str, dest: Path) -> Path:
    """Download and unzip the dataset once; skip if already present."""
    base = dest / "ml-100k"
    if (base / "u.data").exists():
        log.info("Dataset already downloaded at %s - skipping download", base)
        return base
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        z.extractall(dest)
    log.info("Extracted to %s", base)
    return base


def read_raw(base: Path):
    interactions = pd.read_csv(
        base / "u.data", sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"])
    users = pd.read_csv(
        base / "u.user", sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        dtype={"zip_code": str})
    items = pd.read_csv(
        base / "u.item", sep="|", encoding="latin-1",
        names=["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES)
    return users, items, interactions


def run() -> dict:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    base = download(cfg["data"]["url"], raw_dir)
    users, items, interactions = read_raw(base)

    users.to_csv(raw_dir / "users.csv", index=False)
    items.to_csv(raw_dir / "items.csv", index=False)
    interactions.to_csv(raw_dir / "interactions.csv", index=False)

    log.info("Ingested users=%d items=%d interactions=%d",
             len(users), len(items), len(interactions))
    return {"users": len(users), "items": len(items), "interactions": len(interactions)}


if __name__ == "__main__":
    run()
