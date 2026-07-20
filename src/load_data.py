"""
CyberGrid-AI | Step 1: Download and load GEFCom2014 dataset.

Downloads the GEFCom2014 load forecasting competition data, extracts
the load track, and concatenates all 15 task files into one clean
hourly time series of LOAD + 25 temperature stations.

Output: data/processed/gefcom2014_load_temp_clean.csv
  - 60,432 hourly rows, zero missing values
  - Date range: 2005-01-01 to 2011-12-01
  - Columns: LOAD, w1..w25 (temperature stations)

Why timestamps are reconstructed rather than parsed:
  The raw TIMESTAMP field has no delimiter between month/day/year
  (e.g. "112001 1:00" = Jan 1 2001 01:00), making it genuinely
  ambiguous for single-digit months/days. Each task file is
  contiguous hourly data with no gaps at boundaries, so we
  regenerate a clean DatetimeIndex from a known anchor instead.
"""

import zipfile
import urllib.request
from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_DIR        = BASE_DIR / "data" / "raw"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
LOAD_TRACK_DIR = RAW_DIR / "load_track"
MAIN_ZIP       = RAW_DIR / "GEFCom2014.zip"
OUT_CLEAN      = PROCESSED_DIR / "gefcom2014_load_temp_clean.csv"

# ── Constants ──────────────────────────────────────────────────────────
DOWNLOAD_URL = "https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=1"
ANCHOR_START = "2001-01-01 01:00:00"   # first timestamp in Task 1
CLEAN_START  = "2005-01-01 01:00:00"   # load labels only exist from 2005
N_TASKS      = 15


def download_data():
    """Download GEFCom2014 archive if not already present."""
    if MAIN_ZIP.exists():
        print(f"[✓] Already downloaded: {MAIN_ZIP.name}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("[↓] Downloading GEFCom2014 (~120 MB) — this may take a few minutes...")
    urllib.request.urlretrieve(DOWNLOAD_URL, MAIN_ZIP)
    print(f"[✓] Saved to {MAIN_ZIP}")


def extract_data():
    """Extract the load track zip from the main archive."""
    if LOAD_TRACK_DIR.exists():
        print(f"[✓] Already extracted: {LOAD_TRACK_DIR.name}")
        return

    print("[↗] Extracting main archive...")
    with zipfile.ZipFile(MAIN_ZIP, "r") as z:
        load_zip_entry = "GEFCom2014 Data/GEFCom2014-L_V2.zip"
        z.extract(load_zip_entry, RAW_DIR)

    extracted_load_zip = RAW_DIR / "GEFCom2014 Data" / "GEFCom2014-L_V2.zip"
    print("[↗] Extracting load track...")
    with zipfile.ZipFile(extracted_load_zip, "r") as z:
        z.extractall(LOAD_TRACK_DIR)
    print(f"[✓] Load track ready at {LOAD_TRACK_DIR}")


def load_all_tasks() -> pd.DataFrame:
    """Concatenate all 15 task CSV files into one continuous time series."""
    load_dir = LOAD_TRACK_DIR / "Load"
    frames = []

    for t in range(1, N_TASKS + 1):
        path = load_dir / f"Task {t}" / f"L{t}-train.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
        frames.append(df)
        print(f"  Task {t:>2}: {len(df):>6} rows")

    full = pd.concat(frames, ignore_index=True)
    print(f"\n[✓] Total rows: {len(full):,}")

    # Reconstruct unambiguous hourly DatetimeIndex from known anchor
    full["timestamp"] = pd.date_range(
        start=ANCHOR_START, periods=len(full), freq="h"
    )
    full = full.drop(columns=["TIMESTAMP", "ZONEID"])
    full = full.set_index("timestamp")
    return full


def clean_and_save(df: pd.DataFrame):
    """Trim to the fully-labeled window (2005+) and save."""
    df_clean = df.loc[CLEAN_START:].copy()

    missing = df_clean.isna().sum().sum()
    assert missing == 0, f"Unexpected {missing} missing values"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_CLEAN)

    print(f"\n[✓] Saved to: {OUT_CLEAN}")
    print(f"    Rows  : {len(df_clean):,}")
    print(f"    Range : {df_clean.index.min()}  →  {df_clean.index.max()}")
    print(f"    Cols  : LOAD + {len(df_clean.columns)-1} temperature stations")


def main():
    print("=" * 55)
    print("  CyberGrid-AI  |  Step 1: Data Download & Load")
    print("=" * 55)
    download_data()
    extract_data()
    df = load_all_tasks()
    clean_and_save(df)
    print("\n[✓] Done. Next: run src/features.py")


if __name__ == "__main__":
    main()