"""
CyberGrid-AI | Step 2: Feature Engineering

Builds the model-ready feature set from the cleaned GEFCom2014 series.

Matches the original paper's input variables as closely as the dataset allows:
  Paper used : avg temperature, humidity, pressure, wind speed, wind direction
               + day-of-month, day-of-week, day-type, hour-of-day
  We have    : 25 temperature stations only (no humidity/pressure/wind in
               GEFCom2014). Temperature is the only variable the paper attacks,
               so this is sufficient for full replication.

Design decision (documented):
  The 25 station readings are averaged into one composite temperature series,
  matching the paper's single-station setup and keeping the attack surface
  identical to the original experiment.

Outputs:
  data/processed/gefcom2014_features.csv
  Columns: load, temperature, day_of_month, day_of_week, hour_of_day,
           is_holiday, season, target_1h, target_1w
"""

import pandas as pd
import holidays
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
IN_PATH   = BASE_DIR / "data" / "processed" / "gefcom2014_load_temp_clean.csv"
OUT_PATH  = BASE_DIR / "data" / "processed" / "gefcom2014_features.csv"

# US holidays covering the dataset window (ISO New England data)
US_HOLIDAYS = holidays.US(years=range(2005, 2012))

SEASON_MAP = {
    12: "Winter", 1: "Winter",  2: "Winter",
     3: "Spring", 4: "Spring",  5: "Spring",
     6: "Summer", 7: "Summer",  8: "Summer",
     9: "Autumn", 10: "Autumn", 11: "Autumn",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    temp_cols = [c for c in df.columns if c.startswith("w")]
    out = pd.DataFrame(index=df.index)

    # Core signals
    out["load"]        = df["LOAD"]
    out["temperature"] = df[temp_cols].mean(axis=1)   # composite temperature

    # Calendar features (matching the paper's input set)
    out["day_of_month"] = out.index.day
    out["day_of_week"]  = out.index.dayofweek + 1     # 1=Mon ... 7=Sun
    out["hour_of_day"]  = out.index.hour + 1          # 1-24
    out["is_holiday"]   = out.index.to_series().apply(
        lambda d: 1 if (d.date() in US_HOLIDAYS or d.dayofweek >= 5) else 0
    ).values   # 0=workday, 1=weekend/holiday — matches paper's binary day-type

    # Season label (used to split test results by season, matching Tables 3-5)
    out["season"] = out.index.month.map(SEASON_MAP)

    # Forecast targets
    out["target_1h"] = out["load"].shift(-1)     # ultra-short-term: t+1 hour
    out["target_1w"] = out["load"].shift(-168)   # short-term: t+1 week

    # Drop trailing rows where 1-week target is undefined
    out = out.dropna(subset=["target_1w"])
    return out


def main():
    print("=" * 55)
    print("  CyberGrid-AI  |  Step 2: Feature Engineering")
    print("=" * 55)

    df = pd.read_csv(IN_PATH, index_col="timestamp", parse_dates=True)
    print(f"[✓] Loaded {len(df):,} rows from cleaned dataset")

    feats = build_features(df)

    print(f"[✓] Features built")
    print(f"    Columns : {list(feats.columns)}")
    print(f"    Shape   : {feats.shape}")
    print(f"\n    Season breakdown:")
    for season, count in feats["season"].value_counts().items():
        print(f"      {season:<8}: {count:,} rows")

    feats.to_csv(OUT_PATH)
    print(f"\n[✓] Saved to: {OUT_PATH}")
    print("[✓] Done. Next: run src/attacks.py")


if __name__ == "__main__":
    main()