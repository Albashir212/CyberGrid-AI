"""
CyberGrid-AI | Step 5: Training Pipeline

Trains all four models (CNN, LSTM, AE-CNN, AE-CLSTM) on GEFCom2014
load data for two forecast horizons:
  - Ultra-short-term: t+1 hour
  - Short-term      : t+1 week (168 hours)

Experimental design mirrors the paper exactly:
  - Train : 2005-2010 (6 years)
  - Test  : 2011 (1 year, evaluated per season)
  - Attack: scaling attack ±5% on temperature (paper's Scenario 2)

Results are saved to results/ as CSV tables matching
the format of Tables 3-5 in the paper.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
import joblib

# Suppress TF noise
tf.get_logger().setLevel("ERROR")

from models import build_cnn, build_lstm, build_ae_cnn, build_ae_clstm
from attacks import apply_attack

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
IN_PATH      = BASE_DIR / "data" / "processed" / "gefcom2014_features.csv"
MODELS_DIR   = BASE_DIR / "results" / "models"
RESULTS_DIR  = BASE_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
WINDOW_SIZE   = 24          # 24-hour lookback window
TRAIN_END     = "2010-12-31 23:00:00"
TEST_START    = "2011-01-01 00:00:00"
EPOCHS        = 50
BATCH_SIZE    = 32
PATIENCE      = 7           # early stopping patience

FEATURE_COLS  = [
    "load", "temperature",
    "day_of_month", "day_of_week", "hour_of_day", "is_holiday"
]
N_FEATURES    = len(FEATURE_COLS)

SEASON_MAP = {
    1: "Winter", 2: "Winter",  3: "Spring",
    4: "Spring", 5: "Spring",  6: "Summer",
    7: "Summer", 8: "Summer",  9: "Autumn",
   10: "Autumn", 11: "Autumn", 12: "Winter"
}


# ── Helpers ────────────────────────────────────────────────────────────
def make_windows(features: np.ndarray,
                 targets: np.ndarray,
                 window: int):
    """Convert flat arrays into (sample, window, feature) tensors."""
    X, y = [], []
    for i in range(window, len(features)):
        X.append(features[i - window:i])
        y.append(targets[i])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def compute_metrics(y_true, y_pred):
    """R, MSE, MAE, RMSE — matching the paper's Table 3-5 columns."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot
    r    = np.sqrt(max(r2, 0)) * 100   # correlation % like the paper
    return {"R(%)": round(r, 2),
            "MSE":  round(mse, 3),
            "MAE":  round(mae, 3),
            "RMSE": round(rmse, 3)}


def get_season(month):
    return SEASON_MAP[month]


def train_model(model, X_train, y_train, name):
    """Train with Adam + early stopping. Returns trained model."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=0
        )
    ]
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=cb,
        verbose=0
    )
    return model


# ── Main pipeline ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CyberGrid-AI  |  Step 5: Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────
    df = pd.read_csv(IN_PATH, index_col="timestamp", parse_dates=True)
    print(f"[✓] Loaded {len(df):,} rows")

    train_df = df.loc[:TRAIN_END].copy()
    test_df  = df.loc[TEST_START:].copy()
    print(f"[✓] Train: {len(train_df):,} rows  |  "
          f"Test: {len(test_df):,} rows")

    # ── 2. Scale features ──────────────────────────────────────────
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURE_COLS])
    test_scaled  = scaler.transform(test_df[FEATURE_COLS])

    # Save scaler for later use in conformal module
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("[✓] Features scaled (MinMax) and scaler saved")

    # Load scaler for target inverse transform
    load_idx = FEATURE_COLS.index("load")
    load_min  = scaler.data_min_[load_idx]
    load_max  = scaler.data_max_[load_idx]

    def inv_load(arr):
        return arr * (load_max - load_min) + load_min

    results_rows = []

    for horizon, target_col in [("1h", "target_1h"), ("1w", "target_1w")]:
        print(f"\n{'─'*60}")
        print(f"  Horizon: {horizon}  |  target: {target_col}")
        print(f"{'─'*60}")

        # Scale targets with load scaler range
        train_targets = (train_df[target_col].values - load_min) / \
                        (load_max - load_min)
        test_targets  = (test_df[target_col].values - load_min) / \
                        (load_max - load_min)

        # Build windowed arrays
        X_train, y_train = make_windows(train_scaled, train_targets,
                                        WINDOW_SIZE)
        X_test,  y_test  = make_windows(test_scaled,  test_targets,
                                        WINDOW_SIZE)

        print(f"[✓] X_train: {X_train.shape}  X_test: {X_test.shape}")

        # Season labels for test rows (aligned to windowed output)
        test_months = test_df.index[WINDOW_SIZE:].month
        test_seasons = [get_season(m) for m in test_months]

        builders = {
            "CNN"     : build_cnn,
            "LSTM"    : build_lstm,
            "AE_CNN"  : build_ae_cnn,
            "AE_CLSTM": build_ae_clstm,
        }

        for model_name, builder in builders.items():
            print(f"\n  Training {model_name}...", end=" ", flush=True)
            model = builder(WINDOW_SIZE, N_FEATURES)
            model = train_model(model, X_train, y_train, model_name)
            model.save(MODELS_DIR / f"{model_name}_{horizon}.keras")
            print("done")

            # ── Scenario 1: Clean data ─────────────────────────────
            preds_scaled = model.predict(X_test, verbose=0)
            preds = inv_load(preds_scaled)
            truth = inv_load(y_test.reshape(-1, 1))

            for season in ["Spring", "Summer", "Autumn", "Winter"]:
                mask = np.array(test_seasons) == season
                if mask.sum() == 0:
                    continue
                m = compute_metrics(truth[mask], preds[mask])
                results_rows.append({
                    "Scenario": "Clean",
                    "Horizon" : horizon,
                    "Season"  : season,
                    "Model"   : model_name,
                    **m
                })

            # ── Scenario 2: Scaling attack ─────────────────────────
            # Apply ±5% scaling to temperature — paper's exact setup
            for direction, lambda_s in [("decrease", -0.05),
                                        ("increase", +0.05)]:
                test_attacked = apply_attack(
                    test_df, "scaling",
                    feature_col="temperature",
                    lambda_s=lambda_s
                )
                test_att_scaled = scaler.transform(
                    test_attacked[FEATURE_COLS]
                )
                X_att, y_att = make_windows(
                    test_att_scaled, test_targets, WINDOW_SIZE
                )
                preds_att = inv_load(
                    model.predict(X_att, verbose=0)
                )

                for season in ["Spring", "Summer", "Autumn", "Winter"]:
                    mask = np.array(test_seasons) == season
                    if mask.sum() == 0:
                        continue
                    m = compute_metrics(truth[mask], preds_att[mask])
                    results_rows.append({
                        "Scenario": f"FDIA_{direction}_5pct",
                        "Horizon" : horizon,
                        "Season"  : season,
                        "Model"   : model_name,
                        **m
                    })

    # ── 3. Save results ────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows)
    out_path = RESULTS_DIR / "results_table.csv"
    results_df.to_csv(out_path, index=False)

    # ── 4. Print summary (Clean, 1h horizon) ──────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY — Clean Data | 1-hour horizon")
    print(f"{'='*60}")
    clean_1h = results_df[
        (results_df["Scenario"] == "Clean") &
        (results_df["Horizon"]  == "1h")
    ].pivot_table(
        index="Season", columns="Model",
        values="RMSE", aggfunc="mean"
    ).round(3)
    print(clean_1h.to_string())

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY — Scaling Attack -5% | 1-hour horizon")
    print(f"{'='*60}")
    att_1h = results_df[
        (results_df["Scenario"] == "FDIA_decrease_5pct") &
        (results_df["Horizon"]  == "1h")
    ].pivot_table(
        index="Season", columns="Model",
        values="RMSE", aggfunc="mean"
    ).round(3)
    print(att_1h.to_string())

    print(f"\n[✓] Full results saved to: {out_path}")
    print("[✓] Trained models saved to: results/models/")
    print("[✓] Done. Next: run src/conformal.py")


if __name__ == "__main__":
    main()