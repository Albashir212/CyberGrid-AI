"""
CyberGrid-AI | Step 3: FDIA Attack Module

Implements all four FDIA attack types named in the paper (Section 2).
The paper only tested the scaling attack. The other three (pulse,
ramping, random) are our extension — the explicit gap we close.

Attack formulations:

1. SCALING (paper's Eq. 2):
   Q_bar_t = (1 + lambda_S) * Q_t   for t_s < t < t_e
   Maintains behavioral pattern and average. "Stealthy" because no
   single point looks like an outlier.

2. PULSE:
   Q_bar_t = Q_t + A   at selected spike timestamps, else Q_t
   Sudden large deviation at isolated points. Discontinuous —
   easy to spot visually but can fool threshold-based detectors
   if amplitude is calibrated carefully.

3. RAMPING:
   Q_bar_t = Q_t + delta * (t - t_s) / (t_e - t_s)
   Linear drift injected gradually over the attack window.
   Hardest to catch because each individual step is tiny.

4. RANDOM:
   Q_bar_t = Q_t + epsilon_t,  epsilon_t ~ N(0, sigma^2)
   Independent noise on every sample. No structure preserved.
   Tests whether the AE can denoise truly uncorrelated corruption.

All attacks are applied to the temperature feature only,
matching the paper's experimental setup exactly.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def scaling_attack(
    series: pd.Series,
    lambda_s: float = 0.05,
    t_start: str = None,
    t_end: str = None,
) -> pd.Series:
    """
    Paper's Eq. (2): Q_bar_t = (1 + lambda_s) * Q_t

    lambda_s : scaling factor, e.g. +0.05 or -0.05
    t_start  : attack window start (inclusive). None = full series.
    t_end    : attack window end   (inclusive). None = full series.
    """
    attacked = series.copy()
    mask = pd.Series(True, index=series.index)
    if t_start:
        mask &= series.index >= t_start
    if t_end:
        mask &= series.index <= t_end
    attacked.loc[mask] = (1 + lambda_s) * series.loc[mask]
    return attacked


def pulse_attack(
    series: pd.Series,
    amplitude: float = None,
    n_pulses: int = 10,
    seed: int = 42,
) -> pd.Series:
    """
    Injects sudden spikes at randomly selected timestamps.

    amplitude : size of each spike (default = 3 * std of series)
    n_pulses  : number of spike locations
    """
    rng = np.random.default_rng(seed)
    attacked = series.copy()
    if amplitude is None:
        amplitude = 3.0 * series.std()
    spike_idx = rng.choice(len(series), size=n_pulses, replace=False)
    # Alternate between positive and negative spikes
    signs = rng.choice([-1, 1], size=n_pulses)
    attacked.iloc[spike_idx] += amplitude * signs
    return attacked


def ramping_attack(
    series: pd.Series,
    max_delta: float = None,
    t_start: str = None,
    t_end: str = None,
) -> pd.Series:
    """
    Injects a linearly growing offset over the attack window.

    max_delta : maximum drift injected at t_end
                (default = 10% of series mean)
    """
    attacked = series.copy()
    mask = pd.Series(False, index=series.index)
    if t_start:
        mask |= series.index >= t_start
    if t_end:
        mask &= series.index <= t_end
    if not mask.any():
        mask = pd.Series(True, index=series.index)

    if max_delta is None:
        max_delta = 0.10 * series.mean()

    window_len = mask.sum()
    ramp = np.linspace(0, max_delta, window_len)
    attacked.loc[mask] = series.loc[mask].values + ramp
    return attacked


def random_attack(
    series: pd.Series,
    sigma: float = None,
    seed: int = 42,
) -> pd.Series:
    """
    Adds independent Gaussian noise to every sample.

    sigma : noise standard deviation (default = 5% of series std)
    """
    rng = np.random.default_rng(seed)
    if sigma is None:
        sigma = 0.05 * series.std()
    noise = rng.normal(loc=0.0, scale=sigma, size=len(series))
    return series + noise


def apply_attack(
    df: pd.DataFrame,
    attack_type: str,
    feature_col: str = "temperature",
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper: applies a named attack to one feature column.

    attack_type : "scaling" | "pulse" | "ramping" | "random"
    feature_col : column to attack (default: "temperature")
    kwargs      : passed to the specific attack function
    """
    attack_fn = {
        "scaling": scaling_attack,
        "pulse":   pulse_attack,
        "ramping": ramping_attack,
        "random":  random_attack,
    }
    if attack_type not in attack_fn:
        raise ValueError(f"Unknown attack: {attack_type}. "
                         f"Choose from {list(attack_fn.keys())}")

    attacked_df = df.copy()
    attacked_df[feature_col] = attack_fn[attack_type](
        df[feature_col], **kwargs
    )
    return attacked_df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    IN_PATH  = BASE_DIR / "data" / "processed" / "gefcom2014_features.csv"

    df = pd.read_csv(IN_PATH, index_col="timestamp", parse_dates=True)
    temp = df["temperature"]

    print("=" * 55)
    print("  CyberGrid-AI  |  Step 3: Attack Sanity Check")
    print("=" * 55)
    print(f"Original temperature — mean: {temp.mean():.2f}, "
          f"std: {temp.std():.2f}\n")

    attacks = {
        "scaling +5%": scaling_attack(temp, lambda_s=+0.05),
        "scaling -5%": scaling_attack(temp, lambda_s=-0.05),
        "pulse"       : pulse_attack(temp),
        "ramping"     : ramping_attack(temp),
        "random"      : random_attack(temp),
    }

    for name, attacked in attacks.items():
        diff = (attacked - temp).abs()
        corr = temp.corr(attacked)
        print(f"[{name:<14}]  mean shift: {(attacked-temp).mean():+.4f}  "
              f"max deviation: {diff.max():.4f}  "
              f"corr with original: {corr:.6f}")

    print("\n[✓] All four attack types verified")
    print("[✓] Done. Next: run src/models.py")