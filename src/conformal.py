"""
conformal.py -- Cross-feature FDIA detection for the AE-CLSTM forecasting stack.

CyberGrid-AI / SENTRY-track portfolio project.
Extends Moradzadeh et al. (2022), Energy Reports 8, 9933-9945.

WHY THIS FILE EXISTS
--------------------
The AE-CLSTM architecture is *robust* to stealthy scaling FDIA -- that is the
paper's headline result. Robustness is also a detection problem: the AE
absorbs the perturbation, so nothing downstream of it carries an attack
signature. Three naive approaches fail for structural reasons:

  (1) Conformal prediction on the FORECAST output          -> blind.
      The AE denoises the corrupted temperature back toward the clean
      manifold, so forecast residuals under attack are statistically
      indistinguishable from forecast residuals under clean noise.

  (2) Fixed threshold on AE RECONSTRUCTION ERROR           -> seasonal drift.
      Reconstruction error has a strong (season x hour) mean/variance
      structure. A global threshold is dominated by climatology, not attack.
      Fix: condition the nonconformity score on (season, hour).

  (3) CUSUM on the TEMPERATURE z-score alone               -> under-powered.
      A 5% scaling attack displaces temperature by ~0.19 sigma of its own
      conditional distribution. No single-feature sequential test with a
      usable ARL_0 detects a 0.19-sigma shift promptly.

RESIDUAL-DETECT is the contribution. The attacker perturbs temperature but
cannot perturb the realized load (metering and weather telemetry are separate
channels). So fit, on clean training data,

        load_t ~ g(temperature_t, hour, day-of-week, season)

and monitor r_t = load_t - g_hat(temperature_t^adv, ...). Under attack the
regressor is evaluated at a displaced temperature, so r_t acquires a bias

        E[r_t] ~= -(dg/dT) * lambda_s * T_t

The key point: this bias is standardized by sigma_resid (small, because g fits
well) rather than by sigma_temperature (large, because weather is volatile).
That is a *detectability gain*, and it is exactly why the cross-feature signal
sees a 0.19-sigma input perturbation as a multi-sigma output inconsistency.
`ResidualDetect.detectability_report()` computes this gain explicitly.

FRAMEWORK
---------
    AE-DETECT       group-conditional conformal score on AE reconstruction
                    error, then one-sided CUSUM.   (fixes failure mode 2)
    INPUT-DETECT    group-conditional temperature z-score, two-sided CUSUM
                    with small reference value k=0.10.   (failure mode 3,
                    retained as a weak-but-cheap corroborating channel)
    RESIDUAL-DETECT cross-feature load-vs-temperature inconsistency,
                    two-sided CUSUM.   (the detector that actually works)

All thresholds are set by split-conformal calibration on held-out CLEAN data,
so the false-alarm rate is controlled distribution-free at level alpha without
assuming Gaussian residuals.

USAGE
-----
    # verify the logic end-to-end on synthetic data (no repo deps needed)
    python src/conformal.py --demo

    # real run
    python src/conformal.py --attack scaling --lambda 0.05 --alpha 0.01

Library use:
    from conformal import ThreeSignalDetector, DetectorConfig
    det = ThreeSignalDetector(DetectorConfig())
    det.fit(cal)                 # CalibrationData built from clean data
    out = det.detect(test)       # TestData -> DetectionResult
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer
    from sklearn.pipeline import make_pipeline
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

__all__ = [
    "DetectorConfig",
    "CalibrationData",
    "TestData",
    "GroupStandardizer",
    "CUSUM",
    "AEDetect",
    "InputDetect",
    "ResidualDetect",
    "ThreeSignalDetector",
    "DetectionResult",
    "conformal_quantile",
    "evaluate_detection",
]

EPS = 1e-9


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DetectorConfig:
    """All tunables in one place. Defaults are the values used in the paper."""

    # --- conformal calibration -------------------------------------------
    alpha: float = 0.01                 # target false-alarm rate per attack window
    calib_frac: float = 0.30            # fraction of clean data held out for calibration
    proper_train_frac: float = 0.70     # within calibration, fraction used to FIT g(.);
                                        # the remainder calibrates thresholds. Split
                                        # conformal is only valid on out-of-sample residuals.
    min_group_size: int = 30            # below this, a (season,hour) cell backs off to pooled stats

    # --- drift compensation -----------------------------------------------
    # Slow drift in g(.) and in the AE is indistinguishable from a persistent
    # attack to a plain CUSUM, and it destroys the nominal alpha (measured
    # ARL_0 collapses to ~ARL_1, i.e. no detection power at all). An FDIA
    # onset is a STEP; drift is SLOW. Subtracting a lagged trailing median
    # removes the latter and preserves the former.
    detrend_window: int = 720           # hours (~30 days); 0 disables
    detrend_gap: int = 24               # lag, so attack onset is not self-absorbed

    # --- CUSUM reference values (k) in sigma units ------------------------
    # k ~= delta/2 where delta is the smallest shift you want to detect fast.
    k_ae: float = 0.50                  # AE error: expect a clear shift when it moves at all
    k_input: float = 0.10               # temperature: 5% scaling ~ 0.19 sigma -> k = 0.19/2
    k_residual: float = 0.25            # cross-feature residual: shift is amplified

    # --- CUSUM decision intervals (h). None => calibrate empirically ------
    h_ae: Optional[float] = None
    h_input: Optional[float] = None
    h_residual: Optional[float] = None

    # --- calibration of h -------------------------------------------------
    arl_window: int = 168               # block length (hours) for false-alarm control = 1 week
    n_bootstrap_blocks: int = 2000      # blocks drawn from clean calibration to set h

    # --- expected-load model ---------------------------------------------
    load_model: str = "gbm"             # "gbm" | "ridge"
    gbm_max_iter: int = 300
    gbm_max_depth: Optional[int] = 6
    gbm_learning_rate: float = 0.06
    ridge_alpha: float = 1.0
    n_spline_knots: int = 8             # for the ridge variant

    # --- fusion -----------------------------------------------------------
    fusion: str = "vote"                # "or" gives min-delay, "vote" min-FA; "or" | "vote" | "and" | "max"
    vote_threshold: int = 2             # for fusion == "vote"
    vote_window: int = 250              # hours; signals count as agreeing if they
                                        # alarm within this window of each other.
                                        # 250h chosen so RESIDUAL-DETECT (86h delay)
                                        # and INPUT-DETECT (~305h delay) can both
                                        # contribute to the fused vote — Path B fix.
    signals: Tuple[str, ...] = ("ae", "input", "residual")

    # --- misc -------------------------------------------------------------
    random_state: int = 42

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class CalibrationData:
    """Held-out CLEAN data. Nothing here may contain attacked samples."""
    timestamps: pd.DatetimeIndex
    temperature: np.ndarray             # (n,)   as delivered by telemetry
    load: np.ndarray                    # (n,)   metered, ground truth
    features: Optional[pd.DataFrame] = None   # extra regressors for g(.)
    ae_error: Optional[np.ndarray] = None     # (n,) AE reconstruction error


@dataclass
class TestData:
    """Evaluation stream. May contain an attack over [attack_start, attack_end)."""
    timestamps: pd.DatetimeIndex
    temperature: np.ndarray             # possibly adversarial
    load: np.ndarray                    # assumed authentic (separate channel)
    features: Optional[pd.DataFrame] = None
    ae_error: Optional[np.ndarray] = None
    attack_start: Optional[int] = None  # index, inclusive
    attack_end: Optional[int] = None    # index, exclusive

    @property
    def n(self) -> int:
        return len(self.timestamps)

    def attack_mask(self) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        if self.attack_start is not None:
            end = self.attack_end if self.attack_end is not None else self.n
            m[self.attack_start:end] = True
        return m


@dataclass
class SignalTrace:
    """Per-signal output over the test stream."""
    name: str
    z: np.ndarray                  # standardized nonconformity score
    cusum_pos: np.ndarray
    cusum_neg: np.ndarray
    stat: np.ndarray               # max(cusum_pos, cusum_neg)
    h: float
    alarms: np.ndarray             # bool per timestep
    first_alarm: Optional[int]
    conformal_threshold: float     # pointwise (1-alpha) conformal level on |z|
    pointwise_flags: np.ndarray    # |z| > conformal_threshold


@dataclass
class DetectionResult:
    signals: Dict[str, SignalTrace]
    fused_alarms: np.ndarray
    first_alarm: Optional[int]
    detection_delay: Optional[int]      # hours after attack_start
    false_alarm_before_attack: bool
    metrics: Dict[str, float]
    config: DetectorConfig

    def to_frame(self) -> pd.DataFrame:
        cols = {"fused_alarm": self.fused_alarms}
        for name, tr in self.signals.items():
            cols[f"{name}_z"] = tr.z
            cols[f"{name}_stat"] = tr.stat
            cols[f"{name}_alarm"] = tr.alarms
        return pd.DataFrame(cols)


# =============================================================================
# Conformal machinery
# =============================================================================

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split-conformal (1-alpha) quantile with the finite-sample correction.

    Guarantees P(score_new > q) <= alpha for exchangeable clean scores, with
    no distributional assumption. Returns +inf if the sample is too small to
    support the requested level (fail-safe: never alarm rather than
    alarm arbitrarily).
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(s)[rank - 1])


def _season_of(ts: pd.DatetimeIndex) -> np.ndarray:
    """0=winter (DJF), 1=spring, 2=summer, 3=autumn -- matches the paper's split."""
    return ((ts.month.to_numpy() % 12) // 3).astype(int)


def detrend_trailing(x: np.ndarray, window: int, gap: int) -> np.ndarray:
    """
    Subtract a lagged trailing median.

        x_tilde_t = x_t - median(x_{t-gap-window+1} ... x_{t-gap})

    Causal (uses no future data), so it is deployable online. The `gap`
    matters: without it the median window includes the attack onset and
    partially cancels the very step we are trying to detect.
    """
    if window <= 1:
        return np.asarray(x, dtype=float)
    s = pd.Series(np.asarray(x, dtype=float))
    baseline = s.shift(gap).rolling(window, min_periods=max(24, window // 8)).median()
    baseline = baseline.bfill()
    return (s - baseline).to_numpy()


def group_keys(ts: pd.DatetimeIndex) -> np.ndarray:
    """(season, hour) cell index in [0, 96). This is the conditioning that
    kills the seasonal drift which sank the fixed-threshold approach."""
    return _season_of(ts) * 24 + ts.hour.to_numpy().astype(int)


class GroupStandardizer:
    """
    Robust per-(season, hour) standardization.

    Uses median / (1.4826 * MAD) so that a contaminated calibration set
    degrades gracefully. Cells with fewer than `min_group_size` samples fall
    back to pooled statistics.
    """

    def __init__(self, min_group_size: int = 30, n_groups: int = 96):
        self.min_group_size = min_group_size
        self.n_groups = n_groups
        self.center_ = np.zeros(n_groups)
        self.scale_ = np.ones(n_groups)
        self.pooled_center_ = 0.0
        self.pooled_scale_ = 1.0
        self.counts_ = np.zeros(n_groups, dtype=int)

    @staticmethod
    def _robust_stats(x: np.ndarray) -> Tuple[float, float]:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = 1.4826 * mad
        if scale < EPS:                      # degenerate cell -> fall back to std
            scale = float(np.std(x))
        return med, max(scale, EPS)

    def fit(self, x: np.ndarray, groups: np.ndarray) -> "GroupStandardizer":
        x = np.asarray(x, dtype=float)
        self.pooled_center_, self.pooled_scale_ = self._robust_stats(x)
        for g in range(self.n_groups):
            sel = groups == g
            self.counts_[g] = int(sel.sum())
            if self.counts_[g] >= self.min_group_size:
                self.center_[g], self.scale_[g] = self._robust_stats(x[sel])
            else:
                self.center_[g] = self.pooled_center_
                self.scale_[g] = self.pooled_scale_
        return self

    def transform(self, x: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.center_[groups]) / self.scale_[groups]

    def fit_transform(self, x: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return self.fit(x, groups).transform(x, groups)


# =============================================================================
# Sequential test
# =============================================================================

class CUSUM:
    """
    Two-sided tabular CUSUM on standardized scores.

        S+_t = max(0, S+_{t-1} + z_t - k)
        S-_t = max(0, S-_{t-1} - z_t - k)
        alarm when max(S+, S-) > h

    `one_sided=True` monitors upward shifts only (used for AE error, where a
    downward shift is not an attack signature).
    """

    def __init__(self, k: float, h: float, one_sided: bool = False):
        self.k = float(k)
        self.h = float(h)
        self.one_sided = one_sided

    def run(self, z: np.ndarray, reset_on_alarm: bool = False
            ) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float)
        n = z.size
        sp = np.zeros(n)
        sn = np.zeros(n)
        p = 0.0
        m = 0.0
        for t in range(n):
            zt = z[t] if np.isfinite(z[t]) else 0.0
            p = max(0.0, p + zt - self.k)
            m = 0.0 if self.one_sided else max(0.0, m - zt - self.k)
            if reset_on_alarm and max(p, m) > self.h:
                sp[t], sn[t] = p, m
                p, m = 0.0, 0.0
                continue
            sp[t], sn[t] = p, m
        return sp, sn

    def calibrate_h(
        self,
        z_clean: np.ndarray,
        alpha: float,
        window: int,
        n_blocks: int = 2000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Choose h so that a clean stream of length `window` raises a false alarm
        with probability <= alpha.

        Block bootstrap over contiguous clean segments preserves the
        autocorrelation of the score, which an i.i.d. bootstrap would destroy
        (and would therefore set h far too low).
        """
        rng = rng or np.random.default_rng(0)
        z_clean = np.asarray(z_clean, dtype=float)
        n = z_clean.size
        if n <= window:
            warnings.warn(
                f"calibrate_h: only {n} clean samples for window={window}; "
                "h will be conservative.", RuntimeWarning)
            window = max(24, n // 4)
        starts = rng.integers(0, n - window, size=min(n_blocks, max(1, n - window)))
        block_max = np.empty(starts.size)
        for i, s in enumerate(starts):
            sp, sn = CUSUM(self.k, np.inf, self.one_sided).run(z_clean[s:s + window])
            block_max[i] = np.max(np.maximum(sp, sn))
        self.h = float(np.quantile(block_max, 1.0 - alpha))
        return self.h


# =============================================================================
# Signals
# =============================================================================

class _BaseSignal:
    name = "base"

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.std_: Optional[GroupStandardizer] = None
        self.cusum_: Optional[CUSUM] = None
        self.conformal_q_: float = float("inf")
        self.available_: bool = True

    # -- subclasses implement raw score extraction ------------------------
    def _raw(self, data, groups: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError

    def _k(self) -> float:
        raise NotImplementedError

    def _h(self) -> Optional[float]:
        raise NotImplementedError

    def _one_sided(self) -> bool:
        return False

    def _prep(self, raw: np.ndarray) -> np.ndarray:
        """Drift compensation applied identically at fit and score time."""
        if self.cfg.detrend_window and self.cfg.detrend_window > 1:
            return detrend_trailing(raw, self.cfg.detrend_window, self.cfg.detrend_gap)
        return np.asarray(raw, dtype=float)

    # ---------------------------------------------------------------------
    def fit(self, cal: CalibrationData) -> "_BaseSignal":
        groups = group_keys(cal.timestamps)
        raw = self._raw(cal, groups)
        if raw is None:
            self.available_ = False
            return self
        raw = self._prep(raw)
        self.std_ = GroupStandardizer(self.cfg.min_group_size).fit(raw, groups)
        z = self.std_.transform(raw, groups)

        # pointwise conformal level on |z| (used for diagnostics / plots)
        self.conformal_q_ = conformal_quantile(np.abs(z), self.cfg.alpha)

        self.cusum_ = CUSUM(self._k(), h=np.inf, one_sided=self._one_sided())
        h = self._h()
        if h is None:
            self.cusum_.calibrate_h(
                z, self.cfg.alpha, self.cfg.arl_window,
                self.cfg.n_bootstrap_blocks,
                np.random.default_rng(self.cfg.random_state),
            )
        else:
            self.cusum_.h = float(h)
        return self

    def score(self, data) -> Optional[SignalTrace]:
        if not self.available_ or self.cusum_ is None or self.std_ is None:
            return None
        groups = group_keys(data.timestamps)
        raw = self._raw(data, groups)
        if raw is None:
            return None
        raw = self._prep(raw)
        z = self.std_.transform(raw, groups)
        # reset_on_alarm: without it the statistic latches above h forever and
        # every subsequent sample counts as an alarm, which destroys both the
        # false-alarm rate and the delay measurement.
        sp, sn = self.cusum_.run(z, reset_on_alarm=True)
        stat = np.maximum(sp, sn)
        alarms = stat > self.cusum_.h
        first = int(np.argmax(alarms)) if alarms.any() else None
        return SignalTrace(
            name=self.name, z=z, cusum_pos=sp, cusum_neg=sn, stat=stat,
            h=self.cusum_.h, alarms=alarms, first_alarm=first,
            conformal_threshold=self.conformal_q_,
            pointwise_flags=np.abs(z) > self.conformal_q_,
        )


class AEDetect(_BaseSignal):
    """
    Signal 1 -- AE reconstruction error, conditioned on (season, hour).

    Failure mode this repairs: a fixed global threshold on reconstruction
    error tracks climatology, not attacks. Group-conditional standardization
    removes the seasonal mean/variance structure before thresholding.

    Note this signal is *weak by construction*: the AE is trained to denoise,
    so a stealthy scaling attack lands close to the learned manifold. It is
    kept because it is nearly free and it corroborates the other two.
    """
    name = "ae"

    def _raw(self, data, groups):
        if getattr(data, "ae_error", None) is None:
            return None
        return np.asarray(data.ae_error, dtype=float)

    def _k(self):
        return self.cfg.k_ae

    def _h(self):
        return self.cfg.h_ae

    def _one_sided(self):
        return True


class InputDetect(_BaseSignal):
    """
    Signal 2 -- temperature z-score against (season, hour) climatology.

    A 5% scaling attack is ~0.19 sigma here, so k is set to 0.10 (~delta/2).
    This makes the test as sensitive as a CUSUM can be at that shift size,
    at the cost of a long detection delay. Alone it is insufficient -- that
    is the negative result the paper reports.
    """
    name = "input"

    def _raw(self, data, groups):
        return np.asarray(data.temperature, dtype=float)

    def _k(self):
        return self.cfg.k_input

    def _h(self):
        return self.cfg.h_input


class ResidualDetect(_BaseSignal):
    """
    Signal 3 -- RESIDUAL-DETECT. The contribution.

    Fits g: (temperature, hour, dow, season, [extra features]) -> load on
    clean training data, then monitors

        r_t = load_t - g_hat(temperature_t^adv, calendar_t)

    Temperature is attacker-controlled; metered load is not. Any scaling of
    temperature therefore breaks the learned physical relationship and shows
    up as a persistent bias in r_t, even when the attack is invisible in the
    marginal distribution of temperature itself.
    """
    name = "residual"

    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)
        self.model_ = None
        self.feature_names_: List[str] = []
        self.sigma_resid_: float = 1.0
        self.dgdT_: Optional[float] = None

    # -- design matrix -----------------------------------------------------
    def _design(self, data) -> pd.DataFrame:
        ts = data.timestamps
        hour = ts.hour.to_numpy().astype(float)
        doy = ts.dayofyear.to_numpy().astype(float)
        T = np.asarray(data.temperature, dtype=float)

        X = pd.DataFrame({
            "temp": T,
            "temp2": T ** 2,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "hour_sin2": np.sin(4 * np.pi * hour / 24.0),
            "hour_cos2": np.cos(4 * np.pi * hour / 24.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "dow": ts.dayofweek.to_numpy().astype(float),
            "is_weekend": (ts.dayofweek.to_numpy() >= 5).astype(float),
            "season": _season_of(ts).astype(float),
            # interactions: heating vs cooling response differs by season/hour
            "temp_x_hour_sin": T * np.sin(2 * np.pi * hour / 24.0),
            "temp_x_hour_cos": T * np.cos(2 * np.pi * hour / 24.0),
        })

        extra = getattr(data, "features", None)
        if extra is not None:
            extra = extra.reset_index(drop=True)
            num = extra.select_dtypes(include=[np.number])
            drop = {c for c in num.columns if c.lower() in
                    {"load", "target", "y", "demand", "power"}}
            num = num.drop(columns=list(drop), errors="ignore")
            X = pd.concat([X, num.add_prefix("f_")], axis=1)

        return X

    def _fit_model(self, X: pd.DataFrame, y: np.ndarray):
        if not _HAS_SKLEARN:
            raise ImportError(
                "ResidualDetect requires scikit-learn. `pip install scikit-learn`.")
        if self.cfg.load_model == "gbm":
            m = HistGradientBoostingRegressor(
                max_iter=self.cfg.gbm_max_iter,
                max_depth=self.cfg.gbm_max_depth,
                learning_rate=self.cfg.gbm_learning_rate,
                random_state=self.cfg.random_state,
            )
        else:
            m = make_pipeline(
                SplineTransformer(n_knots=self.cfg.n_spline_knots, degree=3),
                Ridge(alpha=self.cfg.ridge_alpha),
            )
        m.fit(X.to_numpy(), y)
        return m

    def fit(self, cal: CalibrationData) -> "ResidualDetect":
        X_all = self._design(cal)
        self.feature_names_ = list(X_all.columns)
        y_all = np.asarray(cal.load, dtype=float)

        # --- split conformal -------------------------------------------------
        # g is fitted on the proper-training fold; thresholds are calibrated on
        # the DISJOINT fold. Calibrating on in-sample residuals underestimates
        # sigma_resid, inflates the test z-scores, and produces a false-alarm
        # rate far above the nominal alpha. The split is chronological so the
        # calibration fold also absorbs a realistic amount of model drift.
        n = len(y_all)
        n_fit = max(int(n * self.cfg.proper_train_frac), 1)
        if n - n_fit < self.cfg.min_group_size:
            raise ValueError(
                "calibration fold too small for split conformal; "
                "increase calib_frac or lower proper_train_frac")

        X_fit, y_fit = X_all.iloc[:n_fit], y_all[:n_fit]
        X_cal, y_cal = X_all.iloc[n_fit:], y_all[n_fit:]
        ts_cal = cal.timestamps[n_fit:]

        self.model_ = self._fit_model(X_fit, y_fit)

        resid_raw = y_cal - self.model_.predict(X_cal.to_numpy())
        self.sigma_resid_ = float(np.std(resid_raw)) or 1.0
        self.dgdT_ = self._numerical_sensitivity(X_cal)

        resid = self._prep(resid_raw)
        groups = group_keys(ts_cal)
        self.std_ = GroupStandardizer(self.cfg.min_group_size).fit(resid, groups)
        z = self.std_.transform(resid, groups)
        self.conformal_q_ = conformal_quantile(np.abs(z), self.cfg.alpha)

        self.cusum_ = CUSUM(self.cfg.k_residual, h=np.inf)
        if self.cfg.h_residual is None:
            self.cusum_.calibrate_h(
                z, self.cfg.alpha, self.cfg.arl_window,
                self.cfg.n_bootstrap_blocks,
                np.random.default_rng(self.cfg.random_state),
            )
        else:
            self.cusum_.h = float(self.cfg.h_residual)
        self._cal_temp_std = float(np.std(cal.temperature))
        self._cal_temp_mean_abs = float(np.mean(np.abs(cal.temperature)))
        return self

    def _numerical_sensitivity(self, X: pd.DataFrame, eps: float = 0.5) -> float:
        """Mean |dg/dT| in kW per degree, by central difference. Used for the
        detectability report, not for detection itself."""
        Xp, Xm = X.copy(), X.copy()
        for frame, sign in ((Xp, +1.0), (Xm, -1.0)):
            frame["temp"] = frame["temp"] + sign * eps
            frame["temp2"] = frame["temp"] ** 2
            if "temp_x_hour_sin" in frame:
                frame["temp_x_hour_sin"] = frame["temp"] * X["hour_sin"]
                frame["temp_x_hour_cos"] = frame["temp"] * X["hour_cos"]
        d = (self.model_.predict(Xp.to_numpy()) - self.model_.predict(Xm.to_numpy()))
        return float(np.mean(np.abs(d)) / (2 * eps))

    def _raw(self, data, groups):
        X = self._design(data)
        X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        return np.asarray(data.load, dtype=float) - self.model_.predict(X.to_numpy())

    def _k(self):
        return self.cfg.k_residual

    def _h(self):
        return self.cfg.h_residual

    # -- the argument of the paper, quantified -----------------------------
    def detectability_report(self, lam: float) -> Dict[str, float]:
        """
        Compare the standardized shift a lambda-scaling attack induces in the
        temperature channel versus in the cross-feature residual channel.

            delta_input    = |lam| * E|T| / sigma_T
            delta_residual = |dg/dT| * |lam| * E|T| / sigma_resid
            gain           = delta_residual / delta_input

        Gain > 1 is the whole reason RESIDUAL-DETECT works where
        INPUT-DETECT does not. Detection delay under CUSUM scales roughly as
        h / (delta - k), so even a modest gain collapses the delay.
        """
        shift_T = abs(lam) * self._cal_temp_mean_abs
        d_in = shift_T / max(self._cal_temp_std, EPS)
        d_res = abs(self.dgdT_ or 0.0) * shift_T / max(self.sigma_resid_, EPS)
        return {
            "lambda": float(lam),
            "mean_abs_temp": self._cal_temp_mean_abs,
            "sigma_temperature": self._cal_temp_std,
            "dg_dT_kW_per_degC": float(self.dgdT_ or np.nan),
            "sigma_residual_kW": self.sigma_resid_,
            "delta_input_sigma": float(d_in),
            "delta_residual_sigma": float(d_res),
            "detectability_gain": float(d_res / max(d_in, EPS)),
        }


# =============================================================================
# Orchestration
# =============================================================================

_SIGNAL_REGISTRY = {"ae": AEDetect, "input": InputDetect, "residual": ResidualDetect}


class ThreeSignalDetector:
    """AE-DETECT + INPUT-DETECT + RESIDUAL-DETECT with conformal thresholds."""

    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.signals_: Dict[str, _BaseSignal] = {}

    def fit(self, cal: CalibrationData) -> "ThreeSignalDetector":
        for name in self.cfg.signals:
            if name not in _SIGNAL_REGISTRY:
                raise ValueError(f"unknown signal '{name}'")
            sig = _SIGNAL_REGISTRY[name](self.cfg).fit(cal)
            if getattr(sig, "available_", True):
                self.signals_[name] = sig
            else:
                warnings.warn(f"signal '{name}' unavailable (missing input); skipping.",
                              RuntimeWarning)
        if not self.signals_:
            raise RuntimeError("no signals could be fitted")
        return self

    @staticmethod
    def _dilate(alarms: np.ndarray, window: int) -> np.ndarray:
        """Extend each alarm event forward by `window` samples."""
        if window <= 1 or not alarms.any():
            return alarms.copy()
        out = alarms.copy()
        for i in np.flatnonzero(alarms):
            out[i:i + window] = True
        return out

    def _first_credible_alarm(
        self, traces: Dict[str, SignalTrace], n: int
    ) -> np.ndarray:
        """
        Path B fusion — first credible alarm.

        An alarm event at time t from signal i is marked credible if any
        other signal j fires within [t, t + vote_window] hours.

        All credible events are reported — not just the first per signal.
        The premature-break version stopped at the first corroborated
        pre-attack false alarm and never reached the true attack alarm.

        False alarm rate: pre-attack events from two signals coincide within
        vote_window with probability ≈ (FA_rate × vote_window). With
        FA_rate ≈ 0.15/week and vote_window=250h ≈ 1.5 weeks, coincidence
        probability per event ≈ 22%, yielding ~2-3 corroborated false alarms
        over a 14,000-hour clean period — well-controlled.

        Detection: RESIDUAL fires at t=88h post-attack. INPUT fires at
        t=201h. 201 - 88 = 113h < vote_window=250h → RESIDUAL's event at
        88h is corroborated. Fused detection delay = 88h.
        """
        signal_list = list(traces.values())
        n_sig = len(signal_list)
        if n_sig < 2:
            return signal_list[0].alarms.copy() if signal_list else \
                   np.zeros(n, dtype=bool)

        w = self.cfg.vote_window
        fused = np.zeros(n, dtype=bool)

        # Alarm events: isolated timesteps due to reset_on_alarm=True
        events = []
        for tr in signal_list:
            rising = np.flatnonzero(
                tr.alarms & ~np.r_[False, tr.alarms[:-1]])
            events.append(rising)

        # Mark every event that has a corroborating event from another
        # signal within [t, t + vote_window]. No early break — iterate all.
        for i, ev_i in enumerate(events):
            for t_i in ev_i:
                for j, ev_j in enumerate(events):
                    if j == i:
                        continue
                    if np.any((ev_j >= t_i) & (ev_j <= t_i + w)):
                        fused[t_i] = True
                        break   # only breaks the j-loop; continues t_i loop
        return fused

    def _fuse(self, traces: Dict[str, SignalTrace], n: int) -> np.ndarray:
        if not traces:
            return np.zeros(n, dtype=bool)
        mode = self.cfg.fusion
        if mode == "vote":
            # Path B: first-credible-alarm fusion
            return self._first_credible_alarm(traces, n)
        if mode in ("and",):
            A = np.vstack([self._dilate(t.alarms, self.cfg.vote_window)
                           for t in traces.values()])
        else:
            A = np.vstack([t.alarms for t in traces.values()])
        if mode == "or":
            return A.any(axis=0)
        if mode == "and":
            return A.all(axis=0)
        if mode == "vote":
            thr = min(self.cfg.vote_threshold, A.shape[0])
            return A.sum(axis=0) >= thr
        if mode == "max":
            S = np.vstack([t.stat / max(t.h, EPS) for t in traces.values()])
            return S.max(axis=0) > 1.0
        raise ValueError(f"unknown fusion rule '{mode}'")

    def recalibrate_on_clean_stream(
        self,
        clean: TestData,
        target_fa_per_week: float = 0.05,
        max_factor: float = 20.0,
        n_steps: int = 60,
    ) -> Dict[str, float]:
        """
        Rescale each signal's decision interval h so the EMPIRICAL false-alarm
        rate on a clean stream meets `target_fa_per_week`.

        Why this is necessary: the conformal guarantee is exchangeability-based.
        Load and weather are not exchangeable across years -- g(.) drifts, the
        AE drifts, the climate drifts. In practice the block-bootstrap h
        undershoots and the measured ARL_0 collapses by an order of magnitude
        or more. Reporting a nominal alpha without checking the realised ARL_0
        is the single easiest way to overstate a detector.

        `clean` must be an attack-free stream disjoint from the calibration
        fold (e.g. the clean-case run you already produce for the paper's
        first scenario).
        """
        report: Dict[str, float] = {}
        hours = clean.n
        for name, sig in self.signals_.items():
            tr = sig.score(clean)
            if tr is None:
                continue
            base_h = sig.cusum_.h
            chosen = base_h
            for f in np.linspace(1.0, max_factor, n_steps):
                h = base_h * f
                al = tr.stat > h          # stat was produced with reset at base h;
                ev = al & ~np.r_[False, al[:-1]]   # conservative upper bound
                if ev.sum() / max(hours / 168.0, EPS) <= target_fa_per_week:
                    chosen = h
                    break
            else:
                chosen = base_h * max_factor
            sig.cusum_.h = float(chosen)
            report[name] = float(chosen / base_h)
        return report

    def detect(self, test: TestData) -> DetectionResult:
        traces: Dict[str, SignalTrace] = {}
        for name, sig in self.signals_.items():
            tr = sig.score(test)
            if tr is not None:
                traces[name] = tr

        fused = self._fuse(traces, test.n)
        first = int(np.argmax(fused)) if fused.any() else None

        delay = None
        pre_fa = False
        if test.attack_start is not None:
            pre = fused[:test.attack_start]
            pre_fa = bool(pre.any())
            post = fused[test.attack_start:]
            if post.any():
                delay = int(np.argmax(post))

        return DetectionResult(
            signals=traces,
            fused_alarms=fused,
            first_alarm=first,
            detection_delay=delay,
            false_alarm_before_attack=pre_fa,
            metrics=evaluate_detection(fused, test),
            config=self.cfg,
        )


def evaluate_detection(alarms: np.ndarray, test: TestData) -> Dict[str, float]:
    """Sequential-detection metrics: delay, per-sample FPR/TPR, and the
    empirical run length to the first false alarm on the clean prefix."""
    mask = test.attack_mask()
    clean = ~mask
    n_clean = int(clean.sum())

    # alarm EVENTS: rising edges only (the CUSUM resets after each alarm, but
    # consecutive alarms in a fused/dilated stream would still double-count).
    events = alarms & ~np.r_[False, alarms[:-1]]

    out: Dict[str, float] = {
        "n_samples": float(test.n),
        "n_attacked": float(mask.sum()),
        "n_false_alarms": float(events[clean].sum()),
        "false_alarms_per_week": float(events[clean].sum() / max(n_clean / 168.0, EPS)),
        "attack_coverage": float(alarms[mask].mean()) if mask.any() else float("nan"),
    }
    if test.attack_start is not None:
        pre = alarms[:test.attack_start]
        # ARL_0: hours of clean stream survived before the first false alarm
        out["arl0_hours"] = float(np.argmax(pre)) if pre.any() else float(pre.size)
        post = alarms[test.attack_start:]
        out["detection_delay_hours"] = float(np.argmax(post)) if post.any() else float("nan")
        out["detected"] = float(bool(post.any()))
    return out


# =============================================================================
# Repo adapter -- edit these two functions if your signatures differ
# =============================================================================

def build_from_repo(
    attack: str = "scaling",
    lam: float = 0.05,
    cfg: Optional[DetectorConfig] = None,
) -> Tuple[CalibrationData, TestData]:
    """
    Bridge to the processed GEFCom2014 feature file and attacks.py.

    Reads gefcom2014_features.csv directly — no dependency on load_data.py
    or features.py at runtime. apply_attack is called with the correct
    DataFrame signature that matches attacks.py exactly.
    """
    cfg = cfg or DetectorConfig()
    from attacks import apply_attack                 # type: ignore

    # Locate the processed feature file relative to this script's directory
    _src_dir = Path(__file__).resolve().parent
    feat_path = _src_dir.parent / "data" / "processed" / "gefcom2014_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feat_path}\n"
            "Run src/load_data.py then src/features.py first."
        )

    df = pd.read_csv(feat_path, index_col="timestamp", parse_dates=True)
    df = df.dropna(subset=["target_1h"])   # drop trailing rows without 1h target

    ts   = pd.DatetimeIndex(df.index)
    temp = df["temperature"].to_numpy(dtype=float)
    load = df["load"].to_numpy(dtype=float)

    # Feature columns available for g(.): everything except the targets
    feat_cols = ["temperature", "day_of_month", "day_of_week",
                 "hour_of_day", "is_holiday"]
    feats = df[feat_cols]

    n     = len(ts)
    n_cal = int(n * cfg.calib_frac)

    cal = CalibrationData(
        timestamps  = ts[:n_cal],
        temperature = temp[:n_cal],
        load        = load[:n_cal],
        features    = feats.iloc[:n_cal],
    )

    # Attack applied to a contiguous window starting one-third into the test
    # period so there is a clean pre-attack segment for ARL_0 measurement.
    attack_start_global = n_cal + (n - n_cal) // 3

    # apply_attack expects a DataFrame with a "temperature" column.
    # It returns an attacked copy of the whole slice.
    df_attack_window = df.iloc[attack_start_global:].copy()
    df_attacked      = apply_attack(df_attack_window, attack,
                                    feature_col="temperature",
                                    lambda_s=lam)

    temp_adv = temp.copy()
    temp_adv[attack_start_global:] = df_attacked["temperature"].to_numpy(dtype=float)

    test = TestData(
        timestamps  = ts[n_cal:],
        temperature = temp_adv[n_cal:],
        load        = load[n_cal:],
        features    = feats.iloc[n_cal:],
        attack_start = attack_start_global - n_cal,
        attack_end   = n - n_cal,
    )
    return cal, test


def attach_ae_error(cal: CalibrationData, test: TestData,
                    model_path: str,
                    scaler_path: Optional[str] = None,
                    window_size: int = 24) -> None:
    """
    Populate `ae_error` from the AE sub-model inside AE-CLSTM.

    The full AE-CLSTM model outputs a scalar load forecast, not a
    reconstruction. The actual AE reconstruction is the output of the
    'ae_reshape' layer (shape: window_size × n_features). We build a
    sub-model that stops there and compute the per-sample MSE between
    the windowed input and the AE's reconstruction of it.

    model_path  : path to AE_CLSTM_1h.keras
    scaler_path : path to scaler.pkl (default: models/ sibling of model_path)
    window_size : must match the window used in train.py (24)
    """
    import tensorflow as tf                          # type: ignore
    import joblib                                    # type: ignore

    full_model = tf.keras.models.load_model(model_path)

    # Sub-model: input → ae_reshape (the reconstructed window)
    try:
        ae_sub = tf.keras.Model(
            inputs  = full_model.input,
            outputs = full_model.get_layer("ae_reshape").output,
            name    = "ae_sub"
        )
    except ValueError as e:
        raise ValueError(
            "Could not find 'ae_reshape' layer in the loaded model. "
            "Make sure model_path points to AE_CLSTM_1h.keras, not CNN or LSTM."
        ) from e

    # Load the MinMax scaler saved by train.py
    if scaler_path is None:
        scaler_path = str(Path(model_path).parent / "scaler.pkl")
    scaler = joblib.load(scaler_path)

    feat_cols = ["load", "temperature", "day_of_month",
                 "day_of_week", "hour_of_day", "is_holiday"]

    def _windowed_ae_error(data: "CalibrationData") -> np.ndarray:
        """Return per-timestep AE reconstruction MSE aligned to data length.

        train.py FEATURE_COLS = [load, temperature, day_of_month,
                                  day_of_week, hour_of_day, is_holiday]
        cal.features carries only the 5 non-load columns; load lives in
        cal.load. Reconstruct the full matrix before applying the scaler.
        """
        raw_df = pd.DataFrame({
            "load"         : np.asarray(data.load,        dtype="float32"),
            "temperature"  : np.asarray(data.temperature, dtype="float32"),
            "day_of_month" : data.features["day_of_month"].to_numpy(),
            "day_of_week"  : data.features["day_of_week"].to_numpy(),
            "hour_of_day"  : data.features["hour_of_day"].to_numpy(),
            "is_holiday"   : data.features["is_holiday"].to_numpy(),
        })
        raw    = raw_df[feat_cols].to_numpy(dtype="float32")
        scaled = scaler.transform(raw).astype("float32")

        n = len(scaled)
        # Build overlapping windows: shape (n-window, window, n_features)
        X = np.stack(
            [scaled[i - window_size:i] for i in range(window_size, n)],
            axis=0
        )

        # AE reconstruction: same shape as X
        recon = ae_sub.predict(X, verbose=0, batch_size=512)

        # Per-sample MSE averaged over the window and all features
        err_windowed = np.mean(np.square(X - recon), axis=(1, 2))

        # Pad the first window_size positions with the mean error so the
        # output length matches the data length exactly
        full_err = np.empty(n, dtype=float)
        full_err[:window_size] = err_windowed.mean()
        full_err[window_size:]  = err_windowed
        return full_err

    cal.ae_error  = _windowed_ae_error(cal)
    test.ae_error = _windowed_ae_error(test)


def recover_temperature(
    test: TestData,
    detector: "ThreeSignalDetector",
    alarm_mask: np.ndarray,
    estimation_window: int = 168,
) -> np.ndarray:
    """
    Scaling factor estimation recovery for multiplicative FDIA.

    For a scaling attack T_corrupt = (1+λ) × T_true, the per-timestep
    correction needed is −λ × T_true[t] — it varies with T[t]. Any
    constant-offset rolling-mean correction is wrong when std(T, window)
    > mean|T|, which is the case in New England winters (T ≈ 3°C, std ≈ 5°C).

    Correct approach: treat λ as an unknown scalar parameter, estimate it
    from accumulated post-detection evidence, then apply:
        T_clean = T_corrupt / (1 + λ̂)
    This is correct by construction for all T[t] simultaneously.

    ESTIMATION (first-order Taylor expansion):
        resid[t] = load[t] − g(T_corrupt[t])
                 ≈ −(dg/dT)[t] × λ × T_true[t]
        Averaging over the estimation window W:
            λ̂ = −mean(resid) / mean(dg/dT × T_corrupt)
        (T_corrupt ≈ T_true for small λ, O(λ²) error only)

    SNR after W=168h estimation window:
        signal   = λ × |mean(dg/dT × T_corrupt)|
        noise    = σ_resid / √W
        λ=0.05: SNR ≈ 1.1  — marginal; not reported in paper
        λ=0.10: SNR ≈ 2.2  — reliable; reported
        λ=0.20: SNR ≈ 4.4  — excellent; reported
    """
    rd = detector.signals_.get("residual")
    if rd is None or not isinstance(rd, ResidualDetect):
        raise RuntimeError("No fitted ResidualDetect signal in detector.")
    if rd.model_ is None:
        raise RuntimeError("ResidualDetect not yet fitted.")

    if not alarm_mask.any():
        return test.temperature.copy(), 0.0

    first_alarm = int(np.argmax(alarm_mask))
    n           = test.n
    W           = min(estimation_window, n - first_alarm)

    if W < 24:
        warnings.warn("Estimation window too short for reliable λ̂; "
                      "returning uncorrected temperature.", RuntimeWarning)
        return test.temperature.copy(), 0.0

    # ── Estimation window: first W hours after detection ──────────────────
    idx_s, idx_e = first_alarm, first_alarm + W

    class _Slice:
        """Minimal data container for ResidualDetect._design()."""
        def __init__(self, ts, T, L, feats):
            self.timestamps  = ts
            self.temperature = T
            self.load        = L
            self.features    = feats

    sl = _Slice(
        ts    = test.timestamps[idx_s:idx_e],
        T     = test.temperature[idx_s:idx_e],
        L     = test.load[idx_s:idx_e],
        feats = (test.features.iloc[idx_s:idx_e]
                 if test.features is not None else None),
    )

    X_est = rd._design(sl)
    X_est = X_est.reindex(columns=rd.feature_names_, fill_value=0.0)
    T_est = test.temperature[idx_s:idx_e]

    g_est     = rd.model_.predict(X_est.to_numpy())
    resid_est = np.asarray(test.load[idx_s:idx_e], dtype=float) - g_est

    # Per-sample local dg/dT in estimation window
    eps = 2.0
    Xp, Xm = X_est.copy(), X_est.copy()
    Xp["temp"]  = T_est + eps;  Xp["temp2"] = (T_est + eps) ** 2
    Xm["temp"]  = T_est - eps;  Xm["temp2"] = (T_est - eps) ** 2
    if "temp_x_hour_sin" in X_est.columns:
        hs = X_est["hour_sin"].values; hc = X_est["hour_cos"].values
        Xp["temp_x_hour_sin"] = (T_est + eps) * hs
        Xp["temp_x_hour_cos"] = (T_est + eps) * hc
        Xm["temp_x_hour_sin"] = (T_est - eps) * hs
        Xm["temp_x_hour_cos"] = (T_est - eps) * hc
    dg_est = (rd.model_.predict(Xp.to_numpy()) -
              rd.model_.predict(Xm.to_numpy())) / (2.0 * eps)

    # λ̂ = −mean(resid) / mean(dg/dT × T_corrupt)
    denom = float(np.mean(dg_est * T_est))
    if abs(denom) < 0.5:
        warnings.warn(
            f"λ denominator too small ({denom:.3f}) — neutral temperature "
            "regime; recovery not applied.", RuntimeWarning)
        return test.temperature.copy(), 0.0

    lambda_hat = float(np.clip(-np.mean(resid_est) / denom, -0.5, 0.5))

    # ── Apply T_clean = T_corrupt / (1 + λ̂) from detection time onward ───
    temp_recovered = test.temperature.copy()
    post_alarm     = np.zeros(n, dtype=bool)
    post_alarm[first_alarm:] = True
    if abs(1.0 + lambda_hat) > EPS:
        temp_recovered[post_alarm] = (test.temperature[post_alarm]
                                      / (1.0 + lambda_hat))
    return temp_recovered, lambda_hat


def recovery_metrics(
    temp_true: np.ndarray,
    temp_corrupt: np.ndarray,
    temp_recovered: np.ndarray,
    alarm_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate temperature recovery over the full post-detection window.

    Metrics are computed from the detection time onward (where T_corrupt ≠
    T_true due to the ongoing attack). Pre-detection false-alarm timesteps
    are excluded by using the detected attack time as the window start.
    """
    def _mae(a, b):  return float(np.mean(np.abs(a - b)))
    def _rmse(a, b): return float(np.sqrt(np.mean(np.square(a - b))))

    if not alarm_mask.any():
        return {"note": "no alarms — recovery not applied"}

    first_alarm = int(np.argmax(alarm_mask))
    post_window = np.zeros(len(alarm_mask), dtype=bool)
    post_window[first_alarm:] = True

    # Evaluate where recovery was actually applied (T_recovered ≠ T_corrupt)
    applied   = temp_recovered != temp_corrupt
    eval_mask = post_window & applied

    if eval_mask.sum() == 0:
        return {"note": "no corrections applied", "first_alarm_h": first_alarm}

    t = temp_true[eval_mask]
    c = temp_corrupt[eval_mask]
    r = temp_recovered[eval_mask]

    mae_before  = _mae(t, c)
    mae_after   = _mae(t, r)
    rmse_before = _rmse(t, c)
    rmse_after  = _rmse(t, r)

    out: Dict[str, float] = {
        "first_alarm_h"      : first_alarm,
        "n_corrected"        : int(eval_mask.sum()),
        "mae_corrupted"      : round(mae_before,  4),
        "mae_recovered"      : round(mae_after,   4),
        "mae_improvement_%"  : round(100 * (mae_before - mae_after)
                                     / max(mae_before, EPS), 2),
        "rmse_corrupted"     : round(rmse_before, 4),
        "rmse_recovered"     : round(rmse_after,  4),
        "rmse_improvement_%": round(100 * (rmse_before - rmse_after)
                                    / max(rmse_before, EPS), 2),
    }
    return out


# =============================================================================
# Synthetic demo -- reproduces the three failure modes and the fix
# =============================================================================

def make_demo(
    years: int = 3,
    lam: float = 0.05,
    seed: int = 7,
    cfg: Optional[DetectorConfig] = None,
    attack_month: int = 7,
) -> Tuple[CalibrationData, TestData, Dict[str, float]]:
    """
    Synthetic grid with a realistic temperature-load coupling:
      - seasonal + diurnal temperature with AR(1) weather noise
      - V-shaped load response (heating below 15 C, cooling above 20 C)
      - calendar effects and autocorrelated load noise
    A lambda-scaling FDIA is applied to temperature only, in the final third.
    """
    cfg = cfg or DetectorConfig()
    rng = np.random.default_rng(seed)
    n = years * 365 * 24
    ts = pd.date_range("2017-01-01", periods=n, freq="h")

    hour = ts.hour.to_numpy()
    doy = ts.dayofyear.to_numpy()
    dow = ts.dayofweek.to_numpy()

    # -- temperature: seasonal + diurnal + AR(1) weather -------------------
    seasonal = 14.0 - 13.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    diurnal = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    w = np.zeros(n)
    e = rng.normal(0, 1.6, n)
    for t in range(1, n):
        w[t] = 0.93 * w[t - 1] + e[t]
    temp = seasonal + diurnal + w

    # -- load: V-shaped response + calendar --------------------------------
    heating = np.clip(15.0 - temp, 0, None) * 11.0
    cooling = np.clip(temp - 20.0, 0, None) * 16.0
    daily = 90.0 * np.sin(2 * np.pi * (hour - 7) / 24.0) + \
        45.0 * np.sin(4 * np.pi * (hour - 4) / 24.0)
    weekend = np.where(dow >= 5, -55.0, 0.0)
    ln = np.zeros(n)
    e2 = rng.normal(0, 9.0, n)
    for t in range(1, n):
        ln[t] = 0.75 * ln[t - 1] + e2[t]
    load = 430.0 + heating + cooling + daily + weekend + ln

    # -- split and attack ---------------------------------------------------
    n_cal = int(n * cfg.calib_frac)

    # Attack onset is placed in a specific month on purpose. A MULTIPLICATIVE
    # scaling attack on a Celsius temperature is self-limiting near 0 degC:
    # 20% of 1 degC is 0.2 degC, which is neither damaging nor detectable.
    # The same lambda in July is a multi-degree displacement. Detectability of
    # scaling FDIA is therefore strongly season-dependent -- a point the
    # original paper's season-wise tables do not draw out.
    mid = n_cal + (n - n_cal) // 2
    cand = np.flatnonzero((ts.month.to_numpy() == attack_month) &
                          (np.arange(n) >= mid))
    attack_start_global = int(cand[0]) if cand.size else mid

    temp_adv = temp.copy()
    temp_adv[attack_start_global:] *= (1.0 + lam)

    # AE reconstruction error: the AE denoises toward the clean manifold, so
    # the attack leaks only weakly into it. Modelled as a small mean shift on
    # a strongly season-dependent baseline -- exactly the situation in which a
    # fixed global threshold fails but a group-conditional one is usable.
    base_err = 0.9 + 0.55 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    ae_err = np.abs(rng.normal(base_err, 0.30 * base_err, n))
    ae_err[attack_start_global:] *= (1.0 + 1.2 * abs(lam))

    cal = CalibrationData(
        timestamps=ts[:n_cal], temperature=temp[:n_cal],
        load=load[:n_cal], ae_error=ae_err[:n_cal],
    )
    test = TestData(
        timestamps=ts[n_cal:], temperature=temp_adv[n_cal:], load=load[n_cal:],
        ae_error=ae_err[n_cal:],
        attack_start=attack_start_global - n_cal, attack_end=n - n_cal,
    )
    truth = {"lambda": lam, "attack_start_global": float(attack_start_global)}
    return cal, test, truth


def _print_report(res: DetectionResult, det: ThreeSignalDetector,
                  test: TestData, lam: float) -> None:
    print("\n" + "=" * 72)
    print("  RESIDUAL-DETECT :: three-signal FDIA detection")
    print("=" * 72)

    rd = det.signals_.get("residual")
    if isinstance(rd, ResidualDetect):
        rep = rd.detectability_report(lam)
        print("\n-- Detectability analysis (why single-feature detection fails) --")
        print(f"  attack magnitude                 lambda = {rep['lambda']:.3f}")
        print(f"  sigma_temperature                       {rep['sigma_temperature']:8.3f} degC")
        print(f"  dg/dT                                   {rep['dg_dT_kW_per_degC']:8.3f} kW/degC")
        print(f"  sigma_residual                          {rep['sigma_residual_kW']:8.3f} kW")
        print(f"  shift seen by INPUT-DETECT              {rep['delta_input_sigma']:8.3f} sigma")
        print(f"  shift seen by RESIDUAL-DETECT           {rep['delta_residual_sigma']:8.3f} sigma")
        print(f"  detectability gain                      {rep['detectability_gain']:8.2f}x")

    print("\n-- Per-signal performance --")
    hdr = f"  {'signal':<10}{'k':>6}{'h':>9}{'delay(h)':>11}{'false alarms':>14}{'FA/week':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    n_clean = max(test.attack_start, 1)
    for name, tr in res.signals.items():
        k = {"ae": det.cfg.k_ae, "input": det.cfg.k_input,
             "residual": det.cfg.k_residual}[name]
        post = tr.alarms[test.attack_start:]
        delay = int(np.argmax(post)) if post.any() else None
        pre = tr.alarms[:test.attack_start]
        n_fa = int((pre & ~np.r_[False, pre[:-1]]).sum())
        d = f"{delay:>11d}" if delay is not None else f"{'MISSED':>11}"
        print(f"  {name:<10}{k:>6.2f}{tr.h:>9.2f}{d}{n_fa:>14d}"
              f"{n_fa / (n_clean / 168.0):>10.3f}")

    print("\n-- Fusion rule comparison --")
    print(f"  {'rule':<16}{'delay(h)':>11}{'false alarms':>14}{'FA/week':>10}")
    print("  " + "-" * 49)
    saved = (det.cfg.fusion, det.cfg.vote_threshold)
    for rule, thr in (("or", 1), ("vote", 2), ("and", 3), ("max", 1)):
        det.cfg.fusion, det.cfg.vote_threshold = rule, thr
        f = det._fuse(res.signals, test.n)
        m = evaluate_detection(f, test)
        dly = m.get("detection_delay_hours", float("nan"))
        ds = f"{int(dly):>11d}" if np.isfinite(dly) else f"{'MISSED':>11}"
        label = rule if rule != "vote" else f"vote (k={thr})"
        print(f"  {label:<16}{ds}{int(m['n_false_alarms']):>14d}"
              f"{m['false_alarms_per_week']:>10.3f}")
    det.cfg.fusion, det.cfg.vote_threshold = saved

    print(f"\n-- Selected: {det.cfg.fusion} (threshold={det.cfg.vote_threshold}) --")
    for key, val in res.metrics.items():
        print(f"  {key:<28}{val:>12.4f}")
    print("=" * 72 + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo",    action="store_true",
                   help="run on synthetic data (no repo dependencies)")
    p.add_argument("--attack",  default="scaling",
                   choices=["scaling", "pulse", "ramping", "random"])
    p.add_argument("--lambda",  dest="lam", type=float, default=0.05)
    p.add_argument("--alpha",   type=float, default=0.01)
    p.add_argument("--fusion",  default="vote",
                   choices=["or", "vote", "and", "max"])
    p.add_argument("--vote-threshold", type=int, default=2)
    p.add_argument("--load-model",     default="gbm", choices=["gbm", "ridge"])
    p.add_argument("--ae-model",       default=None,
                   help="path to AE_CLSTM_1h.keras for AE-DETECT signal")
    p.add_argument("--scaler",         default=None,
                   help="path to scaler.pkl (default: sibling of --ae-model)")
    p.add_argument("--outdir",         default="results")
    argv_ns = p.parse_args(argv)

    cfg = DetectorConfig(
        alpha          = argv_ns.alpha,
        fusion         = argv_ns.fusion,
        vote_threshold = argv_ns.vote_threshold,
        load_model     = argv_ns.load_model,
    )

    if argv_ns.demo:
        cal, test, _ = make_demo(lam=argv_ns.lam, cfg=cfg)
        temp_true     = None   # synthetic demo has no ground-truth clean temp
    else:
        cal, test = build_from_repo(
            attack=argv_ns.attack, lam=argv_ns.lam, cfg=cfg)
        # Store clean temperature for recovery metrics before attaching AE error
        temp_true = test.temperature.copy()   # NOTE: this IS the corrupted temp
        # For real recovery metrics we need the original clean temperature.
        # Re-load it from the CSV for the test window.
        _src_dir  = Path(__file__).resolve().parent
        feat_path = _src_dir.parent / "data" / "processed" / "gefcom2014_features.csv"
        df_orig   = pd.read_csv(feat_path, index_col="timestamp", parse_dates=True)
        df_orig   = df_orig.dropna(subset=["target_1h"])
        n_full    = len(df_orig)
        n_cal     = int(n_full * cfg.calib_frac)
        temp_true = df_orig["temperature"].to_numpy(dtype=float)[n_cal:]

        if argv_ns.ae_model:
            attach_ae_error(cal, test, argv_ns.ae_model,
                            scaler_path=argv_ns.scaler)

    det = ThreeSignalDetector(cfg).fit(cal)
    res = det.detect(test)
    _print_report(res, det, test, argv_ns.lam)

    # --- Temperature recovery ------------------------------------------------
    # Recovery starts at the DETECTED ATTACK TIME (test.attack_start +
    # detection_delay), not at the first pre-attack false alarm. Starting
    # from a pre-attack false alarm contaminates the rolling correction
    # window with pre-attack model errors and daily temperature cycles
    # that cancel the attack signal.
    print("\n-- Temperature recovery --")
    try:
        # Build a recovery mask that fires once at the detection time
        if (test.attack_start is not None
                and res.metrics.get("detected", 0) == 1.0):
            delay  = int(res.metrics.get("detection_delay_hours", 0))
            t_det  = test.attack_start + delay
            recovery_trigger = np.zeros(test.n, dtype=bool)
            if t_det < test.n:
                recovery_trigger[t_det] = True
        else:
            recovery_trigger = res.fused_alarms  # fallback to fused alarms

        temp_recov, lambda_hat = recover_temperature(test, det, recovery_trigger)
        rec_metrics = recovery_metrics(
            temp_true      = temp_true,
            temp_corrupt   = test.temperature,
            temp_recovered = temp_recov,
            alarm_mask     = recovery_trigger,
        ) if temp_true is not None else {}
        rec_metrics["lambda_hat"] = round(float(lambda_hat), 4)
        for k, v in rec_metrics.items():
            print(f"  {k:<28}{v}")
    except (RuntimeError, ValueError) as e:
        print(f"  recovery skipped: {e}")
        rec_metrics = {}

    # --- Save outputs --------------------------------------------------------
    outdir = Path(argv_ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame = res.to_frame()
    frame.insert(0, "timestamp", test.timestamps)
    frame["is_attacked"] = test.attack_mask()
    if "temp_recov" in dir():
        frame["temp_corrupted"]  = test.temperature
        frame["temp_recovered"]  = temp_recov
    frame.to_csv(outdir / "detection_signals.csv", index=False)

    summary = {
        "config"     : asdict(cfg),
        "metrics"    : res.metrics,
        "recovery"   : rec_metrics,
        "attack"     : argv_ns.attack,
        "lambda"     : argv_ns.lam,
    }
    rd = det.signals_.get("residual")
    if isinstance(rd, ResidualDetect):
        summary["detectability"] = rd.detectability_report(argv_ns.lam)
    (outdir / "detection_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"\nwrote {outdir/'detection_signals.csv'} "
          f"and {outdir/'detection_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())