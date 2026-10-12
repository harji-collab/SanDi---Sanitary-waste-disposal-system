"""Synthetic sensor workload for the SANDI smart sanitary-disposal unit.

Each sample is a 3.2 s window (32 steps @10 Hz) of [IR, temp_C, weight_g].
Classes:
  0 IDLE           nobody near, bin stable
  1 DISPOSAL       person approaches, lid opens, weight steps up a little
  2 NEAR_FULL      weight 70-90 % of capacity
  3 FULL           weight > 90 % of capacity -> request emptying
  4 THERMAL_ALERT  chamber temp rising abnormally (incinerator / fire risk)
Swap this file for real logs from src/logger.py once hardware is wired.
"""
import numpy as np

CLASSES = ["IDLE", "DISPOSAL", "NEAR_FULL", "FULL", "THERMAL_ALERT"]
T = 32
CAP_G = 3000.0
# realistic field mix: the bin is idle most of the time
CLASS_MIX = [0.70, 0.15, 0.07, 0.04, 0.04]


def _window(c, rng):
    t = np.arange(T)
    base_w = rng.uniform(0.05, 0.65) * CAP_G
    temp0 = rng.uniform(24, 34)
    ir = (rng.random(T) < 0.02).astype(float)
    temp = temp0 + rng.normal(0, 0.15, T)
    w = base_w + rng.normal(0, 6, T)
    if c == 1:  # disposal
        s = rng.integers(3, 14); e = s + rng.integers(8, 16)
        ir[s:e] = (rng.random(e - s) > 0.1)
        step = rng.uniform(15, 120)
        w[e - 3:] += step
        w[s:e] += rng.normal(0, 25, e - s)  # lid handling noise
    elif c == 2:
        w = rng.uniform(0.70, 0.90) * CAP_G + rng.normal(0, 6, T)
    elif c == 3:
        w = rng.uniform(0.90, 1.05) * CAP_G + rng.normal(0, 6, T)
    elif c == 4:
        slope = rng.uniform(0.25, 1.2)
        temp = temp0 + slope * t + rng.normal(0, 0.3, T)
    # occasional hard cases: disposal while near full, sensor glitches
    if c in (2, 3) and rng.random() < 0.25:
        s = rng.integers(3, 20); ir[s:s + 6] = 1
    if rng.random() < 0.03:
        w[rng.integers(0, T)] += rng.normal(0, 200)
    return np.stack([ir, temp, w], 1)


def make(n, seed=0, mix=CLASS_MIX):
    rng = np.random.default_rng(seed)
    y = rng.choice(len(CLASSES), size=n, p=mix)
    X = np.stack([_window(c, rng) for c in y]).astype(np.float32)
    return X, y


def features(X):
    """Flattened normalised window + a few engineered stats -> 102 dims."""
    ir, tp, w = X[..., 0], X[..., 1], X[..., 2]
    flat = np.concatenate([ir, (tp - 30) / 10, w / CAP_G], 1)
    extra = np.stack([
        ir.mean(1), ir.max(1),
        (tp[:, -4:].mean(1) - tp[:, :4].mean(1)) / 10,
        (w[:, -4:].mean(1) - w[:, :4].mean(1)) / 200,
        np.median(w, 1) / CAP_G, w.std(1) / 100,
    ], 1)
    return np.concatenate([flat, extra], 1).astype(np.float32)


def compact_features(X):
    """Cheap 30-dim features for the optimised student (batch form).
    8-step downsampled channels + 6 stats, no sorts/percentiles."""
    ir, tp, w = X[..., 0], X[..., 1], X[..., 2]
    n = X.shape[0]
    ds = X.reshape(n, 8, 4, 3).mean(2)               # (n, 8, 3)
    d = np.concatenate([ds[..., 0], (ds[..., 1] - 30) / 10, ds[..., 2] / CAP_G], 1)
    extra = np.stack([ir.mean(1), ir.max(1),
                      (ds[:, -1, 1] - ds[:, 0, 1]) / 10,
                      (ds[:, -1, 2] - ds[:, 0, 2]) / 200,
                      w.mean(1) / CAP_G, (w.max(1) - w.min(1)) / 1000], 1)
    return np.concatenate([d, extra], 1).astype(np.float32)
