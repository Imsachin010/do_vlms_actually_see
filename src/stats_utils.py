"""
src/stats_utils.py
Single Source of Truth for Statistical Tests, Robust Regressions, and FWER Adjustments.
"""

import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, theilslopes, t as t_dist
import statsmodels.api as sm


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute Wilson score interval for a binomial proportion.
    Returns (center, lower_bound, upper_bound) in percentage [0, 100].
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)))

    lower = max(0.0, (center - half) * 100.0)
    upper = min(100.0, (center + half) * 100.0)
    pct = p * 100.0
    return pct, lower, upper


def paired_ttest_and_cohen_d(differences: np.ndarray) -> Dict[str, float]:
    """
    Compute paired t-test statistics and Cohen's d on paired difference vector.
    """
    diffs = np.array(differences, dtype=float)
    n = len(diffs)
    if n < 2:
        return {"mean_diff": 0.0, "sd_diff": 0.0, "t_stat": 0.0, "df": 0, "p_value": 1.0, "cohen_d": 0.0}

    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    se_d = sd_d / math.sqrt(n)

    t_stat = mean_d / se_d if se_d > 0 else 0.0
    df_val = n - 1
    p_val = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df=df_val))
    cohen_d = mean_d / sd_d if sd_d > 0 else 0.0

    return {
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "t_stat": t_stat,
        "df": df_val,
        "p_value": p_val,
        "cohen_d": cohen_d
    }


def theil_sen_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Theil-Sen robust regression slope and intercept.
    """
    res = theilslopes(y, x)
    return float(res[0]), float(res[1])


def bootstrap_theil_sen_band(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 95% bootstrap confidence band for Theil-Sen fit across x_grid.
    """
    rng = np.random.default_rng(seed)
    boot_preds = np.zeros((n_boot, len(x_grid)))
    n = len(x)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        xs, ys = x[idx], y[idx]
        res = theilslopes(ys, xs)
        boot_preds[i, :] = res[1] + res[0] * x_grid

    y_low = np.percentile(boot_preds, 2.5, axis=0) * 100.0
    y_high = np.percentile(boot_preds, 97.5, axis=0) * 100.0
    return y_low, y_high


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation rho and two-tailed p-value.
    """
    rho, p_val = spearmanr(x, y)
    return float(rho), float(p_val)


def holm_bonferroni_adjustment(p_values: List[float]) -> List[float]:
    """
    Compute step-down Holm-Bonferroni adjusted p-values.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        multiplier = m - rank
        adj_p = min(1.0, p * multiplier)
        adj_p = max(adj_p, cum_max)
        cum_max = adj_p
        adjusted[orig_idx] = adj_p

    return adjusted
