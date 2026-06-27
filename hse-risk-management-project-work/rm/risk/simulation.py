"""Сценарная симуляция факторов и расчет VaR/ES."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t

from rm.config import ES_LEVEL, VAR_LEVEL
from rm.models import GaussianModel, StudentTModel


def empirical_scenarios(panel: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Исторические h-дневные сценарии как rolling-суммы дневных факторов."""
    if horizon_days < 1:
        raise ValueError("horizon_days должен быть >= 1")
    return panel.rolling(horizon_days).sum().dropna()


def simulate_gaussian(
    panel: pd.DataFrame,
    horizon_days: int,
    n_scenarios: int,
    rng: np.random.Generator,
    cov: pd.DataFrame | np.ndarray | None = None,
    mean: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """h-дневные сценарии из многомерной нормали.

    Если cov передан, используем его как однодневную условную ковариацию
    (EWMA/GARCH); иначе MLE-ковариация берется из самой панели.
    """
    cols = list(panel.columns)
    if mean is None:
        mu = panel.mean().to_numpy(dtype=float)
    else:
        mu = np.asarray(mean, dtype=float)
    if cov is None:
        model = GaussianModel.fit(panel)
        cov_arr = model.cov
        mu = model.mean
    else:
        cov_arr = cov.to_numpy(dtype=float) if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    sample = rng.multivariate_normal(
        horizon_days * mu,
        horizon_days * _nearest_psd(cov_arr),
        size=n_scenarios,
    )
    return pd.DataFrame(sample, columns=cols)


def simulate_student_t(
    panel: pd.DataFrame,
    horizon_days: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    """h-дневные сценарии как сумма h независимых дневных t-реализаций."""
    model = StudentTModel.fit(panel)
    total = np.zeros((n_scenarios, panel.shape[1]))
    for _ in range(horizon_days):
        total += model.simulate(n_scenarios, rng)
    return pd.DataFrame(total, columns=panel.columns), model.df


def simulate_t_with_cov(
    panel: pd.DataFrame,
    horizon_days: int,
    n_scenarios: int,
    rng: np.random.Generator,
    cov: pd.DataFrame | np.ndarray,
    df: float,
    mean: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """t-сценарии с заданной ковариацией.

    scipy задает shape, а не covariance, поэтому shape = cov * (ν-2)/ν.
    Для горизонта h ковариацию масштабируем линейно.
    """
    cols = list(panel.columns)
    cov_arr = cov.to_numpy(dtype=float) if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    cov_arr = _nearest_psd(cov_arr)
    if mean is None:
        loc = horizon_days * panel.mean().to_numpy(dtype=float)
    else:
        loc = horizon_days * np.asarray(mean, dtype=float)
    df = max(float(df), 2.05)
    shape = horizon_days * cov_arr * (df - 2.0) / df
    sample = multivariate_t.rvs(
        loc=loc,
        shape=_nearest_psd(shape),
        df=df,
        size=n_scenarios,
        random_state=rng,
    )
    return pd.DataFrame(np.atleast_2d(sample), columns=cols)


def risk_measures(
    pnl: pd.Series | np.ndarray,
    var_level: float = VAR_LEVEL,
    es_level: float = ES_LEVEL,
) -> dict[str, float]:
    """VaR/ES как положительная величина потерь в рублях."""
    pnl_arr = np.asarray(pnl, dtype=float)
    pnl_arr = pnl_arr[np.isfinite(pnl_arr)]
    if pnl_arr.size == 0:
        raise ValueError("пустой P&L для расчета риска")
    losses = -pnl_arr
    var = float(np.quantile(losses, var_level))
    es_cut = float(np.quantile(losses, es_level))
    tail = losses[losses >= es_cut]
    es = float(tail.mean()) if tail.size else es_cut
    return {
        "mean_pnl": float(pnl_arr.mean()),
        "std_pnl": float(pnl_arr.std(ddof=1)),
        f"VaR_{var_level:.3f}": var,
        f"ES_{es_level:.3f}": es,
        "min_pnl": float(pnl_arr.min()),
        "max_pnl": float(pnl_arr.max()),
    }


def _nearest_psd(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Симметризовать и подрезать отрицательные собственные числа."""
    m = np.asarray(matrix, dtype=float)
    m = (m + m.T) / 2.0
    vals, vecs = np.linalg.eigh(m)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T
