"""
Статистические тесты бэктеста VaR (этап 6).

Ансамбль тестов (так как при 99% VaR и ~250 днях мощность каждого отдельного
теста низкая — ожидается всего ~2.5 пробоя):
  * Kupiec POF (UC)         — корректна ли ЧАСТОТА пробоев (безусловное покрытие);
  * Christoffersen IND      — НЕЗАВИСИМЫ ли пробои (нет кластеризации);
  * Christoffersen CC       — совместный тест частоты И независимости (UC+IND);
  * Dynamic Quantile (DQ)   — Engle–Manganelli: «продвинутый» тест, ловит связь
                              пробоев с прошлыми пробоями и с уровнем самого VaR;
  * Basel traffic light     — светофор Базеля по числу пробоев.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass
class BacktestResult:
    n_obs: int
    n_exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_lr: float
    kupiec_pvalue: float
    christoffersen_lr: float          # independence (IND)
    christoffersen_pvalue: float
    cc_lr: float                      # conditional coverage (UC+IND)
    cc_pvalue: float
    dq_stat: float                    # Dynamic Quantile (Engle–Manganelli)
    dq_pvalue: float
    traffic_light: str


def kupiec_pof(exceptions, tail_prob: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures (UC). H0: частота пробоев = tail_prob."""
    x = np.asarray(exceptions, dtype=bool)
    n = int(x.size)
    k = int(x.sum())
    if n == 0:
        return np.nan, np.nan
    phat = np.clip(k / n, 1e-12, 1 - 1e-12)
    p = np.clip(tail_prob, 1e-12, 1 - 1e-12)
    ll_null = (n - k) * np.log(1 - p) + k * np.log(p)
    ll_alt = (n - k) * np.log(1 - phat) + k * np.log(phat)
    lr = float(-2.0 * (ll_null - ll_alt))
    return lr, float(chi2.sf(lr, 1))


def christoffersen_independence(exceptions) -> tuple[float, float]:
    """Christoffersen IND: H0 — пробои независимы (нет кластеризации). LR~χ²(1)."""
    x = np.asarray(exceptions, dtype=bool)
    if x.size < 2:
        return np.nan, np.nan
    prev = x[:-1]
    curr = x[1:]
    n00 = int((~prev & ~curr).sum())
    n01 = int((~prev & curr).sum())
    n10 = int((prev & ~curr).sum())
    n11 = int((prev & curr).sum())

    pi = np.clip((n01 + n11) / max(n00 + n01 + n10 + n11, 1), 1e-12, 1 - 1e-12)
    pi0 = np.clip(n01 / max(n00 + n01, 1), 1e-12, 1 - 1e-12)
    pi1 = np.clip(n11 / max(n10 + n11, 1), 1e-12, 1 - 1e-12)

    ll_ind = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_markov = (
        n00 * np.log(1 - pi0) + n01 * np.log(pi0)
        + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
    )
    lr = float(-2.0 * (ll_ind - ll_markov))
    return lr, float(chi2.sf(lr, 1))


def christoffersen_cc(exceptions, tail_prob: float) -> tuple[float, float]:
    """Christoffersen CC (conditional coverage) = UC + IND, LR~χ²(2)."""
    lr_uc, _ = kupiec_pof(exceptions, tail_prob)
    lr_ind, _ = christoffersen_independence(exceptions)
    if not (np.isfinite(lr_uc) and np.isfinite(lr_ind)):
        return np.nan, np.nan
    lr = float(lr_uc + lr_ind)
    return lr, float(chi2.sf(lr, 2))


def dq_test(exceptions, var_series, tail_prob: float, lags: int = 4) -> tuple[float, float]:
    """Dynamic Quantile тест Engle–Manganelli (out-of-sample).

    Регрессия центрированной последовательности пробоев Hit_t = 1{пробой} − p
    на константу, L прошлых Hit и СЕГОДНЯШНИЙ VaR. Если эти регрессоры значимы —
    пробои предсказуемы (зависят от прошлого/уровня VaR) => модель плоха.
    Статистика DQ = β̂'(X'X)β̂ / (p(1−p)) ~ χ²(k), k = число регрессоров.
    """
    hit = np.asarray(exceptions, dtype=float) - tail_prob
    var = np.asarray(var_series, dtype=float)
    n = hit.size
    if var.size != n or n <= lags + 2:
        return np.nan, np.nan
    rows, y = [], []
    for t in range(lags, n):
        reg = [1.0] + [hit[t - l] for l in range(1, lags + 1)] + [var[t]]
        rows.append(reg)
        y.append(hit[t])
    X = np.asarray(rows, dtype=float)
    Y = np.asarray(y, dtype=float)
    denom = tail_prob * (1.0 - tail_prob)
    try:
        xtx = X.T @ X
        beta = np.linalg.pinv(xtx) @ X.T @ Y
        stat = float(beta @ xtx @ beta / denom)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return stat, float(chi2.sf(stat, X.shape[1]))


def traffic_light(n_exceptions: int, var_level: float = 0.99) -> str:
    """Basel traffic light для ~250 наблюдений и 99% VaR."""
    if var_level != 0.99:
        return "n/a"
    if n_exceptions <= 4:
        return "green"
    if n_exceptions <= 9:
        return "yellow"
    return "red"


def summarize_backtest(exceptions, var_level: float, var_series=None) -> BacktestResult:
    """Полный ансамбль тестов по последовательности пробоев.

    var_series (опционально) нужен только для DQ-теста; без него DQ = NaN.
    """
    x = np.asarray(exceptions, dtype=bool)
    tail_prob = 1.0 - var_level
    lr_uc, p_uc = kupiec_pof(x, tail_prob)
    lr_ind, p_ind = christoffersen_independence(x)
    lr_cc, p_cc = christoffersen_cc(x, tail_prob)
    if var_series is not None:
        dq_stat, dq_p = dq_test(x, var_series, tail_prob)
    else:
        dq_stat, dq_p = np.nan, np.nan
    n = int(x.size)
    k = int(x.sum())
    return BacktestResult(
        n_obs=n,
        n_exceptions=k,
        expected_exceptions=n * tail_prob,
        exception_rate=k / n if n else np.nan,
        kupiec_lr=lr_uc,
        kupiec_pvalue=p_uc,
        christoffersen_lr=lr_ind,
        christoffersen_pvalue=p_ind,
        cc_lr=lr_cc,
        cc_pvalue=p_cc,
        dq_stat=dq_stat,
        dq_pvalue=dq_p,
        traffic_light=traffic_light(k, var_level),
    )
