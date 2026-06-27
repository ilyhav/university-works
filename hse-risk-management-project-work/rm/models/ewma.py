"""
EWMA-ковариация (RiskMetrics) — простая модель кластеризации волатильности.

Экспоненциально взвешенная ковариация с нулевым средним:
    Σ_t = λ·Σ_{t-1} + (1−λ)·r_{t-1} r_{t-1}ᵀ.
Свежие наблюдения весят больше — ковариация «дышит» вместе с рынком, что
закрывает выявленный ARCH-эффект без подгонки полноценного GARCH.

λ по умолчанию 0.94 (стандарт RiskMetrics для дневных данных), но можно
оценить λ по MLE (максимизация однодневного предиктивного log-lik). Для
риска на завтра используем предиктивную ковариацию next_cov.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

_LOG2PI = np.log(2.0 * np.pi)


def _gauss_loglik(predictive: np.ndarray, V: np.ndarray, burn: int) -> float:
    """Сумма гауссовых log-плотностей N(0, Σ_t) при r_t (ручной расчёт, быстро)."""
    n, p = V.shape
    acc = 0.0
    for t in range(burn, n):
        S = predictive[t]
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return -np.inf
        sol = np.linalg.solve(S, V[t])
        quad = float(V[t] @ sol)
        acc += -0.5 * (p * _LOG2PI + logdet + quad)
    return acc


@dataclass
class EWMAModel:
    lam: float
    columns: list[str]
    index: pd.Index           # календарь панели (для рядов условной σ)
    predictive: np.ndarray    # (n, p, p): Σ_t ДО наблюдения r_t (для log-lik)
    next_cov: np.ndarray      # (p, p): прогноз ковариации на следующий день
    loglik: float
    n_obs: int

    @staticmethod
    def _recurse(V: np.ndarray, lam: float, init: np.ndarray):
        n, p = V.shape
        Sigma = init.copy()
        predictive = np.empty((n, p, p))
        for t in range(n):
            predictive[t] = Sigma                      # прогноз на день t (до r_t)
            r = V[t][:, None]
            Sigma = lam * Sigma + (1.0 - lam) * (r @ r.T)
        return predictive, Sigma                       # Sigma — прогноз на n+1

    @classmethod
    def fit(cls, X: pd.DataFrame, lam: float | None = 0.94,
            burn: int = 20) -> "EWMAModel":
        """lam=None -> оценить λ по MLE; иначе использовать заданное (0.94)."""
        cols = list(X.columns)
        V = X.to_numpy(dtype=float)
        n, p = V.shape
        init = np.cov(V, rowvar=False, ddof=0)

        def neg_ll(l: float) -> float:
            pred, _ = cls._recurse(V, l, init)
            return -_gauss_loglik(pred, V, burn)

        if lam is None:
            res = minimize_scalar(neg_ll, bounds=(0.80, 0.9999), method="bounded")
            lam = float(res.x)

        pred, next_cov = cls._recurse(V, lam, init)
        ll = _gauss_loglik(pred, V, burn)
        return cls(lam=lam, columns=cols, index=X.index, predictive=pred,
                   next_cov=next_cov, loglik=float(ll), n_obs=n - burn)

    @property
    def n_params(self) -> int:
        return 1  # только λ (среднее зафиксировано нулём, init из выборки)

    def factor_vols(self) -> pd.Series:
        """Прогноз дневной σ каждого фактора на следующий день (корень из диагонали)."""
        return pd.Series(np.sqrt(np.diag(self.next_cov)), index=self.columns)

    def conditional_vol_series(self, factor: str) -> pd.Series:
        """Ряд предиктивной σ одного фактора (для графика «дышащей» волатильности)."""
        j = self.columns.index(factor)
        return pd.Series(np.sqrt(self.predictive[:, j, j]), index=self.index,
                         name=f"EWMA σ {factor}")

    def simulate(self, n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
        """Однодневные приращения из N(0, next_cov)."""
        return rng.multivariate_normal(np.zeros(len(self.columns)),
                                       self.next_cov, size=n_scenarios)
