"""
Многомерная нормальная модель приращений факторов (Башелье / GBM-база).

MLE в замкнутом виде: μ̂ — выборочное среднее, Σ̂ — выборочная ковариация
с делителем n (оценка максимального правдоподобия, ddof=0). Это базовая
модель из курса; на ней проверяем, насколько тяжёлые хвосты ломают нормаль.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from rm.models.base import aic, bic, n_cov_params


@dataclass
class GaussianModel:
    mean: np.ndarray
    cov: np.ndarray
    columns: list[str]
    loglik: float
    n_obs: int

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "GaussianModel":
        cols = list(X.columns)
        V = X.to_numpy(dtype=float)
        mu = V.mean(axis=0)
        cov = np.cov(V, rowvar=False, ddof=0)   # MLE-оценка (делитель n)
        ll = float(multivariate_normal(mu, cov, allow_singular=True).logpdf(V).sum())
        return cls(mean=mu, cov=cov, columns=cols, loglik=ll, n_obs=len(V))

    @property
    def n_params(self) -> int:
        p = len(self.mean)
        return p + n_cov_params(p)

    @property
    def aic(self) -> float:
        return aic(self.loglik, self.n_params)

    @property
    def bic(self) -> float:
        return bic(self.loglik, self.n_params, self.n_obs)

    def simulate(self, n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
        """n однодневных приращений факторов из N(μ, Σ). Форма (n, p)."""
        return rng.multivariate_normal(self.mean, self.cov, size=n_scenarios)
