"""
Многомерная t-Стьюдента: тяжёлые хвосты + хвостовая зависимость факторов.

Подгонка — MLE через ECME-алгоритм (вариант EM):
  * E-шаг: веса w_i = (ν+p)/(ν+d_i), d_i — квадрат расстояния Махаланобиса;
  * CM-шаг для μ, Σ (матрица масштаба «scale», НЕ ковариация);
  * условная максимизация по ν (профильный log-lik, одномерная оптимизация).

Важно: для многомерной t матрица Σ — это масштаб; ковариация = Σ·ν/(ν−2)
(существует только при ν>2). Малое ν ⇔ очень тяжёлые хвосты — ровно то,
что показал дескриптивный анализ (эксцесс ≫ 0, df≈3 у рыночного фактора).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_t

from rm.models.base import aic, bic, n_cov_params


@dataclass
class StudentTModel:
    mean: np.ndarray          # μ (loc)
    scale: np.ndarray         # Σ (матрица масштаба, не ковариация)
    df: float                 # ν — число степеней свободы
    columns: list[str]
    loglik: float
    n_obs: int

    @classmethod
    def fit(cls, X: pd.DataFrame, tol: float = 1e-6, max_iter: int = 500) -> "StudentTModel":
        cols = list(X.columns)
        V = X.to_numpy(dtype=float)
        n, p = V.shape

        mu = V.mean(axis=0)
        S = np.cov(V, rowvar=False, ddof=0)
        nu = 8.0
        prev_ll = -np.inf

        for _ in range(max_iter):
            Sinv = np.linalg.pinv(S)
            diff = V - mu
            d = np.einsum("ij,jk,ik->i", diff, Sinv, diff)   # Махаланобис²
            w = (nu + p) / (nu + d)                          # E-шаг

            mu = (w[:, None] * V).sum(axis=0) / w.sum()      # CM-шаг: μ
            diff = V - mu
            S = (w[:, None] * diff).T @ diff / n             # CM-шаг: Σ

            def neg_ll(log_nu: float) -> float:
                nu_ = np.exp(log_nu)
                return -float(multivariate_t.logpdf(V, loc=mu, shape=S, df=nu_).sum())

            res = minimize_scalar(neg_ll, bounds=(np.log(2.05), np.log(250.0)),
                                  method="bounded")
            nu = float(np.exp(res.x))

            ll = float(multivariate_t.logpdf(V, loc=mu, shape=S, df=nu).sum())
            if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol * max(abs(prev_ll), 1.0):
                prev_ll = ll
                break
            prev_ll = ll

        return cls(mean=mu, scale=S, df=nu, columns=cols, loglik=prev_ll, n_obs=n)

    @property
    def cov(self) -> np.ndarray:
        """Ковариация = Σ·ν/(ν−2) (существует только при ν>2)."""
        if self.df <= 2:
            return np.full_like(self.scale, np.inf)
        return self.scale * self.df / (self.df - 2.0)

    @property
    def n_params(self) -> int:
        p = len(self.mean)
        return p + n_cov_params(p) + 1   # +1 за ν

    @property
    def aic(self) -> float:
        return aic(self.loglik, self.n_params)

    @property
    def bic(self) -> float:
        return bic(self.loglik, self.n_params, self.n_obs)

    def simulate(self, n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
        """n однодневных приращений факторов из t_ν(μ, Σ). Форма (n, p)."""
        sample = multivariate_t.rvs(loc=self.mean, shape=self.scale, df=self.df,
                                    size=n_scenarios, random_state=rng)
        return np.atleast_2d(sample)
