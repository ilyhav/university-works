"""
Процесс Орнштейна–Уленбека / AR(1) — возврат к среднему для факторов ставок.

Дискретный AR(1):  x_t = c + φ·x_{t-1} + ε_t,  ε ~ N(0, σ²).
Это точная дискретизация OU; параметры пересчитываются в непрерывные:
  * скорость возврата θ = −ln(φ)/Δt   (Δt = 1 торговый день);
  * долгосрочное среднее  m = c/(1−φ);
  * период полураспада  t½ = ln2 / θ  (за сколько дней отклонение гаснет вдвое).

MLE при гауссовых ε эквивалентен OLS-регрессии x_t на x_{t-1} (условное
правдоподобие). Если φ≈1 — это случайное блуждание (нет возврата), и на
горизонте 1–10 дней OU вырождается в random walk: для риска ставок этого
обычно и достаточно — проверяем явно.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AR1Result:
    name: str
    c: float
    phi: float
    sigma: float            # σ остатков
    phi_se: float           # стандартная ошибка φ
    half_life: float        # дни (inf, если φ≥1)
    long_run_mean: float

    @property
    def mean_reverting(self) -> bool:
        return 0.0 < self.phi < 1.0

    @property
    def rw_tstat(self) -> float:
        """t-статистика H0: φ=1 (близость к случайному блужданию)."""
        return (self.phi - 1.0) / self.phi_se if self.phi_se > 0 else np.nan

    @property
    def wn_tstat(self) -> float:
        """t-статистика H0: φ=0 (близость к белому шуму / i.i.d.)."""
        return self.phi / self.phi_se if self.phi_se > 0 else np.nan

    @property
    def regime(self) -> str:
        """Классификация по двум t-тестам (φ=0 и φ=1), без привязки к смыслу ряда.
        Для приращений «белый шум» = i.i.d.; для уровней «случайное блуждание» =
        единичный корень (нет возврата к среднему)."""
        t0, t1 = self.wn_tstat, self.rw_tstat
        if abs(t0) < 2:
            return "≈ белый шум (φ≈0)"
        if abs(t1) < 2:
            return "≈ случайное блуждание (φ≈1)"
        if 0.0 < self.phi < 1.0:
            return "возврат к среднему"
        return "иное (φ вне [0,1])"


def fit_ar1(series: pd.Series, name: str | None = None) -> AR1Result:
    x = series.dropna().to_numpy(dtype=float)
    y, xl = x[1:], x[:-1]
    n = len(y)
    A = np.column_stack([np.ones(n), xl])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, phi = float(beta[0]), float(beta[1])

    resid = y - A @ beta
    dof = max(n - 2, 1)
    sigma = float(np.sqrt((resid @ resid) / dof))
    # ковариация OLS-оценок -> se(φ)
    xtx_inv = np.linalg.inv(A.T @ A)
    phi_se = float(sigma * np.sqrt(xtx_inv[1, 1]))

    half_life = float(np.log(2) / -np.log(phi)) if 0.0 < phi < 1.0 else np.inf
    long_run = float(c / (1.0 - phi)) if phi != 1.0 else np.nan
    return AR1Result(name=name or series.name, c=c, phi=phi, sigma=sigma,
                     phi_se=phi_se, half_life=half_life, long_run_mean=long_run)


def ar1_table(X: pd.DataFrame) -> pd.DataFrame:
    """AR(1)/OU по каждому столбцу: φ, оба t-теста (φ=0 и φ=1), полураспад, режим."""
    rows = {}
    for col in X.columns:
        r = fit_ar1(X[col], name=col)
        rows[col] = {
            "phi": r.phi,
            "phi_se": r.phi_se,
            "t(φ=0)": r.wn_tstat,
            "t(φ=1)": r.rw_tstat,
            "полураспад_дн": r.half_life,
            "долгоср_среднее": r.long_run_mean,
            "режим": r.regime,
        }
    return pd.DataFrame(rows).T
