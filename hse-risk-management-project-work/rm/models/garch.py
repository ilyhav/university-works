"""
GARCH(1,1) на каждый фактор + постоянная условная корреляция (CCC).

GARCH воспроизводит кластеризацию волатильности явной динамикой дисперсии:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1},     персистентность = α+β (→1 = долгая память).
Инновации — t-Стьюдента (учитывают тяжёлые хвосты сверх GARCH).

Многомерность собираем по схеме CCC (Bollerslev): индивидуальные GARCH-σ плюс
ПОСТОЯННАЯ корреляция стандартизованных остатков:
    Σ_{t+1} = D_{t+1} · R · D_{t+1},   D = diag(σ_i),  R = corr(стд. остатков).
Это компромисс: динамика волатильности у каждого фактора своя, а корреляции
считаем стабильными (упрощение — обсуждается; полный DCC сложнее и для 1–10
дней избыточен).

Числовая устойчивость: arch предпочитает ряды с σ≈1..1000, поэтому каждый
фактор центрируется и масштабируется к σ≈10, а σ-прогноз возвращается в
исходные единицы.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

_SCALE = 10.0  # целевая σ ряда для arch


@dataclass
class GarchFactor:
    name: str
    omega: float
    alpha: float
    beta: float
    persistence: float
    nu: float                 # ν инноваций t (NaN, если dist='normal')
    next_vol: float           # прогноз дневной σ на следующий день, исходные единицы
    loglik: float             # log-lik arch (в масштабированных единицах)


@dataclass
class CCCGarchModel:
    factors: list[GarchFactor]
    corr: np.ndarray              # R — постоянная корреляция стд. остатков
    columns: list[str]
    next_cov: np.ndarray          # Σ на следующий день
    cond_vol: dict[str, pd.Series]  # ряды условной σ (исходные единицы)

    @classmethod
    def fit(cls, X: pd.DataFrame, dist: str = "t") -> "CCCGarchModel":
        from arch import arch_model

        cols = list(X.columns)
        factors: list[GarchFactor] = []
        std_resid = {}
        cond_vol = {}
        next_vol = {}

        for name in cols:
            x = X[name].to_numpy(dtype=float)
            mu, sd = x.mean(), x.std(ddof=1)
            y = (x - mu) / sd * _SCALE          # к σ≈10

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1,
                                dist=dist, rescale=False)
                res = am.fit(disp="off", show_warning=False)

            p = res.params
            alpha = float(p.get("alpha[1]", np.nan))
            beta = float(p.get("beta[1]", np.nan))
            omega = float(p.get("omega", np.nan))
            nu = float(p.get("nu", np.nan))

            fc = res.forecast(horizon=1, reindex=False)
            var_next_y = float(fc.variance.to_numpy()[-1, 0])
            nv = np.sqrt(var_next_y) / _SCALE * sd      # обратно в исходные единицы
            next_vol[name] = nv

            cv = res.conditional_volatility / _SCALE * sd
            cond_vol[name] = pd.Series(cv, index=X.index, name=f"GARCH σ {name}")
            std_resid[name] = res.resid / res.conditional_volatility  # безразмерн.

            factors.append(GarchFactor(
                name=name, omega=omega, alpha=alpha, beta=beta,
                persistence=alpha + beta, nu=nu, next_vol=nv,
                loglik=float(res.loglikelihood),
            ))

        E = pd.DataFrame(std_resid).to_numpy()
        R = np.atleast_2d(np.corrcoef(E, rowvar=False))   # (1,1) при одном факторе
        d = np.array([next_vol[c] for c in cols])
        D = np.diag(d)
        next_cov = D @ R @ D

        return cls(factors=factors, corr=R, columns=cols,
                   next_cov=next_cov, cond_vol=cond_vol)

    def table(self) -> pd.DataFrame:
        """Сводка по факторам: ω, α, β, персистентность, ν, σ на завтра (год.)."""
        rows = {f.name: {
            "omega": f.omega, "alpha": f.alpha, "beta": f.beta,
            "персистентность": f.persistence, "nu": f.nu,
            "σ_завтра_дн": f.next_vol,
            "σ_завтра_год": f.next_vol * np.sqrt(250),
        } for f in self.factors}
        return pd.DataFrame(rows).T

    def simulate(self, n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
        """Однодневные приращения из N(0, next_cov) (условная ковариация на завтра)."""
        return rng.multivariate_normal(np.zeros(len(self.columns)),
                                       self.next_cov, size=n_scenarios)
