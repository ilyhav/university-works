"""
Стохастические модели динамики факторов (этап 3), все на MLE:
  * normal     — многомерная нормаль (база Башелье/GBM);
  * student_t  — многомерная t-Стьюдента (тяжёлые хвосты, оценка ν по ECME/EM);
  * ewma       — EWMA-ковариация RiskMetrics (кластеризация волатильности);
  * garch      — GARCH(1,1) на фактор + постоянная корреляция (CCC);
  * ou         — AR(1)/Орнштейн–Уленбек, возврат к среднему для ставок.

Каждая модель умеет .simulate(n, rng) -> однодневные приращения факторов и
сравнивается по .aic/.bic (см. base). Это вход для риск-движка (этап 5).
"""
from rm.models.base import aic, bic, n_cov_params
from rm.models.normal import GaussianModel
from rm.models.student_t import StudentTModel
from rm.models.ewma import EWMAModel
from rm.models.garch import CCCGarchModel, GarchFactor
from rm.models.ou import AR1Result, ar1_table, fit_ar1

__all__ = [
    "aic", "bic", "n_cov_params",
    "GaussianModel",
    "StudentTModel",
    "EWMAModel",
    "CCCGarchModel", "GarchFactor",
    "AR1Result", "ar1_table", "fit_ar1",
]
