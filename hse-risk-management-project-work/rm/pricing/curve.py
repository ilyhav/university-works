"""
Дисконтная кривая из КБД (кривой бескупонной доходности ЦБ).

КБД задаёт бескупонную (zero-coupon) доходность z(t) на стандартных сроках.
Для произвольного срока t интерполируем линейно по сроку, за пределами узлов —
плоская экстраполяция (короче 0.25 г и длиннее 30 лет ставку держим постоянной).

Соглашение о капитализации (важный пункт для обсуждения): по умолчанию
ЭФФЕКТИВНАЯ ГОДОВАЯ — DF(t) = (1+z)^(−t). Альтернатива — непрерывная,
DF(t) = exp(−z·t); ошибка ценообразования против рынка покажет, какое
соглашение ближе к методике ЦБ. Обе поддержаны параметром compounding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_curve_row(gcurve: pd.DataFrame, asof) -> pd.Series:
    """Срез кривой на дату asof (последняя доступная дата ≤ asof).

    Возвращает Series: индекс — срок (годы, float, по возрастанию), значение —
    доходность в долях.
    """
    asof = pd.Timestamp(asof)
    sub = gcurve.loc[:asof]
    if sub.empty:
        raise ValueError(f"нет данных КБД на дату ≤ {asof.date()}")
    row = sub.iloc[-1].dropna()
    row.index = [float(t) for t in row.index]
    return row.sort_index()


def interp_zero(curve_row: pd.Series, t: float) -> float:
    """Бескупонная доходность на срок t (годы); плоская экстраполяция по краям."""
    tenors = curve_row.index.to_numpy(dtype=float)
    yields = curve_row.to_numpy(dtype=float)
    if t <= tenors[0]:
        return float(yields[0])
    if t >= tenors[-1]:
        return float(yields[-1])
    return float(np.interp(t, tenors, yields))


def discount_factor(t: float, curve_row: pd.Series, compounding: str = "annual") -> float:
    """Дисконт-фактор на срок t лет по кривой. compounding: 'annual'|'continuous'."""
    if t <= 0:
        return 1.0
    z = interp_zero(curve_row, t)
    if compounding == "continuous":
        return float(np.exp(-z * t))
    return float((1.0 + z) ** (-t))
