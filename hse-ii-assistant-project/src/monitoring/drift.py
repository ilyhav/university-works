"""Обнаружение дрейфа: PSI по распределениям, сдвиг каталога, свежесть данных.

Два источника деградации, за которыми мы следим:
  1. дрейф ВХОДА  — меняется профиль запросов клиентов (суммы/сроки);
  2. дрейф ДАННЫХ — меняется рынок (ключевая ставка → ставки вкладов), и если
     каталог не обновляется, рекомендации перестают соответствовать реальности.
"""

from __future__ import annotations

import numpy as np

from ..core.models import Deposit

# Классические пороги PSI.
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Population Stability Index между эталонным и текущим распределениями."""
    expected = np.asarray([x for x in expected if x is not None], dtype=float)
    actual = np.asarray([x for x in actual if x is not None], dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Границы бинов — по квантилям эталона; уникализируем, чтобы не было нулевой ширины.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    eps = 1e-6
    e_pct = np.clip(e_counts / e_counts.sum(), eps, None)
    a_pct = np.clip(a_counts / a_counts.sum(), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_band(value: float) -> str:
    if value < PSI_MODERATE:
        return "нет значимого дрейфа"
    if value < PSI_SIGNIFICANT:
        return "умеренный дрейф"
    return "значимый дрейф"


def catalog_summary(deposits: list[Deposit], currency: str = "RUB") -> dict:
    """Сводные статистики каталога по валюте — для снапшотов и сравнения."""
    rates = [d.nominal_rate for d in deposits if d.currency == currency]
    arr = np.asarray(rates, dtype=float)
    if arr.size == 0:
        return {"currency": currency, "n": 0}
    return {
        "currency": currency,
        "n": int(arr.size),
        "mean_rate": round(float(arr.mean()), 3),
        "median_rate": round(float(np.median(arr)), 3),
        "min_rate": round(float(arr.min()), 3),
        "max_rate": round(float(arr.max()), 3),
        "as_of": max(d.as_of_date for d in deposits).isoformat(),
    }


def catalog_rate_psi(
    baseline: list[Deposit], current: list[Deposit], currency: str = "RUB"
) -> dict:
    """PSI распределения номинальных ставок между двумя версиями каталога."""
    base_rates = [d.nominal_rate for d in baseline if d.currency == currency]
    cur_rates = [d.nominal_rate for d in current if d.currency == currency]
    value = psi(base_rates, cur_rates)
    return {
        "metric": "catalog_rate_psi",
        "currency": currency,
        "psi": round(value, 4),
        "band": psi_band(value),
        "baseline_mean": round(float(np.mean(base_rates)), 3) if base_rates else None,
        "current_mean": round(float(np.mean(cur_rates)), 3) if cur_rates else None,
    }


def freshness_status(freshness_days: int | None, sla_days: int) -> dict:
    """Статус свежести каталога относительно SLA."""
    if freshness_days is None:
        return {"freshness_days": None, "sla_days": sla_days, "stale": False, "level": "unknown"}
    if freshness_days <= sla_days:
        level = "ok"
    elif freshness_days <= 2 * sla_days:
        level = "warning"
    else:
        level = "critical"
    return {
        "freshness_days": freshness_days,
        "sla_days": sla_days,
        "stale": freshness_days > sla_days,
        "level": level,
    }
