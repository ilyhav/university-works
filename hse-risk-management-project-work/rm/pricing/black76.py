"""
Модель Блэка-76 — опционы на фьючерс (бонус, п.8).

Задание специально подчёркивает: опционы выписаны на ФЬЮЧЕРС, поэтому базовый
актив — фьючерсная цена F, и применяется модель Блэка-76 (не Блэка–Шоулза по
спот-цене). Премия дисконтируется по безрисковой ставке r на срок T:

    call = e^(−rT) · [F·Φ(d1) − K·Φ(d2)]
    put  = e^(−rT) · [K·Φ(−d2) − F·Φ(−d1)]
    d1 = [ln(F/K) + ½σ²T] / (σ√T),   d2 = d1 − σ√T.

Калибровка implied vol: численно решаем σ так, чтобы модельная премия совпала
с наблюдаемой ценой опциона (по соседним страйкам строится «улыбка»).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black76_price(F: float, K: float, T: float, sigma: float, r: float,
                  option: str = "call") -> float:
    """Премия опциона на фьючерс по Блэку-76."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if option == "call" else max(K - F, 0.0)
        return float(np.exp(-r * T) * intrinsic)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = np.exp(-r * T)
    if option == "call":
        return float(disc * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def black76_implied_vol(price: float, F: float, K: float, T: float, r: float,
                        option: str = "call") -> float:
    """Implied volatility из наблюдаемой премии (brentq по σ)."""
    from scipy.optimize import brentq

    disc = np.exp(-r * T)
    intrinsic = disc * (max(F - K, 0.0) if option == "call" else max(K - F, 0.0))
    if price <= intrinsic + 1e-12:
        return np.nan
    f = lambda s: black76_price(F, K, T, s, r, option) - price
    try:
        return float(brentq(f, 1e-4, 5.0, maxiter=200))
    except ValueError:
        return np.nan


def black76_greeks(F: float, K: float, T: float, sigma: float, r: float,
                   option: str = "call") -> dict[str, float]:
    """Дельта/гамма/вега по фьючерсу (для контроля чувствительности)."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    disc = np.exp(-r * T)
    delta = disc * (norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1.0)
    gamma = disc * norm.pdf(d1) / (F * sigma * sqrtT)
    vega = disc * F * norm.pdf(d1) * sqrtT
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}
