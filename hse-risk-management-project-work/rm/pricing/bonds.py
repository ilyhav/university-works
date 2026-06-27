"""
Ценообразование ОФЗ от кривой КБД (п.4 — справедливая стоимость).

Справедливая «грязная» цена = приведённая стоимость всех будущих денежных
потоков (купоны + погашение номинала), дисконтированных по бескупонной кривой:

    PV = Σ_i CF_i · DF(t_i),   DF — дисконт-фактор из rm.pricing.curve.

Денежные потоки берём из эмиссионного расписания MOEX (bondization): купоны в
рублях и амортизация/погашение номинала в рублях — так не зависим от соглашений
о номинале и день-счёте купона.

Контроль точности (требование п.4):
  * ошибка модельной цены против рыночной — в рублях и в % номинала;
  * ошибка в доходности — модельная YTM против рыночной (б.п.);
  * дюрация (Маколея/модифицированная) и выпуклость как sanity-check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rm.pricing.curve import discount_factor


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def bond_cashflows(coupons: pd.DataFrame, amorts: pd.DataFrame, asof,
                   day_count: float = 365.0) -> pd.DataFrame:
    """Будущие денежные потоки облигации после даты asof.

    Возвращает DataFrame: date, t (годы до выплаты), cf (руб.), kind.
    Купоны — из блока coupons (колонка value, руб.); погашение номинала —
    из блока amortizations (колонка value, руб.). Берём только выплаты > asof.
    """
    asof = pd.Timestamp(asof)
    rows: list[dict] = []

    if coupons is not None and not coupons.empty:
        for _, c in coupons.iterrows():
            d = pd.Timestamp(c.get("coupondate"))
            if pd.isna(d) or d <= asof:
                continue
            val = _to_float(c.get("value"))
            if np.isnan(val):
                continue
            rows.append({"date": d, "cf": val, "kind": "купон"})

    if amorts is not None and not amorts.empty:
        for _, a in amorts.iterrows():
            d = pd.Timestamp(a.get("amortdate"))
            val = _to_float(a.get("value"))
            if pd.isna(d) or d <= asof or np.isnan(val) or val == 0:
                continue
            rows.append({"date": d, "cf": val, "kind": "номинал"})

    if not rows:
        return pd.DataFrame(columns=["date", "cf", "kind", "t"])
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["t"] = (df["date"] - asof).dt.days / day_count
    return df


def face_value(amorts: pd.DataFrame) -> float:
    """Номинал = сумма всех выплат тела долга (для ОФЗ-ПД — единственное погашение)."""
    if amorts is None or amorts.empty or "value" not in amorts:
        return 1000.0
    total = pd.to_numeric(amorts["value"], errors="coerce").sum()
    return float(total) if total > 0 else 1000.0


def price_from_curve(cashflows: pd.DataFrame, curve_row: pd.Series,
                     compounding: str = "annual") -> float:
    """Грязная цена (руб.) = Σ CF·DF(t) по кривой КБД."""
    if cashflows.empty:
        return 0.0
    dfs = np.array([discount_factor(t, curve_row, compounding) for t in cashflows["t"]])
    return float((cashflows["cf"].to_numpy() * dfs).sum())


def pv_at_yield(cashflows: pd.DataFrame, y: float) -> float:
    """PV при плоской доходности y (для YTM/дюрации). Годовая капитализация."""
    t = cashflows["t"].to_numpy()
    cf = cashflows["cf"].to_numpy()
    return float((cf / (1.0 + y) ** t).sum())


def ytm_from_price(cashflows: pd.DataFrame, dirty_price: float) -> float:
    """Доходность к погашению из грязной цены (численно, brentq)."""
    from scipy.optimize import brentq

    if cashflows.empty or dirty_price <= 0:
        return np.nan
    f = lambda y: pv_at_yield(cashflows, y) - dirty_price
    try:
        return float(brentq(f, -0.5, 3.0, maxiter=200))
    except ValueError:
        return np.nan


def duration_convexity(cashflows: pd.DataFrame, y: float) -> tuple[float, float, float]:
    """Дюрация Маколея, модифицированная дюрация, выпуклость при доходности y."""
    t = cashflows["t"].to_numpy()
    cf = cashflows["cf"].to_numpy()
    pv = cf / (1.0 + y) ** t
    P = pv.sum()
    if P <= 0:
        return np.nan, np.nan, np.nan
    macaulay = float((t * pv).sum() / P)
    modified = macaulay / (1.0 + y)
    convexity = float((t * (t + 1.0) * pv).sum() / (1.0 + y) ** 2 / P)
    return macaulay, modified, convexity


@dataclass
class BondPricing:
    secid: str
    asof: pd.Timestamp
    face: float
    n_flows: int
    model_dirty: float          # модельная грязная цена, руб.
    market_dirty: float         # рыночная грязная цена, руб.
    market_clean_pct: float     # рыночная чистая цена, % номинала
    accrued: float              # НКД, руб.
    err_rub: float              # модель − рынок, руб.
    err_pct_face: float         # ошибка в % номинала
    model_ytm: float            # модельная YTM (из модельной цены)
    market_ytm: float           # рыночная YTM (YIELDCLOSE/100)
    err_ytm_bp: float           # ошибка доходности, б.п.
    macaulay: float
    modified_dur: float
    convexity: float


def price_bond(secid: str, coupons: pd.DataFrame, amorts: pd.DataFrame,
               curve_row: pd.Series, asof, market_clean_pct: float,
               accrued: float, market_ytm: float,
               compounding: str = "annual") -> BondPricing:
    """Полный отчёт по одной облигации на дату asof: модель vs рынок."""
    asof = pd.Timestamp(asof)
    face = face_value(amorts)
    cf = bond_cashflows(coupons, amorts, asof)

    model_dirty = price_from_curve(cf, curve_row, compounding)
    market_dirty = market_clean_pct / 100.0 * face + accrued

    model_ytm = ytm_from_price(cf, model_dirty)
    mac, mod, conv = duration_convexity(cf, market_ytm if not np.isnan(market_ytm)
                                        else model_ytm)

    err_rub = model_dirty - market_dirty
    return BondPricing(
        secid=secid, asof=asof, face=face, n_flows=len(cf),
        model_dirty=model_dirty, market_dirty=market_dirty,
        market_clean_pct=market_clean_pct, accrued=accrued,
        err_rub=err_rub, err_pct_face=err_rub / face * 100.0,
        model_ytm=model_ytm, market_ytm=market_ytm,
        err_ytm_bp=(model_ytm - market_ytm) * 1e4
        if not (np.isnan(model_ytm) or np.isnan(market_ytm)) else np.nan,
        macaulay=mac, modified_dur=mod, convexity=conv,
    )
