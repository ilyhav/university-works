"""
Ценообразование (этап 4 — справедливая стоимость):
  * curve   — дисконтная кривая из КБД (интерполяция, дисконт-факторы);
  * bonds   — ОФЗ от кривой: денежные потоки, PV, YTM, дюрация/выпуклость,
              ошибка модель–рынок;
  * black76 — опционы на фьючерс (бонус, п.8): премия и implied vol.

Акции и валюта оцениваются линейно (стоимость = цена/курс × объём) — отдельного
модуля не требуют; их репрайсинг под факторные шоки соберётся в риск-движке.
"""
from rm.pricing.curve import discount_factor, get_curve_row, interp_zero
from rm.pricing.bonds import (
    BondPricing,
    bond_cashflows,
    duration_convexity,
    face_value,
    price_bond,
    price_from_curve,
    pv_at_yield,
    ytm_from_price,
)
from rm.pricing.black76 import black76_greeks, black76_implied_vol, black76_price

__all__ = [
    "discount_factor", "get_curve_row", "interp_zero",
    "BondPricing", "bond_cashflows", "duration_convexity", "face_value",
    "price_bond", "price_from_curve", "pv_at_yield", "ytm_from_price",
    "black76_greeks", "black76_implied_vol", "black76_price",
]
