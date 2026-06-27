"""
Сборка единой согласованной панели рыночных данных.

Объединяет загрузки из MOEX и ЦБ в один объект MarketData с общим
календарём торговых дней. Это вход для всех последующих шагов
(риск-факторы, модели, ценообразование, риск, бэктест).

Пропуски (например, дни, когда USD/EUR не торговались на MOEX, но курс
ЦБ есть) выравниваются по бизнес-календарю с forward-fill там, где это
экономически корректно (цены), и без заполнения там, где нет (доходности —
лучше оставить NaN и обсудить).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from rm.config import (
    HIST_END,
    HIST_START,
    PORTFOLIO,
    BENCHMARK_TICKERS,
)
from rm.data import cbr, moex

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Контейнер согласованных рядов на общем календаре торговых дней."""

    stock_prices: pd.DataFrame      # дата × SECID, чистая цена закрытия
    bond_clean: pd.DataFrame        # дата × SECID, чистая цена, % номинала
    bond_accint: pd.DataFrame       # дата × SECID, НКД, руб.
    bond_yield: pd.DataFrame        # дата × SECID, YTM, % годовых
    gcurve: pd.DataFrame            # дата × срок(годы), доходность в долях
    fx: pd.DataFrame                # дата × {USD,EUR}, руб. за единицу
    indices: pd.DataFrame           # дата × {IMOEX,RTSI}
    coupons: dict[str, pd.DataFrame]       # SECID -> расписание купонов
    amortizations: dict[str, pd.DataFrame] # SECID -> график погашения

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.stock_prices.index


def _iso(d) -> str:
    return d.strftime("%Y-%m-%d")


def build_market_data(force_reload: bool = False) -> MarketData:
    """Загрузить и склеить всё. Тяжёлая функция — результат кэшируется
    по слоям (каждая загрузка отдельно), поэтому повторный вызов быстрый."""
    s, e = _iso(HIST_START), _iso(HIST_END)
    fr = force_reload

    logger.info("=== Акции ===")
    stock_prices = moex.load_many(
        moex.load_stock_history, PORTFOLIO.stocks, s, e, value_col="CLOSE", force_reload=fr
    )

    logger.info("=== Облигации ===")
    bond_clean = moex.load_many(
        moex.load_bond_history, PORTFOLIO.bonds, s, e, value_col="CLOSE", force_reload=fr
    )
    bond_accint = moex.load_many(
        moex.load_bond_history, PORTFOLIO.bonds, s, e, value_col="ACCINT", force_reload=fr
    )
    bond_yield = moex.load_many(
        moex.load_bond_history, PORTFOLIO.bonds, s, e, value_col="YIELDCLOSE", force_reload=fr
    )
    coupons = {b: moex.load_bond_coupons(b, force_reload=fr) for b in PORTFOLIO.bonds}
    amorts = {b: moex.load_bond_amortization(b, force_reload=fr) for b in PORTFOLIO.bonds}

    logger.info("=== Индексы ===")
    idx = {t: moex.load_index_history(t, s, e, force_reload=fr)["CLOSE"]
           for t in BENCHMARK_TICKERS}
    indices = pd.DataFrame(idx)

    logger.info("=== КБД (ЦБ) ===")
    gcurve = cbr.load_gcurve(s, e, force_reload=fr)
    gcurve = _ensure_decimal_rates(gcurve)

    logger.info("=== Курсы валют (ЦБ) ===")
    fx = pd.concat(
        {c: cbr.load_fx_rate(c, s, e, force_reload=fr)[c] for c in PORTFOLIO.fx}, axis=1
    )

    md = MarketData(
        stock_prices=stock_prices,
        bond_clean=bond_clean,
        bond_accint=bond_accint,
        bond_yield=bond_yield,
        gcurve=gcurve,
        fx=fx,
        indices=indices,
        coupons=coupons,
        amortizations=amorts,
    )
    _align(md)
    return md


def _align(md: MarketData) -> None:
    """Привести всё к общему календарю торговых дней (по акциям)."""
    cal = md.stock_prices.dropna(how="all").index
    md.bond_clean = md.bond_clean.reindex(cal)
    md.bond_accint = md.bond_accint.reindex(cal)
    md.bond_yield = md.bond_yield.reindex(cal)
    md.gcurve = md.gcurve.reindex(cal).ffill()       # кривая — медленная, ffill ок
    md.fx = md.fx.reindex(cal).ffill()               # курс ЦБ есть и в неторговые дни
    md.indices = md.indices.reindex(cal)
    logger.info("выровнено по календарю: %s торговых дней (%s — %s)",
                len(cal), cal.min().date(), cal.max().date())


def _ensure_decimal_rates(df: pd.DataFrame) -> pd.DataFrame:
    """КБД должна быть в долях: 0.14 = 14%.

    Старый кэш мог сохраниться в процентных пунктах (14 вместо 0.14), поэтому
    нормализуем после чтения из parquet, а не только в парсере ЦБ.
    """
    if df.empty:
        return df
    out = df.copy()
    converted = False
    while True:
        values = pd.Series(out.to_numpy().ravel()).dropna()
        if values.empty or values.abs().median() <= 1.0:
            if converted:
                logger.warning("КБД похожа на проценты, перевожу в доли")
            return out
        converted = True
        out = out / 100.0
