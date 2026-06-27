"""
Переоценка портфеля под сценариями риск-факторов.

Факторы приходят из outputs/factors/factor_scores.csv:
  * rate_PC1..3 — PCA-счёты дневных приращений КБД;
  * eq_<SECID> — лог-доходности акций;
  * fx_USD/fx_EUR — лог-доходности официальных курсов ЦБ.

Для ставок восстанавливаем сдвиг всей КБД через PCA-нагрузки:
    dy ~= scores @ loadings.T + mean(dy)
и переоцениваем будущие денежные потоки ОФЗ от новой кривой. Для акций и
валюты P&L линейно следует из exp(log-return)-1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rm.config import (
    BOND_NOTIONAL_EACH,
    EQUITY_NOTIONAL_EACH,
    FX_NOTIONAL_EACH,
    PORTFOLIO,
)
from rm.factors.pca import PCAResult
from rm.pricing.bonds import bond_cashflows, face_value, price_from_curve
from rm.pricing.curve import get_curve_row


RATE_COLUMNS = ("rate_PC1", "rate_PC2", "rate_PC3")


def _asof_row(df: pd.DataFrame, asof) -> pd.Series:
    sub = df.loc[:pd.Timestamp(asof)]
    if sub.empty:
        raise ValueError(f"нет данных на дату <= {pd.Timestamp(asof).date()}")
    return sub.iloc[-1]


@dataclass
class PortfolioState:
    """Состояние портфеля на дату оценки и функция сценарного P&L."""

    asof: pd.Timestamp
    base_curve: pd.Series
    rate_loadings: pd.DataFrame
    rate_mean: pd.Series
    bond_cashflows: dict[str, pd.DataFrame]
    bond_units: dict[str, float]
    bond_base_prices: dict[str, float]
    equity_prices: pd.Series
    equity_units: dict[str, float]
    fx_rates: pd.Series
    fx_units: dict[str, float]
    factor_columns: list[str]
    compounding: str = "annual"

    @classmethod
    def from_market_data(
        cls,
        md,
        asof,
        rate_pca: PCAResult,
        compounding: str = "annual",
    ) -> "PortfolioState":
        asof = pd.Timestamp(asof)
        curve = get_curve_row(md.gcurve, asof)

        loadings = rate_pca.loadings.copy()
        loadings.index = [float(x) for x in loadings.index]
        loadings = loadings.reindex(curve.index)
        if loadings.isna().any().any():
            missing = loadings.index[loadings.isna().any(axis=1)].tolist()
            raise ValueError(f"PCA-нагрузки не покрывают сроки КБД: {missing}")

        rate_mean = rate_pca.mean.copy()
        rate_mean.index = [float(x) for x in rate_mean.index]
        rate_mean = rate_mean.reindex(curve.index).fillna(0.0)

        bond_cf: dict[str, pd.DataFrame] = {}
        bond_units: dict[str, float] = {}
        bond_base: dict[str, float] = {}
        for secid in PORTFOLIO.bonds:
            cf = bond_cashflows(
                md.coupons.get(secid),
                md.amortizations.get(secid),
                asof,
            )
            face = face_value(md.amortizations.get(secid))
            bond_cf[secid] = cf
            bond_units[secid] = BOND_NOTIONAL_EACH / face
            bond_base[secid] = price_from_curve(cf, curve, compounding)

        equity_prices = _asof_row(md.stock_prices, asof).astype(float)
        equity_units = {
            secid: EQUITY_NOTIONAL_EACH / float(equity_prices[secid])
            for secid in PORTFOLIO.stocks
        }

        fx_rates = _asof_row(md.fx, asof).astype(float)
        fx_units = {
            cur: FX_NOTIONAL_EACH / float(fx_rates[cur])
            for cur in PORTFOLIO.fx
        }

        factor_columns = (
            list(RATE_COLUMNS)
            + [f"eq_{secid}" for secid in PORTFOLIO.stocks]
            + [f"fx_{cur}" for cur in PORTFOLIO.fx]
        )

        return cls(
            asof=asof,
            base_curve=curve,
            rate_loadings=loadings,
            rate_mean=rate_mean,
            bond_cashflows=bond_cf,
            bond_units=bond_units,
            bond_base_prices=bond_base,
            equity_prices=equity_prices,
            equity_units=equity_units,
            fx_rates=fx_rates,
            fx_units=fx_units,
            factor_columns=factor_columns,
            compounding=compounding,
        )

    @property
    def base_value(self) -> float:
        bond_value = sum(
            self.bond_units[b] * self.bond_base_prices[b]
            for b in self.bond_base_prices
        )
        equity_value = sum(
            self.equity_units[s] * float(self.equity_prices[s])
            for s in PORTFOLIO.stocks
        )
        fx_value = sum(
            self.fx_units[c] * float(self.fx_rates[c])
            for c in PORTFOLIO.fx
        )
        return float(bond_value + equity_value + fx_value)

    def pnl_components(
        self,
        factor_changes: pd.DataFrame | pd.Series | np.ndarray,
        horizon_days: int = 1,
    ) -> pd.DataFrame:
        """Вернуть P&L по классам активов и total для набора сценариев.

        factor_changes должны быть суммарными изменениями факторов за горизонт:
        для 10 дней это сумма 10 дневных лог-доходностей/PCA-счётов.
        """
        scenarios = self._coerce_scenarios(factor_changes)
        n = len(scenarios)
        bonds = np.zeros(n)
        equities = np.zeros(n)
        fx = np.zeros(n)

        rate_scores = scenarios.loc[:, list(RATE_COLUMNS)].to_numpy(dtype=float)
        dy = rate_scores @ self.rate_loadings.to_numpy().T
        dy = dy + horizon_days * self.rate_mean.to_numpy(dtype=float)
        base_curve_values = self.base_curve.to_numpy(dtype=float)
        shocked_curves = np.clip(base_curve_values[None, :] + dy, -0.95, None)
        tenors = self.base_curve.index.to_numpy(dtype=float)

        for secid, cf in self.bond_cashflows.items():
            shocked_prices = _price_cashflows_vectorized(
                cf,
                tenors,
                shocked_curves,
                self.compounding,
            )
            bonds += self.bond_units[secid] * (
                shocked_prices - self.bond_base_prices[secid]
            )

        for secid in PORTFOLIO.stocks:
            col = f"eq_{secid}"
            ret = scenarios[col].to_numpy(dtype=float)
            base = float(self.equity_prices[secid])
            equities += self.equity_units[secid] * base * np.expm1(ret)

        for cur in PORTFOLIO.fx:
            col = f"fx_{cur}"
            ret = scenarios[col].to_numpy(dtype=float)
            base = float(self.fx_rates[cur])
            fx += self.fx_units[cur] * base * np.expm1(ret)

        out = pd.DataFrame({
            "bonds": bonds,
            "equities": equities,
            "fx": fx,
        }, index=scenarios.index)
        out["total"] = out.sum(axis=1)
        return out

    def pnl(
        self,
        factor_changes: pd.DataFrame | pd.Series | np.ndarray,
        horizon_days: int = 1,
    ) -> pd.Series:
        return self.pnl_components(factor_changes, horizon_days)["total"]

    def _coerce_scenarios(
        self,
        factor_changes: pd.DataFrame | pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        if isinstance(factor_changes, pd.Series):
            df = factor_changes.to_frame().T
        elif isinstance(factor_changes, pd.DataFrame):
            df = factor_changes.copy()
        else:
            arr = np.asarray(factor_changes, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            df = pd.DataFrame(arr, columns=self.factor_columns)

        missing = [c for c in self.factor_columns if c not in df.columns]
        if missing:
            raise ValueError(f"не хватает факторов для переоценки: {missing}")
        return df.loc[:, self.factor_columns].astype(float)


def _price_cashflows_vectorized(
    cashflows: pd.DataFrame,
    tenors: np.ndarray,
    curves: np.ndarray,
    compounding: str,
) -> np.ndarray:
    """PV денежных потоков для всех сценариев сразу.

    curves: n_scenarios × n_tenors, значения zero-rate в долях.
    """
    n = curves.shape[0]
    if cashflows.empty:
        return np.zeros(n)
    price = np.zeros(n)
    for t, cf in zip(cashflows["t"].to_numpy(dtype=float), cashflows["cf"].to_numpy(dtype=float)):
        z = _interp_zero_vectorized(tenors, curves, float(t))
        if compounding == "continuous":
            df = np.exp(-z * t)
        else:
            df = (1.0 + z) ** (-t)
        price += cf * df
    return price


def _interp_zero_vectorized(tenors: np.ndarray, curves: np.ndarray, t: float) -> np.ndarray:
    if t <= tenors[0]:
        return curves[:, 0]
    if t >= tenors[-1]:
        return curves[:, -1]
    right = int(np.searchsorted(tenors, t, side="right"))
    left = right - 1
    w = (t - tenors[left]) / (tenors[right] - tenors[left])
    return curves[:, left] * (1.0 - w) + curves[:, right] * w
