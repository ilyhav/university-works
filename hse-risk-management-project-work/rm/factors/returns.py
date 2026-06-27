"""
Преобразование панелей рыночных данных в доходности / приращения факторов.

Ключевое методологическое решение (это обсуждается на защите):
  * цены акций, индексы, курсы валют  -> ЛОГ-доходности  r = ln(P_t / P_{t-1});
  * ставки кривой бескупонной доходности (КБД) -> АБСОЛЮТНЫЕ приращения
    Δy = y_t − y_{t-1}.

Почему ставки иначе: лог-доходность ставки бессмысленна (ставка может быть
около нуля или меняться скачком в разы при шоках 2022 г.), а риск облигаций
управляется именно изменением доходности Δy (дюрация × Δy). Поэтому факторы
кривой — это приращения ставок, а не их относительные изменения. Это и есть
подход Литтермана–Шейнкмана (PCA по Δy).

Все функции чистые: принимают и возвращают pandas-объекты, сети не трогают.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Базовые преобразования                                                      #
# --------------------------------------------------------------------------- #
def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Лог-доходности r_t = ln(P_t / P_{t-1}). Первая строка — NaN (выпадает)."""
    return np.log(prices).diff()


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Простые доходности (P_t / P_{t-1} − 1). Нужны для проверок P&L."""
    return prices.pct_change()


def curve_increments(gcurve: pd.DataFrame) -> pd.DataFrame:
    """Абсолютные дневные приращения ставок КБД: Δy_t = y_t − y_{t-1}.

    Колонки — сроки (годы), упорядочиваются по возрастанию срока.
    Значения в долях (0.001 = +10 б.п.).
    """
    ordered = gcurve.reindex(sorted(gcurve.columns, key=float), axis=1)
    return ordered.diff()


# --------------------------------------------------------------------------- #
# Склейка истории сменивших листинг тикеров                                   #
# --------------------------------------------------------------------------- #
def splice_returns(successor: pd.Series, predecessor: pd.Series) -> pd.Series:
    """Непрерывный ряд доходностей: берём доходность нового тикера, а там,
    где её ещё нет (до переезда), подставляем доходность старого.

    Склеиваем ДОХОДНОСТИ, а не цены: разрыв уровня цен на стыке не превращается
    в фантомную доходность — переходный день у нового тикера всё равно NaN
    (нет вчерашней цены) и выпадает.
    """
    return successor.combine_first(predecessor)


def build_equity_returns(
    stock_prices: pd.DataFrame,
    predecessor_prices: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Лог-доходности акций с непрерывной историей.

    stock_prices       — широкая таблица дата × SECID (чистые цены закрытия).
    predecessor_prices — {новый_тикер: ряд_цен_старого_тикера}; для каждого
                         такого тикера доходности склеиваются (см. splice_returns).
                         Старый ряд должен быть выровнен по тому же календарю.
    """
    rets = log_returns(stock_prices)
    if predecessor_prices:
        for successor, pred_px in predecessor_prices.items():
            if successor not in rets.columns:
                logger.warning("склейка: тикера %s нет в портфеле — пропуск", successor)
                continue
            pred_ret = log_returns(pred_px.reindex(stock_prices.index))
            before = int(rets[successor].notna().sum())
            rets[successor] = splice_returns(rets[successor], pred_ret)
            after = int(rets[successor].notna().sum())
            logger.info("склейка %s: доходностей %d -> %d (+%d из предшественника)",
                        successor, before, after, after - before)
    return rets


# --------------------------------------------------------------------------- #
# Подготовка матрицы под PCA / ковариацию                                     #
# --------------------------------------------------------------------------- #
@dataclass
class CoverageReport:
    """Диагностика покрытия по колонкам матрицы доходностей."""

    table: pd.DataFrame          # по колонке: n_obs, доля_NaN, первая/последняя дата
    rows_total: int              # строк всего
    rows_complete: int           # строк без единого NaN (попадут в PCA listwise)

    def dropped_rows(self) -> int:
        return self.rows_total - self.rows_complete


def coverage(returns: pd.DataFrame) -> CoverageReport:
    """Сколько наблюдений у каждой колонки и сколько строк полностью без NaN."""
    rows = []
    for col in returns.columns:
        s = returns[col]
        valid = s.dropna()
        rows.append({
            "колонка": col,
            "n_obs": int(s.notna().sum()),
            "доля_NaN": round(float(s.isna().mean()), 4),
            "первая": str(valid.index.min().date()) if not valid.empty else "—",
            "последняя": str(valid.index.max().date()) if not valid.empty else "—",
        })
    tbl = pd.DataFrame(rows).set_index("колонка")
    complete = int(returns.dropna(how="any").shape[0])
    return CoverageReport(table=tbl, rows_total=len(returns), rows_complete=complete)


def clean_for_pca(
    returns: pd.DataFrame,
    min_coverage: float = 0.5,
) -> tuple[pd.DataFrame, CoverageReport]:
    """Подготовить матрицу к PCA: отбросить колонки с покрытием ниже порога,
    затем listwise-удаление строк с NaN. Возвращает (чистая матрица, отчёт).

    listwise (а не попарные ковариации) — чтобы матрица ковариаций гарантированно
    была положительно полуопределённой и собственные числа не уходили в минус.
    Для дневных доходностей 2021–2025 редкие дни простоя выпадают почти без потерь;
    отчёт фиксирует, сколько строк ушло (это надо упомянуть в обсуждении).
    """
    cov_by_col = returns.notna().mean()
    keep = cov_by_col[cov_by_col >= min_coverage].index.tolist()
    dropped_cols = [c for c in returns.columns if c not in keep]
    if dropped_cols:
        logger.warning("PCA: колонки с покрытием < %.0f%% отброшены: %s",
                       100 * min_coverage, dropped_cols)
    sub = returns[keep]
    rep = coverage(sub)
    clean = sub.dropna(how="any")
    if rep.dropped_rows():
        logger.info("PCA: listwise-удаление строк с NaN: %d из %d (-%d)",
                    rep.rows_complete, rep.rows_total, rep.dropped_rows())
    return clean, rep
