"""
Дескриптивный анализ риск-факторов (п.2c задания).

Что считаем и зачем (это всё — материал для критического обсуждения):
  * моменты: среднее, σ, годовая волатильность, асимметрия, эксцесс;
  * нормальность: тест Жарка–Бера (по асимметрии и эксцессу);
  * тяжесть хвостов: индекс хвоста Хилла (отдельно левый/правый хвост);
  * стационарность: ADF (H0: единичный корень) + KPSS (H0: стационарность) —
    два теста с противоположными гипотезами дают устойчивый вывод;
  * кластеризация волатильности: тест Льюнга–Бокса по КВАДРАТАМ доходностей
    (значимая автокорреляция квадратов = ARCH-эффект, нужен GARCH/EWMA);
  * корреляции между факторами.

Все функции возвращают аккуратные DataFrame/словари — скрипт сохраняет их в CSV.
Зависимости: scipy.stats, statsmodels (есть в requirements).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from rm.config import TRADING_DAYS_PER_YEAR


# --------------------------------------------------------------------------- #
# Моменты + нормальность                                                      #
# --------------------------------------------------------------------------- #
def moments_table(returns: pd.DataFrame) -> pd.DataFrame:
    """По каждому ряду: моменты, годовая волатильность и тест Жарка–Бера."""
    rows = {}
    for col in returns.columns:
        x = returns[col].dropna().to_numpy()
        if len(x) < 20:
            continue
        jb_stat, jb_p = stats.jarque_bera(x)
        rows[col] = {
            "n": len(x),
            "среднее": float(np.mean(x)),
            "σ_дн": float(np.std(x, ddof=1)),
            "σ_год": float(np.std(x, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "асимметрия": float(stats.skew(x)),
            "эксцесс_изб": float(stats.kurtosis(x, fisher=True)),  # избыточный (норм=0)
            "JB_стат": float(jb_stat),
            "JB_pvalue": float(jb_p),
            "норм?": "нет" if jb_p < 0.05 else "не отвергнута",
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Тяжесть хвостов                                                             #
# --------------------------------------------------------------------------- #
def hill_tail_index(x: np.ndarray, tail: str = "right", k_frac: float = 0.05) -> float:
    """Оценка Хилла индекса хвоста α (меньше α — тяжелее хвост).

    Для нормального распределения α велик (хвост лёгкий, экспоненциальный);
    для дневных доходностей акций обычно α ≈ 3–5 (тяжёлые степенные хвосты).
    k_frac — доля верхних порядковых статистик в оценке (классически 5–10%).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if tail == "left":
        x = -x
    x = x[x > 0]
    if len(x) < 30:
        return float("nan")
    xs = np.sort(x)[::-1]
    k = int(np.clip(int(k_frac * len(xs)), 10, len(xs) - 1))
    top = xs[: k + 1]
    logs = np.log(top[:k]) - np.log(top[k])
    mean_log = logs.mean()
    return float(1.0 / mean_log) if mean_log > 0 else float("nan")


def tail_table(returns: pd.DataFrame, k_frac: float = 0.05) -> pd.DataFrame:
    """Индексы хвоста Хилла слева и справа + ориентир для нормального хвоста."""
    rows = {}
    for col in returns.columns:
        x = returns[col].dropna().to_numpy()
        rows[col] = {
            "α_левый": hill_tail_index(x, "left", k_frac),
            "α_правый": hill_tail_index(x, "right", k_frac),
        }
    return pd.DataFrame(rows).T


def fit_student_t(x: np.ndarray) -> dict[str, float]:
    """Подгонка t-Стьюдента (df, loc, scale) по MLE. Малое df = тяжёлые хвосты."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    df, loc, scale = stats.t.fit(x)
    return {"df": float(df), "loc": float(loc), "scale": float(scale)}


# --------------------------------------------------------------------------- #
# Стационарность: ADF + KPSS                                                  #
# --------------------------------------------------------------------------- #
def stationarity_table(series: pd.DataFrame) -> pd.DataFrame:
    """ADF и KPSS по каждому ряду + согласованный вывод.

    ADF  H0: единичный корень (нестационарен) -> p<0.05 => стационарен.
    KPSS H0: стационарен                       -> p<0.05 => нестационарен.
    Согласие обоих тестов даёт уверенный вывод; расхождение — повод для
    обсуждения (например, near-unit-root после шока 2022 г.).
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    rows = {}
    for col in series.columns:
        x = series[col].dropna().to_numpy()
        if len(x) < 30:
            continue
        adf_stat, adf_p = adfuller(x, autolag="AIC")[:2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # KPSS пишет о p вне таблицы — это норм
            kpss_stat, kpss_p = kpss(x, regression="c", nlags="auto")[:2]
        adf_stat = "стационарен" if adf_p < 0.05 else "нет"
        kpss_concl = "стационарен" if kpss_p >= 0.05 else "нет"
        rows[col] = {
            "ADF_pvalue": round(float(adf_p), 4),
            "ADF_вывод": adf_stat,
            "KPSS_pvalue": round(float(kpss_p), 4),
            "KPSS_вывод": kpss_concl,
            "согласие": "да" if adf_stat == kpss_concl else "расхождение",
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Кластеризация волатильности (ARCH-эффект)                                   #
# --------------------------------------------------------------------------- #
def volatility_clustering_table(returns: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Тест Льюнга–Бокса по КВАДРАТАМ доходностей: H0 — нет автокорреляции.

    p<0.05 => квадраты автокоррелированы => кластеризация волатильности
    (тихие/бурные периоды группируются) => нужен GARCH/EWMA, а не i.i.d.-нормаль.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows = {}
    for col in returns.columns:
        x = returns[col].dropna()
        if len(x) < lags + 10:
            continue
        sq = (x - x.mean()) ** 2
        lb = acorr_ljungbox(sq, lags=[lags], return_df=True)
        stat = float(lb["lb_stat"].iloc[0])
        pval = float(lb["lb_pvalue"].iloc[0])
        rows[col] = {
            f"LB({lags})_стат": round(stat, 2),
            "LB_pvalue": round(pval, 4),
            "ARCH-эффект": "есть" if pval < 0.05 else "нет",
        }
    return pd.DataFrame(rows).T


def squared_return_acf(returns: pd.Series, nlags: int = 40) -> pd.Series:
    """ACF квадратов доходностей (для графика кластеризации волатильности)."""
    from statsmodels.tsa.stattools import acf

    x = returns.dropna()
    sq = ((x - x.mean()) ** 2).to_numpy()
    vals = acf(sq, nlags=nlags, fft=True)
    return pd.Series(vals, index=range(len(vals)), name="ACF(r²)")


# --------------------------------------------------------------------------- #
# Корреляции и сезонность                                                     #
# --------------------------------------------------------------------------- #
def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Матрица корреляций факторов (по умолчанию Пирсон)."""
    return returns.corr(method=method)


def weekday_seasonality(returns: pd.Series) -> pd.DataFrame:
    """Средняя доходность и σ по дням недели — грубая проверка сезонности."""
    x = returns.dropna()
    names = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт"}
    g = x.groupby(x.index.dayofweek)
    out = pd.DataFrame({"среднее": g.mean(), "σ": g.std(ddof=1), "n": g.size()})
    out.index = [names.get(i, str(i)) for i in out.index]
    return out
