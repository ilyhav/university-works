"""
Слой риск-факторов: доходности/приращения, PCA, дескриптивный анализ.

Этап 2 проекта:
  * returns     — лог-доходности акций/валют/индексов, приращения кривой КБД,
                  склейка истории сменивших листинг тикеров, подготовка под PCA;
  * pca         — компоненты кривой (уровень/наклон/кривизна), рыночный фактор
                  акций, факторы валют; восстановление рядов для риск-движка;
  * descriptive — моменты, тяжёлые хвосты (Хилл, t-Стьюдент), стационарность
                  (ADF+KPSS), кластеризация волатильности (Льюнг–Бокс), корреляции.
"""
from rm.factors.returns import (
    CoverageReport,
    build_equity_returns,
    clean_for_pca,
    coverage,
    curve_increments,
    log_returns,
    simple_returns,
    splice_returns,
)
from rm.factors.pca import (
    PCAResult,
    fit_pca,
    interpret_curve_components,
)
from rm.factors.descriptive import (
    correlation_matrix,
    fit_student_t,
    hill_tail_index,
    moments_table,
    squared_return_acf,
    stationarity_table,
    tail_table,
    volatility_clustering_table,
    weekday_seasonality,
)

__all__ = [
    "CoverageReport",
    "build_equity_returns",
    "clean_for_pca",
    "coverage",
    "curve_increments",
    "log_returns",
    "simple_returns",
    "splice_returns",
    "PCAResult",
    "fit_pca",
    "interpret_curve_components",
    "correlation_matrix",
    "fit_student_t",
    "hill_tail_index",
    "moments_table",
    "squared_return_acf",
    "stationarity_table",
    "tail_table",
    "volatility_clustering_table",
    "weekday_seasonality",
]
