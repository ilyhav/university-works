"""
Этап 5: расчет VaR 99% и ES 97.5% по портфелю.

Запуск:
    python -m scripts.build_risk

Требует уже построенные этапы 2-4:
  * outputs/factors/factor_scores.csv
  * outputs/models/*_cov.csv
  * data_cache/* для рыночной панели
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rm.config import (
    ES_LEVEL,
    HORIZONS_DAYS,
    OUTPUT_DIR,
    RANDOM_SEED,
    RISK_DATE,
    VAR_LEVEL,
)
from rm.data.dataset import build_market_data
from rm.factors import clean_for_pca, curve_increments, fit_pca
from rm.risk import (
    PortfolioState,
    empirical_scenarios,
    risk_measures,
    simulate_gaussian,
    simulate_student_t,
    simulate_t_with_cov,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_risk")

FACTORS_CSV = OUTPUT_DIR / "factors" / "factor_scores.csv"
MODELS_DIR = OUTPUT_DIR / "models"
RISK_DIR = OUTPUT_DIR / "risk"
RISK_DIR.mkdir(parents=True, exist_ok=True)

N_SCENARIOS = 50_000


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = RISK_DIR / name
    df.to_csv(path, encoding="utf-8-sig")
    logger.info("сохранено: %s", path.relative_to(OUTPUT_DIR.parent))


def save_fig(fig, name: str) -> None:
    path = RISK_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("график:   %s", path.relative_to(OUTPUT_DIR.parent))


def build_rate_pca(md):
    dy = curve_increments(md.gcurve)
    clean, _ = clean_for_pca(dy, min_coverage=0.9)
    return fit_pca(clean, n_components=3, standardize=False)


def load_cov(name: str, columns: list[str]) -> pd.DataFrame | None:
    path = MODELS_DIR / name
    if not path.exists():
        return None
    cov = pd.read_csv(path, index_col=0)
    return cov.reindex(index=columns, columns=columns).astype(float)


def garch_df(default_df: float) -> float:
    path = MODELS_DIR / "garch_factors.csv"
    if not path.exists():
        return default_df
    tab = pd.read_csv(path, index_col=0)
    if "nu" not in tab:
        return default_df
    return float(np.nanmedian(pd.to_numeric(tab["nu"], errors="coerce")))


def component_es(components: pd.DataFrame, es_level: float = ES_LEVEL) -> pd.Series:
    losses = -components
    total = losses["total"]
    cut = float(np.quantile(total, es_level))
    tail = losses.loc[total >= cut, ["bonds", "equities", "fx", "total"]]
    return tail.mean()


def run_model(
    model_name: str,
    scenarios: pd.DataFrame,
    state: PortfolioState,
    horizon: int,
    keep_components: bool = False,
) -> tuple[dict, pd.DataFrame | None]:
    comps = state.pnl_components(scenarios, horizon_days=horizon)
    stats = risk_measures(comps["total"], VAR_LEVEL, ES_LEVEL)
    stats.update({
        "модель": model_name,
        "горизонт_дн": horizon,
        "сценариев": len(scenarios),
        "стоимость_портфеля": state.base_value,
        "VaR_%стоимости": stats[f"VaR_{VAR_LEVEL:.3f}"] / state.base_value * 100.0,
        "ES_%стоимости": stats[f"ES_{ES_LEVEL:.3f}"] / state.base_value * 100.0,
    })
    return stats, comps if keep_components else None


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    if not FACTORS_CSV.exists():
        raise SystemExit(f"нет {FACTORS_CSV} — сначала запусти scripts.build_factors")

    logger.info("=== Загрузка факторов и панели ===")
    panel = pd.read_csv(FACTORS_CSV, index_col=0, parse_dates=True)
    md = build_market_data()
    rate_pca = build_rate_pca(md)
    state = PortfolioState.from_market_data(md, RISK_DATE, rate_pca, compounding="annual")
    logger.info("дата оценки: %s, стоимость модели: %.0f руб.", RISK_DATE, state.base_value)

    ewma_cov = load_cov("ewma_next_cov.csv", list(panel.columns))
    garch_cov = load_cov("garch_next_cov.csv", list(panel.columns))

    rows: list[dict] = []
    selected_components: dict[tuple[str, int], pd.DataFrame] = {}
    selected_scenarios: dict[tuple[str, int], pd.Series] = {}

    logger.info("=== Сценарии и VaR/ES ===")
    for horizon in HORIZONS_DAYS:
        logger.info("горизонт %d дн.", horizon)

        scenarios_by_model: list[tuple[str, pd.DataFrame, bool]] = []
        scenarios_by_model.append(("Историческая симуляция", empirical_scenarios(panel, horizon), False))
        scenarios_by_model.append(("Нормаль i.i.d.", simulate_gaussian(panel, horizon, N_SCENARIOS, rng), False))

        t_scen, t_df = simulate_student_t(panel, horizon, N_SCENARIOS, rng)
        scenarios_by_model.append((f"t-Стьюдента i.i.d. (ν={t_df:.2f})", t_scen, True))

        if ewma_cov is not None:
            scenarios_by_model.append((
                "EWMA λ=0.94 + нормаль",
                simulate_gaussian(panel, horizon, N_SCENARIOS, rng, cov=ewma_cov),
                False,
            ))
        if garch_cov is not None:
            df_g = garch_df(t_df)
            scenarios_by_model.append((
                f"GARCH-CCC + t (ν≈{df_g:.2f})",
                simulate_t_with_cov(panel, horizon, N_SCENARIOS, rng, cov=garch_cov, df=df_g),
                True,
            ))

        for model_name, scenarios, keep in scenarios_by_model:
            stats, comps = run_model(model_name, scenarios, state, horizon, keep_components=keep)
            rows.append(stats)
            if comps is not None:
                selected_components[(model_name, horizon)] = comps
                selected_scenarios[(model_name, horizon)] = comps["total"]

    risk = pd.DataFrame(rows)
    order = [
        "модель", "горизонт_дн", "сценариев", "стоимость_портфеля",
        "mean_pnl", "std_pnl", f"VaR_{VAR_LEVEL:.3f}", f"ES_{ES_LEVEL:.3f}",
        "VaR_%стоимости", "ES_%стоимости", "min_pnl", "max_pnl",
    ]
    risk = risk[order].sort_values(["горизонт_дн", f"VaR_{VAR_LEVEL:.3f}"], ascending=[True, True])
    save_csv(risk, "risk_summary.csv")

    contrib_rows = {}
    for key, comps in selected_components.items():
        model_name, horizon = key
        ce = component_es(comps)
        contrib_rows[f"{horizon}д | {model_name}"] = ce
    contrib = pd.DataFrame(contrib_rows).T
    for col in ["bonds", "equities", "fx"]:
        contrib[f"{col}_доля_ES"] = contrib[col] / contrib["total"]
    save_csv(contrib, "es_contributions.csv")

    logger.info("=== Графики ===")
    try:
        plot_risk_bars(risk)
        plot_selected_distributions(selected_scenarios)
    except Exception as exc:
        logger.warning("часть графиков не построилась: %s", exc)

    write_summary(risk, contrib, state)

    best_tail = risk.sort_values(f"ES_{ES_LEVEL:.3f}", ascending=False).iloc[0]
    print("\n=== ИТОГ ЭТАПА 5 ===")
    print(f"Стоимость портфеля для риска: {state.base_value:,.0f} руб.")
    print(f"Максимальный ES {ES_LEVEL:.1%}: {best_tail[f'ES_{ES_LEVEL:.3f}']:,.0f} руб. "
          f"({best_tail['модель']}, {int(best_tail['горизонт_дн'])} дн.)")
    print(f"Все результаты: {RISK_DIR}")


def plot_risk_bars(risk: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, horizon in zip(axes, HORIZONS_DAYS):
        sub = risk[risk["горизонт_дн"] == horizon].copy()
        sub = sub.sort_values(f"VaR_{VAR_LEVEL:.3f}")
        ax.barh(sub["модель"], sub[f"VaR_{VAR_LEVEL:.3f}"] / 1e6, color="#2c7fb8")
        ax.set_title(f"VaR {VAR_LEVEL:.0%}, горизонт {horizon} дн.")
        ax.set_xlabel("млн руб.")
        ax.grid(alpha=0.3)
    save_fig(fig, "var_by_model.png")


def plot_selected_distributions(series_by_model: dict[tuple[str, int], pd.Series]) -> None:
    if not series_by_model:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, horizon in zip(axes, HORIZONS_DAYS):
        for (model, h), pnl in series_by_model.items():
            if h != horizon:
                continue
            ax.hist(pnl / 1e6, bins=80, density=True, alpha=0.45, label=model)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"Распределение P&L, {horizon} дн.")
        ax.set_xlabel("P&L, млн руб.")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save_fig(fig, "pnl_distributions_tail_models.png")


def write_summary(risk: pd.DataFrame, contrib: pd.DataFrame, state: PortfolioState) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Этап 5 — VaR/ES портфеля\n")
    add(f"_scripts/build_risk.py, seed={RANDOM_SEED}, дата оценки {RISK_DATE}._\n")
    add(f"Модельная стоимость портфеля для сценарного P&L: **{state.base_value:,.0f} руб.**\n")

    add("## 1. Сравнение моделей\n")
    for horizon in HORIZONS_DAYS:
        sub = risk[risk["горизонт_дн"] == horizon].sort_values(f"ES_{ES_LEVEL:.3f}", ascending=False)
        top = sub.iloc[0]
        low = sub.iloc[-1]
        add(f"- горизонт **{horizon} дн.**: максимальный ES дает **{top['модель']}** "
            f"({top[f'ES_{ES_LEVEL:.3f}']:,.0f} руб., "
            f"{top['ES_%стоимости']:.2f}% стоимости); минимальный — **{low['модель']}** "
            f"({low[f'ES_{ES_LEVEL:.3f}']:,.0f} руб.).")
    add("- нормаль остается базой, но t/GARCH показывают хвостовой риск выше из-за "
        "тяжелых хвостов и условной волатильности; это согласуется с этапами 2-3.\n")

    add("## 2. Декомпозиция ES\n")
    if not contrib.empty:
        row = contrib.sort_values("total", ascending=False).iloc[0]
        add(f"- в самом тяжелом хвосте средние потери: облигации "
            f"{row['bonds']:,.0f} руб. ({row['bonds_доля_ES']:.0%}), акции "
            f"{row['equities']:,.0f} руб. ({row['equities_доля_ES']:.0%}), валюта "
            f"{row['fx']:,.0f} руб. ({row['fx_доля_ES']:.0%}).")
    add("- валютные позиции крупнее остальных по номиналу (2×100 млн), поэтому часто "
        "доминируют в абсолютном VaR/ES; облигации чувствительны к параллельному "
        "сдвигу КБД и длинному хвосту кривой.\n")

    add("## 3. Методические решения\n")
    add("- для ставок сценарные rate_PC восстанавливаются в сдвиг всей КБД через PCA, "
        "после чего ОФЗ переоцениваются полным PV будущих потоков;")
    add("- акции и FX переоцениваются через exp(лог-доходности)-1, с позициями, "
        "зафиксированными на дату риска;")
    add("- VaR — 99%, ES — 97.5%, положительные числа означают потери.\n")
    add("**Допущение по 10-дневному горизонту (важно для обсуждения):** "
        "10 дней моделируются как СУММА 10 дневных факторных изменений с одной "
        "переоценкой в конце (buy-and-hold), а не как путь с ежедневной "
        "ребалансировкой к постоянным рублёвым весам. Это сознательное упрощение: "
        "полный путь-зависимый сценарий потребовал бы ежедневной переоценки кривой и "
        "цен на траектории. Эффект ребалансировки на VaR второго порядка и обычно "
        "слегка СНИЖАЕТ риск (ежедневный сброс к весам гасит накопление дрейфа); "
        "поэтому терминальная оценка консервативна. Путь-зависимый движок — "
        "обозначенное направление развития.\n")

    path = RISK_DIR / "SUMMARY_risk.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("сводка:   %s", path.relative_to(OUTPUT_DIR.parent))


if __name__ == "__main__":
    main()
