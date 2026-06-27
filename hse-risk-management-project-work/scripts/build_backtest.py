"""
Этап 6: бэктест VaR 99% за 2025 год.

Модель для бэктеста: rolling historical simulation, окно 500 торговых дней.
На каждый день 2025 портфель переоценивается на предыдущую дату, VaR считается
по прошлым однодневным сценариям факторов, затем сравнивается с фактическим
P&L следующего дня. Тесты делаются для total, bonds, equities, fx.

Запуск:
    python -m scripts.build_backtest
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rm.backtest import summarize_backtest
from rm.config import (
    BACKTEST_END,
    BACKTEST_START,
    OUTPUT_DIR,
    RANDOM_SEED,
    VAR_LEVEL,
)
from rm.data.dataset import build_market_data
from rm.factors import clean_for_pca, curve_increments, fit_pca
from rm.risk import PortfolioState, risk_measures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_backtest")

FACTORS_CSV = OUTPUT_DIR / "factors" / "factor_scores.csv"
BACKTEST_DIR = OUTPUT_DIR / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 500
MIN_WINDOW = 250


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = BACKTEST_DIR / name
    df.to_csv(path, encoding="utf-8-sig")
    logger.info("сохранено: %s", path.relative_to(OUTPUT_DIR.parent))


def save_fig(fig, name: str) -> None:
    path = BACKTEST_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("график:   %s", path.relative_to(OUTPUT_DIR.parent))


def build_rate_pca(md):
    dy = curve_increments(md.gcurve)
    clean, _ = clean_for_pca(dy, min_coverage=0.9)
    return fit_pca(clean, n_components=3, standardize=False)


def main() -> None:
    if not FACTORS_CSV.exists():
        raise SystemExit(f"нет {FACTORS_CSV} — сначала запусти scripts.build_factors")

    logger.info("=== Загрузка данных ===")
    panel = pd.read_csv(FACTORS_CSV, index_col=0, parse_dates=True)
    md = build_market_data()
    rate_pca = build_rate_pca(md)

    bt_dates = panel.loc[pd.Timestamp(BACKTEST_START):pd.Timestamp(BACKTEST_END)].index
    rows: list[dict] = []

    logger.info("=== Rolling historical VaR, окно %d дней ===", WINDOW)
    for date in bt_dates:
        pos = panel.index.get_loc(date)
        if not isinstance(pos, int) or pos == 0:
            continue
        start = max(0, pos - WINDOW)
        window = panel.iloc[start:pos]
        if len(window) < MIN_WINDOW:
            continue

        prev_date = panel.index[pos - 1]
        state = PortfolioState.from_market_data(md, prev_date, rate_pca, compounding="annual")

        scen_components = state.pnl_components(window, horizon_days=1)
        realized_components = state.pnl_components(panel.iloc[pos], horizon_days=1).iloc[0]

        row = {
            "date": date,
            "prev_date": prev_date,
            "window": len(window),
            "portfolio_value": state.base_value,
        }
        for book in ("total", "bonds", "equities", "fx"):
            rm = risk_measures(scen_components[book], var_level=VAR_LEVEL)
            realized = float(realized_components[book])
            var = rm[f"VaR_{VAR_LEVEL:.3f}"]
            row[f"realized_pnl_{book}"] = realized
            row[f"VaR_{VAR_LEVEL:.3f}_{book}"] = var
            row[f"exception_{book}"] = realized < -var
        rows.append(row)

    series = pd.DataFrame(rows).set_index("date")
    if series.empty:
        raise SystemExit("не получилось построить бэктест: нет дат/окна")
    save_csv(series, "backtest_series.csv")

    summary_rows = {}
    for book in ("total", "bonds", "equities", "fx"):
        result = summarize_backtest(
            series[f"exception_{book}"], VAR_LEVEL,
            var_series=series[f"VaR_{VAR_LEVEL:.3f}_{book}"],
        )
        summary_rows[book] = result.__dict__
    summary = pd.DataFrame(summary_rows).T
    save_csv(summary, "backtest_summary.csv")

    logger.info("=== Графики ===")
    try:
        plot_backtest(series)
    except Exception as exc:
        logger.warning("график не построился: %s", exc)

    write_summary(series, summary)

    print("\n=== ИТОГ ЭТАПА 6 ===")
    result = summary.loc["total"]
    print(f"Наблюдений: {int(result.n_obs)}, пробоев VaR total: {int(result.n_exceptions)} "
          f"(ожидалось {result.expected_exceptions:.1f})")
    print(f"Kupiec total p-value={result.kupiec_pvalue:.3f}, "
          f"Christoffersen total p-value={result.christoffersen_pvalue:.3f}, "
          f"traffic light={result.traffic_light}")
    print(f"Все результаты: {BACKTEST_DIR}")


def plot_backtest(series: pd.DataFrame) -> None:
    var_col = f"VaR_{VAR_LEVEL:.3f}_total"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(series.index, series["realized_pnl_total"] / 1e6, color="#2c3e50", lw=1.0, label="P&L")
    ax.plot(series.index, -series[var_col] / 1e6, color="#c0392b", lw=1.2, label=f"-VaR {VAR_LEVEL:.0%}")
    exc = series[series["exception_total"]]
    if not exc.empty:
        ax.scatter(exc.index, exc["realized_pnl_total"] / 1e6, color="#e74c3c", zorder=5, label="пробой")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Бэктест VaR: фактический P&L против порога")
    ax.set_ylabel("млн руб.")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "var_backtest.png")


def write_summary(series: pd.DataFrame, summary: pd.DataFrame) -> None:
    r = summary.loc["total"]
    lines: list[str] = []
    add = lines.append
    add("# Этап 6 — бэктест VaR\n")
    add(f"_scripts/build_backtest.py, seed={RANDOM_SEED}, rolling historical simulation, окно {WINDOW} дней._\n")

    add("## 1. Результаты\n")
    add(f"- весь портфель: наблюдений в 2025: **{int(r['n_obs'])}**; пробоев VaR 99%: "
        f"**{int(r['n_exceptions'])}** при ожидаемых {r['expected_exceptions']:.1f}.")
    add(f"- частота пробоев: {r['exception_rate']:.2%}; Basel traffic light: "
        f"**{r['traffic_light']}**.")
    add(f"- Kupiec POF (UC): LR={r['kupiec_lr']:.2f}, p-value={r['kupiec_pvalue']:.3f}.")
    add(f"- Christoffersen IND: LR={r['christoffersen_lr']:.2f}, "
        f"p-value={r['christoffersen_pvalue']:.3f}.")
    add(f"- Christoffersen CC (UC+IND): LR={r['cc_lr']:.2f}, p-value={r['cc_pvalue']:.3f}.")
    add(f"- Dynamic Quantile (Engle–Manganelli): stat={r['dq_stat']:.2f}, "
        f"p-value={r['dq_pvalue']:.3f}.\n")

    add("**Подпортфели (UC / IND / CC / DQ):**")
    names = {"bonds": "облигации", "equities": "акции", "fx": "валюта"}
    for book, label in names.items():
        rb = summary.loc[book]
        add(f"- {label}: пробоев {int(rb['n_exceptions'])}/{int(rb['n_obs'])}; "
            f"Kupiec={rb['kupiec_pvalue']:.3f}, IND={rb['christoffersen_pvalue']:.3f}, "
            f"CC={rb['cc_pvalue']:.3f}, DQ={rb['dq_pvalue']:.3f}; "
            f"traffic light={rb['traffic_light']}.")
    add("")

    add("## 2. Интерпретация\n")
    if r["kupiec_pvalue"] >= 0.05:
        add("- Kupiec (UC) не отвергает корректную частоту пробоев на 5% уровне.")
    else:
        add("- Kupiec (UC) отвергает корректную частоту пробоев: модель требует калибровки.")
    if r["christoffersen_pvalue"] >= 0.05:
        add("- IND не показывает значимой кластеризации пробоев.")
    else:
        add("- IND: пробои кластеризуются — VaR медленно реагирует на режимы волатильности.")
    if r["cc_pvalue"] >= 0.05:
        add("- CC (совместный) не отвергает модель.")
    else:
        add("- CC (совместный) отвергает модель — провал по частоте и/или независимости.")
    if np.isfinite(r["dq_pvalue"]):
        add("- DQ " + ("не выявляет" if r["dq_pvalue"] >= 0.05 else "выявляет")
            + " значимой предсказуемости пробоев по прошлым пробоям и уровню VaR.")
    add("- при 99% VaR и одном календарном году мощность тестов низкая: ожидается всего "
        "около 2-3 пробоев, поэтому ансамбль тестов дополняется анализом ES/хвостов (этап 5).\n")

    worst = series.nsmallest(5, "realized_pnl_total")
    add("## 3. Худшие дни P&L\n")
    for date, row in worst.iterrows():
        add(f"- {date.date()}: P&L {row['realized_pnl_total']:,.0f} руб., "
            f"VaR {row[f'VaR_{VAR_LEVEL:.3f}_total']:,.0f} руб., "
            f"пробой={'да' if row['exception_total'] else 'нет'}.")
    add("")

    path = BACKTEST_DIR / "SUMMARY_backtest.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("сводка:   %s", path.relative_to(OUTPUT_DIR.parent))


if __name__ == "__main__":
    main()
