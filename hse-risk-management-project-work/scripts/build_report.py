"""
Финальная сборка markdown-отчета из результатов этапов 2-6.

Запуск:
    python -m scripts.build_report
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rm.config import OUTPUT_DIR, RISK_DATE, VAR_LEVEL, ES_LEVEL


REPORT = OUTPUT_DIR / "FINAL_REPORT.md"


def money(x: float) -> str:
    return f"{x:,.0f} руб.".replace(",", " ")


def pct(x: float) -> str:
    return f"{x:.2f}%"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"_Нет файла {path}_\n"


def main() -> None:
    pricing = pd.read_csv(OUTPUT_DIR / "pricing" / "bond_pricing.csv", index_col=0)
    risk = pd.read_csv(OUTPUT_DIR / "risk" / "risk_summary.csv", index_col=0)
    bt = pd.read_csv(OUTPUT_DIR / "backtest" / "backtest_summary.csv", index_col=0).iloc[0]

    var_col = f"VaR_{VAR_LEVEL:.3f}"
    es_col = f"ES_{ES_LEVEL:.3f}"
    one_day = risk[risk["горизонт_дн"] == 1].sort_values(var_col, ascending=False).iloc[0]
    ten_day = risk[risk["горизонт_дн"] == 10].sort_values(var_col, ascending=False).iloc[0]

    lines: list[str] = []
    add = lines.append
    add("# Проект по рыночному риску — финальный отчет\n")
    add(f"_Собрано scripts/build_report.py. Дата оценки риска: {RISK_DATE}._\n")

    add("## Executive summary\n")
    add(f"- Портфель для риск-движка: **{money(float(risk['стоимость_портфеля'].iloc[0]))}**.")
    add(f"- Pricing ОФЗ исправлен и проверен: средняя абсолютная ошибка на дату риска "
        f"**{pricing['ошибка_%ном'].abs().mean():.3f}% номинала**, средняя ошибка YTM "
        f"**{pricing['ошибка_YTM_бп'].abs().mean():.1f} б.п.**.")
    add(f"- Максимальный 1-дневный VaR {VAR_LEVEL:.0%}: **{money(float(one_day[var_col]))}** "
        f"({one_day['модель']}); соответствующий ES {ES_LEVEL:.1%}: "
        f"**{money(float(one_day[es_col]))}**.")
    add(f"- Максимальный 10-дневный VaR {VAR_LEVEL:.0%}: **{money(float(ten_day[var_col]))}** "
        f"({ten_day['модель']}); соответствующий ES {ES_LEVEL:.1%}: "
        f"**{money(float(ten_day[es_col]))}**.")
    add(f"- Бэктест 2025: **{int(bt['n_exceptions'])}** пробоя из "
        f"{int(bt['n_obs'])} наблюдений при ожидаемых {bt['expected_exceptions']:.1f}; "
        f"Kupiec p-value={bt['kupiec_pvalue']:.3f}, "
        f"Christoffersen p-value={bt['christoffersen_pvalue']:.3f}, "
        f"traffic light={bt['traffic_light']}.\n")

    add("## Файлы результатов\n")
    add("- `outputs/factors/` — риск-факторы, PCA, дескриптивная статистика.")
    add("- `outputs/models/` — MLE-модели, ковариации EWMA/GARCH, сравнение AIC/BIC.")
    add("- `outputs/pricing/` — справедливая стоимость ОФЗ, точность pricing, графики.")
    add("- `outputs/risk/` — VaR/ES, вклад классов активов в ES, распределения P&L.")
    add("- `outputs/backtest/` — ряд бэктеста, тесты Kupiec/Christoffersen, график пробоев.\n")

    add("---\n")
    add(read_text(OUTPUT_DIR / "factors" / "SUMMARY.md"))
    add("\n---\n")
    add(read_text(OUTPUT_DIR / "models" / "SUMMARY_models.md"))
    add("\n---\n")
    add(read_text(OUTPUT_DIR / "pricing" / "SUMMARY_pricing.md"))
    add("\n---\n")
    add(read_text(OUTPUT_DIR / "risk" / "SUMMARY_risk.md"))
    add("\n---\n")
    add(read_text(OUTPUT_DIR / "backtest" / "SUMMARY_backtest.md"))

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Финальный отчет сохранен: {REPORT}")


if __name__ == "__main__":
    main()
