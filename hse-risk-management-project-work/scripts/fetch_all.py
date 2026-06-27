"""
Прогон всего слоя данных: загрузка + сборка панели + сохранение снапшота.

Запусти ЛОКАЛЬНО (нужен интернет):
    python -m scripts.fetch_all

После выполнения в data_cache/ лягут parquet-кэши всех загрузок,
а в outputs/ — сводка по покрытию (что загрузилось, сколько дней, NaN).
Повторный запуск будет читать из кэша (мгновенно). Для обновления
истории до 01.01.2026 запусти с флагом --reload.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from rm.config import OUTPUT_DIR
from rm.data.dataset import build_market_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def coverage_report(md) -> pd.DataFrame:
    rows = []

    def add(name, df):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        rows.append({
            "блок": name,
            "колонок": df.shape[1],
            "дней": df.shape[0],
            "первая_дата": str(df.index.min().date()) if len(df) else "—",
            "последняя_дата": str(df.index.max().date()) if len(df) else "—",
            "доля_NaN": round(float(df.isna().mean().mean()), 4) if df.size else 1.0,
        })

    add("акции (CLOSE)", md.stock_prices)
    add("облигации (clean %)", md.bond_clean)
    add("облигации (НКД)", md.bond_accint)
    add("облигации (YTM)", md.bond_yield)
    add("КБД", md.gcurve)
    add("курсы USD/EUR", md.fx)
    add("индексы", md.indices)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reload", action="store_true", help="принудительно перезагрузить")
    args = ap.parse_args()

    md = build_market_data(force_reload=args.reload)
    rep = coverage_report(md)
    print("\n=== Покрытие данных ===\n")
    print(rep.to_string(index=False))

    out = OUTPUT_DIR / "data_coverage.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nСводка сохранена: {out}")
    print(f"Всего торговых дней в панели: {len(md.calendar)}")
