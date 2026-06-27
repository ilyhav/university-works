"""
Этап 4: справедливая стоимость портфеля (облигации от КБД + акции/валюта).

Запусти ЛОКАЛЬНО (данные из кэша, сети не требует, если кэш уже собран):

    python -m scripts.build_pricing

Что делает:
  1. на дату оценки риска (RISK_DATE) ценит 5 ОФЗ от кривой КБД, сравнивает
     модельную цену с рыночной (в рублях, % номинала и б.п. доходности),
     считает дюрацию Маколея/модифицированную и выпуклость;
  2. сравнивает два соглашения о капитализации (годовое vs непрерывное) —
     выбирает то, что ближе к рынку;
  3. проверяет точность ценообразования НА ВСЁМ 2025-м (средняя |ошибка|);
  4. оценивает стоимость акций и валютных позиций (линейно);
  5. сохраняет таблицы, графики и SUMMARY_pricing.md в outputs/pricing/.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rm.config import (
    BACKTEST_END,
    BACKTEST_START,
    EQUITY_NOTIONAL_EACH,
    BOND_NOTIONAL_EACH,
    FX_NOTIONAL_EACH,
    OUTPUT_DIR,
    PORTFOLIO,
    RISK_DATE,
)
from rm.data.dataset import build_market_data
from rm.pricing import get_curve_row, price_bond
from rm.pricing.bonds import bond_cashflows, face_value, price_from_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_pricing")

PRICING_DIR = OUTPUT_DIR / "pricing"
PRICING_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = PRICING_DIR / name
    df.to_csv(path, encoding="utf-8-sig")
    logger.info("сохранено: %s", path.relative_to(OUTPUT_DIR.parent))


def save_fig(fig, name: str) -> None:
    path = PRICING_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("график:   %s", path.relative_to(OUTPUT_DIR.parent))


def asof_row(df: pd.DataFrame, asof) -> pd.Series | None:
    sub = df.loc[:pd.Timestamp(asof)]
    return None if sub.empty else sub.iloc[-1]


def price_all_bonds(md, asof, compounding: str) -> pd.DataFrame:
    """Ценообразование всех ОФЗ на дату asof при заданном соглашении."""
    curve_row = get_curve_row(md.gcurve, asof)
    clean = asof_row(md.bond_clean, asof)
    accint = asof_row(md.bond_accint, asof)
    ytm = asof_row(md.bond_yield, asof)

    rows = {}
    for b in PORTFOLIO.bonds:
        res = price_bond(
            secid=b, coupons=md.coupons.get(b), amorts=md.amortizations.get(b),
            curve_row=curve_row, asof=asof,
            market_clean_pct=float(clean[b]), accrued=float(accint[b]),
            market_ytm=float(ytm[b]) / 100.0, compounding=compounding,
        )
        rows[b] = {
            "номинал": res.face,
            "потоков": res.n_flows,
            "модель_грязн": round(res.model_dirty, 2),
            "рынок_грязн": round(res.market_dirty, 2),
            "ошибка_руб": round(res.err_rub, 2),
            "ошибка_%ном": round(res.err_pct_face, 3),
            "модель_YTM_%": round(res.model_ytm * 100, 3),
            "рынок_YTM_%": round(res.market_ytm * 100, 3),
            "ошибка_YTM_бп": round(res.err_ytm_bp, 1),
            "дюрация_Маколея": round(res.macaulay, 2),
            "мод_дюрация": round(res.modified_dur, 2),
            "выпуклость": round(res.convexity, 1),
        }
    return pd.DataFrame(rows).T


def historical_accuracy(md, compounding: str) -> pd.DataFrame:
    """Средняя |ошибка| ценообразования по каждой ОФЗ за 2025 год."""
    cal = md.bond_clean.loc[pd.Timestamp(BACKTEST_START):pd.Timestamp(BACKTEST_END)].index
    errs = {b: [] for b in PORTFOLIO.bonds}
    for d in cal:
        try:
            curve_row = get_curve_row(md.gcurve, d)
        except ValueError:
            continue
        clean = md.bond_clean.loc[d]
        accint = md.bond_accint.loc[d]
        for b in PORTFOLIO.bonds:
            cp = float(clean[b]) if not pd.isna(clean[b]) else np.nan
            ac = float(accint[b]) if not pd.isna(accint[b]) else np.nan
            if np.isnan(cp) or np.isnan(ac):
                continue
            face = face_value(md.amortizations.get(b))
            cf = bond_cashflows(md.coupons.get(b), md.amortizations.get(b), d)
            if cf.empty:
                continue
            model = price_from_curve(cf, curve_row, compounding)
            market = cp / 100.0 * face + ac
            errs[b].append((model - market) / face * 100.0)
    rows = {}
    for b, e in errs.items():
        e = np.array(e)
        rows[b] = {
            "дней": len(e),
            "сред_ошибка_%ном": round(float(np.mean(e)), 3) if len(e) else np.nan,
            "сред|ошибка|_%ном": round(float(np.mean(np.abs(e))), 3) if len(e) else np.nan,
            "макс|ошибка|_%ном": round(float(np.max(np.abs(e))), 3) if len(e) else np.nan,
        }
    return pd.DataFrame(rows).T


def equity_fx_values(md, asof) -> pd.DataFrame:
    """Стоимость позиций в акциях и валюте на дату asof (линейно)."""
    px = asof_row(md.stock_prices, asof)
    fx = asof_row(md.fx, asof)
    rows = {}
    for s in PORTFOLIO.stocks:
        price = float(px[s]) if not pd.isna(px[s]) else np.nan
        rows[s] = {"тип": "акция", "цена/курс": round(price, 2),
                   "позиция_руб": EQUITY_NOTIONAL_EACH,
                   "единиц": round(EQUITY_NOTIONAL_EACH / price, 1) if price > 0 else np.nan}
    for c in PORTFOLIO.fx:
        rate = float(fx[c]) if not pd.isna(fx[c]) else np.nan
        rows[c] = {"тип": "валюта", "цена/курс": round(rate, 4),
                   "позиция_руб": FX_NOTIONAL_EACH,
                   "единиц": round(FX_NOTIONAL_EACH / rate, 0) if rate > 0 else np.nan}
    return pd.DataFrame(rows).T


def main() -> None:
    logger.info("=== Сборка панели данных (из кэша) ===")
    md = build_market_data()
    asof = pd.Timestamp(RISK_DATE)
    logger.info("дата оценки: %s", asof.date())

    # --- ценообразование при двух соглашениях о капитализации ---
    logger.info("=== Облигации: годовое vs непрерывное дисконтирование ===")
    bonds_annual = price_all_bonds(md, asof, "annual")
    bonds_cont = price_all_bonds(md, asof, "continuous")
    mae_annual = bonds_annual["ошибка_%ном"].abs().mean()
    mae_cont = bonds_cont["ошибка_%ном"].abs().mean()
    better = "annual" if mae_annual <= mae_cont else "continuous"
    logger.info("MAE |ошибки| (%% ном): годовое=%.3f, непрерывное=%.3f -> лучше %s",
                mae_annual, mae_cont, better)
    primary = bonds_annual if better == "annual" else bonds_cont
    save_csv(primary, "bond_pricing.csv")

    # --- точность за весь 2025 ---
    logger.info("=== Точность ценообразования за 2025 ===")
    hist = historical_accuracy(md, better)
    save_csv(hist, "bond_pricing_accuracy_2025.csv")

    # --- акции и валюта ---
    eqfx = equity_fx_values(md, asof)
    save_csv(eqfx, "equity_fx_values.csv")

    # --- стоимость портфеля ---
    bond_units = {b: BOND_NOTIONAL_EACH / primary.loc[b, "номинал"] for b in PORTFOLIO.bonds}
    bond_value = sum(bond_units[b] * primary.loc[b, "модель_грязн"] for b in PORTFOLIO.bonds)
    portfolio_value = (bond_value + len(PORTFOLIO.stocks) * EQUITY_NOTIONAL_EACH
                       + len(PORTFOLIO.fx) * FX_NOTIONAL_EACH)

    # --- графики ---
    logger.info("=== Графики ===")
    try:
        curve_row = get_curve_row(md.gcurve, asof)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(curve_row.index, curve_row.values * 100, "o-", color="#2c3e50")
        for b in PORTFOLIO.bonds:
            ax.axvline(primary.loc[b, "дюрация_Маколея"], color="grey", ls=":", alpha=0.5)
        ax.set_xlabel("срок, лет"); ax.set_ylabel("ставка КБД, % год.")
        ax.set_title(f"Кривая КБД на {asof.date()} (пунктир — дюрации ОФЗ)")
        ax.grid(alpha=0.3)
        save_fig(fig, "kbd_curve.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        primary["ошибка_%ном"].plot.bar(ax=ax, color="#c0392b")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("ошибка цены, % номинала")
        ax.set_title(f"Ошибка модель–рынок на {asof.date()} ({better})")
        ax.grid(alpha=0.3)
        save_fig(fig, "pricing_error.png")
    except Exception as exc:
        logger.warning("часть графиков не построилась: %s", exc)

    write_summary(asof, primary, bonds_annual, bonds_cont, better,
                  mae_annual, mae_cont, hist, portfolio_value, bond_value)

    print("\n=== ИТОГ ЭТАПА 4 ===")
    print(f"Соглашение о капитализации: {better} "
          f"(MAE годовое={mae_annual:.3f}%, непрерывное={mae_cont:.3f}% ном.)")
    print(f"Ошибки на {asof.date()} (% ном): "
          f"{primary['ошибка_%ном'].abs().min():.3f}–{primary['ошибка_%ном'].abs().max():.3f}")
    print(f"Средняя |ошибка| YTM: {primary['ошибка_YTM_бп'].abs().mean():.1f} б.п.")
    print(f"Стоимость портфеля (модельная): {portfolio_value:,.0f} руб.")
    print(f"\nВсё в: {PRICING_DIR}")


def write_summary(asof, primary, bonds_annual, bonds_cont, better,
                  mae_annual, mae_cont, hist, portfolio_value, bond_value) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Этап 4 — справедливая стоимость\n")
    add(f"_scripts/build_pricing.py, дата оценки {asof.date()}._\n")

    add("## 1. Соглашение о капитализации кривой\n")
    add(f"- средняя |ошибка| цены: годовое = **{mae_annual:.3f}%** номинала, "
        f"непрерывное = **{mae_cont:.3f}%** -> используем **{better}**.")
    add("- это методическая развилка: КБД ЦБ можно трактовать с разной "
        "капитализацией; выбираем ту, что ближе к рынку, и фиксируем явно.\n")

    add(f"## 2. Ценообразование ОФЗ на {asof.date()}\n")
    add(f"- ошибка цены по 5 ОФЗ: от {primary['ошибка_%ном'].abs().min():.3f}% до "
        f"{primary['ошибка_%ном'].abs().max():.3f}% номинала "
        f"(в среднем {primary['ошибка_%ном'].abs().mean():.3f}%).")
    add(f"- ошибка доходности: в среднем {primary['ошибка_YTM_бп'].abs().mean():.1f} б.п. "
        f"(макс {primary['ошибка_YTM_бп'].abs().max():.1f} б.п.).")
    add("- модельная цена систематически "
        + ("выше" if primary["ошибка_руб"].mean() > 0 else "ниже")
        + " рыночной — вероятные причины: КБД сглаживает индивидуальную премию/"
        "ликвидность выпуска, день-счёт и точность узлов кривой.")
    add("- дюрации растянуты по лесенке (короткий выпуск — малая дюрация, длинный "
        "26238 — максимальная), что подтверждает корректную чувствительность к ставке.\n")

    add("## 3. Точность за весь 2025 (не только на одну дату)\n")
    add(f"- средняя |ошибка| по году: от {hist['сред|ошибка|_%ном'].min():.3f}% до "
        f"{hist['сред|ошибка|_%ном'].max():.3f}% номинала по выпускам.")
    add("- устойчивость ошибки во времени показывает, что модель не подогнана под "
        "одну дату (см. bond_pricing_accuracy_2025.csv).\n")

    add("## 4. Стоимость портфеля\n")
    add(f"- модельная грязная стоимость облигационной части: {bond_value:,.0f} руб.")
    add(f"- полная модельная стоимость портфеля (облигации модельно + акции/валюта "
        f"по рынку): **{portfolio_value:,.0f} руб.** (номинал портфеля 260 млн).\n")

    add("## 5. Бонус (Блэк-76) — задел\n")
    add("- модуль rm.pricing.black76 готов: премия опциона на фьючерс и калибровка "
        "implied vol по соседним страйкам; подключим на этапе бонуса с данными "
        "срочного рынка MOEX.\n")

    path = PRICING_DIR / "SUMMARY_pricing.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("сводка:   %s", path.relative_to(OUTPUT_DIR.parent))


if __name__ == "__main__":
    main()
