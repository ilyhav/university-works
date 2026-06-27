"""
Презентационные ассеты: дополнительные графики-схемы + единый HTML-дашборд.

Запуск (после остальных build_*; данные берёт из outputs/):
    python -m scripts.build_slides

Создаёт в outputs/slides/:
  * portfolio_donut.png   — структура портфеля (почему валюта доминирует);
  * es_contribution.png   — вклад классов активов в хвостовой ES;
  * kpi_dashboard.png     — карточки ключевых чисел (готовый слайд);
  * pipeline.png          — схема пайплайна данные → … → бэктест;
  * risk_engine_flow.png  — как сценарий факторов превращается в P&L → VaR/ES;
  * dashboard.html        — самодостаточная HTML-страница (все графики + KPI
                            встроены в base64): открой в браузере, скриншоть в
                            слайды или показывай вживую.

Скрипт устойчив к отсутствию части файлов: чего нет — пропускает, числа берёт
из outputs/* с разумными запасными значениями.
"""
from __future__ import annotations

import base64
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from rm.config import (
    BOND_NOTIONAL_EACH,
    EQUITY_NOTIONAL_EACH,
    ES_LEVEL,
    FX_NOTIONAL_EACH,
    OUTPUT_DIR,
    PORTFOLIO,
    VAR_LEVEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_slides")

SLIDES = OUTPUT_DIR / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)

# единая палитра
NAVY = "#1f3a5f"
BLUE = "#2c7fb8"
TEAL = "#2a9d8f"
RED = "#c0392b"
AMBER = "#e08e0b"
GREY = "#6b7280"
BG = "#f5f7fa"


def load_csv(path, **kw):
    try:
        return pd.read_csv(path, **kw) if path.exists() else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("не прочитал %s: %s", path.name, exc)
        return None


def save(fig, name: str) -> None:
    p = SLIDES / name
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("график:   %s", p.relative_to(OUTPUT_DIR.parent))


# --------------------------------------------------------------------------- #
# Сбор ключевых чисел из outputs/                                             #
# --------------------------------------------------------------------------- #
def collect_kpis() -> dict:
    k = {  # запасные значения (если CSV нет)
        "value": 248_543_753.0, "var1": 7.38e6, "es1": 7.47e6,
        "var10": 25.43e6, "es10": 24.12e6, "pricing_mae": 0.573,
        "ytm_bp": 22.4, "bt_exc": 3, "bt_obs": 250, "bt_light": "green",
        "bt_kupiec": 0.758, "nu": 4.02,
    }
    risk = load_csv(OUTPUT_DIR / "risk" / "risk_summary.csv")
    if risk is not None:
        vc, ec = f"VaR_{VAR_LEVEL:.3f}", f"ES_{ES_LEVEL:.3f}"
        r1 = risk[risk["горизонт_дн"] == 1]
        r10 = risk[risk["горизонт_дн"] == 10]
        if not r1.empty:
            k["var1"], k["es1"] = float(r1[vc].max()), float(r1[ec].max())
            k["value"] = float(r1["стоимость_портфеля"].iloc[0])
        if not r10.empty:
            k["var10"], k["es10"] = float(r10[vc].max()), float(r10[ec].max())
    bp = load_csv(OUTPUT_DIR / "pricing" / "bond_pricing.csv", index_col=0)
    if bp is not None and "ошибка_%ном" in bp:
        k["pricing_mae"] = float(bp["ошибка_%ном"].abs().mean())
        if "ошибка_YTM_бп" in bp:
            k["ytm_bp"] = float(bp["ошибка_YTM_бп"].abs().mean())
    bt = load_csv(OUTPUT_DIR / "backtest" / "backtest_summary.csv", index_col=0)
    if bt is not None and "total" in bt.index:
        t = bt.loc["total"]
        k["bt_exc"], k["bt_obs"] = int(t["n_exceptions"]), int(t["n_obs"])
        k["bt_light"] = str(t["traffic_light"])
        k["bt_kupiec"] = float(t["kupiec_pvalue"])
    return k


# --------------------------------------------------------------------------- #
# Графики                                                                     #
# --------------------------------------------------------------------------- #
def plot_donut() -> None:
    parts = {
        "Облигации": len(PORTFOLIO.bonds) * BOND_NOTIONAL_EACH,
        "Акции": len(PORTFOLIO.stocks) * EQUITY_NOTIONAL_EACH,
        "Валюта USD/EUR": len(PORTFOLIO.fx) * FX_NOTIONAL_EACH,
    }
    vals = np.array(list(parts.values())) / 1e6
    total = vals.sum()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    wedges, _ = ax.pie(vals, startangle=90, counterclock=False,
                       colors=[BLUE, TEAL, RED],
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.text(0, 0.12, f"{total:.0f}", ha="center", va="center",
            fontsize=30, fontweight="bold", color=NAVY)
    ax.text(0, -0.18, "млн руб.", ha="center", va="center", fontsize=13, color=GREY)
    for w, (name, v) in zip(wedges, zip(parts, vals)):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        x, y = 1.15 * np.cos(ang), 1.15 * np.sin(ang)
        ax.text(x, y, f"{name}\n{v:.0f} млн ({v/total*100:.0f}%)",
                ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Структура портфеля: 200 из 260 млн — валюта",
                 fontsize=14, fontweight="bold", color=NAVY, pad=14)
    save(fig, "portfolio_donut.png")


def plot_es_contribution() -> None:
    df = load_csv(OUTPUT_DIR / "risk" / "es_contributions.csv", index_col=0)
    shares = {"bonds": 0.01, "equities": 0.0, "fx": 0.99}
    if df is not None and "total" in df:
        worst = df.sort_values("total", ascending=False).iloc[0]
        for b in ("bonds", "equities", "fx"):
            col = f"{b}_доля_ES"
            if col in df.columns:
                shares[b] = float(worst[col])
    labels = ["Облигации", "Акции", "Валюта"]
    vals = np.array([shares["bonds"], shares["equities"], shares["fx"]]) * 100
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    left = 0.0
    for v, lab, c in zip(vals, labels, [BLUE, TEAL, RED]):
        ax.barh(0, v, left=left, color=c, edgecolor="white")
        if v > 4:
            ax.text(left + v / 2, 0, f"{lab}\n{v:.0f}%", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
        left += v
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("вклад в хвостовой ES, %")
    ax.set_title("Откуда берётся хвостовой риск: почти весь ES — валюта",
                 fontsize=13, fontweight="bold", color=NAVY)
    save(fig, "es_contribution.png")


def plot_kpi_dashboard(k: dict) -> None:
    cards = [
        ("Стоимость портфеля", f"{k['value']/1e6:.1f} млн ₽", "номинал 260 млн", BLUE),
        (f"VaR 99% · 1 день", f"{k['var1']/1e6:.2f} млн ₽", f"ES 97.5% {k['es1']/1e6:.2f} млн", RED),
        (f"VaR 99% · 10 дней", f"{k['var10']/1e6:.2f} млн ₽", f"ES 97.5% {k['es10']/1e6:.1f} млн", RED),
        ("Ошибка цены ОФЗ", f"{k['pricing_mae']:.3f}% ном.", f"{k['ytm_bp']:.1f} б.п. по YTM", TEAL),
        ("Бэктест 2025", f"{k['bt_exc']} / {k['bt_obs']}", f"Kupiec p={k['bt_kupiec']:.3f} · {k['bt_light']}", AMBER),
        ("t-Стьюдента ν", f"≈ {k['nu']:.1f}", "тяжёлые хвосты, t > нормали", NAVY),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 3); ax.set_ylim(0, 2); ax.axis("off")
    for i, (title, big, sub, color) in enumerate(cards):
        cx, cy = i % 3, 1 - i // 3
        box = FancyBboxPatch((cx + 0.06, cy + 0.08), 0.88, 0.84,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=0, facecolor="white",
                             edgecolor="none", mutation_aspect=1)
        box.set_zorder(1)
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch((cx + 0.06, cy + 0.08), 0.05, 0.84,
                     boxstyle="round,pad=0,rounding_size=0.02",
                     facecolor=color, edgecolor="none", zorder=2))
        ax.text(cx + 0.16, cy + 0.74, title, fontsize=11.5, color=GREY, va="center")
        ax.text(cx + 0.16, cy + 0.46, big, fontsize=23, color=NAVY,
                fontweight="bold", va="center")
        ax.text(cx + 0.16, cy + 0.22, sub, fontsize=10.5, color=color, va="center")
    fig.patch.set_facecolor(BG)
    ax.set_title("Ключевые результаты", fontsize=16, fontweight="bold",
                 color=NAVY, pad=10)
    save(fig, "kpi_dashboard.png")


def _flow(ax, boxes, color):
    n = len(boxes)
    w, gap = 1.0 / n * 0.82, 1.0 / n * 0.18
    y = 0.5
    centers = []
    for i, label in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y - 0.22), w, 0.44,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     facecolor=color, edgecolor="white", linewidth=2))
        ax.text(x + w / 2, y, label, ha="center", va="center", color="white",
                fontsize=10.5, fontweight="bold", wrap=True)
        centers.append((x, x + w))
        if i > 0:
            ax.add_patch(FancyArrowPatch((centers[i-1][1], y), (x, y),
                         arrowstyle="-|>", mutation_scale=18, color=NAVY, lw=1.6))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis("off")


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 2.4))
    _flow(ax, ["Данные\nMOEX / ЦБ", "Риск-факторы\nPCA", "Модели\nMLE",
               "Цены ОФЗ\nот КБД", "VaR / ES\nМонте-Карло", "Бэктест\n2025"], BLUE)
    ax.set_title("Пайплайн проекта", fontsize=14, fontweight="bold", color=NAVY)
    save(fig, "pipeline.png")


def plot_risk_engine() -> None:
    fig, ax = plt.subplots(figsize=(13, 2.4))
    _flow(ax, ["Сценарий\nфакторов", "Шок\nКБД / акции / FX", "Переоценка\nпортфеля",
               "Распределение\nP&L", "VaR 99%\nES 97.5%"], TEAL)
    ax.set_title("Как сценарий факторов превращается в риск", fontsize=14,
                 fontweight="bold", color=NAVY)
    save(fig, "risk_engine_flow.png")


def plot_var_es_concept() -> None:
    """Концептуальная диаграмма: плотность P&L, порог VaR и хвост ES."""
    from scipy import stats
    x = np.linspace(-4.2, 3.6, 700)
    pdf = stats.t.pdf(x, df=4)
    var99 = stats.t.ppf(0.01, df=4)
    es_cut = stats.t.ppf(0.025, df=4)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(x, pdf, color=NAVY, lw=2.6)
    mask = x <= es_cut
    ax.fill_between(x[mask], pdf[mask], color=RED, alpha=0.28, zorder=1)
    ax.axvline(var99, color=RED, lw=2, ls="--")
    ax.annotate("VaR 99%", xy=(var99, stats.t.pdf(var99, 4)),
                xytext=(var99 - 0.15, 0.20), ha="right", color=RED,
                fontsize=12.5, fontweight="bold")
    ax.annotate("ES 97.5%\nсреднее в хвосте", xy=(-3.0, 0.012),
                xytext=(-4.1, 0.105), color="#a93226", fontsize=10.5,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#a93226", lw=1.2))
    ax.set_xlabel("P&L портфеля  ·  потери ←   → прибыль", fontsize=10.5, color=GREY)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_ylim(0, pdf.max() * 1.18)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.set_title("VaR — порог потерь, ES — средние потери за порогом",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
    save(fig, "var_es_concept.png")


def plot_data_timeline() -> None:
    """Таймлайн выборки 2021–2026 с двумя структурными разрывами."""
    events = [
        (2021.02, "Старт выборки\n01.2021", 1, BLUE),
        (2022.15, "Шок 2022\nставка 20%, рубль", -1, RED),
        (2024.45, "13.06.2024\nстоп биржевых\nUSD/EUR (MOEX)", 1, AMBER),
        (2025.92, "Оценка риска\n02.12.2025", -1, TEAL),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    x0, x1 = 2020.8, 2026.2
    ax.hlines(0, x0, x1, color=NAVY, lw=3, zorder=1)
    for x, label, side, color in events:
        ax.plot(x, 0, "o", ms=14, color=color, mec="white", mew=2, zorder=3)
        ax.annotate(label, xy=(x, 0), xytext=(x, 0.62 * side), ha="center",
                    va="bottom" if side > 0 else "top",
                    fontsize=10.5, fontweight="bold", color=NAVY,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.3))
    ax.set_xlim(x0, x1); ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([])
    ax.set_xticks(range(2021, 2027))
    ax.tick_params(axis="x", labelsize=11, colors=GREY)
    for sp in ("left", "right", "top", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Окно 2021–2026: два структурных разрыва",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    save(fig, "data_timeline.png")


# --------------------------------------------------------------------------- #
# HTML-дашборд                                                                #
# --------------------------------------------------------------------------- #
def _img_b64(path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_html(k: dict) -> None:
    o = OUTPUT_DIR
    sections = [
        ("Риск-факторы", [
            (o / "slides" / "pipeline.png", "Пайплайн: данные → факторы → модели → цены → риск → бэктест"),
            (o / "factors" / "curve_loadings.png", "Кривая КБД: уровень / наклон / кривизна (3 PC = 95.6%)"),
            (o / "factors" / "scree_equity.png", "Акции: PCA не сжимается (PC1 ≈ 54%)"),
            (o / "factors" / "qq_market_factor.png", "Тяжёлые хвосты рыночного фактора"),
            (o / "factors" / "vol_clustering_market.png", "Кластеризация волатильности (ARCH-эффект)"),
            (o / "factors" / "correlation_heatmap.png", "Корреляции 15 риск-факторов"),
        ]),
        ("Стохастические модели", [
            (o / "models" / "aic_normal_vs_t.png", "Выбор распределения: t-Стьюдента бьёт нормаль"),
            (o / "models" / "conditional_vol.png", "Условная волатильность: EWMA vs GARCH"),
            (o / "models" / "garch_persistence.png", "GARCH: персистентность α+β ≈ 0.99"),
        ]),
        ("Справедливая стоимость", [
            (o / "pricing" / "kbd_curve.png", "Кривая КБД на дату оценки риска"),
            (o / "pricing" / "pricing_error.png", "Ошибка модель–рынок по 5 ОФЗ"),
        ]),
        ("Риск VaR / ES", [
            (o / "slides" / "risk_engine_flow.png", "Сценарий факторов → P&L → VaR/ES"),
            (o / "slides" / "portfolio_donut.png", "Структура портфеля"),
            (o / "slides" / "es_contribution.png", "Вклад классов активов в хвостовой ES"),
            (o / "risk" / "var_by_model.png", "VaR по моделям и горизонтам"),
            (o / "risk" / "pnl_distributions_tail_models.png", "Распределения P&L"),
        ]),
        ("Бэктест", [
            (o / "backtest" / "var_backtest.png", "Пробои VaR в 2025: P&L против порога"),
        ]),
    ]

    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-t">{t}</div>'
        f'<div class="kpi-v">{v}</div><div class="kpi-s">{s}</div></div>'
        for t, v, s in [
            ("Стоимость портфеля", f"{k['value']/1e6:.1f} млн ₽", "номинал 260 млн"),
            ("VaR 99% · 1 день", f"{k['var1']/1e6:.2f} млн ₽", f"ES {k['es1']/1e6:.2f} млн"),
            ("VaR 99% · 10 дней", f"{k['var10']/1e6:.2f} млн ₽", f"ES {k['es10']/1e6:.1f} млн"),
            ("Ошибка цены ОФЗ", f"{k['pricing_mae']:.3f}%", f"{k['ytm_bp']:.1f} б.п. YTM"),
            ("Бэктест", f"{k['bt_exc']}/{k['bt_obs']}", f"Kupiec {k['bt_kupiec']:.3f} · {k['bt_light']}"),
        ]
    )

    body = []
    for title, items in sections:
        cards = []
        for path, cap in items:
            b64 = _img_b64(path)
            if b64 is None:
                continue
            cards.append(f'<figure><img src="data:image/png;base64,{b64}" '
                         f'alt="{cap}"><figcaption>{cap}</figcaption></figure>')
        if cards:
            body.append(f'<section><h2>{title}</h2><div class="grid">'
                        + "".join(cards) + "</div></section>")

    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Рыночный риск — дашборд</title>
<style>
  :root {{ --navy:{NAVY}; --bg:{BG}; --grey:{GREY}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;
          background:var(--bg); color:#1b2430; }}
  header {{ background:linear-gradient(135deg,#1f3a5f,#2c7fb8); color:#fff;
            padding:34px 40px; }}
  header h1 {{ margin:0 0 6px; font-size:26px; }}
  header p {{ margin:0; opacity:.85; font-size:14px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
           gap:14px; padding:24px 40px; }}
  .kpi {{ background:#fff; border-radius:14px; padding:16px 18px;
          box-shadow:0 4px 14px rgba(20,40,80,.08); border-left:5px solid var(--navy); }}
  .kpi-t {{ font-size:12.5px; color:var(--grey); }}
  .kpi-v {{ font-size:24px; font-weight:700; color:var(--navy); margin:4px 0; }}
  .kpi-s {{ font-size:12px; color:#3a7; }}
  section {{ padding:8px 40px 18px; }}
  section h2 {{ color:var(--navy); border-bottom:2px solid #dde5ee; padding-bottom:6px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
  figure {{ margin:0; background:#fff; border-radius:14px; overflow:hidden;
            box-shadow:0 4px 14px rgba(20,40,80,.08); }}
  figure img {{ width:100%; display:block; }}
  figcaption {{ padding:10px 14px; font-size:13.5px; color:#33415c; font-weight:600; }}
  footer {{ padding:24px 40px 40px; color:var(--grey); font-size:12.5px; }}
</style></head><body>
<header>
  <h1>Оценка рыночного риска портфеля — VaR&nbsp;99% / ES&nbsp;97.5%</h1>
  <p>5 ОФЗ · 10 акций · USD/EUR · 260 млн руб. · дата оценки 02.12.2025 · данные MOEX&nbsp;ISS + ЦБ&nbsp;РФ</p>
</header>
<div class="kpis">{kpi_html}</div>
{''.join(body)}
<footer>Сгенерировано scripts/build_slides.py из outputs/. Открой в браузере,
скриншоть панели в слайды или показывай вживую. Числа синхронны с FINAL_REPORT.md.</footer>
</body></html>"""

    out = SLIDES / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    logger.info("дашборд:  %s", out.relative_to(OUTPUT_DIR.parent))


def main() -> None:
    k = collect_kpis()
    for fn in (plot_donut, plot_es_contribution, lambda: plot_kpi_dashboard(k),
               plot_pipeline, plot_risk_engine, plot_data_timeline,
               plot_var_es_concept):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("график не построился: %s", exc)
    build_html(k)
    print("\n=== ПРЕЗЕНТАЦИОННЫЕ АССЕТЫ ГОТОВЫ ===")
    print(f"Графики и dashboard.html: {SLIDES}")
    print("Открой outputs/slides/dashboard.html в браузере "
          "(двойной клик) — листай и скриншоть в слайды.")


if __name__ == "__main__":
    main()
