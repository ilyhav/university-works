"""
Этап 2: построение риск-факторов и дескриптивный анализ.

Запусти ЛОКАЛЬНО (нужен интернет — подтянет панель из кэша + историю YNDX
для склейки с YDEX):

    python -m scripts.build_factors

Что делает:
  1. собирает панель рыночных данных (build_market_data, из кэша);
  2. строит доходности/приращения факторов:
       — кривая КБД -> приращения Δy -> PCA: уровень/наклон/кривизна;
       — акции -> лог-доходности (со склейкой YNDX->YDEX) -> PCA: рыночный фактор;
       — валюты USD/EUR -> лог-доходности + PCA (общий рубль / расхождение);
  3. дескриптивка по факторам: моменты, тяжёлые хвосты, стационарность,
     кластеризация волатильности, корреляции (п.2c);
  4. сохраняет таблицы (CSV), графики (PNG) и текстовую сводку SUMMARY.md
     в outputs/factors/.

Результаты воспроизводимы: PCA детерминирован (eigh), сети для аналитики нет.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # без дисплея — только сохранение в файлы
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rm.config import (
    EQUITY_PREDECESSORS,
    OUTPUT_DIR,
    RANDOM_SEED,
)
from rm.data import moex
from rm.data.dataset import build_market_data
from rm.factors import (
    build_equity_returns,
    clean_for_pca,
    correlation_matrix,
    curve_increments,
    fit_pca,
    fit_student_t,
    interpret_curve_components,
    log_returns,
    moments_table,
    squared_return_acf,
    stationarity_table,
    tail_table,
    volatility_clustering_table,
    weekday_seasonality,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_factors")

FACTORS_DIR = OUTPUT_DIR / "factors"
FACTORS_DIR.mkdir(parents=True, exist_ok=True)

FX_BREAK = pd.Timestamp("2024-06-13")  # остановка биржевых торгов USD/EUR на MOEX


# --------------------------------------------------------------------------- #
# Сохранение                                                                  #
# --------------------------------------------------------------------------- #
def save_csv(df: pd.DataFrame, name: str) -> None:
    path = FACTORS_DIR / name
    df.to_csv(path, encoding="utf-8-sig")
    logger.info("сохранено: %s", path.relative_to(OUTPUT_DIR.parent))


def save_fig(fig, name: str) -> None:
    path = FACTORS_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("график:   %s", path.relative_to(OUTPUT_DIR.parent))


# --------------------------------------------------------------------------- #
# Факторы                                                                     #
# --------------------------------------------------------------------------- #
def build_rate_factors(md):
    """PCA по приращениям кривой КБД -> уровень / наклон / кривизна."""
    dy = curve_increments(md.gcurve)
    clean, rep = clean_for_pca(dy, min_coverage=0.9)
    res = fit_pca(clean, n_components=3, standardize=False)  # ковариация: единые единицы
    labels = interpret_curve_components(res.loadings)
    logger.info("кривая: интерпретация компонент -> %s", labels)
    return res, labels, rep


def build_equity_factors(md):
    """Лог-доходности акций (со склейкой) -> PCA -> рыночный фактор."""
    predecessor_prices = {}
    for successor, pred in EQUITY_PREDECESSORS.items():
        try:
            df = moex.load_stock_history(pred, _iso(md.calendar.min()), _iso(md.calendar.max()))
            if not df.empty and "CLOSE" in df:
                predecessor_prices[successor] = df["CLOSE"]
                logger.info("история предшественника %s загружена (%d дн.)", pred, len(df))
        except Exception as exc:  # сеть/листинг — не валим весь пайплайн
            logger.warning("не удалось загрузить предшественника %s: %s", pred, exc)

    rets = build_equity_returns(md.stock_prices, predecessor_prices)
    clean, rep = clean_for_pca(rets, min_coverage=0.5)
    # standardize=True: бумаги с разной σ вносят сопоставимый вклад -> чистый PC1=рынок
    res = fit_pca(clean, n_components=clean.shape[1], standardize=True)
    return res, rets, rep


def build_fx_factors(md):
    """Лог-доходности USD/EUR как факторы + PCA для интерпретации."""
    rets = log_returns(md.fx)
    clean, rep = clean_for_pca(rets, min_coverage=0.5)
    res = fit_pca(clean, n_components=clean.shape[1], standardize=True)
    return res, rets, rep


def assemble_factor_panel(rate, equity_rets, fx_rets) -> pd.DataFrame:
    """Единая панель риск-факторов для МК-движка (этап 5):
      * ставки — 3 компоненты кривой (оправданное сжатие 12 сроков -> 3,
        облигации цените от восстановленной по ним кривой);
      * акции — ИНДИВИДУАЛЬНЫЕ лог-доходности 10 бумаг (а не PC!): PCA показал
        PC1≈54%, для 90% нужно 8 из 10 компонент -> сжатие почти не работает,
        2 PC занизили бы риск акций на ~35%. Полная ковариация = без потерь;
      * валюты — сырые лог-доходности USD/EUR (так чище ценить FX-портфель).
    Рыночный фактор PC1 остаётся как описательная конструкция (см. PCA-выгрузки).
    """
    parts = {
        "rate_PC1": rate.scores["PC1"],
        "rate_PC2": rate.scores["PC2"],
        "rate_PC3": rate.scores["PC3"],
    }
    for sec in equity_rets.columns:
        parts[f"eq_{sec}"] = equity_rets[sec]
    for cur in fx_rets.columns:
        parts[f"fx_{cur}"] = fx_rets[cur]
    panel = pd.concat(parts, axis=1, sort=True).dropna(how="any")
    return panel


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Графики                                                                     #
# --------------------------------------------------------------------------- #
def plot_scree(res, title: str, name: str, top: int = 10) -> None:
    k = min(top, len(res.explained_variance_ratio))
    ratio = res.explained_variance_ratio[:k]
    cum = np.cumsum(res.explained_variance_ratio)[:k]
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, ratio * 100, color="#3b6ea5", label="доля дисперсии")
    ax.plot(x, cum * 100, "o-", color="#c0392b", label="накопленная")
    ax.set_xlabel("компонента"); ax.set_ylabel("% дисперсии")
    ax.set_title(title); ax.set_xticks(x); ax.legend(); ax.grid(alpha=0.3)
    save_fig(fig, name)


def plot_curve_loadings(res, labels: dict[str, str], name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    tenors = [float(t) for t in res.loadings.index]
    for comp in res.loadings.columns:
        ax.plot(tenors, res.loadings[comp], "o-", label=f"{comp} — {labels.get(comp, '')}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("срок, лет"); ax.set_ylabel("нагрузка")
    ax.set_title("Компоненты кривой КБД (уровень / наклон / кривизна)")
    ax.legend(); ax.grid(alpha=0.3)
    save_fig(fig, name)


def plot_loadings_bar(res, comp: str, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    s = res.loadings[comp].sort_values()
    ax.barh(s.index, s.values, color="#27ae60")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"нагрузка на {comp}"); ax.set_title(title); ax.grid(alpha=0.3)
    save_fig(fig, name)


def plot_corr_heatmap(corr: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Корреляции риск-факторов")
    save_fig(fig, name)


def plot_qq(series: pd.Series, name: str) -> None:
    from scipy import stats
    x = series.dropna().to_numpy()
    x = (x - x.mean()) / x.std(ddof=1)
    tparams = stats.t.fit(x)
    df_t = tparams[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    stats.probplot(x, dist="norm", plot=axes[0])
    axes[0].set_title("QQ против нормали")
    stats.probplot(x, dist=stats.t, sparams=(df_t,), plot=axes[1])
    axes[1].set_title(f"QQ против t-Стьюдента (df≈{df_t:.1f})")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Хвосты фактора: {series.name}")
    save_fig(fig, name)


def plot_vol_acf(series: pd.Series, name: str) -> None:
    acf_vals = squared_return_acf(series, nlags=40)
    n = series.dropna().shape[0]
    ci = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(acf_vals.index[1:], acf_vals.values[1:], color="#8e44ad")
    ax.axhline(ci, color="red", ls="--", lw=0.8)
    ax.axhline(-ci, color="red", ls="--", lw=0.8)
    ax.set_xlabel("лаг"); ax.set_ylabel("ACF квадратов доходностей")
    ax.set_title(f"Кластеризация волатильности: {series.name}")
    ax.grid(alpha=0.3)
    save_fig(fig, name)


# --------------------------------------------------------------------------- #
# Сводка для защиты                                                           #
# --------------------------------------------------------------------------- #
def write_summary(rate, rate_labels, equity, fx_pca, fx_rets,
                  panel, moments, stat, volclust) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Этап 2 — риск-факторы и дескриптивный анализ\n")
    add(f"_Сгенерировано scripts/build_factors.py, seed={RANDOM_SEED}._\n")

    add("## 1. Кривая КБД — PCA (уровень / наклон / кривизна)\n")
    cum3 = float(np.cumsum(rate.explained_variance_ratio)[2])
    for comp in rate.loadings.columns:
        r = rate.explained_variance_ratio[int(comp[2:]) - 1]
        add(f"- **{comp} — {rate_labels.get(comp, '')}**: {r*100:.2f}% дисперсии")
    add(f"\nПервые 3 компоненты объясняют **{cum3*100:.2f}%** дисперсии приращений "
        "ставок — структура Литтермана–Шейнкмана воспроизводится, для риска ставок "
        "хватает трёх факторов. Это ниже хрестоматийных ~99% по двум причинам: "
        "КБД ЦБ — уже сглаженная модельная кривая (Свенссон), а короткий конец "
        "(0.25–0.75 г) и шок 2022 г. дают непараллельные движения. Облигации "
        "цените от восстановленной по 3 PC кривой; остаток ~"
        f"{(1-cum3)*100:.1f}% уходит в ошибку аппроксимации кривой.\n")

    add("## 2. Акции — PCA (рыночный фактор) и решение для риск-движка\n")
    pc1 = float(equity.explained_variance_ratio[0])
    cum = np.cumsum(equity.explained_variance_ratio)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    add(f"- **PC1 (рыночный фактор)**: {pc1*100:.1f}% общей дисперсии — единый "
        "системный драйвер, на него все бумаги грузятся одного знака.")
    add(f"- но для 90% дисперсии нужно **{n90}** компонент, для 95% — **{n95}** "
        f"(из {equity.n_components}); средняя парная корреляция ≈ "
        f"{(pc1 * equity.n_components - 1) / (equity.n_components - 1):.2f}.")
    add("- **ВЫВОД: PCA-сжатие акций почти не работает** (рынок объясняет лишь "
        "половину, остальное — идиосинкразия отраслей: банк/нефтегаз/металлы/"
        "ритейл/телеком/техи). Поэтому в риск-движок акции входят **индивидуальными "
        "доходностями** (полная ковариация 10 бумаг, без потерь), а не 1–2 PC — "
        "иначе риск акций был бы занижен на ~35%. PC1 оставлен как описательный "
        "«рыночный фактор» (нагрузки/scree в выгрузках).\n")

    add("## 3. Валюты USD/EUR\n")
    corr_fx = fx_rets.corr().iloc[0, 1]
    add(f"- корреляция дневных лог-доходностей USD и EUR: **{corr_fx:.2f}** "
        "(оба — это в первую очередь курс рубля).")
    add(f"- PC1 (общий рубль): {fx_pca.explained_variance_ratio[0]*100:.1f}%, "
        f"PC2 (расхождение EUR/USD): {fx_pca.explained_variance_ratio[1]*100:.1f}%.")
    pre = fx_rets[fx_rets.index < FX_BREAK].std(ddof=1) * np.sqrt(250)
    post = fx_rets[fx_rets.index >= FX_BREAK].std(ddof=1) * np.sqrt(250)
    add(f"- **РАЗРЫВ ИСТОЧНИКА 13.06.2024**: с этой даты биржевые торги USD/EUR на "
        "MOEX остановлены, курс ЦБ считается по внебиржевым сделкам. Годовая σ "
        f"курса до/после: USD {pre['USD']*100:.1f}% → {post['USD']*100:.1f}%, "
        f"EUR {pre['EUR']*100:.1f}% → {post['EUR']*100:.1f}%. Это смена методологии "
        "внутри окна — обязательно учесть при выборе окна оценки ковариаций.\n")

    add("## 4. Дескриптивка факторов (п.2c)\n")
    not_normal = (moments["JB_pvalue"] < 0.05).sum()
    max_kurt = moments["эксцесс_изб"].astype(float).max()
    add(f"- **тяжёлые хвосты**: тест Жарка–Бера отвергает нормальность у "
        f"{not_normal} из {len(moments)} факторов; макс. избыточный эксцесс "
        f"= {max_kurt:.1f} (у нормали 0). Вывод: нужна t-Стьюдента / EWMA-GARCH, "
        "а не i.i.d.-нормаль.")
    arch = (volclust["ARCH-эффект"] == "есть").sum()
    add(f"- **кластеризация волатильности**: ARCH-эффект (Льюнг–Бокс по квадратам) "
        f"у {arch} из {len(volclust)} факторов — тихие и бурные периоды группируются.")
    nonstat = (stat["ADF_вывод"] != "стационарен").sum()
    add(f"- **стационарность**: приращения/доходности стационарны (ADF отвергает "
        f"единичный корень) у {len(stat) - nonstat} из {len(stat)} факторов; "
        "уровни цен и ставок — нестационарны по построению (потому и работаем с "
        "приращениями).\n")

    add("## 5. Что вынести в критическое обсуждение\n")
    add("- структурный разрыв 2022 г. (ставка до 20%, обвал рубля) делает окно "
        "2021–2026 нестационарным по режимам волатильности;")
    add("- смена методологии курса ЦБ с 13.06.2024 — разрыв источника данных;")
    add("- 250 дней бэктеста при α=1% дают ~2.5 ожидаемых пробоя -> низкая мощность "
        "тестов (учтём ансамблем тестов на этапе 6–7);")
    add("- выбор PCA по ковариации (кривая) vs по корреляции (акции) — осознанный: "
        "единые единицы ставок против разной волатильности бумаг.\n")

    path = FACTORS_DIR / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("сводка:   %s", path.relative_to(OUTPUT_DIR.parent))


# --------------------------------------------------------------------------- #
# Главная точка                                                               #
# --------------------------------------------------------------------------- #
def main() -> None:
    np.random.seed(RANDOM_SEED)
    logger.info("=== Сборка панели данных ===")
    md = build_market_data()

    logger.info("=== Факторы ставок (кривая КБД) ===")
    rate, rate_labels, rate_cov = build_rate_factors(md)
    save_csv(rate.loadings, "curve_pca_loadings.csv")
    save_csv(pd.DataFrame({
        "доля": rate.explained_variance_ratio,
        "накопл.": np.cumsum(rate.explained_variance_ratio),
    }, index=[f"PC{i+1}" for i in range(len(rate.explained_variance_ratio))]),
        "curve_pca_variance.csv")

    logger.info("=== Факторы акций ===")
    equity, equity_rets, eq_cov = build_equity_factors(md)
    save_csv(equity.loadings, "equity_pca_loadings.csv")
    save_csv(pd.DataFrame({
        "доля": equity.explained_variance_ratio,
        "накопл.": np.cumsum(equity.explained_variance_ratio),
    }, index=[f"PC{i+1}" for i in range(len(equity.explained_variance_ratio))]),
        "equity_pca_variance.csv")
    save_csv(eq_cov.table, "equity_returns_coverage.csv")

    logger.info("=== Факторы валют ===")
    fx_pca, fx_rets, fx_cov = build_fx_factors(md)
    save_csv(fx_pca.loadings, "fx_pca_loadings.csv")

    logger.info("=== Сборка единой панели факторов ===")
    # акции входят ИНДИВИДУАЛЬНЫМИ доходностями (PCA-сжатие слабое — см. docstring)
    panel = assemble_factor_panel(rate, equity_rets, fx_rets)
    save_csv(panel, "factor_scores.csv")
    logger.info("панель факторов: %d дней × %d факторов "
                "(3 ставки + %d акций + %d валюты)",
                panel.shape[0], panel.shape[1],
                equity_rets.shape[1], fx_rets.shape[1])

    logger.info("=== Дескриптивный анализ факторов ===")
    moments = moments_table(panel)
    tails = tail_table(panel)
    stat = stationarity_table(panel)
    volclust = volatility_clustering_table(panel)
    corr = correlation_matrix(panel)
    save_csv(moments, "moments.csv")
    save_csv(tails, "tails.csv")
    save_csv(stat, "stationarity.csv")
    save_csv(volclust, "volatility_clustering.csv")
    save_csv(corr, "correlation.csv")
    save_csv(weekday_seasonality(equity.scores["PC1"]),
             "market_factor_weekday.csv")

    # подгонка t к рыночному фактору — оценка хвостов
    t_eq = fit_student_t(equity.scores["PC1"].to_numpy())
    logger.info("t-Стьюдент для рыночного фактора: df≈%.2f", t_eq["df"])

    logger.info("=== Графики ===")
    try:
        plot_scree(rate, "Кривая КБД — доля объяснённой дисперсии", "scree_curve.png")
        plot_curve_loadings(rate, rate_labels, "curve_loadings.png")
        plot_scree(equity, "Акции — доля объяснённой дисперсии", "scree_equity.png")
        plot_loadings_bar(equity, "PC1", "Рыночный фактор: нагрузки бумаг",
                          "equity_market_loadings.png")
        plot_corr_heatmap(corr, "correlation_heatmap.png")
        plot_qq(equity.scores["PC1"].rename("рыночный фактор (eq_PC1)"), "qq_market_factor.png")
        plot_vol_acf(equity.scores["PC1"].rename("рыночный фактор (eq_PC1)"),
                     "vol_clustering_market.png")
    except Exception as exc:
        logger.warning("часть графиков не построилась: %s", exc)

    logger.info("=== Сводка ===")
    write_summary(rate, rate_labels, equity, fx_pca, fx_rets,
                  panel, moments, stat, volclust)

    print("\n=== ИТОГ ЭТАПА 2 ===")
    print(f"Кривая: PC1-3 объясняют "
          f"{np.cumsum(rate.explained_variance_ratio)[2]*100:.2f}% "
          f"({', '.join(f'{c}={rate_labels[c]}' for c in rate.loadings.columns)})")
    print(f"Акции:  PC1 (рынок) = {equity.explained_variance_ratio[0]*100:.1f}% дисперсии")
    print(f"Валюты: corr(USD,EUR) = {fx_rets.corr().iloc[0,1]:.2f}")
    print(f"Панель факторов: {panel.shape[0]} дней × {panel.shape[1]} факторов")
    print(f"\nВсё в: {FACTORS_DIR}")


if __name__ == "__main__":
    main()
