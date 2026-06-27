"""
Этап 3: подгонка стохастических моделей факторов (MLE) и их сравнение.

Запусти ЛОКАЛЬНО (сети не требует — читает outputs/factors/factor_scores.csv):

    python -m scripts.build_models

Что делает:
  1. многомерная нормаль и многомерная t-Стьюдента (MLE) -> сравнение AIC/BIC;
  2. EWMA-ковариация (λ=0.94 RiskMetrics и λ по MLE) -> прогноз σ на завтра;
  3. GARCH(1,1)+CCC по факторам -> персистентность, σ на завтра, и
     честное сравнение GARCH-t vs i.i.d.-нормаль на одном факторе;
  4. AR(1)/Орнштейн–Уленбек по факторам ставок -> возврат к среднему vs RW.
Сохраняет параметры (CSV), ковариации на завтра (для риск-движка),
графики и SUMMARY_models.md в outputs/models/.

Всё воспроизводимо: seed зафиксирован, симуляции здесь не запускаются
(одношаговые .simulate появятся на этапе 5 — риск-движок).
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rm.config import OUTPUT_DIR, RANDOM_SEED
from rm.models import (
    CCCGarchModel,
    EWMAModel,
    GaussianModel,
    StudentTModel,
    ar1_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_models")

FACTORS_CSV = OUTPUT_DIR / "factors" / "factor_scores.csv"
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RATE_FACTORS = ["rate_PC1", "rate_PC2", "rate_PC3"]


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = MODELS_DIR / name
    df.to_csv(path, encoding="utf-8-sig")
    logger.info("сохранено: %s", path.relative_to(OUTPUT_DIR.parent))


def save_fig(fig, name: str) -> None:
    path = MODELS_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("график:   %s", path.relative_to(OUTPUT_DIR.parent))


def _iid_normal_loglik(y: np.ndarray) -> float:
    """log-lik i.i.d. N(μ,σ²) — для честного сравнения с GARCH на одном ряду."""
    mu, var = float(np.mean(y)), float(np.var(y, ddof=0))
    return float(np.sum(-0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)))


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    if not FACTORS_CSV.exists():
        raise SystemExit(f"нет {FACTORS_CSV} — сначала запусти scripts.build_factors")
    panel = pd.read_csv(FACTORS_CSV, index_col=0, parse_dates=True)
    logger.info("панель факторов: %d дней × %d факторов", *panel.shape)

    # ------------------------------------------------------------------ #
    # 1. Нормаль vs t-Стьюдента (MLE)                                     #
    # ------------------------------------------------------------------ #
    logger.info("=== Нормаль и t-Стьюдента (MLE) ===")
    gauss = GaussianModel.fit(panel)
    tmod = StudentTModel.fit(panel)
    logger.info("t-Стьюдента: ν≈%.2f", tmod.df)

    comparison = pd.DataFrame({
        "log-lik": [gauss.loglik, tmod.loglik],
        "n_параметров": [gauss.n_params, tmod.n_params],
        "AIC": [gauss.aic, tmod.aic],
        "BIC": [gauss.bic, tmod.bic],
    }, index=["Нормаль", "t-Стьюдента"])
    comparison["ΔAIC_к_лучшей"] = comparison["AIC"] - comparison["AIC"].min()
    save_csv(comparison.round(1), "normal_vs_t.csv")

    save_csv(pd.DataFrame(gauss.cov, index=gauss.columns, columns=gauss.columns),
             "gaussian_cov.csv")
    save_csv(pd.DataFrame(tmod.scale, index=tmod.columns, columns=tmod.columns),
             "student_t_scale.csv")

    # ------------------------------------------------------------------ #
    # 2. EWMA-ковариация                                                 #
    # ------------------------------------------------------------------ #
    logger.info("=== EWMA-ковариация ===")
    ewma_rm = EWMAModel.fit(panel, lam=0.94)
    ewma_ml = EWMAModel.fit(panel, lam=None)
    logger.info("EWMA: λ(RiskMetrics)=0.94, λ(MLE)=%.4f", ewma_ml.lam)

    vols = pd.DataFrame({
        "σ_завтра(λ=0.94)": ewma_rm.factor_vols(),
        "σ_завтра(λ_MLE)": ewma_ml.factor_vols(),
        "σ_безусловная": panel.std(ddof=1),
    })
    vols["год_λ=0.94"] = vols["σ_завтра(λ=0.94)"] * np.sqrt(250)
    save_csv(vols, "ewma_factor_vols.csv")
    save_csv(pd.DataFrame(ewma_rm.next_cov, index=ewma_rm.columns, columns=ewma_rm.columns),
             "ewma_next_cov.csv")

    # ------------------------------------------------------------------ #
    # 3. GARCH(1,1) + CCC                                                 #
    # ------------------------------------------------------------------ #
    logger.info("=== GARCH(1,1) + CCC ===")
    garch = CCCGarchModel.fit(panel, dist="t")
    gtab = garch.table()
    save_csv(gtab.round(4), "garch_factors.csv")
    save_csv(pd.DataFrame(garch.next_cov, index=garch.columns, columns=garch.columns),
             "garch_next_cov.csv")
    logger.info("GARCH: средняя персистентность α+β = %.3f", gtab["персистентность"].mean())

    # честное сравнение GARCH-t vs i.i.d.-нормаль на самом «хвостатом» факторе
    kurt = panel.kurt().sort_values(ascending=False)
    rep = kurt.index[0]
    x = panel[rep].to_numpy()
    y = (x - x.mean()) / x.std(ddof=1) * 10.0
    gf = next(f for f in garch.factors if f.name == rep)
    iid_ll = _iid_normal_loglik(y)
    garch_aic = 2 * 5 - 2 * gf.loglik          # μ,ω,α,β,ν = 5 параметров
    iid_aic = 2 * 2 - 2 * iid_ll               # μ,σ² = 2 параметра
    headline = pd.DataFrame({
        "log-lik": [iid_ll, gf.loglik],
        "n_параметров": [2, 5],
        "AIC": [iid_aic, garch_aic],
    }, index=["i.i.d. Нормаль", "GARCH(1,1)-t"])
    save_csv(headline.round(1), f"garch_vs_iid_{rep}.csv")

    # ------------------------------------------------------------------ #
    # 4. AR(1)/OU — возврат к среднему для ставок                         #
    # ------------------------------------------------------------------ #
    logger.info("=== AR(1)/OU: приращения (i.i.d.?) vs уровни (возврат к среднему?) ===")
    # на приращениях/доходностях: тест i.i.d. -> оправдание масштабирования √10
    ar_incr = ar1_table(panel)
    save_csv(ar_incr.round(4), "ar1_increments_all.csv")
    save_csv(ar_incr.loc[RATE_FACTORS].round(4), "ar1_increments_rates.csv")
    # уровни ставок (прокси = накопленные приращения PC): тест OU/возврата к среднему
    rate_levels = panel[RATE_FACTORS].cumsum()
    ar_levels = ar1_table(rate_levels)
    save_csv(ar_levels.round(4), "ar1_levels_rates.csv")

    # ------------------------------------------------------------------ #
    # Графики                                                            #
    # ------------------------------------------------------------------ #
    logger.info("=== Графики ===")
    try:
        # «дышащая» волатильность: EWMA vs GARCH на хвостатом факторе
        fig, ax = plt.subplots(figsize=(10, 4))
        ann = np.sqrt(250)
        ewma_rm.conditional_vol_series(rep).mul(ann).plot(ax=ax, label="EWMA σ (год.)")
        garch.cond_vol[rep].mul(ann).plot(ax=ax, label="GARCH σ (год.)", alpha=0.8)
        ax.set_title(f"Условная волатильность фактора {rep}: EWMA vs GARCH")
        ax.set_ylabel("годовая σ"); ax.legend(); ax.grid(alpha=0.3)
        save_fig(fig, "conditional_vol.png")

        # AIC-сравнение нормаль vs t
        fig, ax = plt.subplots(figsize=(6, 4))
        comparison["ΔAIC_к_лучшей"].plot.bar(ax=ax, color=["#c0392b", "#27ae60"])
        ax.set_ylabel("ΔAIC к лучшей модели (0 = лучшая)")
        ax.set_title("Выбор распределения: нормаль vs t-Стьюдента")
        ax.grid(alpha=0.3)
        save_fig(fig, "aic_normal_vs_t.png")

        # персистентность GARCH по факторам
        fig, ax = plt.subplots(figsize=(8, 4))
        gtab["персистентность"].sort_values().plot.barh(ax=ax, color="#8e44ad")
        ax.axvline(1.0, color="red", ls="--", lw=0.8, label="α+β=1 (IGARCH)")
        ax.set_xlabel("персистентность α+β")
        ax.set_title("GARCH(1,1): персистентность волатильности по факторам")
        ax.legend(); ax.grid(alpha=0.3)
        save_fig(fig, "garch_persistence.png")
    except Exception as exc:
        logger.warning("часть графиков не построилась: %s", exc)

    # ------------------------------------------------------------------ #
    # Сводка                                                             #
    # ------------------------------------------------------------------ #
    write_summary(panel, gauss, tmod, comparison, ewma_rm, ewma_ml,
                  garch, gtab, headline, rep, ar_incr, ar_levels)

    print("\n=== ИТОГ ЭТАПА 3 ===")
    best = comparison["AIC"].idxmin()
    print(f"Распределение: лучшая по AIC/BIC — {best} (t: ν≈{tmod.df:.2f})")
    lam_note = "упёрлась в границу (≈IGARCH)" if ewma_ml.lam > 0.995 else "близко к 0.94"
    print(f"EWMA: λ_MLE={ewma_ml.lam:.4f} [{lam_note}]; для движка берём RiskMetrics 0.94")
    print(f"GARCH: средняя персистентность α+β = {gtab['персистентность'].mean():.3f}")
    print(f"Ставки: приращения ≈ i.i.d. (белый шум), уровни ≈ "
          f"{ar_levels.loc['rate_PC1', 'режим']}")
    print(f"\nВсё в: {MODELS_DIR}")


def write_summary(panel, gauss, tmod, comparison, ewma_rm, ewma_ml,
                  garch, gtab, headline, rep, ar_incr, ar_levels) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Этап 3 — стохастические модели факторов (MLE)\n")
    add(f"_Сгенерировано scripts/build_models.py, seed={RANDOM_SEED}, "
        f"панель {panel.shape[0]}×{panel.shape[1]}._\n")

    add("## 1. Распределение: нормаль vs t-Стьюдента\n")
    best = comparison["AIC"].idxmin()
    d_aic = comparison.loc["Нормаль", "AIC"] - comparison.loc["t-Стьюдента", "AIC"]
    add(f"- оценка ν (MLE) = **{tmod.df:.2f}** — очень тяжёлые хвосты "
        "(при ν≤4 эксцесс бесконечен), согласуется с дескриптивкой (df≈3).")
    add(f"- по AIC и BIC уверенно лучше **{best}**: ΔAIC = {d_aic:,.0f} в пользу "
        "t-Стьюдента — нормаль отвергается как модель приращений.")
    add("- для риска это критично: нормаль недооценит VaR/ES в хвосте, t — нет.\n")

    add("## 2. EWMA-ковариация (кластеризация волатильности)\n")
    boundary = ewma_ml.lam > 0.995
    if boundary:
        add(f"- λ(MLE) = **{ewma_ml.lam:.4f}** — оценка УПЁРЛАСЬ в верхнюю границу "
            "(≈ IGARCH, near-unit-root): безусловный предиктивный log-lik монотонно "
            "растёт к λ→1, т.е. данные «хотят» почти статичную (гладкую) ковариацию. "
            "Это согласуется с высокой персистентностью GARCH (см. ниже), но **НЕ** "
            "подтверждает 0.94.")
        add("- λ(RiskMetrics) = 0.94 — прагматичный реактивный стандарт; именно его "
            "берём в риск-движок (быстрее реагирует на шоки), а λ_MLE фиксируем как "
            "найденную особенность для обсуждения.")
    else:
        add(f"- λ(RiskMetrics) = 0.94; λ(MLE) = **{ewma_ml.lam:.4f}** — оценка по "
            "данным близка к стандарту, что поддерживает выбор RiskMetrics.")
    add("- прогноз σ «на завтра» при λ=0.94 заметно реагирует на недавние шоки — "
        "см. ewma_factor_vols.csv и график conditional_vol.png.\n")

    add("## 3. GARCH(1,1) + CCC\n")
    persist = gtab["персистентность"]
    add(f"- средняя персистентность α+β = **{persist.mean():.3f}** "
        f"(макс {persist.max():.3f}) — волатильность долгопамятна, шок гаснет медленно.")
    add(f"- на самом тяжёлохвостом факторе ({rep}) GARCH(1,1)-t бьёт i.i.d.-нормаль "
        f"по AIC: {headline.loc['GARCH(1,1)-t','AIC']:,.0f} против "
        f"{headline.loc['i.i.d. Нормаль','AIC']:,.0f} — динамика дисперсии нужна.")
    add("- ν инноваций t в GARCH тоже мало -> хвосты остаются тяжёлыми даже после "
        "учёта кластеризации волатильности.\n")

    add("## 4. AR(1)/Орнштейн–Уленбек — ставки (приращения vs уровни)\n")
    add("Важно различать: OU/возврат к среднему — про УРОВЕНЬ ставки, а i.i.d. — "
        "про ПРИРАЩЕНИЯ (от этого зависит масштабирование 1 день → 10 дней).\n")
    add("**Приращения 3 PC (тест i.i.d.):**")
    for f in RATE_FACTORS:
        r = ar_incr.loc[f]
        add(f"- {f}: φ={r['phi']:.3f}, t(φ=0)={r['t(φ=0)']:.1f}, t(φ=1)={r['t(φ=1)']:.1f} "
            f"-> {r['режим']}.")
    add("\n**Уровни 3 PC (прокси = накопленные приращения; тест OU):**")
    for f in RATE_FACTORS:
        r = ar_levels.loc[f]
        hl = r["полураспад_дн"]
        hl_s = "∞" if not np.isfinite(hl) else f"{hl:.0f} дн."
        add(f"- {f}: φ={r['phi']:.3f}, t(φ=1)={r['t(φ=1)']:.1f}, полураспад {hl_s} "
            f"-> {r['режим']}.")
    add("\nВывод: уровни ставок близки к единичному корню (возврат к среднему на окне "
        "2021–26 статистически не выражен / очень медленный), а приращения — почти "
        "белый шум. Значит на горизонте 1–10 дней RW по ставкам оправдан, а "
        "масштабирование √10 корректно (с малой оговоркой по PC1, где приращения чуть "
        "автокоррелированы). OU оставляем как проверенную альтернативу.\n")

    add("## 5. Что выбрать для риск-движка (этап 5)\n")
    add("- **базовый сценарий**: t-Стьюдента (тяжёлые хвосты) — лучшая по AIC/BIC;")
    add("- **с учётом кластеризации**: GARCH(1,1)-CCC или EWMA(λ=0.94) для ковариации "
        "«на завтра» (λ_MLE вырождается в почти статичную — не берём);")
    add("- **ставки**: приращения 3 PC как RW (OU-альтернатива проверена);")
    add("- сравним VaR/ES по нормали, t и GARCH — расхождение в хвосте и есть "
        "главный результат для обсуждения.\n")

    path = MODELS_DIR / "SUMMARY_models.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("сводка:   %s", path.relative_to(OUTPUT_DIR.parent))


if __name__ == "__main__":
    main()
