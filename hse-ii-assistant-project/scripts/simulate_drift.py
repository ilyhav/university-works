"""Симуляция деградации качества из-за дрейфа данных и её устранения (criterion 5).

Сюжет: ЦБ снизил ключевую ставку и «промо-войны» на рынке закончились, но каталог
в проде не обновлялся. Мы показываем, что: (1) детектор дрейфа и SLA по свежести
ловят проблему, (2) качество ранжирования относительно АКТУАЛЬНОГО рынка падает,
(3) критерии эксплуатации дают вердикт REFRESH, (4) после обновления каталога
метрики восстанавливаются.

Запуск:  python -m scripts.simulate_drift  (или make drift)
"""

from __future__ import annotations

import json
from datetime import timedelta

from src.config import RUNTIME_DIR
from src.core.effective_rate import compute_yield
from src.core.filtering import filter_feasible
from src.core.models import Deposit
from src.core.recommender import recommend
from src.data_access import load_deposits, load_golden, request_from_dict
from src.monitoring import decommission
from src.monitoring.drift import catalog_rate_psi, catalog_summary, freshness_status

# «Сегодня» симуляции отсчитывается от даты каталога, поэтому свежесть всегда ровно
# DRIFT_AGE_DAYS вне зависимости от того, когда был сгенерирован каталог (make data).
DRIFT_AGE_DAYS = 107                 # каталог в проде не обновлялся ~3,5 месяца
SLA_DAYS = 14
# Рынок сместился: ставки вниз (−5 п.п. к ключевой) и промо-надбавка почти исчезла.
MARKET_KEY_RATE = 11.0
MARKET_SHIFT = {"promo": -3.0, "max": 0.5}


def income_optimum(req, deposits: list[Deposit]) -> str | None:
    feasible = filter_feasible(req, deposits)
    if not feasible:
        return None
    return max(feasible, key=lambda d: compute_yield(d, req.amount).total_interest).id


def top1_vs_market(served: list[Deposit], market: list[Deposit], today) -> float:
    """Доля max_income-кейсов, где #1 обслуживающего каталога = оптимум рынка."""
    hits = total = 0
    for g in load_golden():
        if g.get("goal") != "max_income" or g.get("expected_status") != "ok":
            continue
        req = request_from_dict(g)
        true_best = income_optimum(req, market)
        if true_best is None:
            continue
        total += 1
        resp = recommend(req, served, today=today)
        if resp.recommendations and resp.recommendations[0].deposit.id == true_best:
            hits += 1
    return round(hits / total, 4) if total else None


def rate_mae(served: list[Deposit], market: list[Deposit], currency: str = "RUB") -> float:
    m = {d.id: d.nominal_rate for d in market}
    diffs = [abs(d.nominal_rate - m[d.id]) for d in served if d.currency == currency and d.id in m]
    return round(sum(diffs) / len(diffs), 3) if diffs else 0.0


def _snapshot(top1: float, freshness_days: int) -> dict:
    return {
        "ranking_top1": top1,
        "refusal_precision": 1.0,   # guardrails не зависят от рыночных ставок
        "refusal_recall": 1.0,
        "groundedness_share": None,
        "llm_calls": 0,
        "latency_p95_ms": 1.0,
        "freshness_days": freshness_days,
    }


def main() -> None:
    served = load_deposits()                                   # каталог в проде
    served_as_of = max(d.as_of_date for d in served)
    sim_today = served_as_of + timedelta(days=DRIFT_AGE_DAYS)   # каталог давно не трогали
    market = build_market(sim_today)                           # актуальный рынок
    fresh_days_stale = (sim_today - served_as_of).days          # == DRIFT_AGE_DAYS

    psi = catalog_rate_psi(served, market)
    mae = rate_mae(served, market)
    top1_before = top1_vs_market(served, market, sim_today)
    fr_before = freshness_status(fresh_days_stale, SLA_DAYS)
    verdict_before = decommission.evaluate(_snapshot(top1_before, fresh_days_stale))

    # ── Противодействие деградации: обновляем каталог до актуального рынка ──
    refreshed = market
    top1_after = top1_vs_market(refreshed, market, sim_today)
    fr_after = freshness_status(0, SLA_DAYS)
    verdict_after = decommission.evaluate(_snapshot(top1_after, 0))

    report = {
        "sim_today": sim_today.isoformat(),
        "served_catalog": catalog_summary(served),
        "market_catalog": catalog_summary(market),
        "drift": {"rate_psi": psi, "rate_mae_pp": mae},
        "before_refresh": {"top1_vs_market": top1_before, "freshness": fr_before,
                           "verdict": verdict_before["verdict"], "action": verdict_before["action"]},
        "after_refresh": {"top1_vs_market": top1_after, "freshness": fr_after,
                          "verdict": verdict_after["verdict"], "action": verdict_after["action"]},
    }
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    (RUNTIME_DIR / "drift_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("═" * 66)
    print("СИМУЛЯЦИЯ ДРЕЙФА ДАННЫХ И ПРОТИВОДЕЙСТВИЯ ДЕГРАДАЦИИ")
    print("═" * 66)
    print(f"Каталог в проде:  ставки μ={report['served_catalog']['mean_rate']}%, "
          f"as_of={report['served_catalog']['as_of']}")
    print(f"Актуальный рынок: ставки μ={report['market_catalog']['mean_rate']}%, "
          f"as_of={report['market_catalog']['as_of']}")
    print(f"Дрейф ставок:     PSI={psi['psi']} ({psi['band']}), "
          f"MAE по ставке={mae} п.п.")
    print("─" * 66)
    print(f"{'':22}{'ДО обновления':>20}{'ПОСЛЕ обновления':>22}")
    print(f"{'top-1 vs рынок':22}{str(top1_before):>20}{str(top1_after):>22}")
    print(f"{'свежесть, дн.':22}{str(fr_before['freshness_days']):>20}{str(fr_after['freshness_days']):>22}")
    print(f"{'уровень свежести':22}{fr_before['level']:>20}{fr_after['level']:>22}")
    print("─" * 66)
    print(f"ВЕРДИКТ ДО:    {verdict_before['verdict']}")
    print(f"ВЕРДИКТ ПОСЛЕ: {verdict_after['verdict']}")
    print("═" * 66)
    print(f"Отчёт сохранён: {RUNTIME_DIR / 'drift_report.json'}")


def build_market(today) -> list[Deposit]:
    # Импорт здесь, чтобы scripts.catalog_builder не тянулся при импорте модуля как библиотеки.
    from .catalog_builder import build_deposits

    return build_deposits(key_rate=MARKET_KEY_RATE, as_of=today, archetype_shift=MARKET_SHIFT)


if __name__ == "__main__":
    main()
