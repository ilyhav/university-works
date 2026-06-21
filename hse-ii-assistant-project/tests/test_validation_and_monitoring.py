"""Тесты валидации на golden-наборе, дрейфа и критериев вывода модели."""

from __future__ import annotations

from datetime import date

from scripts.catalog_builder import build_deposits
from src.monitoring import decommission
from src.monitoring.drift import catalog_rate_psi, freshness_status, psi
from src.validation.golden import evaluate_golden


def test_golden_all_pass(deposits, today):
    report = evaluate_golden(deposits, today=today)
    assert report["pass_rate"] == 1.0, report["cases"]
    assert report["ranking"]["top1_accuracy"] == 1.0
    assert report["refusal"]["precision"] == 1.0
    assert report["refusal"]["recall"] == 1.0
    assert report["refusal"]["code_accuracy"] == 1.0


def test_psi_zero_for_identical_distributions():
    xs = [float(i) for i in range(100)]
    assert psi(xs, xs) < 1e-6


def test_psi_detects_shift():
    base = [float(i) for i in range(100)]
    shifted = [x + 50 for x in base]
    assert psi(base, shifted) > 0.25  # значимый дрейф


def test_catalog_drift_significant_after_rate_cut():
    base = build_deposits(key_rate=16.0, as_of=date(2026, 6, 5))
    market = build_deposits(key_rate=11.0, as_of=date(2026, 9, 20),
                            archetype_shift={"promo": -3.0, "max": 0.5})
    res = catalog_rate_psi(base, market)
    assert res["psi"] > 0.25
    assert res["band"] == "значимый дрейф"
    assert res["baseline_mean"] > res["current_mean"]


def test_freshness_levels():
    assert freshness_status(5, 14)["level"] == "ok"
    assert freshness_status(20, 14)["level"] == "warning"
    assert freshness_status(40, 14)["level"] == "critical"


def test_decommission_ok_when_healthy():
    v = decommission.evaluate({
        "ranking_top1": 1.0, "refusal_precision": 1.0, "refusal_recall": 1.0,
        "groundedness_share": None, "llm_calls": 0, "latency_p95_ms": 5.0,
        "freshness_days": 8,
    })
    assert v["action"] == "NONE"


def test_decommission_refresh_when_stale():
    v = decommission.evaluate({
        "ranking_top1": 0.8, "refusal_precision": 1.0, "refusal_recall": 1.0,
        "groundedness_share": None, "llm_calls": 0, "latency_p95_ms": 5.0,
        "freshness_days": 107,
    })
    assert v["action"] == "REFRESH"


def test_decommission_llm_off_on_hallucinations():
    v = decommission.evaluate({
        "ranking_top1": 1.0, "refusal_precision": 1.0, "refusal_recall": 1.0,
        "groundedness_share": 0.5, "llm_calls": 40, "latency_p95_ms": 5.0,
        "freshness_days": 8,
    })
    assert v["action"] == "LLM_OFF"


def test_decommission_full_on_ranking_collapse():
    v = decommission.evaluate({
        "ranking_top1": 0.5, "refusal_precision": 1.0, "refusal_recall": 1.0,
        "groundedness_share": None, "llm_calls": 0, "latency_p95_ms": 5.0,
        "freshness_days": 8,
    })
    assert v["action"] == "DECOMMISSION"
