"""Тесты проверки обоснованности (anti-hallucination)."""

from __future__ import annotations

from src.core.models import ClientRequest
from src.core.recommender import recommend
from src.llm.explainer import render_template
from src.llm.groundedness import check_groundedness


def _resp(deposits, today):
    req = ClientRequest(amount=300_000, term_months=12, goal="max_income")
    return recommend(req, deposits, today=today)


def test_template_is_fully_grounded(deposits, today):
    resp = _resp(deposits, today)
    text = render_template(resp)
    g = check_groundedness(text, resp)
    assert g["score"] == 1.0, f"шаблон содержит необоснованные числа: {g['ungrounded']}"


def test_detects_invented_rate(deposits, today):
    resp = _resp(deposits, today)
    text = "Лучший вклад даёт эффективную ставку 99.7% годовых — отличное предложение!"
    g = check_groundedness(text, resp)
    assert g["score"] < 1.0
    assert any(abs(v - 99.7) < 0.01 for v in g["ungrounded"])


def test_detects_invented_income(deposits, today):
    resp = _resp(deposits, today)
    text = "Вы заработаете 777 777 рублей за год."
    g = check_groundedness(text, resp)
    assert g["score"] < 1.0


def test_no_numbers_is_grounded(deposits, today):
    resp = _resp(deposits, today)
    g = check_groundedness("Это надёжный вклад с понятными условиями.", resp)
    assert g["score"] == 1.0
    assert g["total_numbers"] == 0


def test_rounding_within_tolerance_ok(deposits, today):
    resp = _resp(deposits, today)
    eff = resp.recommendations[0].effective_rate  # напр. 19.444
    text = f"Эффективная ставка около {eff:.1f}% годовых."
    g = check_groundedness(text, resp)
    assert g["score"] == 1.0
