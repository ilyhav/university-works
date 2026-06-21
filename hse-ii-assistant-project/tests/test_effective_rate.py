"""Тесты расчёта эффективной доходности — формулы должны быть точны."""

from __future__ import annotations

from datetime import date

import pytest

from src.core.effective_rate import compute_yield
from src.core.models import Deposit


def _dep(**kw) -> Deposit:
    base = dict(
        id="X", bank="B", product="P", nominal_rate=12.0, term_months=12,
        min_amount=0, capitalization="none", payout="at_end",
        as_of_date=date(2026, 6, 5),
    )
    base.update(kw)
    return Deposit(**base)


def test_simple_interest_no_capitalization():
    y = compute_yield(_dep(nominal_rate=12.0, term_months=12, capitalization="none"), 100_000)
    assert y.total_interest == pytest.approx(12_000, abs=1)
    assert y.effective_rate == pytest.approx(12.0, abs=1e-6)
    assert y.future_value == pytest.approx(112_000, abs=1)


def test_monthly_capitalization_beats_nominal():
    y = compute_yield(_dep(nominal_rate=12.0, term_months=12, capitalization="monthly"), 100_000)
    # (1 + 0.12/12)^12 - 1 ≈ 12.6825 %
    assert y.effective_rate == pytest.approx(12.6825, abs=1e-3)
    assert y.total_interest > 12_000


def test_monthly_payout_disables_compounding():
    """Ежемесячная ВЫПЛАТА процентов → без капитализации, эффективная = номинальной."""
    y = compute_yield(
        _dep(nominal_rate=12.0, term_months=12, capitalization="monthly", payout="monthly"),
        100_000,
    )
    assert y.effective_rate == pytest.approx(12.0, abs=1e-6)


def test_quarterly_capitalization():
    y = compute_yield(_dep(nominal_rate=12.0, term_months=12, capitalization="quarterly"), 100_000)
    # (1 + 0.12/4)^4 - 1 = 12.55 %
    assert y.effective_rate == pytest.approx(12.5509, abs=1e-3)


def test_half_year_term_simple():
    y = compute_yield(_dep(nominal_rate=10.0, term_months=6, capitalization="none"), 200_000)
    assert y.total_interest == pytest.approx(10_000, abs=1)  # 200k * 10% * 0.5


def test_quarterly_capitalization_non_multiple_of_three():
    """Закрепляем соглашение о дробном числе периодов при сроке не кратном кварталу."""
    y = compute_yield(_dep(nominal_rate=12.0, term_months=5, capitalization="quarterly"), 100_000)
    # k = 5 / (12/4) = 1.6667 периода; FV = 100000 * (1+0.03)^1.6667 ≈ 105 049.8
    assert y.future_value == pytest.approx(105_049.8, abs=5)
    assert y.total_interest > 0


def test_non_positive_term_raises():
    with pytest.raises(ValueError):
        compute_yield(_dep(term_months=0), 100_000)


def test_non_positive_amount_raises():
    with pytest.raises(ValueError):
        compute_yield(_dep(term_months=12), 0)
