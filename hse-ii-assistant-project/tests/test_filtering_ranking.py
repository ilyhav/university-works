"""Тесты фильтрации и ранжирования."""

from __future__ import annotations

from datetime import date

from src.core.effective_rate import compute_yield
from src.core.filtering import feasible, filter_feasible
from src.core.models import ClientRequest, Deposit
from src.core.ranking import rank


def _req(**kw) -> ClientRequest:
    base = dict(amount=300_000, term_months=12, goal="max_income")
    base.update(kw)
    return ClientRequest(**base)


def test_filter_respects_hard_constraints(deposits):
    req = _req(need_replenishment=True)
    out = filter_feasible(req, deposits)
    assert out, "должны быть пополняемые вклады на 12 мес."
    assert all(d.replenishment for d in out)
    assert all(d.term_months == 12 for d in out)
    assert all(feasible(req, d) for d in out)


def test_amount_range_enforced(deposits):
    req = _req(amount=1_000)  # ниже минимума почти всех вкладов
    out = filter_feasible(req, deposits)
    assert all(d.min_amount <= 1_000 for d in out)


def test_max_income_ranks_income_optimum_first(deposits):
    req = _req(amount=500_000, term_months=12)
    feasibles = filter_feasible(req, deposits)
    ranked = rank(req, feasibles)
    income_opt = max(feasibles, key=lambda d: compute_yield(d, req.amount).total_interest)
    assert ranked[0].deposit.id == income_opt.id


def test_ranks_are_sequential(deposits):
    ranked = rank(_req(), filter_feasible(_req(), deposits))
    assert [s.rank for s in ranked] == list(range(1, len(ranked) + 1))


def test_flexible_goal_prefers_flexible_product(deposits):
    req = _req(goal="flexible", need_replenishment=True, need_withdrawal=True, amount=200_000)
    ranked = rank(req, filter_feasible(req, deposits))
    assert ranked[0].deposit.replenishment and ranked[0].deposit.partial_withdrawal


def test_uninsured_flag_above_asv_limit(deposits):
    req = _req(amount=2_000_000, goal="capital_protection")
    ranked = rank(req, filter_feasible(req, deposits))
    top = ranked[0]
    assert top.uninsured_amount == 600_000  # 2.0M - 1.4M
    assert any("АСВ" in f for f in top.risk_flags)


def _synthetic(did: str, rate: float, promo: bool) -> Deposit:
    return Deposit(id=did, bank="B", product=did, nominal_rate=rate, term_months=12,
                   min_amount=0, capitalization="monthly", promo=promo,
                   as_of_date=date(2026, 6, 5))


def test_max_income_invariant_holds_above_asv_limit():
    """top-1 для max_income == доходный оптимум даже когда промо-вклад выгоднее."""
    high = _synthetic("HIGH", 16.10, promo=True)   # выше доход
    low = _synthetic("LOW", 16.00, promo=False)    # чуть ниже доход, но «надёжнее»
    req = ClientRequest(amount=28_000_000, term_months=12, goal="max_income")
    ranked = rank(req, [low, high])
    assert ranked[0].deposit.id == "HIGH"


def test_asv_not_applied_to_foreign_currency(deposits):
    req = _req(currency="USD", amount=2_000_000, term_months=12)
    ranked = rank(req, filter_feasible(req, deposits))
    assert ranked, "должны быть валютные вклады на 12 мес."
    top = ranked[0]
    assert top.uninsured_amount == 0.0
    assert not any("АСВ" in f for f in top.risk_flags)
