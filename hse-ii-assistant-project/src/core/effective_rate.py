"""Расчёт эффективной доходности вклада с учётом капитализации.

Это «скрытое условие» №1, которое путает клиентов: при одинаковой номинальной
ставке вклад с ежемесячной капитализацией приносит больше, чем вклад с выплатой
в конце. Ядро считает это детерминированно, а LLM лишь пересказывает результат.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Deposit


@dataclass(frozen=True)
class Yield:
    total_interest: float  # доход за срок вклада, ₽
    future_value: float  # сумма к концу срока, ₽
    effective_rate: float  # эффективная годовая ставка (APY), %


def _periods_per_year(capitalization: str) -> int | None:
    """Сколько раз в год капитализируется процент. None — без капитализации."""
    return {"monthly": 12, "quarterly": 4}.get(capitalization)


def compute_yield(deposit: Deposit, amount: float) -> Yield:
    """Доход и эффективная ставка для суммы `amount` на собственном сроке вклада.

    Без капитализации (или при ежемесячной выплате процентов) проценты не
    реинвестируются → простой процент, эффективная ставка равна номинальной.
    С капитализацией проценты добавляются к телу вклада → сложный процент.
    """
    if deposit.term_months <= 0 or amount <= 0:
        raise ValueError("term_months и amount должны быть положительными")
    r = deposit.nominal_rate / 100.0
    term_years = deposit.term_months / 12.0
    n = _periods_per_year(deposit.capitalization)

    if n is None or deposit.payout == "monthly":
        # Простой процент: проценты не присоединяются к телу вклада.
        total_interest = amount * r * term_years
        future_value = amount + total_interest
        effective_rate = deposit.nominal_rate
    else:
        # Сложный процент: i — ставка за период капитализации, k — число периодов.
        i = r / n
        k = deposit.term_months / (12 / n)
        future_value = amount * (1 + i) ** k
        total_interest = future_value - amount
        # APY приводим к годовому виду через фактический срок вклада.
        effective_rate = ((future_value / amount) ** (1 / term_years) - 1) * 100.0

    return Yield(
        total_interest=round(total_interest, 2),
        future_value=round(future_value, 2),
        effective_rate=round(effective_rate, 3),
    )
