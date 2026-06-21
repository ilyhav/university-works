"""Жёсткая фильтрация каталога по неоспоримым условиям запроса.

«Жёсткие» условия (валюта, сумма в диапазоне, срок, требуемые опции) не подлежат
смягчению ранжированием: вклад, который физически не подходит, не должен попасть
в выдачу ни при каком скоринге. Это граница между фильтрацией и ранжированием.
"""

from __future__ import annotations

from .models import ClientRequest, Deposit

TERM_TOLERANCE_MONTHS = 3


def term_matches(request: ClientRequest, d: Deposit) -> bool:
    if request.horizon_flexible:
        return abs(d.term_months - request.term_months) <= TERM_TOLERANCE_MONTHS
    return d.term_months == request.term_months


def amount_matches(request: ClientRequest, d: Deposit) -> bool:
    if request.amount < d.min_amount:
        return False
    if d.max_amount is not None and request.amount > d.max_amount:
        return False
    return True


def feasible(request: ClientRequest, d: Deposit) -> bool:
    """Все жёсткие условия выполнены одновременно."""
    if d.currency != request.currency:
        return False
    if not term_matches(request, d):
        return False
    if not amount_matches(request, d):
        return False
    if request.need_replenishment and not d.replenishment:
        return False
    if request.need_withdrawal and not d.partial_withdrawal:
        return False
    return True


def filter_feasible(request: ClientRequest, deposits: list[Deposit]) -> list[Deposit]:
    return [d for d in deposits if feasible(request, d)]
