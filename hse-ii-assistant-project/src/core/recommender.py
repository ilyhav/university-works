"""Оркестрация детерминированного ядра: вход → отказ-или-выдача.

Порядок проверок важен: сначала дешёвые guardrails (вход, тема, ожидания), затем
жёсткая фильтрация и ранжирование. Возвращается AssistantResponse БЕЗ объяснения —
текст на естественном языке добавляет LLM-слой (assistant.py).
"""

from __future__ import annotations

import time
from datetime import date

from ..config import TOP_N
from . import guards
from .filtering import filter_feasible
from .models import AssistantResponse, ClientRequest, Deposit
from .ranking import rank


def recommend(
    request: ClientRequest,
    deposits: list[Deposit],
    top_n: int = TOP_N,
    today: date | None = None,
) -> AssistantResponse:
    """Главная функция ядра. Детерминирована и не обращается к сети."""
    started = time.perf_counter()
    today = today or date.today()

    catalog_as_of = max((d.as_of_date for d in deposits), default=None)
    freshness_days = (today - catalog_as_of).days if catalog_as_of else None

    def _finish(resp: AssistantResponse) -> AssistantResponse:
        resp.catalog_as_of = catalog_as_of
        resp.freshness_days = freshness_days
        resp.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return resp

    # 1. Guardrails до обращения к каталогу.
    refusal = (
        guards.check_input(request)
        or guards.check_scope(request)
        or guards.check_expectation(request, deposits)
    )
    if refusal is not None:
        return _finish(
            AssistantResponse(status="refused", request=request, refusal=refusal)
        )

    # 2. Жёсткая фильтрация.
    feasible = filter_feasible(request, deposits)
    if not feasible:
        return _finish(
            AssistantResponse(
                status="refused",
                request=request,
                refusal=guards.classify_infeasible(request, deposits),
            )
        )

    # 3. Ранжирование и срез top-N.
    ranked = rank(request, feasible)
    return _finish(
        AssistantResponse(
            status="ok",
            request=request,
            recommendations=ranked[:top_n],
        )
    )
