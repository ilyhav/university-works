"""Генерация демонстрационного трафика в журнал мониторинга.

Запуск:  python -m scripts.demo_traffic
Наполняет data/runtime/requests.jsonl реалистичным потоком запросов (с долей
отказов и лёгким дрейфом сумм в последние дни), чтобы вкладка «Мониторинг» в
веб-форме показывала живые данные даже без обращений к LLM.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from src.config import REQUESTS_LOG
from src.core.models import ClientRequest
from src.core.recommender import recommend
from src.data_access import load_deposits
from src.llm.explainer import explain
from src.monitoring.logging_store import log_response

N = 180
DAYS = 12
GOALS = ["max_income", "max_income", "flexible", "short_term", "capital_protection"]
TERMS = [3, 6, 9, 12, 12, 18, 24, 36]
OUT_OF_SCOPE = [
    "посоветуйте, какие акции купить",
    "что выгоднее — ипотека или аренда?",
    "стоит ли вкладываться в крипту?",
]


def _make_request(rng: random.Random, day_frac: float) -> ClientRequest:
    # Лёгкий дрейф: в последние дни клиенты приходят с большими суммами.
    scale = 1.0 + 0.8 * day_frac
    amount = round(rng.lognormvariate(12.4, 0.6) * scale, -3)
    amount = float(min(max(amount, 5_000), 6_000_000))
    kind = rng.random()
    if kind < 0.06:  # вне компетенции
        return ClientRequest(amount=amount, term_months=rng.choice(TERMS),
                             goal="max_income", free_text_question=rng.choice(OUT_OF_SCOPE))
    if kind < 0.10:  # завышенное ожидание
        return ClientRequest(amount=amount, term_months=rng.choice(TERMS),
                             goal="max_income", expected_rate=rng.choice([35, 40, 50]))
    if kind < 0.13:  # нестандартный срок → отказ
        return ClientRequest(amount=amount, term_months=rng.choice([1, 4, 7, 30]),
                             goal="max_income")
    return ClientRequest(
        amount=amount, term_months=rng.choice(TERMS), goal=rng.choice(GOALS),
        need_replenishment=rng.random() < 0.25, need_withdrawal=rng.random() < 0.15,
        horizon_flexible=rng.random() < 0.2,
    )


def main() -> None:
    rng = random.Random(7)
    deposits = load_deposits()
    if REQUESTS_LOG.exists():
        REQUESTS_LOG.unlink()  # чистый демо-журнал

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=DAYS)
    refused = 0
    for i in range(N):
        day_frac = i / N
        ts = start + timedelta(seconds=day_frac * DAYS * 86400 + rng.uniform(0, 3600))
        req = _make_request(rng, day_frac)
        resp = recommend(req, deposits, today=now.date())
        explain(resp, client=None)  # офлайн-объяснение
        log_response(resp, ts=ts.isoformat())
        refused += int(resp.status == "refused")

    print(f"Записано {N} запросов в {REQUESTS_LOG}")
    print(f"Из них отказов: {refused} ({refused / N:.0%})")


if __name__ == "__main__":
    main()
