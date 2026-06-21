"""Журналирование каждого ответа ассистента в JSONL — основа мониторинга.

Пишем плоскую запись (без персональных данных — только параметры запроса и
метрики ответа), чтобы поверх лога строить дашборд, считать дрейф и проверять
критерии деградации.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import REQUESTS_LOG
from ..core.models import AssistantResponse
from ..core.ranking import RANKING_VERSION


def response_to_record(resp: AssistantResponse, ts: str | None = None) -> dict:
    """Преобразовать ответ в плоскую запись лога."""
    top = resp.recommendations[0] if resp.recommendations else None
    g = resp.groundedness or {}
    return {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "ranking_version": RANKING_VERSION,
        "status": resp.status,
        "refusal_code": resp.refusal.code.value if resp.refusal else None,
        "goal": resp.request.goal,
        "amount": resp.request.amount,
        "term_months": resp.request.term_months,
        "currency": resp.request.currency,
        "need_replenishment": resp.request.need_replenishment,
        "need_withdrawal": resp.request.need_withdrawal,
        "n_recommendations": len(resp.recommendations),
        "top_id": top.deposit.id if top else None,
        "top_effective_rate": top.effective_rate if top else None,
        "top_income": top.total_interest if top else None,
        "llm_used": resp.llm_used,
        "groundedness_source": g.get("source"),
        "groundedness_score": g.get("score"),
        "ungrounded_count": len(g.get("ungrounded", [])),
        "latency_ms": resp.latency_ms,
        "freshness_days": resp.freshness_days,
        "catalog_as_of": resp.catalog_as_of.isoformat() if resp.catalog_as_of else None,
    }


def log_response(resp: AssistantResponse, ts: str | None = None, path: Path = REQUESTS_LOG) -> dict:
    record = response_to_record(resp, ts=ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_records(path: Path = REQUESTS_LOG) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_log_df(path: Path = REQUESTS_LOG) -> pd.DataFrame:
    records = read_records(path)
    return pd.DataFrame(records)
