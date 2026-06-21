"""Загрузка учебного каталога вкладов, клиентских сценариев и golden-набора."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from .config import CATALOG_PATH, GOLDEN_PATH, SCENARIOS_PATH
from .core.models import ClientRequest, Deposit

_BOOL_TRUE = {"true", "1", "yes", "да"}


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in _BOOL_TRUE


def _to_float_or_none(v: str) -> float | None:
    v = (v or "").strip()
    return float(v) if v else None


def load_deposits(path: Path = CATALOG_PATH) -> list[Deposit]:
    """Прочитать каталог вкладов из CSV в валидированные модели Deposit."""
    deposits: list[Deposit] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            deposits.append(
                Deposit(
                    id=row["id"],
                    bank=row["bank"],
                    product=row["product"],
                    currency=row["currency"],
                    nominal_rate=float(row["nominal_rate"]),
                    term_months=int(row["term_months"]),
                    min_amount=float(row["min_amount"]),
                    max_amount=_to_float_or_none(row["max_amount"]),
                    capitalization=row["capitalization"],
                    payout=row["payout"],
                    replenishment=_to_bool(row["replenishment"]),
                    partial_withdrawal=_to_bool(row["partial_withdrawal"]),
                    early_termination=row["early_termination"],
                    online_only=_to_bool(row["online_only"]),
                    promo=_to_bool(row["promo"]),
                    as_of_date=date.fromisoformat(row["as_of_date"]),
                    notes=row.get("notes", ""),
                )
            )
    return deposits


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict]:
    """Типовые клиентские сценарии для демонстрации (человекочитаемые карточки)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """Размеченный golden-набор для валидации качества (вход + ожидаемый исход)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def request_from_dict(d: dict) -> ClientRequest:
    """Собрать ClientRequest из словаря сценария/golden (только релевантные поля)."""
    fields = ClientRequest.model_fields.keys()
    return ClientRequest(**{k: v for k, v in d.items() if k in fields})
