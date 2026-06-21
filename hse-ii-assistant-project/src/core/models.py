"""Доменные модели: продукт-вклад, запрос клиента, рекомендация, ответ ассистента."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Capitalization = Literal["none", "monthly", "quarterly"]
Payout = Literal["at_end", "monthly"]
EarlyTermination = Literal["loss_of_interest", "reduced_rate", "penalty_free"]
Goal = Literal["max_income", "flexible", "short_term", "capital_protection"]


class Deposit(BaseModel):
    """Один вклад из учебного каталога."""

    id: str
    bank: str
    product: str
    currency: str = "RUB"
    nominal_rate: float = Field(..., description="Номинальная годовая ставка, %")
    term_months: int
    min_amount: float
    max_amount: float | None = None
    capitalization: Capitalization = "none"
    payout: Payout = "at_end"
    replenishment: bool = False
    partial_withdrawal: bool = False
    early_termination: EarlyTermination = "loss_of_interest"
    online_only: bool = False
    promo: bool = False
    as_of_date: date
    notes: str = ""

    @field_validator("nominal_rate")
    @classmethod
    def _rate_sane(cls, v: float) -> float:
        if not (0 < v < 100):
            raise ValueError(f"подозрительная ставка {v}")
        return v


class ClientRequest(BaseModel):
    """Параметры, которые задаёт пользователь в веб-форме."""

    amount: float
    term_months: int
    currency: str = "RUB"
    need_replenishment: bool = False
    need_withdrawal: bool = False
    goal: Goal = "max_income"
    horizon_flexible: bool = False
    expected_rate: float | None = Field(
        default=None, description="Ожидаемая клиентом ставка, % (для анти-мисселинг-проверки)"
    )
    free_text_question: str | None = None


class RefusalCode(str, Enum):
    """Машиночитаемые причины отказа — для метрик и мониторинга."""

    INVALID_INPUT = "INVALID_INPUT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNREALISTIC_EXPECTATION = "UNREALISTIC_EXPECTATION"
    AMOUNT_OUT_OF_RANGE = "AMOUNT_OUT_OF_RANGE"
    TERM_UNAVAILABLE = "TERM_UNAVAILABLE"
    CURRENCY_UNAVAILABLE = "CURRENCY_UNAVAILABLE"
    CONSTRAINTS_UNAVAILABLE = "CONSTRAINTS_UNAVAILABLE"
    PROMPT_INJECTION = "PROMPT_INJECTION"


class Refusal(BaseModel):
    code: RefusalCode
    message: str


class ScoredDeposit(BaseModel):
    """Вклад с рассчитанными показателями и местом в выдаче."""

    deposit: Deposit
    rank: int
    effective_rate: float = Field(..., description="Эффективная годовая ставка с учётом капитализации, %")
    total_interest: float = Field(..., description="Доход за срок вклада, ₽")
    future_value: float = Field(..., description="Сумма к концу срока, ₽")
    score: float
    insured_amount: float
    uninsured_amount: float
    risk_flags: list[str] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    """Итоговый ответ ассистента — то, что видит пользователь и пишется в лог."""

    status: Literal["ok", "refused"]
    request: ClientRequest
    recommendations: list[ScoredDeposit] = Field(default_factory=list)
    refusal: Refusal | None = None
    explanation: str = ""
    llm_used: bool = False
    groundedness: dict = Field(default_factory=dict)
    catalog_as_of: date | None = None
    freshness_days: int | None = None
    latency_ms: float | None = None
