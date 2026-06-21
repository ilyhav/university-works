"""Скоринг и ранжирование подходящих вкладов под цель клиента.

Ранжирование работает только над уже отфильтрованными (выполнимыми) вкладами и
смешивает доходность, гибкость и защищённость с весами, зависящими от цели. Веса
вынесены в GOAL_WEIGHTS, чтобы их можно было версионировать как «конфиг модели».
"""

from __future__ import annotations

from ..config import ASV_INSURANCE_LIMIT
from .effective_rate import compute_yield
from .models import ClientRequest, Deposit, ScoredDeposit

# Веса композитного скоринга по целям клиента. Это и есть «обучаемая» часть модели:
# конфиг ранжирования версионируется и валидируется на golden-наборе (см. validation).
GOAL_WEIGHTS: dict[str, dict[str, float]] = {
    # Для цели «максимальный доход» ранжируем строго по доходу — это гарантирует
    # инвариант «top-1 == доходный оптимум» и исключает переупорядочивание гибкостью.
    "max_income": {"income": 1.0, "flex": 0.0, "protection": 0.0, "liquidity": 0.0},
    "flexible": {"income": 0.45, "flex": 0.45, "protection": 0.10, "liquidity": 0.0},
    "short_term": {"income": 0.45, "flex": 0.15, "protection": 0.0, "liquidity": 0.40},
    "capital_protection": {"income": 0.40, "flex": 0.15, "protection": 0.45, "liquidity": 0.0},
}

# Версия конфигурации скоринга — пишется в лог каждого ответа (управление моделью).
RANKING_VERSION = "ranking-1.0.0"


def _flex_score(d: Deposit) -> float:
    parts = [d.replenishment, d.partial_withdrawal, d.early_termination == "penalty_free"]
    return sum(parts) / len(parts)


def _liquidity_score(d: Deposit) -> float:
    return 1.0 if d.payout == "monthly" or d.partial_withdrawal else 0.0


def _risk_flags(d: Deposit, amount: float, uninsured: float) -> list[str]:
    flags: list[str] = []
    if uninsured > 0:
        flags.append(
            f"Сумма выше лимита АСВ 1,4 млн ₽: застраховано "
            f"{ASV_INSURANCE_LIMIT:,.0f} ₽, не застраховано {uninsured:,.0f} ₽".replace(",", " ")
        )
    if d.early_termination == "loss_of_interest":
        flags.append("При досрочном закрытии проценты сгорают")
    elif d.early_termination == "reduced_rate":
        flags.append("При досрочном закрытии ставка падает до ~0,01%")
    if d.capitalization == "none" and d.payout == "at_end":
        flags.append("Без капитализации: проценты не присоединяются к телу вклада")
    if d.promo:
        flags.append("Акционная ставка — действует при выполнении условий (см. примечания)")
    if d.online_only:
        flags.append("Оформление только онлайн")
    if not d.replenishment:
        flags.append("Пополнение не предусмотрено")
    return flags


def rank(request: ClientRequest, deposits: list[Deposit]) -> list[ScoredDeposit]:
    """Вернуть выполнимые вклады, отсортированные по композитному скору (убыв.)."""
    if not deposits:
        return []

    weights = GOAL_WEIGHTS[request.goal]
    yields = {d.id: compute_yield(d, request.amount) for d in deposits}
    max_interest = max(y.total_interest for y in yields.values()) or 1.0

    scored: list[ScoredDeposit] = []
    for d in deposits:
        y = yields[d.id]
        if request.currency == "RUB":
            insured = min(request.amount, ASV_INSURANCE_LIMIT)
            uninsured = max(0.0, request.amount - ASV_INSURANCE_LIMIT)
            protection = insured / request.amount if request.amount else 0.0
        else:
            # АСВ страхует вклады в рамках рублёвого лимита; к валютным здесь не применяем,
            # чтобы не сравнивать сумму в USD/CNY с лимитом 1,4 млн ₽ как с рублёвой.
            insured = uninsured = 0.0
            protection = 1.0
        if not d.promo:
            protection = min(1.0, protection + 0.05)  # без акционных условий надёжнее

        score = (
            weights["income"] * (y.total_interest / max_interest)
            + weights["flex"] * _flex_score(d)
            + weights["protection"] * protection
            + weights["liquidity"] * _liquidity_score(d)
        )
        scored.append(
            ScoredDeposit(
                deposit=d,
                rank=0,
                effective_rate=y.effective_rate,
                total_interest=y.total_interest,
                future_value=y.future_value,
                score=round(score, 6),
                insured_amount=round(insured, 2),
                uninsured_amount=round(uninsured, 2),
                risk_flags=_risk_flags(d, request.amount, uninsured),
            )
        )

    # Стабильная сортировка: при равном скоре выше доход, затем меньший min_amount.
    scored.sort(key=lambda s: (-s.score, -s.total_interest, s.deposit.min_amount))
    for i, s in enumerate(scored, start=1):
        s.rank = i
    return scored
