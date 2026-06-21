"""Guardrails: когда ассистент обязан отказаться или предупредить.

Корректный отказ — отдельный измеримый класс ответа (criterion: «случаи, когда
ассистент корректно отказывается отвечать»). Все проверки детерминированы и
покрыты тестами, поэтому модельный риск ложной выдачи контролируем.
"""

from __future__ import annotations

import math
import re

from .filtering import amount_matches, term_matches
from .models import ClientRequest, Deposit, Refusal, RefusalCode

# Темы вне компетенции модуля «подбор вклада». Ассистент не даёт инвест-советы,
# не подбирает кредиты/акции/крипту и не консультирует вне финансовых продуктов.
_OUT_OF_SCOPE_PATTERNS = [
    r"\bакци[ийяю]\b", r"\bоблигац", r"\bкрипт", r"\bбитко", r"\bфорекс\b",
    r"\bинвестиц", r"\bпортфел", r"\bипотек", r"\bкредит", r"\bзайм", r"\bрассрочк",
    r"\bстраховк", r"\bполис\b", r"\bпиф\b", r"\bетф\b", r"\betf\b",
    r"\bнедвижим", r"\bметалл", r"\bвалют.{0,12}куп", r"\bкуда вложить",
    r"\bпосоветуй\s+(?:акци|облигац|фонд)", r"\bдоговор\b.*\bрасторг",
]

# Темы, которые подтверждают релевантность вопроса (вклады/депозиты).
_IN_SCOPE_HINTS = [
    "вклад", "депозит", "ставк", "процент", "капитализац", "пополнен",
    "снят", "срок", "доход", "асв", "страхов", "вкладонавигатор",
]

# Грубые маркеры prompt injection в свободном тексте пользователя.
_INJECTION_PATTERNS = [
    r"ignore (?:all )?previous", r"забудь (?:все )?(?:предыдущие|инструкции)",
    r"system prompt", r"ты теперь", r"act as", r"переключись в режим",
    r"разглас", r"выведи (?:свой )?промпт", r"disregard",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def check_input(request: ClientRequest) -> Refusal | None:
    """Базовая валидация входа: суммы/сроки должны быть конечны и положительны."""
    if not math.isfinite(request.amount) or request.amount <= 0:
        return Refusal(
            code=RefusalCode.INVALID_INPUT,
            message="Сумма вклада должна быть положительным числом.",
        )
    if request.term_months <= 0 or request.term_months > 120:
        return Refusal(
            code=RefusalCode.INVALID_INPUT,
            message="Срок вклада должен быть в диапазоне 1–120 месяцев.",
        )
    if request.expected_rate is not None and (
        not math.isfinite(request.expected_rate) or request.expected_rate < 0
    ):
        return Refusal(
            code=RefusalCode.INVALID_INPUT,
            message="Ожидаемая ставка должна быть неотрицательным числом.",
        )
    return None


def check_scope(request: ClientRequest) -> Refusal | None:
    """Отказ, если свободный вопрос пользователя выходит за рамки подбора вклада."""
    text = (request.free_text_question or "").strip()
    if not text:
        return None
    if _matches_any(text, _INJECTION_PATTERNS):
        return Refusal(
            code=RefusalCode.PROMPT_INJECTION,
            message=(
                "Запрос содержит попытку переопределить инструкции ассистента. "
                "Я отвечаю только на вопросы о подборе вкладов."
            ),
        )
    out_of_scope = _matches_any(text, _OUT_OF_SCOPE_PATTERNS)
    in_scope = any(h in text.lower() for h in _IN_SCOPE_HINTS)
    if out_of_scope and not in_scope:
        return Refusal(
            code=RefusalCode.OUT_OF_SCOPE,
            message=(
                "Я помогаю только с подбором банковских вкладов и депозитов. "
                "Инвестиционные, кредитные и страховые продукты — вне моей компетенции; "
                "по ним лучше обратиться к профильному консультанту."
            ),
        )
    return None


def check_expectation(request: ClientRequest, deposits: list[Deposit]) -> Refusal | None:
    """Анти-мисселинг: не обещаем доходность выше рынка.

    Если клиент ждёт ставку существенно выше максимальной на рынке — честно
    отказываемся «подтвердить» такую доходность, а не подбираем под завышенное
    ожидание. Это снижает регуляторный риск (мисселинг) и риск разочарования.
    """
    if request.expected_rate is None or not deposits:
        return None
    same_currency = [d for d in deposits if d.currency == request.currency]
    if not same_currency:
        return None
    market_max = max(d.nominal_rate for d in same_currency)
    if request.expected_rate > market_max + 3.0:
        return Refusal(
            code=RefusalCode.UNREALISTIC_EXPECTATION,
            message=(
                f"Ставку {request.expected_rate:.1f}% годовых по вкладу в "
                f"{request.currency} сейчас предложить нельзя: максимум на рынке "
                f"учебного каталога — {market_max:.1f}%. Обещания доходности выше "
                "рыночной — признак недобросовестного предложения."
            ),
        )
    return None


def classify_infeasible(request: ClientRequest, deposits: list[Deposit]) -> Refusal:
    """Когда после фильтрации не осталось ни одного вклада — объяснить почему.

    Разбираем причину (сумма / срок / валюта / ограничения), чтобы отказ был
    конкретным и полезным, а не «ничего не найдено».
    """
    cur = [d for d in deposits if d.currency == request.currency]
    if not cur:
        return Refusal(
            code=RefusalCode.CURRENCY_UNAVAILABLE,
            message=f"В каталоге нет вкладов в валюте {request.currency}.",
        )

    by_term = [d for d in cur if term_matches(request, d)]
    if not by_term:
        terms = sorted({d.term_months for d in cur})
        return Refusal(
            code=RefusalCode.TERM_UNAVAILABLE,
            message=(
                f"Нет вкладов на срок {request.term_months} мес. "
                f"Доступные сроки: {', '.join(map(str, terms))} мес. "
                "Можно включить «гибкий горизонт», чтобы рассмотреть близкие сроки."
            ),
        )

    by_amount = [d for d in by_term if amount_matches(request, d)]
    if not by_amount:
        cap = [d.max_amount for d in by_term if d.max_amount is not None]
        too_high = bool(cap) and request.amount > max(cap)
        if too_high:
            hi = max(cap)
            tail = f"максимальная сумма среди доступных вкладов — до {hi:,.0f}."
        else:
            lo = min(d.min_amount for d in by_term)
            tail = f"минимальный взнос среди доступных вкладов — от {lo:,.0f}."
        return Refusal(
            code=RefusalCode.AMOUNT_OUT_OF_RANGE,
            message=(
                f"Сумма {request.amount:,.0f} {request.currency} не подходит под "
                f"выбранные условия: {tail}"
            ).replace(",", " "),
        )

    needs = []
    if request.need_replenishment:
        needs.append("пополнение")
    if request.need_withdrawal:
        needs.append("частичное снятие")
    return Refusal(
        code=RefusalCode.CONSTRAINTS_UNAVAILABLE,
        message=(
            "Под выбранные дополнительные условия ("
            + (", ".join(needs) if needs else "ограничения")
            + ") подходящих вкладов нет. Попробуйте снять часть требований."
        ),
    )
