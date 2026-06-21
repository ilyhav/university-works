"""Формирование объяснения: GigaChat при наличии, иначе детерминированный шаблон.

Контракт безопасности: показанный пользователю текст ВСЕГДА обоснован числами ядра.
Если GigaChat вернул текст с «придуманными» числами (groundedness < порога), ответ
отбраковывается, показывается шаблон, а факт отбраковки фиксируется для мониторинга
деградации качества.
"""

from __future__ import annotations

from ..core.models import AssistantResponse
from . import prompts
from .gigachat_client import GigaChatClient, LLMUnavailable
from .groundedness import check_groundedness

# Принимаем ответ LLM только если ВСЕ числа обоснованы.
ACCEPT_THRESHOLD = 1.0

_GOAL_LABEL = {
    "max_income": "максимальный доход",
    "flexible": "гибкость (пополнение и снятие)",
    "short_term": "короткий срок",
    "capital_protection": "сохранность капитала",
}
_REFUSAL_HINT = {
    "AMOUNT_OUT_OF_RANGE": "Попробуйте увеличить сумму или выбрать другой срок.",
    "TERM_UNAVAILABLE": "Выберите доступный срок или включите «гибкий горизонт».",
    "CONSTRAINTS_UNAVAILABLE": "Снимите часть дополнительных требований.",
    "OUT_OF_SCOPE": "Задайте вопрос строго про вклады и депозиты.",
    "UNREALISTIC_EXPECTATION": "Ориентируйтесь на реальные рыночные ставки.",
    "INVALID_INPUT": "Проверьте корректность введённых чисел.",
    "PROMPT_INJECTION": "Уберите служебные команды из текста запроса.",
}


def _fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def render_template(resp: AssistantResponse) -> str:
    """Детерминированное объяснение — всегда обосновано по построению."""
    req = resp.request
    if resp.status == "refused" and resp.refusal:
        hint = _REFUSAL_HINT.get(resp.refusal.code.value, "")
        return f"{resp.refusal.message} {hint}".strip()

    top = resp.recommendations[0]
    d = top.deposit
    goal = _GOAL_LABEL.get(req.goal, req.goal)
    parts = [
        f"Под ваш запрос (сумма {_fmt(req.amount)} {req.currency}, срок "
        f"{req.term_months} мес., цель — {goal}) подобрано {len(resp.recommendations)} "
        f"вариант(ов).",
        f"Лучший: {d.bank} «{d.product}» — эффективная ставка {top.effective_rate}% "
        f"(номинальная {d.nominal_rate}%), доход за срок {_fmt(top.total_interest)} "
        f"{req.currency}, к концу срока {_fmt(top.future_value)} {req.currency}.",
    ]
    if top.risk_flags:
        parts.append("На что обратить внимание: " + "; ".join(top.risk_flags) + ".")
    if top.uninsured_amount > 0:
        parts.append(
            f"Сумма превышает лимит страхования АСВ: не застраховано "
            f"{_fmt(top.uninsured_amount)} {req.currency} — возможно, стоит разнести "
            "вклад по разным банкам."
        )
    parts.append("Остальные варианты и их условия — в сравнительной таблице ниже.")
    return " ".join(parts)


def explain(resp: AssistantResponse, client: GigaChatClient | None = None) -> AssistantResponse:
    """Заполнить explanation / llm_used / groundedness у ответа ядра."""
    template = render_template(resp)

    if client is None or not client.available:
        resp.explanation = template
        resp.llm_used = False
        resp.groundedness = {"source": "offline_template", "score": 1.0}
        return resp

    try:
        text = client.chat(prompts.SYSTEM_PROMPT, prompts.build_user_prompt(resp))
    except LLMUnavailable:
        resp.explanation = template
        resp.llm_used = False
        resp.groundedness = {"source": "offline_template", "score": 1.0}
        return resp

    g = check_groundedness(text, resp)
    if g["score"] >= ACCEPT_THRESHOLD:
        resp.explanation = text
        resp.llm_used = True
        resp.groundedness = {"source": "gigachat", **g}
    else:
        # Галлюцинация поймана — показываем шаблон, но фиксируем инцидент.
        resp.explanation = template
        resp.llm_used = False
        resp.groundedness = {"source": "gigachat_rejected", **g}
    return resp
