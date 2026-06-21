"""Проверка обоснованности (groundedness) текста LLM относительно данных ядра.

Идея: собрать закрытый список чисел, которые посчитало ядро, затем извлечь все
числа из текста модели и проверить, что каждое из них соответствует разрешённому
(с допуском на округление). Любое «новое» число — признак галлюцинации.
"""

from __future__ import annotations

import re

from ..config import ASV_INSURANCE_LIMIT
from ..core.models import AssistantResponse

# Любые виды пробелов (включая NBSP / narrow NBSP) приводим к обычному пробелу.
_SPACES_RE = re.compile(r"[   \s]")
# Число с разделителями разрядов (пробел) либо обычное, опц. единица измерения.
_NUM_RE = re.compile(
    r"(\d{1,3}(?: \d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*"
    r"(млн|миллион\w*|тыс\w*|%)?",
    re.IGNORECASE,
)


def allowed_numbers(resp: AssistantResponse) -> list[float]:
    """Все числа, которые ассистенту разрешено называть (из результата ядра)."""
    vals: set[float] = {ASV_INSURANCE_LIMIT, 1.4, len(resp.recommendations)}
    req = resp.request
    vals.update({req.amount, float(req.term_months)})
    if req.expected_rate is not None:
        vals.add(req.expected_rate)
    if resp.freshness_days is not None:
        vals.add(float(resp.freshness_days))
    for s in resp.recommendations:
        vals.update(
            {
                s.deposit.nominal_rate,
                s.effective_rate,
                s.total_interest,
                s.future_value,
                s.insured_amount,
                s.uninsured_amount,
                float(s.rank),
                float(s.deposit.term_months),
                s.deposit.min_amount,
            }
        )
        if s.deposit.max_amount is not None:
            vals.add(s.deposit.max_amount)
    return [v for v in vals if v is not None]


def _structural_numbers(resp: AssistantResponse) -> set[int]:
    """Целые, которые модель может называть как «структурные»: сроки, ранги, кол-во.

    В отличие от ставок/сумм эти значения не несут риска дезинформации. Но процент
    (значение с «%») структурным НЕ считается — он обязан совпасть со ставкой ядра.
    """
    nums: set[int] = {len(resp.recommendations), resp.request.term_months}
    for s in resp.recommendations:
        nums.add(s.rank)
        nums.add(s.deposit.term_months)
    return nums


def _normalize_spaces(text: str) -> str:
    """Схлопнуть любые пробелы в обычный, чтобы «194 441» парсилось как одно число."""
    return _SPACES_RE.sub(" ", text)


def _extract_numbers(text: str) -> list[tuple[float, bool]]:
    """Список (значение, это_процент). Масштаб «млн»/«тыс» учитывается."""
    text = _normalize_spaces(text)
    out: list[tuple[float, bool]] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(1).replace(" ", "").replace(",", ".")
        try:
            num = float(raw)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        is_percent = unit == "%"
        if unit.startswith(("млн", "миллион")):
            num *= 1_000_000
        elif unit.startswith("тыс"):
            num *= 1_000
        out.append((num, is_percent))
    return out


def _is_grounded(value: float, is_percent: bool, allowed: list[float], structural: set[int]) -> bool:
    # Процент обязан совпасть с разрешённой ставкой — структурного послабления нет.
    if is_percent:
        return any(abs(value - a) <= max(0.05, 0.01 * abs(a)) for a in allowed)
    # Структурные целые (срок/ранг/количество) — без риска дезинформации.
    if float(value).is_integer() and int(value) in structural:
        return True
    return any(abs(value - a) <= max(0.5, 0.01 * abs(a)) for a in allowed)


def check_groundedness(text: str, resp: AssistantResponse) -> dict:
    """Вернуть {'score', 'total_numbers', 'ungrounded': [...]}.

    score == 1.0 → все числа обоснованы. Меньше → есть «придуманные» цифры.
    """
    allowed = allowed_numbers(resp)
    structural = _structural_numbers(resp)
    numbers = _extract_numbers(text)
    if not numbers:
        return {"score": 1.0, "total_numbers": 0, "ungrounded": []}
    ungrounded = [
        round(v, 3) for v, is_pct in numbers if not _is_grounded(v, is_pct, allowed, structural)
    ]
    score = 1.0 - len(ungrounded) / len(numbers)
    return {"score": round(score, 4), "total_numbers": len(numbers), "ungrounded": ungrounded}
