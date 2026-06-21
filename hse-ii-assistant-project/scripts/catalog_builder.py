"""Конструктор учебного каталога вкладов.

Ставки вкладов привязаны к ключевой ставке ЦБ: rate = base(key_rate) + надбавки.
Поэтому сдвиг ключевой ставки порождает согласованный сдвиг всего каталога —
это используется в симуляции дрейфа (scripts/simulate_drift.py) как реалистичная
причина деградации качества рекомендаций при устаревшем каталоге.
"""

from __future__ import annotations

import random
from datetime import date

from src.core.models import Deposit

# Премия к ставке за срок (мес.): короткие и очень длинные вклады — чуть ниже пика.
TERM_PREMIUM = {3: -1.5, 6: -0.5, 9: 0.0, 12: 0.6, 18: 0.4, 24: 0.1, 36: -0.4}

# Архетипы продуктов: набор условий + надбавка к ставке + текст «скрытых условий».
ARCHETYPES = {
    "max": dict(
        capitalization="monthly", payout="at_end", replenishment=False,
        partial_withdrawal=False, early_termination="loss_of_interest",
        online_only=False, promo=False, premium=1.2,
        notes="Максимальная ставка ценой гибкости: без пополнения и снятия, "
        "при досрочном закрытии проценты сгорают.",
    ),
    "promo": dict(
        capitalization="none", payout="at_end", replenishment=False,
        partial_withdrawal=False, early_termination="reduced_rate",
        online_only=True, promo=True, premium=2.3,
        notes="Акционная ставка действует только для новых денег и при онлайн-"
        "оформлении; по истечении промо-срока ставка снижается.",
    ),
    "flex": dict(
        capitalization="monthly", payout="at_end", replenishment=True,
        partial_withdrawal=True, early_termination="reduced_rate",
        online_only=False, promo=False, premium=-0.8,
        notes="Пополнение и частичное снятие до неснижаемого остатка; ставка ниже "
        "максимальной — плата за гибкость.",
    ),
    "save": dict(
        capitalization="none", payout="monthly", replenishment=True,
        partial_withdrawal=False, early_termination="reduced_rate",
        online_only=False, promo=False, premium=-0.3,
        notes="Проценты выплачиваются ежемесячно на карту (без капитализации), "
        "вклад можно пополнять.",
    ),
    "protect": dict(
        capitalization="monthly", payout="at_end", replenishment=False,
        partial_withdrawal=False, early_termination="penalty_free",
        online_only=False, promo=False, premium=-0.5,
        notes="Сохранение процентов при досрочном закрытии — приоритет надёжности "
        "над ставкой.",
    ),
}

# (bank, product, archetype, term_months, min_amount, max_amount, currency)
SPECS: list[tuple] = [
    ("Банк Восход", "Максимальный", "max", 12, 50_000, None, "RUB"),
    ("Банк Восход", "Максимальный-6", "max", 6, 50_000, None, "RUB"),
    ("Банк Восход", "Надёжный", "protect", 12, 30_000, None, "RUB"),
    ("Банк Восход", "Пенсионный", "save", 12, 1_000, None, "RUB"),
    ("СеверКредит", "Промо-Старт", "promo", 3, 100_000, 5_000_000, "RUB"),
    ("СеверКредит", "Доходный", "max", 18, 100_000, None, "RUB"),
    ("СеверКредит", "Пополняй", "flex", 12, 50_000, None, "RUB"),
    ("ПримФинанс", "Свободный", "flex", 6, 30_000, None, "RUB"),
    ("ПримФинанс", "Сберегательный", "save", 12, 10_000, None, "RUB"),
    ("ПримФинанс", "Длинный", "max", 36, 100_000, None, "RUB"),
    ("Капитал-Банк", "Премиальный", "max", 12, 1_400_000, None, "RUB"),
    ("Капитал-Банк", "Стандарт", "save", 6, 30_000, None, "RUB"),
    ("Капитал-Банк", "Гибкий", "flex", 24, 50_000, None, "RUB"),
    ("ГринБанк", "Онлайн-Промо", "promo", 6, 100_000, 3_000_000, "RUB"),
    ("ГринБанк", "Зелёный", "save", 12, 15_000, None, "RUB"),
    ("ГринБанк", "Эко-Доход", "max", 9, 50_000, None, "RUB"),
    ("МосВклад", "Столичный", "max", 12, 50_000, None, "RUB"),
    ("МосВклад", "Управляй", "flex", 18, 50_000, None, "RUB"),
    ("МосВклад", "Копилка", "save", 3, 5_000, None, "RUB"),
    ("МосВклад", "Промо-Лето", "promo", 3, 50_000, 2_000_000, "RUB"),
    ("УралДепозит", "Уральский", "max", 24, 100_000, None, "RUB"),
    ("УралДепозит", "Надёжный+", "protect", 36, 50_000, None, "RUB"),
    ("УралДепозит", "Быстрый", "promo", 3, 30_000, 1_500_000, "RUB"),
    ("ОнлайнКопилка", "Турбо", "promo", 6, 10_000, 1_000_000, "RUB"),
    ("ОнлайнКопилка", "Ровный", "save", 9, 10_000, None, "RUB"),
    ("ОнлайнКопилка", "Защита", "protect", 12, 30_000, None, "RUB"),
    ("Капитал-Банк", "Долларовый", "save", 12, 1_000, None, "USD"),
    ("ПримФинанс", "Долларовый+", "max", 6, 1_000, None, "USD"),
    ("ГринБанк", "Юаневый", "save", 12, 10_000, None, "CNY"),
]

# Базовая ставка по валютам. Для RUB — функция ключевой ставки ЦБ.
FX_BASE = {"USD": 4.0, "CNY": 2.6}


def _base_rate(currency: str, key_rate: float) -> float:
    return key_rate if currency == "RUB" else FX_BASE[currency]


def build_deposits(
    key_rate: float = 16.0,
    as_of: date = date(2026, 6, 5),
    archetype_shift: dict[str, float] | None = None,
) -> list[Deposit]:
    """Построить детерминированный каталог под заданную ключевую ставку и дату.

    archetype_shift сдвигает надбавку отдельных архетипов — это моделирует
    НЕравномерные изменения рынка (например, «промо-ставки исчезли»), которые
    меняют не только уровень, но и порядок выгодности вкладов.
    """
    shift = archetype_shift or {}
    rng = random.Random(20260605)  # фиксированный сид → воспроизводимый каталог
    deposits: list[Deposit] = []
    for idx, (bank, product, arch, term, min_amt, max_amt, currency) in enumerate(SPECS, 1):
        a = ARCHETYPES[arch]
        noise = rng.uniform(-0.2, 0.2)
        premium = a["premium"] + shift.get(arch, 0.0)
        rate = _base_rate(currency, key_rate) + TERM_PREMIUM[term] + premium + noise
        rate = round(max(0.5, rate), 1)
        deposits.append(
            Deposit(
                id=f"D{idx:02d}",
                bank=bank,
                product=product,
                currency=currency,
                nominal_rate=rate,
                term_months=term,
                min_amount=float(min_amt),
                max_amount=float(max_amt) if max_amt is not None else None,
                capitalization=a["capitalization"],
                payout=a["payout"],
                replenishment=a["replenishment"],
                partial_withdrawal=a["partial_withdrawal"],
                early_termination=a["early_termination"],
                online_only=a["online_only"],
                promo=a["promo"],
                as_of_date=as_of,
                notes=a["notes"],
            )
        )
    return deposits


CSV_FIELDS = [
    "id", "bank", "product", "currency", "nominal_rate", "term_months",
    "min_amount", "max_amount", "capitalization", "payout", "replenishment",
    "partial_withdrawal", "early_termination", "online_only", "promo",
    "as_of_date", "notes",
]


def deposit_to_row(d: Deposit) -> dict:
    return {
        "id": d.id, "bank": d.bank, "product": d.product, "currency": d.currency,
        "nominal_rate": d.nominal_rate, "term_months": d.term_months,
        "min_amount": int(d.min_amount),
        "max_amount": "" if d.max_amount is None else int(d.max_amount),
        "capitalization": d.capitalization, "payout": d.payout,
        "replenishment": str(d.replenishment).lower(),
        "partial_withdrawal": str(d.partial_withdrawal).lower(),
        "early_termination": d.early_termination,
        "online_only": str(d.online_only).lower(),
        "promo": str(d.promo).lower(),
        "as_of_date": d.as_of_date.isoformat(), "notes": d.notes,
    }
