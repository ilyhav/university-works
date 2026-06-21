"""Критерии вывода модели из промышленной среды (criterion 6 задания).

Решение принимается по формальным порогам, а не «на глаз». Различаем два масштаба
вмешательства:
  • LLM_OFF      — отключить только LLM-слой, оставить детерминированное ядро
                   (есть безопасный fallback, поэтому это не полный вывод);
  • DECOMMISSION — вывести весь модуль из эксплуатации (отказ ядра по качеству);
  • REFRESH      — данные устарели/рынок сместился: требуется обновление каталога;
  • MONITOR/OK   — работаем штатно.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Пороги. Вынесены в одно место, версионируются вместе с моделью.
THRESHOLDS = {
    "groundedness_share_min": 0.98,      # доля полностью обоснованных ответов LLM
    "groundedness_share_critical": 0.90,
    "refusal_precision_min": 0.90,
    "refusal_precision_critical": 0.70,
    "refusal_recall_min": 0.85,
    "ranking_top1_min": 0.90,
    "ranking_top1_critical": 0.75,
    "latency_p95_ms_max": 1500.0,
    "freshness_warning_days": 14,
    "freshness_critical_days": 28,
}

_OK, _WARN, _BREACH = "ok", "warn", "breach"


@dataclass
class Criterion:
    name: str
    status: str           # ok | warn | breach
    value: float | None
    threshold: float | None
    action: str           # NONE | MONITOR | REFRESH | LLM_OFF | DECOMMISSION
    note: str


def _grounded(share: float | None, llm_calls: int) -> Criterion:
    if not llm_calls or share is None:
        return Criterion("groundedness", _OK, share, THRESHOLDS["groundedness_share_min"],
                         "NONE", "LLM-слой не использовался — нечего оценивать.")
    if share < THRESHOLDS["groundedness_share_critical"]:
        return Criterion("groundedness", _BREACH, share, THRESHOLDS["groundedness_share_critical"],
                         "LLM_OFF", "Слишком много галлюцинаций: отключить LLM-слой, "
                         "оставить детерминированные объяснения.")
    if share < THRESHOLDS["groundedness_share_min"]:
        return Criterion("groundedness", _WARN, share, THRESHOLDS["groundedness_share_min"],
                         "MONITOR", "Рост доли необоснованных ответов — усилить контроль.")
    return Criterion("groundedness", _OK, share, THRESHOLDS["groundedness_share_min"],
                     "NONE", "Объяснения обоснованы числами ядра.")


def _refusal(precision: float | None, recall: float | None) -> Criterion:
    if precision is None:
        return Criterion("refusal_quality", _OK, None, None, "NONE", "Нет данных по отказам.")
    if precision < THRESHOLDS["refusal_precision_critical"]:
        return Criterion("refusal_quality", _BREACH, precision,
                         THRESHOLDS["refusal_precision_critical"], "DECOMMISSION",
                         "Ассистент массово отказывает/выдаёт некорректно — вывести из эксплуатации.")
    if precision < THRESHOLDS["refusal_precision_min"] or (
        recall is not None and recall < THRESHOLDS["refusal_recall_min"]
    ):
        return Criterion("refusal_quality", _WARN, precision,
                         THRESHOLDS["refusal_precision_min"], "REFRESH",
                         "Качество отказов ниже нормы — пересмотреть guardrails.")
    return Criterion("refusal_quality", _OK, precision, THRESHOLDS["refusal_precision_min"],
                     "NONE", "Отказы корректны.")


def _ranking(top1: float | None) -> Criterion:
    if top1 is None:
        return Criterion("ranking_quality", _OK, None, None, "NONE", "Нет данных по ранжированию.")
    if top1 < THRESHOLDS["ranking_top1_critical"]:
        return Criterion("ranking_quality", _BREACH, top1, THRESHOLDS["ranking_top1_critical"],
                         "DECOMMISSION", "Ранжирование не находит лучший вклад — вывести из эксплуатации.")
    if top1 < THRESHOLDS["ranking_top1_min"]:
        return Criterion("ranking_quality", _WARN, top1, THRESHOLDS["ranking_top1_min"],
                         "REFRESH", "Точность top-1 просела — переобучить конфиг скоринга / обновить данные.")
    return Criterion("ranking_quality", _OK, top1, THRESHOLDS["ranking_top1_min"],
                     "NONE", "Ранжирование совпадает с эталоном.")


def _freshness(days: int | None) -> Criterion:
    if days is None:
        return Criterion("data_freshness", _OK, None, None, "NONE", "Свежесть неизвестна.")
    if days > THRESHOLDS["freshness_critical_days"]:
        return Criterion("data_freshness", _BREACH, float(days),
                         float(THRESHOLDS["freshness_critical_days"]), "REFRESH",
                         "Каталог критически устарел — рекомендации недостоверны, требуется обновление.")
    if days > THRESHOLDS["freshness_warning_days"]:
        return Criterion("data_freshness", _WARN, float(days),
                         float(THRESHOLDS["freshness_warning_days"]), "REFRESH",
                         "Каталог устаревает — запланировать обновление ставок.")
    return Criterion("data_freshness", _OK, float(days),
                     float(THRESHOLDS["freshness_warning_days"]), "NONE", "Каталог свежий.")


def _latency(p95: float | None) -> Criterion:
    if p95 is None:
        return Criterion("latency", _OK, None, None, "NONE", "Нет данных по задержкам.")
    if p95 > THRESHOLDS["latency_p95_ms_max"]:
        return Criterion("latency", _WARN, p95, THRESHOLDS["latency_p95_ms_max"],
                         "MONITOR", "p95 задержки выше цели — проверить LLM-слой/инфраструктуру.")
    return Criterion("latency", _OK, p95, THRESHOLDS["latency_p95_ms_max"],
                     "NONE", "Задержки в норме.")


# Приоритет вердиктов: чем выше — тем серьёзнее.
_ACTION_RANK = {"NONE": 0, "MONITOR": 1, "REFRESH": 2, "LLM_OFF": 3, "DECOMMISSION": 4}
_VERDICT = {
    0: "OK — эксплуатация в штатном режиме",
    1: "MONITOR — наблюдать, штатных действий не требуется",
    2: "REFRESH — обновить данные/конфиг (плановое противодействие деградации)",
    3: "LLM_OFF — отключить LLM-слой, работать на детерминированном ядре",
    4: "DECOMMISSION — вывести модуль из промышленной среды",
}


def evaluate(snapshot: dict) -> dict:
    """Оценить критерии вывода модели по агрегированному снапшоту мониторинга.

    Ожидаемые ключи snapshot (любые могут отсутствовать → критерий 'ok/нет данных'):
      groundedness_share, llm_calls, refusal_precision, refusal_recall,
      ranking_top1, latency_p95_ms, freshness_days.
    """
    criteria = [
        _grounded(snapshot.get("groundedness_share"), snapshot.get("llm_calls", 0)),
        _refusal(snapshot.get("refusal_precision"), snapshot.get("refusal_recall")),
        _ranking(snapshot.get("ranking_top1")),
        _freshness(snapshot.get("freshness_days")),
        _latency(snapshot.get("latency_p95_ms")),
    ]
    worst = max(_ACTION_RANK[c.action] for c in criteria)
    return {
        "verdict": _VERDICT[worst],
        "action": next(a for a, r in _ACTION_RANK.items() if r == worst),
        "criteria": [asdict(c) for c in criteria],
    }
