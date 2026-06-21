"""Валидация качества на golden-наборе + проверка критериев вывода модели.

Запуск:  python -m scripts.run_eval  (или make eval)

Печатает отчёт и сохраняет его в data/runtime/eval_report.json. Метрики подаются
в monitoring.decommission — так связываются валидация и эксплуатация.
"""

from __future__ import annotations

import json
import time
from datetime import date

import numpy as np

from src.config import RUNTIME_DIR
from src.core.recommender import recommend
from src.data_access import load_deposits, load_golden, request_from_dict
from src.monitoring import decommission
from src.monitoring.logging_store import read_records
from src.validation.golden import evaluate_golden


def _latency_p95(deposits, today) -> float:
    lat = []
    for g in load_golden():
        t0 = time.perf_counter()
        recommend(request_from_dict(g), deposits, today=today)
        lat.append((time.perf_counter() - t0) * 1000)
    return round(float(np.percentile(lat, 95)), 3)


def _groundedness_from_log() -> tuple[float | None, int]:
    """Доля полностью обоснованных ответов LLM по журналу (если LLM использовался)."""
    recs = [r for r in read_records() if r.get("groundedness_source", "").startswith("gigachat")]
    if not recs:
        return None, 0
    grounded = sum(1 for r in recs if r["groundedness_source"] == "gigachat")
    return round(grounded / len(recs), 4), len(recs)


def main() -> None:
    today = date.today()
    deposits = load_deposits()
    report = evaluate_golden(deposits, today=today)

    catalog_as_of = max(d.as_of_date for d in deposits)
    freshness_days = (today - catalog_as_of).days
    g_share, llm_calls = _groundedness_from_log()

    snapshot = {
        "ranking_top1": report["ranking"]["top1_accuracy"],
        "refusal_precision": report["refusal"]["precision"],
        "refusal_recall": report["refusal"]["recall"],
        "groundedness_share": g_share,
        "llm_calls": llm_calls,
        "latency_p95_ms": _latency_p95(deposits, today),
        "freshness_days": freshness_days,
    }
    verdict = decommission.evaluate(snapshot)
    report["snapshot"] = snapshot
    report["decommission"] = verdict

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNTIME_DIR / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("═" * 64)
    print(f"ВАЛИДАЦИЯ КАЧЕСТВА  (golden: {report['n_cases']} кейсов, дата {today})")
    print("═" * 64)
    print(f"Пройдено кейсов:        {report['passed']}/{report['n_cases']} "
          f"(pass_rate={report['pass_rate']})")
    r = report["ranking"]
    print(f"Ранжирование top-1:     {r['top1_accuracy']}  (MRR={r['mrr']}, "
          f"кейсов max_income={r['max_income_cases']})")
    f = report["refusal"]
    print(f"Отказы precision/recall: {f['precision']}/{f['recall']}  "
          f"(F1={f['f1']}, точность кода={f['code_accuracy']})")
    print(f"  confusion: {f['confusion']}")
    c = report["constraints"]
    print(f"Доп. условия:           {c['satisfied']}/{c['checked']} (rate={c['rate']})")
    print(f"Свежесть каталога:      {freshness_days} дн. (LLM-вызовов в логе: {llm_calls})")
    print(f"Задержка p95 (ядро):    {snapshot['latency_p95_ms']} мс")
    print("─" * 64)
    print(f"ВЕРДИКТ ЭКСПЛУАТАЦИИ:   {verdict['verdict']}")
    for cr in verdict["criteria"]:
        mark = {"ok": "✓", "warn": "⚠", "breach": "✗"}[cr["status"]]
        print(f"  {mark} {cr['name']:<16} {cr['action']:<12} {cr['note']}")

    failed = [x for x in report["cases"] if not x["passed"]]
    if failed:
        print("─" * 64)
        print("НЕ ПРОЙДЕНЫ:")
        for x in failed:
            print(f"  • {x['id']} {x['title']}: {x['reason']}")
    print("═" * 64)
    print(f"Отчёт сохранён: {out}")


if __name__ == "__main__":
    main()
