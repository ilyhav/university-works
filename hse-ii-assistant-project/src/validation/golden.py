"""Прогон ассистента по golden-набору и агрегация метрик качества."""

from __future__ import annotations

from datetime import date

from ..core.models import Deposit
from ..core.recommender import recommend
from ..data_access import load_golden, request_from_dict
from .metrics import Confusion, precision_recall_f1, ratio, reciprocal_rank


def _case_passed(g: dict, resp) -> tuple[bool, str]:
    """Прошёл ли кейс полностью + краткая причина расхождения."""
    exp_status = g["expected_status"]
    if resp.status != exp_status:
        return False, f"статус {resp.status} ≠ ожидаемого {exp_status}"

    if exp_status == "refused":
        code = g.get("expected_refusal_code")
        if code and (resp.refusal is None or resp.refusal.code.value != code):
            got = resp.refusal.code.value if resp.refusal else None
            return False, f"код отказа {got} ≠ {code}"
        return True, "ок"

    # exp_status == "ok"
    if not resp.recommendations:
        return False, "пустая выдача при ожидаемом ok"
    top = resp.recommendations[0]
    if g.get("expected_top_id") and top.deposit.id != g["expected_top_id"]:
        return False, f"top-1 {top.deposit.id} ≠ {g['expected_top_id']}"
    for k, v in (g.get("expected_constraints") or {}).items():
        if getattr(top.deposit, k) != v:
            return False, f"условие {k}={getattr(top.deposit, k)} ≠ {v}"
    sub = g.get("expect_risk_flag_substring")
    if sub and not any(sub in f for f in top.risk_flags):
        return False, f"нет риск-флага со словом «{sub}»"
    return True, "ок"


def evaluate_golden(deposits: list[Deposit], today: date | None = None) -> dict:
    """Полный отчёт по golden-набору: ранжирование, отказы, ограничения, кейсы."""
    golden = load_golden()
    conf = Confusion()
    code_total = code_correct = 0
    rank_total = rank_hits = 0
    rr_sum = 0.0
    cons_total = cons_ok = 0
    passed = 0
    cases: list[dict] = []

    for g in golden:
        resp = recommend(request_from_dict(g), deposits, today=today)
        exp_refused = g["expected_status"] == "refused"
        conf.add(exp_refused, resp.status == "refused")

        if exp_refused and resp.status == "refused" and g.get("expected_refusal_code"):
            code_total += 1
            if resp.refusal and resp.refusal.code.value == g["expected_refusal_code"]:
                code_correct += 1

        if g["expected_status"] == "ok" and g.get("expected_top_id"):
            rank_total += 1
            ids = [s.deposit.id for s in resp.recommendations]
            if ids and ids[0] == g["expected_top_id"]:
                rank_hits += 1
            rr_sum += reciprocal_rank(ids, g["expected_top_id"])

        if g["expected_status"] == "ok" and g.get("expected_constraints"):
            cons_total += 1
            top = resp.recommendations[0].deposit if resp.recommendations else None
            if top and all(getattr(top, k) == v for k, v in g["expected_constraints"].items()):
                cons_ok += 1

        ok, reason = _case_passed(g, resp)
        passed += int(ok)
        cases.append({
            "id": g["id"], "title": g.get("title", ""),
            "expected": g["expected_status"], "predicted": resp.status,
            "passed": ok, "reason": reason,
        })

    return {
        "n_cases": len(golden),
        "passed": passed,
        "pass_rate": ratio(passed, len(golden)),
        "ranking": {
            "max_income_cases": rank_total,
            "top1_accuracy": ratio(rank_hits, rank_total),
            "mrr": round(rr_sum / rank_total, 4) if rank_total else None,
        },
        "refusal": {
            **precision_recall_f1(conf),
            "code_accuracy": ratio(code_correct, code_total),
        },
        "constraints": {
            "checked": cons_total,
            "satisfied": cons_ok,
            "rate": ratio(cons_ok, cons_total),
        },
        "cases": cases,
    }
