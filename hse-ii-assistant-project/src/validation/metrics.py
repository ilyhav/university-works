"""Чистые функции метрик качества ассистента."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Confusion:
    tp: int = 0  # ожидался отказ и был отказ
    fp: int = 0  # отказ там, где ждали выдачу
    fn: int = 0  # выдача там, где ждали отказ
    tn: int = 0  # ожидалась выдача и была выдача

    def add(self, expected_refused: bool, predicted_refused: bool) -> None:
        if expected_refused and predicted_refused:
            self.tp += 1
        elif not expected_refused and predicted_refused:
            self.fp += 1
        elif expected_refused and not predicted_refused:
            self.fn += 1
        else:
            self.tn += 1


def precision_recall_f1(c: Confusion) -> dict:
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else None
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else None
    if precision and recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "confusion": {"tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn},
    }


def reciprocal_rank(ranked_ids: list[str], target_id: str) -> float:
    """1/позиция целевого элемента в выдаче (0, если его нет)."""
    if target_id in ranked_ids:
        return 1.0 / (ranked_ids.index(target_id) + 1)
    return 0.0


def ratio(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None
