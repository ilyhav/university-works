"""Каталог на диске должен совпадать с детерминированным генератором."""

from __future__ import annotations

from datetime import date

from scripts.catalog_builder import build_deposits
from src.data_access import load_deposits


def test_csv_matches_builder():
    on_disk = {d.id: d for d in load_deposits()}
    built = {d.id: d for d in build_deposits(key_rate=16.0, as_of=date(2026, 6, 5))}
    assert on_disk.keys() == built.keys(), "набор id в CSV разошёлся с генератором — пересоздайте data"
    for did, d in built.items():
        got = on_disk[did]
        assert got.nominal_rate == d.nominal_rate, f"{did}: ставка в CSV ≠ генератору (make data)"
        assert got.term_months == d.term_months
        assert got.capitalization == d.capitalization
        assert got.early_termination == d.early_termination
