"""Общие фикстуры тестов."""

from __future__ import annotations

from datetime import date

import pytest

from scripts.catalog_builder import build_deposits


@pytest.fixture(scope="session")
def deposits():
    """Детерминированный каталог (как на диске): ключевая ставка 16, as_of 2026-06-05."""
    return build_deposits(key_rate=16.0, as_of=date(2026, 6, 5))


@pytest.fixture(scope="session")
def today():
    return date(2026, 6, 13)
