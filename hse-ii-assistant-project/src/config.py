"""Единая конфигурация приложения (читается из окружения / .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = DATA_DIR / "runtime"
SNAPSHOTS_DIR = RUNTIME_DIR / "snapshots"

CATALOG_PATH = DATA_DIR / "deposits.csv"
SCENARIOS_PATH = DATA_DIR / "client_scenarios.json"
GOLDEN_PATH = DATA_DIR / "golden_set.json"
REQUESTS_LOG = RUNTIME_DIR / "requests.jsonl"

# Сумма страхового возмещения АСВ (ст. 11 ФЗ-177): 1,4 млн ₽ на банк.
ASV_INSURANCE_LIMIT = 1_400_000.0

# Сколько вариантов показывать в сравнительной таблице.
TOP_N = 5


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


@dataclass(frozen=True)
class GigaChatConfig:
    credentials: str | None
    scope: str
    model: str
    verify_ssl: bool
    temperature: float
    # Жёсткое отключение LLM-слоя (только ядро) — независимо от наличия ключа.
    disabled: bool

    @property
    def enabled(self) -> bool:
        return bool(self.credentials) and not self.disabled


@dataclass(frozen=True)
class AppConfig:
    gigachat: GigaChatConfig
    catalog_freshness_sla_days: int


def load_config() -> AppConfig:
    return AppConfig(
        gigachat=GigaChatConfig(
            credentials=os.getenv("GIGACHAT_CREDENTIALS") or None,
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"),
            verify_ssl=_flag("GIGACHAT_VERIFY_SSL", False),
            temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.2")),
            disabled=_flag("ASSISTANT_LLM_DISABLED", False),
        ),
        catalog_freshness_sla_days=int(os.getenv("CATALOG_FRESHNESS_SLA_DAYS", "14")),
    )


CONFIG = load_config()
