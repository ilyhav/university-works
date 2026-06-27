"""
Прозрачное кэширование загрузок в parquet.

Идея: любая загрузка из сети (MOEX/ЦБ) проходит через @disk_cache.
Первый вызов идёт в сеть и кладёт результат в data_cache/<key>.parquet,
последующие — читают с диска. Это даёт:
  * воспроизводимость (данные «замораживаются» на момент первого прогона);
  * быстрый повторный запуск всего пайплайна;
  * возможность обращаться к промежуточным результатам (требование отчёта).

Ключ кэша строится из имени функции и её аргументов, поэтому разные
тикеры/даты кэшируются отдельно.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from rm.config import CACHE_DIR

logger = logging.getLogger(__name__)


def _make_key(func_name: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps(
        {"fn": func_name, "args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in kwargs.items()}},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    safe_name = func_name.replace(".", "_")
    return f"{safe_name}__{digest}"


def disk_cache(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Декоратор: кэширует возвращаемый DataFrame в parquet.

    Управляющий флаг ``force_reload=True`` в kwargs принудительно
    перезагружает данные (полезно при обновлении истории до 01.01.2026).
    """

    @functools.wraps(func)
    def wrapper(*args, force_reload: bool = False, **kwargs) -> pd.DataFrame:
        key = _make_key(func.__qualname__, args, kwargs)
        path: Path = CACHE_DIR / f"{key}.parquet"

        if path.exists() and not force_reload:
            logger.info("cache hit  : %s -> %s", func.__qualname__, path.name)
            return pd.read_parquet(path)

        logger.info("cache miss : %s (fetching from network)", func.__qualname__)
        df = func(*args, **kwargs)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{func.__qualname__} должна возвращать DataFrame, получено {type(df)}")
        df.to_parquet(path)
        logger.info("cached     : %s rows -> %s", len(df), path.name)
        return df

    return wrapper


def clear_cache(pattern: str = "*.parquet") -> int:
    """Удалить кэш (по умолчанию — весь). Возвращает число удалённых файлов."""
    n = 0
    for p in CACHE_DIR.glob(pattern):
        p.unlink()
        n += 1
    logger.info("cleared %s cache files (%s)", n, pattern)
    return n
