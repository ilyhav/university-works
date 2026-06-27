"""
Загрузчики данных Московской биржи через ISS API (iss.moex.com).

ISS — Informational & Statistical Server, свободный доступ, без ключа.
Документация: https://iss.moex.com/iss/reference/

Покрывает пункты задания:
  1c — рыночные котировки облигаций (history);
  1d — котировки 10 акций (history);
  1e — индексы МосБиржи и РТС (history);
  1b — расписания выплат по облигациям (securities/<sec>/bondization — купоны/амортизация);
  1f — фьючерс и опционы на фьючерс (для бонуса).

Все публичные функции кэшируются в parquet (см. cache.disk_cache).

ВАЖНО про «цепочку до первоисточника» (требование защиты):
  MOEX публикует на ISS те же данные, что показываются в торговом терминале;
  первоисточник — итоги торгов самой биржи (organized trading). Для облигаций
  «чистая» цена выражена в % от номинала (колонка LEGALCLOSEPRICE / CLOSE),
  накопленный купонный доход (НКД) — в колонке ACCINT.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import pandas as pd
import requests

from rm.data.cache import disk_cache

logger = logging.getLogger(__name__)

ISS_BASE = "https://iss.moex.com/iss"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "hse-risk-project/1.0"})
_TIMEOUT = 30
_PAGE = 100  # ISS отдаёт history постранично; шаг курсора


# --------------------------------------------------------------------------- #
# Низкоуровневый клиент                                                       #
# --------------------------------------------------------------------------- #
def _get(url: str, params: dict[str, Any] | None = None) -> dict:
    params = dict(params or {})
    params.setdefault("iss.meta", "off")
    for attempt in range(4):
        resp = _SESSION.get(url, params=params, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("ISS %s -> %s (attempt %s)", url, resp.status_code, attempt + 1)
        time.sleep(1.5 * (attempt + 1))
    resp.raise_for_status()
    return {}


def _block_to_df(payload: dict, block: str) -> pd.DataFrame:
    """ISS отдаёт {block: {'columns': [...], 'data': [[...], ...]}}."""
    node = payload.get(block, {})
    cols = node.get("columns", [])
    data = node.get("data", [])
    return pd.DataFrame(data, columns=cols)


def _paged_history(url: str, params: dict[str, Any], block: str = "history") -> pd.DataFrame:
    """Скачать весь history-блок, листая курсор start=0,100,200,..."""
    frames: list[pd.DataFrame] = []
    start = 0
    while True:
        p = dict(params, start=start)
        payload = _get(url, p)
        df = _block_to_df(payload, block)
        if df.empty:
            break
        frames.append(df)
        if len(df) < _PAGE:
            break
        start += _PAGE
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Акции (engine=stock, market=shares, board=TQBR)                             #
# --------------------------------------------------------------------------- #
@disk_cache
def load_stock_history(secid: str, start: str, end: str, board: str = "TQBR") -> pd.DataFrame:
    """Дневная история по акции. Возвращает TRADEDATE, CLOSE, VOLUME, ...

    start/end — строки 'YYYY-MM-DD'.
    """
    url = f"{ISS_BASE}/history/engines/stock/markets/shares/boards/{board}/securities/{secid}.json"
    params = {"from": start, "till": end}
    df = _paged_history(url, params)
    return _clean_history(df, secid)


# --------------------------------------------------------------------------- #
# Облигации (engine=stock, market=bonds)                                      #
# --------------------------------------------------------------------------- #
@disk_cache
def load_bond_history(secid: str, start: str, end: str, board: str = "TQOB") -> pd.DataFrame:
    """Дневная история по облигации (TQOB — основной режим для ОФЗ).

    Ключевые колонки:
      CLOSE / LEGALCLOSEPRICE — чистая цена в % от номинала;
      ACCINT                  — НКД (руб. на облигацию);
      YIELDCLOSE              — доходность к погашению на закрытие, % годовых;
      FACEVALUE               — номинал.
    «Грязная» цена = CLOSE% * FACEVALUE/100 + ACCINT.
    """
    url = f"{ISS_BASE}/history/engines/stock/markets/bonds/boards/{board}/securities/{secid}.json"
    params = {"from": start, "till": end}
    df = _paged_history(url, params)
    return _clean_history(df, secid)


@disk_cache
def load_bond_coupons(secid: str) -> pd.DataFrame:
    """Расписание купонов (п.1b). Блок 'coupons' из bondization.

    Колонки: coupondate, value (руб.), valueprc (% годовых), startdate, ...
    """
    url = f"{ISS_BASE}/securities/{secid}/bondization.json"
    payload = _get(url, {"iss.only": "coupons", "limit": "unlimited"})
    df = _block_to_df(payload, "coupons")
    if not df.empty and "coupondate" in df:
        df["coupondate"] = pd.to_datetime(df["coupondate"], errors="coerce")
    return df


@disk_cache
def load_bond_amortization(secid: str) -> pd.DataFrame:
    """График амортизации/погашения номинала. Блок 'amortizations'."""
    url = f"{ISS_BASE}/securities/{secid}/bondization.json"
    payload = _get(url, {"iss.only": "amortizations", "limit": "unlimited"})
    df = _block_to_df(payload, "amortizations")
    if not df.empty and "amortdate" in df:
        df["amortdate"] = pd.to_datetime(df["amortdate"], errors="coerce")
    return df


@disk_cache
def load_security_meta(secid: str) -> pd.DataFrame:
    """Паспорт бумаги: MATDATE (погашение), OFFERDATE (оферта), COUPONPERIOD,
    FACEVALUE, ISIN и т.д. Нужен для отбора ОФЗ-ПД без оферт (п.1b)."""
    url = f"{ISS_BASE}/securities/{secid}.json"
    payload = _get(url, {"iss.only": "description"})
    df = _block_to_df(payload, "description")
    return df


# --------------------------------------------------------------------------- #
# Индексы (engine=stock, market=index)                                        #
# --------------------------------------------------------------------------- #
@disk_cache
def load_index_history(secid: str, start: str, end: str) -> pd.DataFrame:
    """История индекса (IMOEX, RTSI). Колонка CLOSE."""
    url = f"{ISS_BASE}/history/engines/stock/markets/index/securities/{secid}.json"
    params = {"from": start, "till": end}
    df = _paged_history(url, params)
    return _clean_history(df, secid)


# --------------------------------------------------------------------------- #
# Срочный рынок: фьючерсы и опционы (engine=futures) — бонус (п.1f, п.8)       #
# --------------------------------------------------------------------------- #
@disk_cache
def load_futures_history(secid: str, start: str, end: str) -> pd.DataFrame:
    """История фьючерса (market=forts). CLOSE, OPENPOSITION, ..."""
    url = f"{ISS_BASE}/history/engines/futures/markets/forts/securities/{secid}.json"
    params = {"from": start, "till": end}
    df = _paged_history(url, params)
    return _clean_history(df, secid)


@disk_cache
def load_options_board(asset: str, on_date: str) -> pd.DataFrame:
    """Срез опционов на фьючерс на один торговый день (для бонуса п.1f/8).

    asset — код базового фьючерса (например, 'Si' / 'BR' / конкретный SECID).
    Возвращает доступные страйки/типы (Call/Put) и цены расчёта на дату.
    Реализацию SECID-фильтра уточнишь под выбранный актив — ISS options
    лежат в engine=futures, market=options.
    """
    url = f"{ISS_BASE}/history/engines/futures/markets/options/securities.json"
    params = {"date": on_date, "assetcode": asset}
    df = _paged_history(url, params)
    return df


# --------------------------------------------------------------------------- #
# Хелперы                                                                     #
# --------------------------------------------------------------------------- #
def _clean_history(df: pd.DataFrame, secid: str) -> pd.DataFrame:
    """Привести history к единому виду: индекс по дате, числовые колонки, SECID."""
    if df.empty:
        logger.warning("пустая история для %s", secid)
        return df
    if "TRADEDATE" in df.columns:
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
        df = df.dropna(subset=["TRADEDATE"]).set_index("TRADEDATE").sort_index()
    for col in df.columns:
        if col not in ("SECID", "BOARDID", "SHORTNAME"):
            try:
                df[col] = pd.to_numeric(df[col])
            except (TypeError, ValueError):
                pass
    df["SECID"] = secid
    return df


def load_many(
    loader,
    secids: Iterable[str],
    start: str,
    end: str,
    value_col: str = "CLOSE",
    **kwargs,
) -> pd.DataFrame:
    """Собрать по нескольким бумагам широкую таблицу: индекс=дата, колонки=SECID.

    Пример:
        prices = load_many(load_stock_history, PORTFOLIO.stocks, '2021-01-01', '2026-01-01')
    """
    series = {}
    for sec in secids:
        df = loader(sec, start, end, **kwargs)
        if df.empty or value_col not in df.columns:
            logger.warning("нет колонки %s для %s — пропуск", value_col, sec)
            continue
        series[sec] = df[value_col]
    if not series:
        return pd.DataFrame()
    wide = pd.concat(series, axis=1)
    wide.columns = list(series.keys())
    return wide.sort_index()
