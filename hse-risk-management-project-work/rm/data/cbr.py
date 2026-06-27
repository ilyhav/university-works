"""
Загрузчики данных Банка России (cbr.ru).

Покрывает:
  1a — процентные ставки на сроки 0–30 лет: КБД (кривая бескупонной
       доходности ОФЗ, она же G-curve). ЦБ публикует параметры
       Нельсона–Сигеля–Свенссона и значения на стандартных сроках.
  1e — официальные курсы USD и EUR (ежедневные фиксинги ЦБ).
       NB: с 13.06.2024 биржевые торги USD/EUR на MOEX остановлены,
       и курс ЦБ с этого момента считается по внебиржевым данным —
       это смена методологии в середине выборки, обязательно отметить
       в критическом обсуждении (п.2).

Цепочка до первоисточника (для защиты):
  КБД -> рассчитывается ЦБ по сделкам с ОФЗ на МосБирже (методика ЦБ
  на основе модели Свенссона) -> первоисточник: котировки ОФЗ.
  Курс ЦБ -> до 06.2024 по итогам биржевых торгов MOEX; после — по
  отчётности банков о внебиржевых сделках.

Эндпоинты ЦБ менялись; ниже — рабочие на момент написания. Если структура
ответа изменится, поправь парсер в одном месте (функции _parse_*).
"""
from __future__ import annotations

import logging
import re
from io import StringIO

import pandas as pd
import requests

from rm.data.cache import disk_cache

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "hse-risk-project/1.0"})
_TIMEOUT = 30

# Стандартные сроки КБД ЦБ (годы). 0.25 ≈ 3 мес — ближайшее к «0 лет».
GCURVE_TENORS: tuple[float, ...] = (
    0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0,
)


# --------------------------------------------------------------------------- #
# КБД / G-curve (п.1a)                                                        #
# --------------------------------------------------------------------------- #
@disk_cache
def load_gcurve(start: str, end: str) -> pd.DataFrame:
    """История кривой бескупонной доходности.

    Возвращает широкую таблицу: индекс — дата, колонки — сроки (годы),
    значения — доходность в долях (0.12 = 12% годовых).

    Источник: ЦБ публикует параметры КБД (B0,B1,B2,T1,...) и/или готовые
    значения. Здесь дёргаем datateam-эндпоинт ЦБ с готовыми ставками.
    Если он недоступен — есть запасной путь через параметры Свенссона
    (см. svensson_yield ниже): тянем параметры и считаем кривую сами.
    """
    url = "https://www.cbr.ru/hd_base/zcyc_params/"
    # ЦБ отдаёт HTML-таблицу со ставками по срокам или параметрами; парсим оба формата.
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": _ru_date(start),
        "UniDbQuery.To": _ru_date(end),
    }
    resp = _SESSION.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    try:
        return _ensure_decimal_rates(_parse_zcyc_curve_values(tables))
    except ValueError:
        logger.info("готовая таблица КБД не найдена, пробую параметры Свенссона")

    params_df = _parse_zcyc_params(tables)
    # Строим значения кривой на стандартных сроках из параметров Свенссона.
    rows = {}
    for date, p in params_df.iterrows():
        rows[date] = {t: svensson_yield(t, p) for t in GCURVE_TENORS}
    curve = pd.DataFrame.from_dict(rows, orient="index")
    curve.index.name = "date"
    return _ensure_decimal_rates(curve.sort_index())


def svensson_yield(t: float, p: "pd.Series") -> float:
    """Доходность на срок t (годы) по параметрам Свенссона/Нельсона–Сигеля.

    Параметры ЦБ обозначаются B0,B1,B2,B3,T1,T2 (могут называться
    beta0.. и tau..). Формула КБД ЦБ:
        g(t) = b0 + (b1+b2)*(tau1/t)*(1-exp(-t/tau1))
                  - b2*exp(-t/tau1) + b3*(tau2/t)*(1-exp(-t/tau2))
                  - b3*exp(-t/tau2)
    (ЦБ использует расширенную форму с дополнительными g-членами; при
    необходимости сверь с актуальной методикой ЦБ. Для риска на 1–10 дней
    точность параметризации некритична — важны ПРИРАЩЕНИЯ кривой.)

    Возвращает доходность в долях.
    """
    import math

    b0 = float(p.get("B0", p.get("beta0", 0.0)))
    b1 = float(p.get("B1", p.get("beta1", 0.0)))
    b2 = float(p.get("B2", p.get("beta2", 0.0)))
    b3 = float(p.get("B3", p.get("beta3", 0.0)))
    tau1 = float(p.get("T1", p.get("tau1", 1.0))) or 1.0
    tau2 = float(p.get("T2", p.get("tau2", 1.0))) or 1.0

    term1 = (b1 + b2) * (tau1 / t) * (1 - math.exp(-t / tau1))
    term2 = -b2 * math.exp(-t / tau1)
    term3 = b3 * (tau2 / t) * (1 - math.exp(-t / tau2))
    term4 = -b3 * math.exp(-t / tau2)
    g = b0 + term1 + term2 + term3 + term4
    # ЦБ обычно отдаёт параметры в процентах -> в доли.
    return g / 100.0 if abs(g) > 1.0 else g


# --------------------------------------------------------------------------- #
# Курсы валют (п.1e)                                                          #
# --------------------------------------------------------------------------- #
@disk_cache
def load_fx_rate(currency: str, start: str, end: str) -> pd.DataFrame:
    """Официальный курс ЦБ (руб. за 1 ед. валюты). currency in {'USD','EUR'}.

    Использует XML-сервис ЦБ XML_dynamic.asp по внутреннему коду валюты.
    """
    code = {"USD": "R01235", "EUR": "R01239"}[currency.upper()]
    url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
    params = {
        "date_req1": _ru_date(start),
        "date_req2": _ru_date(end),
        "VAL_NM_RQ": code,
    }
    resp = _SESSION.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    df = _parse_fx_xml(resp.content, currency)
    return df


# --------------------------------------------------------------------------- #
# Ключевая ставка (контекст для обсуждения режимов 2022/2024)                  #
# --------------------------------------------------------------------------- #
@disk_cache
def load_key_rate(start: str, end: str) -> pd.DataFrame:
    """История ключевой ставки ЦБ — для иллюстрации смен режима волатильности."""
    url = "https://www.cbr.ru/hd_base/KeyRate/"
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": _ru_date(start),
        "UniDbQuery.To": _ru_date(end),
    }
    resp = _SESSION.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), decimal=",", thousands=" ")
    df = tables[0]
    df.columns = ["date", "key_rate"]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["key_rate"] = pd.to_numeric(df["key_rate"], errors="coerce") / 100.0
    return df.dropna().set_index("date").sort_index()


# --------------------------------------------------------------------------- #
# Парсеры (вынесены отдельно — единственное место правок при смене формата)    #
# --------------------------------------------------------------------------- #
def _parse_zcyc_curve_values(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Из HTML-таблиц ЦБ вытащить готовые значения КБД по срокам.

    Актуальная страница zcyc_params публикует не параметры B0/B1/..., а
    готовые ставки на сроки 0.25, 0.5, ..., 30 лет. Возвращаем их в долях.
    """
    for raw in tables:
        t = _normalise_table_columns(raw)
        date_col = _find_date_column(t.columns)
        if date_col is None:
            continue

        tenor_cols: dict[str, float] = {}
        for col in t.columns:
            if col == date_col:
                continue
            tenor = _parse_tenor_label(col)
            if tenor is not None and tenor not in tenor_cols.values():
                tenor_cols[col] = tenor
        if len(tenor_cols) < 3:
            continue

        out = pd.DataFrame({"date": pd.to_datetime(t[date_col], dayfirst=True, errors="coerce")})
        for source_col, tenor in tenor_cols.items():
            out[tenor] = _to_numeric_ru(t[source_col])

        out = out.dropna(subset=["date"]).set_index("date").sort_index()
        value_cols = [tenor for tenor in GCURVE_TENORS if tenor in out.columns]
        curve = out[value_cols].dropna(how="all")
        if curve.empty:
            continue

        curve.index.name = "date"
        return _ensure_decimal_rates(curve)

    raise ValueError("не нашёл таблицу значений КБД — проверь формат cbr.ru/hd_base/zcyc_params")


def _parse_zcyc_params(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Из HTML-таблиц ЦБ вытащить параметры КБД по датам. Формат у ЦБ
    периодически меняется, поэтому ищем таблицу с колонкой даты и B0/B1...."""
    for raw in tables:
        t = _normalise_table_columns(raw)
        cols = [str(c).strip().upper() for c in t.columns]
        if any(c in ("ДАТА", "DATE") for c in cols) and any("B0" in c or "BETA0" in c for c in cols):
            t = t.copy()
            t.columns = cols
            date_col = "ДАТА" if "ДАТА" in cols else "DATE"
            t[date_col] = pd.to_datetime(t[date_col], dayfirst=True, errors="coerce")
            # нормализуем числа (запятая -> точка)
            for c in t.columns:
                if c != date_col:
                    t[c] = _to_numeric_ru(t[c])
            rename = {"BETA0": "B0", "BETA1": "B1", "BETA2": "B2", "BETA3": "B3",
                      "TAU1": "T1", "TAU2": "T2"}
            t = t.rename(columns={k: v for k, v in rename.items() if k in t.columns})
            return t.dropna(subset=[date_col]).set_index(date_col).sort_index()
    raise ValueError("не нашёл таблицу параметров КБД — проверь формат cbr.ru/hd_base/zcyc_params")


def _normalise_table_columns(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    t.columns = [_flatten_column_name(c) for c in t.columns]
    return t


def _flatten_column_name(column) -> str:
    if isinstance(column, tuple):
        parts = []
        for part in column:
            text = str(part).strip()
            if not text or text.lower().startswith("unnamed:"):
                continue
            if text not in parts:
                parts.append(text)
        return " ".join(parts)
    return str(column).strip()


def _find_date_column(columns) -> str | None:
    for col in columns:
        upper = str(col).strip().upper()
        if "ДАТА" in upper or upper == "DATE":
            return col
    return None


def _parse_tenor_label(column) -> float | None:
    text = str(column).replace("\xa0", " ").strip()
    upper = text.upper().replace(" ", "")
    if "ДАТА" in upper or upper == "DATE":
        return None
    if "BETA" in upper or "TAU" in upper or re.fullmatch(r"[BT]\d+", upper):
        return None

    nums = re.findall(r"\d+(?:[,.]\d+)?", text)
    if not nums:
        return None
    value = float(nums[-1].replace(",", "."))
    for tenor in GCURVE_TENORS:
        if abs(value - tenor) < 1e-9:
            return tenor
    return value if 0 < value <= 100 else None


def _to_numeric_ru(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _ensure_decimal_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Привести ставки к долям: 0.14 = 14%.

    Страница ЦБ может попадать из read_html как 13.5 или как 1350 из-за
    десятичной запятой. Поэтому нормализуем повторно, пока масштаб не станет
    похож на доли.
    """
    out = df.copy()
    while True:
        values = pd.Series(out.to_numpy().ravel()).dropna()
        if values.empty or values.abs().median() <= 1.0:
            return out
        out = out / 100.0


def _parse_fx_xml(content: bytes, currency: str) -> pd.DataFrame:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(content)
    recs = []
    for rec in root.findall("Record"):
        date = pd.to_datetime(rec.attrib["Date"], dayfirst=True, errors="coerce")
        nominal = float(rec.findtext("Nominal", "1").replace(",", "."))
        value = float(rec.findtext("Value", "0").replace(",", "."))
        recs.append((date, value / nominal))
    df = pd.DataFrame(recs, columns=["date", currency]).dropna()
    return df.set_index("date").sort_index()


def _ru_date(iso: str) -> str:
    """'YYYY-MM-DD' -> 'DD.MM.YYYY' (формат ЦБ)."""
    y, m, d = iso.split("-")
    return f"{d}.{m}.{y}"
