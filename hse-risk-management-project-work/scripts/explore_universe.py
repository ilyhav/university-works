"""
Помощник для отбора 5 ОФЗ-ПД под критерии задания (п.1b):
  * постоянный (известный) купон, не привязанный к показателям  -> ОФЗ-ПД;
  * без оферт (OFFERDATE пуст);
  * погашение (MATDATE) после 01.01.2026.

Запусти ЛОКАЛЬНО (нужен доступ в интернет к moex.ru):
    python -m scripts.explore_universe

Скрипт вытащит список всех ОФЗ, отфильтрует по критериям и распечатает
кандидатов, отсортированных по разным срокам погашения (полезно взять
бумаги с РАЗНЫМИ дюрациями — это богаче нагружает кривую и PCA).
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
ISS = "https://iss.moex.com/iss"


def list_ofz() -> pd.DataFrame:
    """Все облигации федерального займа, торгуемые в режиме TQOB."""
    url = f"{ISS}/engines/stock/markets/bonds/boards/TQOB/securities.json"
    r = requests.get(url, params={"iss.meta": "off"}, timeout=30)
    r.raise_for_status()
    js = r.json()["securities"]
    df = pd.DataFrame(js["data"], columns=js["columns"])
    return df


def filter_pd_no_offer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MATDATE" in df:
        df["MATDATE"] = pd.to_datetime(df["MATDATE"], errors="coerce")
    if "OFFERDATE" in df:
        df["OFFERDATE"] = pd.to_datetime(df["OFFERDATE"], errors="coerce")
    # ОФЗ-ПД: в SHORTNAME обычно есть 'ОФЗ' и фиксированный купон;
    # надёжнее — по SECTYPE/префиксу SU...RMFS и отсутствию оферты.
    mask = df["MATDATE"] > pd.Timestamp("2026-01-01")
    if "OFFERDATE" in df:
        mask &= df["OFFERDATE"].isna()
    cols = [c for c in ("SECID", "SHORTNAME", "MATDATE", "OFFERDATE",
                        "COUPONPERCENT", "COUPONVALUE", "FACEVALUE") if c in df]
    out = df.loc[mask, cols].sort_values("MATDATE")
    return out


def suggest_ladder(cand: pd.DataFrame, target_years=(1, 3, 6, 9, 15),
                   asof=pd.Timestamp("2025-12-02")) -> pd.DataFrame:
    """Подобрать по одной бумаге, ближайшей к каждому целевому сроку, —
    получится «лесенка» по кривой для богатого PCA. Без повторов."""
    cand = cand.copy()
    cand["years"] = (cand["MATDATE"] - asof).dt.days / 365.25
    picked, used = [], set()
    for ty in target_years:
        avail = cand[~cand["SECID"].isin(used)]
        if avail.empty:
            break
        i = (avail["years"] - ty).abs().idxmin()
        picked.append(i)
        used.add(avail.loc[i, "SECID"])
    return cand.loc[picked].sort_values("years")


if __name__ == "__main__":
    ofz = list_ofz()
    cand = filter_pd_no_offer(ofz)
    pd.set_option("display.max_rows", 200, "display.width", 160)
    print(f"\nКандидатов ОФЗ (погашение>2026-01-01, без оферты): {len(cand)}\n")
    print(cand.to_string(index=False))

    ladder = suggest_ladder(cand)
    print("\n=== Предлагаемая лесенка (по 1 бумаге на срок ~1/3/6/9/15 лет) ===")
    cols = [c for c in ("SECID", "SHORTNAME", "MATDATE", "years", "COUPONPERCENT") if c in ladder]
    print(ladder[cols].to_string(index=False))
    print("\nГотовый список для rm/config.py -> PortfolioSpec.bonds:")
    print("    bonds = (" + ", ".join(f'"{s}"' for s in ladder["SECID"]) + ",)")
    print("\nВАЖНО: проверь, что у выбранных бумаг есть история с 2021 г.")
    print("(после fetch_all смотри outputs/data_coverage.csv -> 'первая_дата').")
