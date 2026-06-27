"""
Локальный smoke-тест БЕЗ сети: проверяет, что всё импортируется, кэш
работает, парсеры и выравнивание корректны на синтетических данных.
Сетевые загрузки здесь подменяются заглушками (monkeypatch).

    python -m scripts.smoke_test
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rm.config import PORTFOLIO, RANDOM_SEED, VAR_LEVEL, ES_LEVEL
from rm.data import cache, cbr, moex


def test_config():
    assert PORTFOLIO.total_value() == 5 * 10_000_000 + 10 * 1_000_000 + 2 * 100_000_000
    assert VAR_LEVEL == 0.99 and ES_LEVEL == 0.975
    assert RANDOM_SEED == 20251202
    print("ok  config: суммарный портфель =", f"{PORTFOLIO.total_value():,.0f} руб.")


def test_cache(tmp_calls={"n": 0}):
    original_cache_dir = cache.CACHE_DIR
    tmp_calls["n"] = 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            cache.CACHE_DIR = Path(tmp)

            @cache.disk_cache
            def fake_loader(x):
                tmp_calls["n"] += 1
                return pd.DataFrame({"v": [x, x + 1]})

            a = fake_loader(7)
            b = fake_loader(7)  # должен прийти из кэша, не увеличив счётчик
            assert tmp_calls["n"] == 1, "кэш не сработал — функция вызвана дважды"
            assert a.equals(b)
        finally:
            cache.CACHE_DIR = original_cache_dir
    print("ok  cache : второй вызов прочитан из parquet (сеть не дёрнута)")


def test_svensson_monotone_sanity():
    # параметры с положительным уровнем -> неотрицательные ставки на всех сроках
    p = pd.Series({"B0": 8.0, "B1": -1.0, "B2": 0.5, "B3": 0.0, "T1": 1.5, "T2": 5.0})
    ys = [cbr.svensson_yield(t, p) for t in (0.25, 1, 5, 10, 30)]
    assert all(np.isfinite(ys)), "Свенссон вернул NaN/inf"
    assert all(0 < y < 0.5 for y in ys), f"ставки вне разумного диапазона: {ys}"
    print("ok  curve : Свенссон даёт разумные ставки:",
          {t: round(cbr.svensson_yield(t, p), 4) for t in (0.25, 1, 5, 10, 30)})


def test_fx_xml_parser():
    xml = (
        b"<?xml version='1.0' encoding='windows-1251'?>"
        b"<ValCurs><Record Date='02.12.2025'><Nominal>1</Nominal>"
        b"<Value>92,5000</Value></Record>"
        b"<Record Date='03.12.2025'><Nominal>1</Nominal>"
        b"<Value>93,1000</Value></Record></ValCurs>"
    )
    df = cbr._parse_fx_xml(xml, "USD")
    assert list(df["USD"]) == [92.5, 93.1]
    assert df.index[0] == pd.Timestamp("2025-12-02")
    print("ok  fx    : XML ЦБ распарсен ->", df["USD"].tolist())


def test_cbr_zcyc_curve_values_parser():
    table = pd.DataFrame({
        "Дата": ["25.06.2026", "26.06.2026"],
        "0,25": ["13,82", "13,80"],
        "0,5": ["13,65", "13,61"],
        "1": ["13,39", "13,33"],
        "2": ["12,91", "12,88"],
    })
    curve = cbr._parse_zcyc_curve_values([table])
    assert curve.index[0] == pd.Timestamp("2026-06-25")
    assert list(curve.columns) == [0.25, 0.5, 1.0, 2.0]
    assert round(float(curve.loc[pd.Timestamp("2026-06-25"), 0.25]), 4) == 0.1382
    print("ok  zcyc  : готовая таблица КБД ЦБ распарсена в долях")


def test_gcurve_cached_unit_normalisation():
    from rm.data.dataset import _ensure_decimal_rates

    cached = pd.DataFrame({0.25: [13.5], 1.0: [14.0]})
    fixed = _ensure_decimal_rates(cached)
    assert round(float(fixed.iloc[0, 0]), 3) == 0.135
    assert round(float(fixed.iloc[0, 1]), 3) == 0.140
    html_decimal_lost = pd.DataFrame({0.25: [1350], 1.0: [1400]})
    fixed2 = _ensure_decimal_rates(html_decimal_lost)
    assert round(float(fixed2.iloc[0, 0]), 3) == 0.135
    assert round(float(fixed2.iloc[0, 1]), 3) == 0.140
    print("ok  zcyc  : старый percent-кэш КБД нормализуется")


def test_moex_clean_history_numeric_conversion():
    raw = pd.DataFrame({
        "TRADEDATE": ["2025-01-03"],
        "BOARDID": ["TQBR"],
        "CLOSE": ["123.45"],
        "TEXT_NOTE": ["not a number"],
    })
    df = moex._clean_history(raw, "SBER")
    assert df.index[0] == pd.Timestamp("2025-01-03")
    assert df["CLOSE"].iloc[0] == 123.45
    assert df["TEXT_NOTE"].iloc[0] == "not a number"
    assert df["SECID"].iloc[0] == "SBER"
    print("ok  moex  : history очистка совместима с pandas 3")


def test_align_logic():
    # имитируем dataset._align на синтетике
    from rm.data.dataset import MarketData, _align
    cal = pd.bdate_range("2025-01-01", periods=10)
    stock = pd.DataFrame(np.random.rand(10, 2), index=cal, columns=["AAA", "BBB"])
    # fx покрывает весь календарь, но с внутренним пропуском (день 5 отсутствует)
    fx_dates = cal.delete(5)
    fx_raw = pd.DataFrame(np.random.rand(len(fx_dates), 1), index=fx_dates, columns=["USD"])
    md = MarketData(stock, stock, stock, stock, fx_raw, fx_raw, stock, {}, {})
    _align(md)
    assert md.fx.index.equals(cal), "fx не выровнен по календарю акций"
    # внутренний пропуск должен закрыться ffill; цена дня 5 == цена дня 4
    assert md.fx.notna().all().all(), "ffill не закрыл внутренний пропуск курса"
    assert md.fx["USD"].iloc[5] == md.fx["USD"].iloc[4], "ffill сработал неверно"
    print("ok  align : ряды выровнены, внутренние пропуски курса закрыты ffill")


# --------------------------------------------------------------------------- #
# Этап 2: факторы (returns / pca / descriptive) — на синтетике, без сети       #
# --------------------------------------------------------------------------- #
def test_returns_basics():
    from rm.factors import curve_increments, log_returns

    px = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    r = log_returns(px)["A"]
    assert np.isnan(r.iloc[0])
    assert abs(r.iloc[1] - np.log(1.1)) < 1e-12
    # приращения кривой: колонки упорядочены по сроку, Δy = y_t − y_{t-1}
    curve = pd.DataFrame({5.0: [0.10, 0.11], 1.0: [0.08, 0.085], 10.0: [0.11, 0.10]})
    dy = curve_increments(curve)
    assert list(dy.columns) == [1.0, 5.0, 10.0], "сроки должны идти по возрастанию"
    assert abs(dy[1.0].iloc[1] - 0.005) < 1e-12
    print("ok  factor: лог-доходности и приращения кривой считаются верно")


def test_splice_returns():
    from rm.factors import build_equity_returns

    cal = pd.bdate_range("2024-01-01", periods=10)
    # новый тикер торгуется только с 7-го дня, старый — всю историю
    succ = pd.Series([np.nan] * 6 + [200.0, 202.0, 204.0, 206.0], index=cal)
    pred = pd.Series(100.0 + np.arange(10), index=cal)
    prices = pd.DataFrame({"YDEX": succ, "OTHER": 50.0 + np.arange(10)})
    rets = build_equity_returns(prices, {"YDEX": pred})

    raw_valid = int(np.log(succ).diff().notna().sum())          # только хвост
    spliced_valid = int(rets["YDEX"].notna().sum())
    assert spliced_valid > raw_valid, "склейка не добавила истории из предшественника"
    # ранний день берётся из предшественника, поздний — из нового тикера
    pred_ret = np.log(pred).diff()
    succ_ret = np.log(succ).diff()
    assert abs(rets["YDEX"].iloc[2] - pred_ret.iloc[2]) < 1e-12
    assert abs(rets["YDEX"].iloc[8] - succ_ret.iloc[8]) < 1e-12
    print("ok  splice: YNDX->YDEX склейка по доходностям непрерывна")


def test_clean_for_pca_reports():
    from rm.factors import clean_for_pca

    cal = pd.bdate_range("2024-01-01", periods=10)
    good = pd.Series(np.random.randn(10), index=cal)
    sparse = pd.Series([np.nan] * 7 + list(np.random.randn(3)), index=cal)  # 30% покрытие
    df = pd.DataFrame({"GOOD": good, "SPARSE": sparse})
    clean, rep = clean_for_pca(df, min_coverage=0.5)
    assert "SPARSE" not in clean.columns, "колонка с низким покрытием не отброшена"
    assert "GOOD" in clean.columns
    assert clean.notna().all().all()
    print("ok  clean : низкое покрытие отброшено, отчёт о строках сформирован")


def test_pca_curve_level_slope_curvature():
    """PCA должен восстановить уровень/наклон/кривизну из синтетической кривой,
    собранной из трёх ортогональных форм с убывающей дисперсией."""
    from rm.factors import fit_pca, interpret_curve_components

    rng = np.random.default_rng(RANDOM_SEED)
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    # ортонормированный базис: константа / линейный / квадратичный тренды
    basis, _ = np.linalg.qr(np.vstack([np.ones_like(tenors), tenors, tenors**2]).T)
    n = 1500
    scores = rng.normal(size=(n, 3)) * np.array([0.010, 0.004, 0.0015])  # σ убывает
    dy = scores @ basis.T + rng.normal(scale=1e-5, size=(n, len(tenors)))
    X = pd.DataFrame(dy, columns=tenors)

    res = fit_pca(X, n_components=3, standardize=False)
    labels = interpret_curve_components(res.loadings)
    cum3 = float(np.cumsum(res.explained_variance_ratio)[2])
    assert cum3 > 0.99, f"3 компоненты должны давать >99%, дали {cum3:.3f}"
    assert labels["PC1"] == "уровень", labels
    assert labels["PC2"] == "наклон", labels
    assert labels["PC3"] == "кривизна", labels
    print("ok  pca   : кривая -> уровень/наклон/кривизна, 3 PC =", f"{cum3*100:.2f}%")


def test_pca_reconstruct_roundtrip():
    """Полный набор компонент должен точно восстанавливать исходные ряды
    (в т.ч. в режиме standardize — проверка возврата к исходным единицам)."""
    from rm.factors import fit_pca

    rng = np.random.default_rng(RANDOM_SEED)
    n = 800
    f = rng.normal(size=(n, 2))
    data = np.column_stack([
        2.0 + 0.5 * f[:, 0],
        -1.0 + 0.3 * f[:, 0] + 0.2 * f[:, 1],
        5.0 + 0.1 * f[:, 1] + 0.05 * rng.normal(size=n),
    ])
    X = pd.DataFrame(data, columns=["X1", "X2", "X3"])
    for standardize in (False, True):
        res = fit_pca(X, n_components=3, standardize=standardize)
        recon = res.reconstruct()
        err = float(np.abs(recon.to_numpy() - X.to_numpy()).max())
        assert err < 1e-8, f"reconstruct (standardize={standardize}) ошибка {err}"
    print("ok  pca   : reconstruct точно возвращает ряды (обе нормировки)")


def test_descriptive_heavy_tails_and_normality():
    from rm.factors import hill_tail_index, moments_table

    rng = np.random.default_rng(RANDOM_SEED)
    norm = rng.normal(size=5000)
    heavy = rng.standard_t(df=3, size=5000)  # тяжёлые хвосты
    df = pd.DataFrame({"norm": norm, "heavy": heavy})
    m = moments_table(df)
    assert m.loc["heavy", "JB_pvalue"] < 0.05, "JB должен отвергнуть нормальность у t(3)"
    assert m.loc["heavy", "эксцесс_изб"] > m.loc["norm", "эксцесс_изб"]
    a_heavy = hill_tail_index(heavy, "right")
    a_norm = hill_tail_index(norm, "right")
    assert np.isfinite(a_heavy) and a_heavy < a_norm, "хвост t(3) должен быть тяжелее"
    print("ok  tails : JB ловит ненормальность, Хилл — более тяжёлый хвост t(3)")


def test_descriptive_stationarity():
    from rm.factors import stationarity_table

    rng = np.random.default_rng(RANDOM_SEED)
    eps = rng.normal(size=1000)
    rw = np.cumsum(eps)                      # случайное блуждание — нестационарно
    df = pd.DataFrame({"returns": eps, "random_walk": rw})
    st = stationarity_table(df)
    assert st.loc["returns", "ADF_вывод"] == "стационарен"
    assert st.loc["random_walk", "ADF_вывод"] != "стационарен"
    print("ok  adf   : доходности стационарны, случайное блуждание — нет")


def test_descriptive_vol_clustering():
    from rm.factors import volatility_clustering_table

    rng = np.random.default_rng(RANDOM_SEED)
    n = 2000
    # простой ARCH-подобный процесс: дисперсия зависит от прошлого шока
    e = np.zeros(n)
    sig2 = np.ones(n)
    for t in range(1, n):
        sig2[t] = 0.1 + 0.85 * e[t - 1] ** 2
        e[t] = np.sqrt(sig2[t]) * rng.normal()
    df = pd.DataFrame({"garch": e})
    vc = volatility_clustering_table(df, lags=10)
    assert vc.loc["garch", "ARCH-эффект"] == "есть", "кластеризация волатильности не поймана"
    print("ok  arch  : Льюнг–Бокс по квадратам ловит кластеризацию волатильности")


# --------------------------------------------------------------------------- #
# Этап 3: стохастические модели (normal / t / ewma / garch / ou)               #
# --------------------------------------------------------------------------- #
def test_gaussian_model():
    from rm.models import GaussianModel

    rng = np.random.default_rng(RANDOM_SEED)
    mu = np.array([0.001, -0.002, 0.0])
    cov = np.array([[4e-4, 1e-4, 0.0], [1e-4, 9e-4, 0.0], [0.0, 0.0, 1e-4]])
    X = pd.DataFrame(rng.multivariate_normal(mu, cov, size=4000), columns=list("ABC"))
    m = GaussianModel.fit(X)
    assert np.allclose(m.mean, X.mean().to_numpy(), atol=1e-12)
    assert np.isfinite(m.loglik) and np.isfinite(m.aic) and np.isfinite(m.bic)
    sim = m.simulate(500, rng)
    assert sim.shape == (500, 3)
    print("ok  gauss : нормаль подогнана, log-lik/AIC/BIC и симуляция корректны")


def test_student_t_beats_normal_on_heavy_tails():
    from scipy.stats import multivariate_t
    from rm.models import GaussianModel, StudentTModel

    rng = np.random.default_rng(RANDOM_SEED)
    shape = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]]) * 1e-3
    data = multivariate_t.rvs(loc=[0, 0, 0], shape=shape, df=5, size=4000,
                              random_state=rng)
    X = pd.DataFrame(data, columns=list("ABC"))
    g = GaussianModel.fit(X)
    t = StudentTModel.fit(X)
    assert 2.5 < t.df < 12, f"оценка ν вне разумного диапазона: {t.df}"
    assert t.loglik > g.loglik, "t должна давать больший log-lik на тяжёлых хвостах"
    assert t.aic < g.aic, "t должна быть лучше по AIC"
    print(f"ok  t     : ν≈{t.df:.1f}, t-Стьюдента бьёт нормаль по AIC на t(5)-данных")


def test_ewma_recursion():
    from rm.models import EWMAModel

    cal = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(RANDOM_SEED)
    X = pd.DataFrame(rng.normal(scale=0.01, size=(60, 2)), index=cal, columns=["A", "B"])
    m = EWMAModel.fit(X, lam=0.94, burn=5)
    # ручная проверка рекурсии последнего шага
    V = X.to_numpy()
    Sigma = np.cov(V, rowvar=False, ddof=0)
    for t in range(len(V)):
        r = V[t][:, None]
        Sigma = 0.94 * Sigma + 0.06 * (r @ r.T)
    assert np.allclose(m.next_cov, Sigma, atol=1e-12), "EWMA next_cov не совпал с ручной рекурсией"
    assert (m.factor_vols() > 0).all()
    print("ok  ewma  : рекурсия RiskMetrics и прогноз σ на завтра корректны")


def test_ar1_recovers_phi():
    from rm.models import fit_ar1

    rng = np.random.default_rng(RANDOM_SEED)
    n, phi_true = 4000, 0.90
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.0 + phi_true * x[t - 1] + rng.normal(scale=0.5)
    r = fit_ar1(pd.Series(x, name="f"))
    assert abs(r.phi - phi_true) < 0.05, f"φ не восстановлен: {r.phi}"
    assert r.mean_reverting and 4 < r.half_life < 10  # ln2/(-ln0.9) ≈ 6.6
    print(f"ok  ar1   : φ≈{r.phi:.3f}, полураспад≈{r.half_life:.1f} дн.")


def test_garch_persistence():
    from rm.models import CCCGarchModel

    rng = np.random.default_rng(RANDOM_SEED)
    n = 1500
    omega, alpha, beta = 0.05, 0.10, 0.85   # персистентность 0.95
    e = np.zeros(n)
    sig2 = np.full(n, omega / (1 - alpha - beta))
    for t in range(1, n):
        sig2[t] = omega + alpha * e[t - 1] ** 2 + beta * sig2[t - 1]
        e[t] = np.sqrt(sig2[t]) * rng.normal()
    cal = pd.bdate_range("2020-01-01", periods=n)
    X = pd.DataFrame({"f": e}, index=cal)
    m = CCCGarchModel.fit(X, dist="normal")
    pers = m.factors[0].persistence
    assert 0.6 < pers < 1.0, f"персистентность вне диапазона: {pers}"
    assert m.next_cov.shape == (1, 1) and m.next_cov[0, 0] > 0
    print(f"ok  garch : α+β≈{pers:.3f} (истинная 0.95), CCC-ковариация положительна")


# --------------------------------------------------------------------------- #
# Этап 4: ценообразование (curve / bonds / black76)                            #
# --------------------------------------------------------------------------- #
def test_curve_interp_and_discount():
    from rm.pricing import discount_factor, interp_zero

    curve = pd.Series({0.25: 0.08, 1.0: 0.08, 5.0: 0.08, 10.0: 0.08, 30.0: 0.08})
    assert abs(interp_zero(curve, 3.0) - 0.08) < 1e-12
    assert abs(interp_zero(curve, 100.0) - 0.08) < 1e-12   # плоская экстраполяция
    assert abs(discount_factor(2.0, curve, "annual") - 1.08 ** -2) < 1e-12
    assert abs(discount_factor(2.0, curve, "continuous") - np.exp(-0.08 * 2)) < 1e-12
    print("ok  curve : интерполяция и дисконт-факторы корректны")


def test_bond_par_pricing():
    from rm.pricing.bonds import bond_cashflows, price_from_curve

    asof = pd.Timestamp("2025-01-01")
    cdates = [asof + pd.Timedelta(days=365 * k) for k in (1, 2, 3)]
    coupons = pd.DataFrame({"coupondate": cdates, "value": [80.0, 80.0, 80.0]})
    amorts = pd.DataFrame({"amortdate": [cdates[-1]], "value": [1000.0]})
    curve = pd.Series({0.25: 0.08, 1.0: 0.08, 5.0: 0.08, 30.0: 0.08})
    cf = bond_cashflows(coupons, amorts, asof)
    assert len(cf) == 4 and cf["kind"].tolist().count("номинал") == 1
    price = price_from_curve(cf, curve, "annual")
    assert abs(price - 1000.0) < 0.5, f"8% облигация при 8% кривой ≠ номинал: {price}"
    print(f"ok  bond  : купон=ставке -> цена ≈ номинал ({price:.2f})")


def test_bond_duration_and_ytm():
    from rm.pricing.bonds import duration_convexity, pv_at_yield, ytm_from_price

    # бескупонная: дюрация Маколея = срок до погашения
    zcb = pd.DataFrame({"date": [pd.Timestamp("2030-01-01")], "cf": [1000.0],
                        "kind": ["номинал"], "t": [5.0]})
    mac, mod, conv = duration_convexity(zcb, 0.08)
    assert abs(mac - 5.0) < 1e-9 and abs(mod - 5.0 / 1.08) < 1e-9
    # YTM round-trip
    y = ytm_from_price(zcb, pv_at_yield(zcb, 0.10))
    assert abs(y - 0.10) < 1e-6
    print("ok  dur   : дюрация ZCB = сроку, YTM восстанавливается")


def test_black76_parity_and_iv():
    from rm.pricing.black76 import black76_implied_vol, black76_price

    F, K, T, sigma, r = 105.0, 100.0, 0.5, 0.25, 0.05
    call = black76_price(F, K, T, sigma, r, "call")
    put = black76_price(F, K, T, sigma, r, "put")
    parity = np.exp(-r * T) * (F - K)
    assert abs((call - put) - parity) < 1e-9, "пут-колл паритет нарушен"
    iv = black76_implied_vol(call, F, K, T, r, "call")
    assert abs(iv - sigma) < 1e-6, f"implied vol не восстановлен: {iv}"
    print(f"ok  black76: пут-колл паритет и implied vol (σ={iv:.3f}) корректны")


def test_risk_measures_sign_and_tail():
    from rm.risk import risk_measures

    pnl = pd.Series([100.0, 50.0, 0.0, -10.0, -25.0, -100.0])
    stats = risk_measures(pnl, var_level=0.80, es_level=0.70)
    assert stats["VaR_0.800"] > 0
    assert stats["ES_0.700"] > 0
    assert stats["min_pnl"] == -100.0
    print("ok  risk  : VaR/ES считаются как положительные потери")


def test_portfolio_state_equity_fx_pnl():
    from rm.config import EQUITY_NOTIONAL_EACH, FX_NOTIONAL_EACH, PORTFOLIO
    from rm.risk.portfolio import RATE_COLUMNS, PortfolioState

    equity_prices = pd.Series({s: 100.0 for s in PORTFOLIO.stocks})
    fx_rates = pd.Series({c: 100.0 for c in PORTFOLIO.fx})
    state = PortfolioState(
        asof=pd.Timestamp("2025-01-01"),
        base_curve=pd.Series({0.25: 0.10, 1.0: 0.10, 2.0: 0.10}),
        rate_loadings=pd.DataFrame(np.eye(3), index=[0.25, 1.0, 2.0], columns=["PC1", "PC2", "PC3"]),
        rate_mean=pd.Series({0.25: 0.0, 1.0: 0.0, 2.0: 0.0}),
        bond_cashflows={},
        bond_units={},
        bond_base_prices={},
        equity_prices=equity_prices,
        equity_units={s: EQUITY_NOTIONAL_EACH / 100.0 for s in PORTFOLIO.stocks},
        fx_rates=fx_rates,
        fx_units={c: FX_NOTIONAL_EACH / 100.0 for c in PORTFOLIO.fx},
        factor_columns=list(RATE_COLUMNS) + [f"eq_{s}" for s in PORTFOLIO.stocks] + [f"fx_{c}" for c in PORTFOLIO.fx],
    )
    scenario = pd.Series(0.0, index=state.factor_columns)
    scenario["eq_SBER"] = np.log(1.01)
    scenario["fx_USD"] = np.log(0.99)
    comps = state.pnl_components(scenario)
    assert round(float(comps["equities"].iloc[0])) == round(EQUITY_NOTIONAL_EACH * 0.01)
    assert round(float(comps["fx"].iloc[0])) == round(FX_NOTIONAL_EACH * -0.01)
    assert comps["total"].iloc[0] < 0
    print("ok  pnl   : сценарный P&L акций/FX имеет правильный знак")


def test_backtest_kupiec_smoke():
    from rm.backtest import summarize_backtest

    exceptions = np.array([False] * 248 + [True] * 2)
    var_series = np.full(250, 7.0e6)
    res = summarize_backtest(exceptions, var_level=0.99, var_series=var_series)
    assert res.n_obs == 250 and res.n_exceptions == 2
    for p in (res.kupiec_pvalue, res.christoffersen_pvalue, res.cc_pvalue, res.dq_pvalue):
        assert np.isnan(p) or 0.0 <= p <= 1.0
    print("ok  bt    : Kupiec/IND/CC/DQ summary формируется")


def test_backtest_independence_detects_clustering():
    from rm.backtest import christoffersen_independence

    clustered = np.array([False] * 240 + [True] * 10)   # 10 пробоев подряд
    _, p_ind = christoffersen_independence(clustered)
    assert p_ind < 0.05, f"кластеризация пробоев должна отвергаться IND: p={p_ind}"
    print("ok  ind   : Christoffersen ловит кластеризацию пробоев")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    test_config()
    test_cache()
    test_svensson_monotone_sanity()
    test_fx_xml_parser()
    test_cbr_zcyc_curve_values_parser()
    test_gcurve_cached_unit_normalisation()
    test_moex_clean_history_numeric_conversion()
    test_align_logic()
    # --- этап 2: факторы ---
    test_returns_basics()
    test_splice_returns()
    test_clean_for_pca_reports()
    test_pca_curve_level_slope_curvature()
    test_pca_reconstruct_roundtrip()
    test_descriptive_heavy_tails_and_normality()
    test_descriptive_stationarity()
    test_descriptive_vol_clustering()
    # --- этап 3: модели ---
    test_gaussian_model()
    test_student_t_beats_normal_on_heavy_tails()
    test_ewma_recursion()
    test_ar1_recovers_phi()
    test_garch_persistence()
    # --- этап 4: ценообразование ---
    test_curve_interp_and_discount()
    test_bond_par_pricing()
    test_bond_duration_and_ytm()
    test_black76_parity_and_iv()
    # --- этапы 5-6: риск и бэктест ---
    test_risk_measures_sign_and_tail()
    test_portfolio_state_equity_fx_pnl()
    test_backtest_kupiec_smoke()
    test_backtest_independence_detects_clustering()
    print("\nВСЕ SMOKE-ТЕСТЫ ПРОЙДЕНЫ ✓  (сетевые загрузки не выполнялись)")
