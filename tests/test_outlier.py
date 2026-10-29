import numpy as np
import pandas as pd
import polars as pl
import pytest

from ebt.data.outlier import detect_outliers


def _trades(n=3000, spike_at=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="1s")
    price = 100 + np.cumsum(rng.normal(0, 0.01, n))
    price[spike_at] *= 1.5
    return pd.DataFrame({"price": price, "volume": rng.integers(1, 10, n).astype(float)}, index=idx)


def _bars(n=3000, spike_at=2000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="1min")
    close = 100 + np.cumsum(rng.normal(0, 0.02, n))
    close[spike_at] *= 1.5
    return pd.DataFrame({"open": close, "high": close + 0.05, "low": close - 0.05, "close": close,
                         "volume": rng.integers(1, 100, n).astype(float)}, index=idx)


def _quotes(n=3000, spike_at=2000, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="1s")
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    bid, ask = mid - 0.01, mid + 0.01
    bid[spike_at] *= 1.5
    ask[spike_at] *= 1.5
    return pd.DataFrame({"bid_price": bid, "ask_price": ask}, index=idx)


def _rejected(mask):
    return np.flatnonzero(~np.asarray(mask)).tolist()


# --- Rust detectors ---

@pytest.mark.parametrize("make,kind", [(_trades, "trades"), (_bars, "bars"), (_quotes, "quotes")])
def test_rust_rejects_spike(make, kind):
    df = make()
    mask = detect_outliers(df)  # kind auto-detected
    assert mask.dtype == bool and len(mask) == len(df)
    assert _rejected(mask) == [2000]  # the spike and nothing else
    np.testing.assert_array_equal(detect_outliers(df, kind=kind), mask)


def test_rust_polars_and_column_mapping():
    df = _trades()
    ref = detect_outliers(df)
    pldf = pl.from_pandas(df.rename_axis("ts").reset_index().rename(columns={"price": "px", "volume": "qty"}))
    mask = detect_outliers(pldf, kind="trades", columns={"time": "ts", "price": "px", "volume": "qty"})
    np.testing.assert_array_equal(mask, ref)
    dbg = detect_outliers(pldf, kind="trades", columns={"price": "px", "volume": "qty"}, return_debug=True)
    assert isinstance(dbg, pl.DataFrame) and dbg.height == len(df)
    assert {"accepted", "level", "votes_against", "regime_switch"} <= set(dbg.columns)
    np.testing.assert_array_equal(dbg["accepted"].to_numpy(), ref)


def test_rust_missing_rows_skipped():
    df = _trades()
    with_nan = df.copy()
    with_nan.iloc[1000, 0] = np.nan
    mask = detect_outliers(with_nan)
    assert not mask[1000]
    np.testing.assert_array_equal(np.delete(mask, 1000), detect_outliers(df.drop(df.index[1000])))


def test_rust_bar_consistency_and_debug():
    df = _bars()
    df.iloc[500, df.columns.get_loc("high")] = df["low"].iloc[500] - 1  # high < low
    dbg = detect_outliers(df, kind="bars", return_debug=True)
    assert isinstance(dbg, pd.DataFrame) and dbg.index.equals(df.index)
    assert not dbg["accepted"].iloc[500]
    assert dbg["bar_invalid_reason"].iloc[500] == 2.0
    assert {"accepted", "bar_invalid_reason", "level", "votes_against", "regime_switch"} <= set(dbg.columns)
    assert dbg.drop(columns="accepted").dtypes.eq(float).all()


def test_rust_config_passthrough():
    df = _trades()
    strict = detect_outliers(df, zscore_threshold=0.5, iqr_multiplier=0.1, min_votes_to_reject=1)
    assert strict.sum() < detect_outliers(df).sum()
    with pytest.raises(TypeError):
        detect_outliers(df, not_a_setting=1)


# --- regression tests ---

@pytest.mark.parametrize("make,kind", [(_trades, "trades"), (_bars, "bars"), (_quotes, "quotes")])
def test_decisions_are_point_in_time(make, kind):
    # The decision for row i must not depend on rows after i: a prefix gives the same prefix of decisions.
    df = make()
    full = detect_outliers(df, kind=kind)
    for k in (1500, 2001, 2500):
        np.testing.assert_array_equal(detect_outliers(df.iloc[:k], kind=kind), full[:k])


@pytest.mark.parametrize("make,kind", [(_trades, "trades"), (_bars, "bars"), (_quotes, "quotes")])
def test_unsorted_time_is_rejected(make, kind):
    df = make().sample(frac=1, random_state=0)
    with pytest.raises(ValueError, match="sorted"):
        detect_outliers(df, kind=kind)
    # a NaT row is skipped, not treated as going backwards
    df = make().rename_axis("ts").reset_index()
    df.loc[1000, "ts"] = pd.NaT
    mask = detect_outliers(df, kind=kind)
    assert not mask[1000] and _rejected(mask) == [1000, 2000]


@pytest.mark.parametrize("make,kind", [(_trades, "trades"), (_bars, "bars"), (_quotes, "quotes")])
def test_negative_spike_rejected(make, kind):
    df = make()
    for c in [c for c in df.columns if c != "volume"]:
        df.iloc[2000, df.columns.get_loc(c)] /= 1.5 * 1.5  # x1/1.5 instead of x1.5
    assert _rejected(detect_outliers(df, kind=kind)) == [2000]


def test_rust_recovers_from_level_shift():
    df = _trades()
    p = df["price"].to_numpy().copy(); p[2000] /= 1.5; p[2000:] *= 1.03; df["price"] = p
    mask = detect_outliers(df, kind="trades")
    rejected = _rejected(mask)
    assert rejected and max(rejected) < 2010  # a few rows after the shift, then the new level is accepted
    assert mask[2010:].all()


# --- level shifts, bursts and warm-up ---

def _random_walk(sigma, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="1s")
    return pd.DataFrame({"price": 100 + np.cumsum(rng.normal(0, sigma, n)), "volume": 1.0}, index=idx)


@pytest.mark.parametrize("sigma", [0.001, 0.01, 0.1])
def test_sensitivity_scales_with_the_noise(sigma):
    # The acceptance band follows the learned variance, not a fixed percentage: a 100σ print
    # is rejected whether σ is 0.001% or 0.1% per tick (before: nothing under ~3% was).
    df = _random_walk(sigma)
    df.iloc[2000, 0] += 100 * sigma
    assert _rejected(detect_outliers(df)) == [2000]
    clean = _random_walk(sigma)
    assert _rejected(detect_outliers(clean)) == []


@pytest.mark.parametrize("make,kind", [(_trades, "trades"), (_bars, "bars"), (_quotes, "quotes")])
@pytest.mark.parametrize("mult", [1.01, 0.98])
def test_small_spike_rejected(make, kind, mult):
    df = make()
    for c in [c for c in df.columns if c != "volume"]:
        df.iloc[2000, df.columns.get_loc(c)] = df.iloc[2000][c] / 1.5 * mult
    assert _rejected(detect_outliers(df, kind=kind)) == [2000]


@pytest.mark.parametrize("make,kind", [(_trades, "trades"), (_bars, "bars"), (_quotes, "quotes")])
def test_recovers_from_level_shift(make, kind):
    df = make()
    for c in [c for c in df.columns if c != "volume"]:
        v = df[c].to_numpy().copy(); v[2000] /= 1.5; v[2000:] *= 1.03; df[c] = v
    rejected = _rejected(detect_outliers(df, kind=kind))
    assert rejected and rejected[0] == 2000 and max(rejected) <= 2002  # at most the first n_switch rows


def test_burst_of_bad_prints():
    df = _trades()
    v = df["price"].to_numpy().copy(); v[2001:2003] *= 1.5; df["price"] = v  # rows 2000..2002 are bad
    # default: 3 confirmations make a regime, so the 3rd bad print is accepted and the
    # next two good ones rejected until the level switches back
    assert _rejected(detect_outliers(df)) == [2000, 2001, 2003, 2004]
    # requiring the confirmations to span 10 s (or 5 of them) rejects the whole burst only
    assert _rejected(detect_outliers(df, min_switch_seconds=10)) == [2000, 2001, 2002]
    assert _rejected(detect_outliers(df, n_switch=5)) == [2000, 2001, 2002]
    # a real level shift still gets through with min_switch_seconds, after 10 s
    shift = _trades(); v = shift["price"].to_numpy().copy(); v[2000] /= 1.5; v[2000:] *= 1.03; shift["price"] = v
    rejected = _rejected(detect_outliers(shift, min_switch_seconds=10))
    assert rejected[0] == 2000 and max(rejected) < 2015 and detect_outliers(shift, min_switch_seconds=10)[2015:].all()


def test_warmup_hard_limit():
    # During warm-up (< warmup_threshold accepted points) the vote threshold is relaxed;
    # a print further than warmup_hard_limit from the last accepted price is rejected anyway.
    assert _rejected(detect_outliers(_trades(spike_at=5))) == [5]
    assert _rejected(detect_outliers(_trades(spike_at=5), warmup_hard_limit=1.0)) == []  # no hard limit
    small = _trades(spike_at=5); small.iloc[5, 0] = small.iloc[5, 0] / 1.5 * 1.05
    assert _rejected(detect_outliers(small)) == []
    assert _rejected(detect_outliers(_quotes(spike_at=5), kind="quotes")) == [5]


def test_crossed_quote_is_invalid():
    q = _quotes()
    q.iloc[2000, :] = [q.iloc[2000]["ask_price"] / 1.5, q.iloc[2000]["bid_price"] / 1.5]  # bid > ask
    assert _rejected(detect_outliers(q, kind="quotes")) == [2000]
    assert _rejected(detect_outliers(q, kind="quotes", allow_crossed=True)) == []
    locked = _quotes(); mid = locked.iloc[2000].mean() / 1.5; locked.iloc[2000, :] = [mid, mid]  # bid == ask at the true level
    assert _rejected(detect_outliers(locked, kind="quotes")) == []


def test_quiet_series_then_one_tick_move_is_accepted():
    # A long run of identical prices must not shrink the band below one tick.
    idx = pd.date_range("2024-01-02", periods=600, freq="1s")
    p = np.full(600, 100.0); p[500:] = 100.25; p[550:] = 100.0
    assert _rejected(detect_outliers(pd.DataFrame({"price": p, "volume": 1.0}, index=idx))) == []
    q = pd.DataFrame({"bid_price": np.full(600, 99.75), "ask_price": np.full(600, 100.0)}, index=idx)
    q.iloc[500:, :] += 0.25
    assert _rejected(detect_outliers(q, kind="quotes")) == []
