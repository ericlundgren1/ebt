import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ebt.futures import (
    build_schedule, select_contracts, detect_rolls, contract_mapping, continuous_prices,
    continuous_weights, decompose_positions, interpolate_daily, daily_ohlcv, weighted_returns,
    map_to_continuous, top_n_weights, roll_spreads, apply_panama, microprice,
)

# Three quarterly contracts; each trades at base(t) + a fixed offset, so roll gaps are known.
EXPIRIES = {"A": dt.date(2024, 3, 15), "B": dt.date(2024, 6, 14), "C": dt.date(2024, 9, 20)}
OFFSETS = {"A": 0.0, "B": 2.0, "C": 5.0}


def _base(ts):
    return 100.0 + np.arange(len(ts)) * 0.01


@pytest.fixture
def bars():
    ts = pd.date_range("2024-01-02 09:00", "2024-07-31 16:00", freq="1h")
    ts = ts[(ts.dayofweek < 5) & (ts.hour >= 9) & (ts.hour <= 16)]
    base = _base(ts)
    frames = []
    for c, exp in EXPIRIES.items():
        live = ts.date <= exp
        frames.append(pl.DataFrame({
            "time": ts[live], "contract": c, "close": base[live] + OFFSETS[c],
            "volume": np.full(live.sum(), 10.0 if c != "C" else 1.0),
        }))
    return pl.concat(frames).sort(["time", "contract"])


def _front(bars, **kw):
    return contract_mapping(bars, EXPIRIES, min_dte=7, **kw)


def test_schedule_live_from_first_observation_to_expiry(bars):
    s = build_schedule(bars, EXPIRIES)
    a = s.filter(pl.col("contract") == "A")
    assert a["date"].min() == dt.date(2024, 1, 2)
    assert a["date"].max() == dt.date(2024, 3, 15)
    row = s.filter((pl.col("contract") == "B") & (pl.col("date") == dt.date(2024, 6, 3)))
    assert row["days_to_expiry"][0] == 11


def test_front_rolls_when_min_dte_reached(bars):
    m = _front(bars)
    rolls = detect_rolls(m)
    assert rolls["roll_date"].to_list() == [dt.date(2024, 3, 11), dt.date(2024, 6, 10)]
    assert rolls["old_contract"].to_list() == ["A", "B"]
    assert rolls["new_contract"].to_list() == ["B", "C"]
    assert m.filter(pl.col("date") == dt.date(2024, 3, 8))["contract"][0] == "A"  # 7 days left: still A


def test_expiry_rank_and_target_days(bars):
    s = build_schedule(bars, EXPIRIES)
    r2 = select_contracts(s, expiry_rank=[2], min_dte=7, name="X")
    assert r2["series"].unique().to_list() == ["X(rank2)"]
    assert r2.filter(pl.col("date") == dt.date(2024, 2, 1))["contract"][0] == "B"
    t = select_contracts(s, target_days=[120], min_dte=7)
    # 2024-02-01: B has 134 days, C has 233 → B is closest to 120
    assert t.filter(pl.col("date") == dt.date(2024, 2, 1))["contract"][0] == "B"
    with pytest.raises(ValueError):
        select_contracts(s, target_days=[30], expiry_rank=[1])


def test_monotonic_expiration_blocks_backward_roll(bars):
    # Z expires before C but first appears after the front has moved to C.
    late = bars.filter((pl.col("contract") == "C") & (pl.col("time") >= pd.Timestamp("2024-07-01")))
    late = late.with_columns(pl.lit("Z").alias("contract"))
    exp = {**EXPIRIES, "Z": dt.date(2024, 8, 30)}
    m = contract_mapping(pl.concat([bars, late]), exp, min_dte=7)
    after = m.filter(pl.col("date") >= dt.date(2024, 7, 1))
    assert set(after["contract"].to_list()) == {"C"}


def test_instant_roll_with_panama_is_seamless(bars):
    px = continuous_prices(bars, EXPIRIES, roll="instant", panama=True, min_dte=7, name="X")
    ts = px["time"].to_pandas()
    full_ts = bars.filter(pl.col("contract") == "A")["time"].to_pandas()
    base = pd.Series(_base(pd.DatetimeIndex(bars["time"].unique().sort().to_pandas())),
                     index=bars["time"].unique().sort().to_pandas())
    expected = base.reindex(ts).values + OFFSETS["C"]
    np.testing.assert_allclose(px["price"].to_numpy(), expected, atol=1e-9)
    assert px["series"].unique().to_list() == ["X(front)"]
    assert len(full_ts) > 0


def test_unadjusted_and_blend(bars):
    raw = continuous_prices(bars, EXPIRIES, roll=None, min_dte=7)
    day = raw.filter(pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11))
    base = bars.filter((pl.col("contract") == "B") & (pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11)))
    np.testing.assert_allclose(day["price"].to_numpy(), base["close"].to_numpy())

    bl = continuous_prices(bars, EXPIRIES, roll="blend", panama=False, min_dte=7)
    d = bl.filter(pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11))
    a = bars.filter((pl.col("contract") == "A") & (pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11)))
    b = bars.filter((pl.col("contract") == "B") & (pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11)))
    w = np.linspace(0, 1, d.height)
    np.testing.assert_allclose(d["price"].to_numpy(), (1 - w) * a["close"].to_numpy() + w * b["close"].to_numpy())


def test_blend_keeps_other_series_on_roll_days(bars):
    # front rolls on 03-11 and 06-10; the 120d series rolls on other days
    px = continuous_prices(bars, EXPIRIES, target_days=[0, 120], roll="blend", min_dte=7)
    front_rolls = {dt.date(2024, 3, 11), dt.date(2024, 6, 10)}
    m = contract_mapping(bars, EXPIRIES, target_days=[120], min_dte=7)
    assert not front_rolls & set(detect_rolls(m)["roll_date"].to_list())
    dates = set(px.filter(pl.col("series") == "120d")["time"].cast(pl.Date).to_list())
    assert front_rolls <= dates


def test_column_mapping_and_pandas_input(bars):
    pdf = bars.rename({"time": "ts", "contract": "symbol", "close": "px"}).to_pandas()
    exp = pd.DataFrame({"symbol": list(EXPIRIES), "expiration": list(EXPIRIES.values())})
    cols = {"time": "ts", "contract": "symbol", "price": "px", "expiry": "expiration"}
    px = continuous_prices(pdf, exp, roll="instant", panama=True, min_dte=7, columns=cols)
    ref = continuous_prices(bars, EXPIRIES, roll="instant", panama=True, min_dte=7)
    assert px.columns == ["ts", "series", "price"]
    np.testing.assert_allclose(px["price"].to_numpy(), ref["price"].to_numpy())
    m = contract_mapping(pdf, exp, min_dte=7, columns=cols)
    assert "symbol" in m.columns


def test_interpolation_weights(bars):
    w = continuous_weights(bars, EXPIRIES, target_days=[60], min_dte=7, name="X")
    sums = w.group_by("date").agg(pl.col("weight").sum())
    np.testing.assert_allclose(sums["weight"].to_numpy(), 1.0)
    day = w.filter(pl.col("date") == dt.date(2024, 2, 1)).sort("days_to_expiry")
    # A: 43 days, B: 134 days → w_A = (134-60)/(134-43)
    assert day["contract"].to_list() == ["A", "B"]
    np.testing.assert_allclose(day["weight"].to_list(), [74 / 91, 17 / 91])


def test_top_n_weights_lagged_and_ratcheted(bars):
    oi = pl.DataFrame({
        "time": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-02-02", "2024-02-02"]) + pd.Timedelta(hours=17),
        "contract": ["A", "B", "A", "B"],
        "open_interest": [60.0, 40.0, 80.0, 20.0],
    })
    w = continuous_weights(bars, EXPIRIES, top_n=2, values=oi, min_dte=7, name="X")
    assert w["series"].unique().to_list() == ["X(top2)"]
    d1 = w.filter(pl.col("date") == dt.date(2024, 2, 2)).sort("contract")  # from 02-01 OI
    np.testing.assert_allclose(d1["weight"].to_list(), [0.6, 0.4])
    d2 = w.filter(pl.col("date") == dt.date(2024, 2, 5)).sort("contract")  # front A capped at 0.6
    np.testing.assert_allclose(d2["weight"].to_list(), [0.6, 0.4])

    vol = continuous_weights(bars, EXPIRIES, top_n=1, min_dte=7, value_agg="sum", label="vol",
                             columns={"weight": "volume"})
    assert vol["series"][0] == "vol1"
    np.testing.assert_allclose(vol.group_by("date").agg(pl.col("weight").sum())["weight"].to_numpy(), 1.0)


def test_decompose_positions_round_trip(bars):
    m = _front(bars, name="X")
    idx = pd.date_range("2024-03-07 09:00", "2024-03-12 16:00", freq="1h")
    pos = pd.DataFrame({"X(front)": 1.0}, index=idx)
    out = decompose_positions(pos, m)
    assert {"A", "B"} <= set(out.columns)
    assert out.loc["2024-03-08 16:00", "A"] == 1.0
    assert out.loc["2024-03-09 00:00", "A"] == 0.0  # close marker after the active period
    assert np.isnan(out.loc["2024-03-10 00:00", "A"])
    assert out.loc["2024-03-11 09:00", "B"] == 1.0
    assert np.isnan(out.loc["2024-03-08 16:00", "B"])


def test_interpolate_daily_and_daily_ohlcv(bars):
    d = daily_ohlcv(bars.with_columns(open=pl.col("close"), high=pl.col("close"), low=pl.col("close")))
    d = d.join(build_schedule(bars, EXPIRIES).select("date", "contract", "days_to_expiry"), on=["date", "contract"])
    out = interpolate_daily(d, target_days=[60], name="X")
    row = out.filter(pl.col("date") == dt.date(2024, 2, 1))
    a = d.filter((pl.col("date") == dt.date(2024, 2, 1)) & (pl.col("contract") == "A"))["close"][0]
    b = d.filter((pl.col("date") == dt.date(2024, 2, 1)) & (pl.col("contract") == "B"))["close"][0]
    np.testing.assert_allclose(row["close"][0], 74 / 91 * a + 17 / 91 * b)


def test_weighted_returns_matches_single_contract(bars):
    w = continuous_weights(bars, EXPIRIES, expiry_rank=[1], min_dte=7)
    r = weighted_returns(bars, w)
    mapped = map_to_continuous(bars, contract_mapping(bars, EXPIRIES, expiry_rank=[1], min_dte=7))
    diffs = mapped.with_columns(pl.col("close").diff().over(pl.col("time").cast(pl.Date)).alias("d"))
    np.testing.assert_allclose(r["return"].fill_null(0).to_numpy(), diffs["d"].fill_null(0).to_numpy())


# ── regression tests ──────────────────────────────────────────────────

def _seamless_error(px, bars):
    """continuous price minus the true seamless level (base + C's offset), per row."""
    full = bars["time"].unique().sort().to_pandas()
    base = pd.Series(_base(pd.DatetimeIndex(full)), index=full)
    return px["price"].to_numpy() - (base.reindex(px["time"].to_pandas()).values + OFFSETS["C"])


def test_blend_with_panama_is_seamless_on_roll_days(bars):
    # default arguments: roll='blend', panama=True. Before the fix the roll day opened
    # `spread` below the previous close and drifted back to 0 by the close.
    px = continuous_prices(bars, EXPIRIES, min_dte=7)
    np.testing.assert_allclose(_seamless_error(px, bars), 0.0, atol=1e-9)
    roll_day = px.filter(pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11))
    assert roll_day.height == 8  # every common timestamp of the roll day survives the blend
    # blend without panama still starts on the old contract and ends on the new one
    raw = continuous_prices(bars, EXPIRIES, min_dte=7, panama=False)
    d = raw.filter(pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11))["price"].to_numpy()
    a = bars.filter((pl.col("contract") == "A") & (pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11)))["close"].to_numpy()
    b = bars.filter((pl.col("contract") == "B") & (pl.col("time").cast(pl.Date) == dt.date(2024, 3, 11)))["close"].to_numpy()
    assert d[0] == a[0] and d[-1] == b[-1]


def test_trading_date_uses_local_date_for_tz_aware_stamps(bars):
    # 18:00 Chicago on Jan 2 is Jan 3 in UTC; the trading date must be Jan 2.
    local = bars.with_columns(pl.col("time").dt.replace_time_zone("America/Chicago"))
    s_local = build_schedule(local, EXPIRIES)
    s_naive = build_schedule(bars, EXPIRIES)
    assert s_local.equals(s_naive)
    evening = pl.DataFrame({"time": pd.to_datetime(["2024-01-02 18:00", "2024-01-02 23:30"]).tz_localize("America/Chicago"),
                            "contract": "A", "close": 1.0})
    sched = build_schedule(evening, {"A": dt.date(2024, 3, 15)})
    assert sched["date"].to_list() == [dt.date(2024, 1, 2)]
    # decompose_positions puts tz-aware stamps on the same trading date as the bars
    m = contract_mapping(local, EXPIRIES, min_dte=7, name="X")
    idx = pd.date_range("2024-03-08 09:00", "2024-03-11 16:00", freq="1h", tz="America/Chicago")
    out = decompose_positions(pd.DataFrame({"X(front)": 1.0}, index=idx), m)
    assert out.loc["2024-03-08 23:00", "A"] == 1.0 and np.isnan(out.loc["2024-03-08 23:00", "B"])
    assert out.loc["2024-03-11 09:00", "B"] == 1.0


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_right_labelled_midnight_rule_is_resolution_independent(unit):
    # Only a stamp exactly at midnight moves to the previous day; anything after it stays.
    # (Before: 1µs was subtracted, which is 0 in a datetime[ms] column — right_labelled silently did nothing.)
    stamps = pd.DatetimeIndex(["2024-01-02 23:00", "2024-01-03 00:00", "2024-01-03 00:00:00.000001", "2024-01-03 01:00"])
    df = (pl.DataFrame({"time": stamps, "contract": "A", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
          .with_columns(pl.col("time").dt.cast_time_unit(unit)))
    def counts(rl):
        d = daily_ohlcv(df, right_labelled=rl)
        return dict(zip(d["date"].to_list(), d["volume"].to_list()))
    assert counts(False) == {dt.date(2024, 1, 2): 1.0, dt.date(2024, 1, 3): 3.0}
    # in ms the 1µs stamp *is* 00:00, so it is a midnight stamp too
    assert counts(True) == ({dt.date(2024, 1, 2): 3.0, dt.date(2024, 1, 3): 1.0} if unit == "ms"
                            else {dt.date(2024, 1, 2): 2.0, dt.date(2024, 1, 3): 2.0})


def test_right_labelled_midnight_stamp_stays_with_previous_day():
    # hourly right-labelled bars covering 24/7, two contracts, roll at midnight.
    ts = pd.date_range("2024-03-07 01:00", "2024-03-13 00:00", freq="1h")  # stamps at the END of each hour
    exp = {"A": dt.date(2024, 3, 15), "B": dt.date(2024, 6, 14)}
    frames = [pl.DataFrame({"time": ts, "contract": c, "close": 100.0 + i}) for i, c in enumerate(exp)]
    bars24 = pl.concat(frames).sort(["time", "contract"])
    m = contract_mapping(bars24, exp, min_dte=7, name="X", right_labelled=True)
    rolls = detect_rolls(m)
    assert rolls["roll_date"].to_list() == [dt.date(2024, 3, 9)]  # A has 6 days left on 03-09 → B
    # the bar stamped 03-09 00:00 is the last bar of 03-08 and still belongs to A …
    px = continuous_prices(bars24, exp, min_dte=7, roll="instant", panama=False, right_labelled=True)
    assert px.filter(pl.col("time") == pd.Timestamp("2024-03-09 00:00"))["price"][0] == 100.0
    assert px.filter(pl.col("time") == pd.Timestamp("2024-03-09 01:00"))["price"][0] == 101.0
    # … and so does the position at that stamp
    pos = pd.DataFrame({"X(front)": 1.0}, index=ts)
    out = decompose_positions(pos, m, right_labelled=True)
    assert out.loc["2024-03-09 00:00", "A"] == 1.0 and np.isnan(out.loc["2024-03-09 00:00", "B"])
    assert out.loc["2024-03-09 01:00", "A"] == 0.0 and out.loc["2024-03-09 01:00", "B"] == 1.0
    # without right_labelled the same stamp is attributed to B: one bar on the wrong contract
    wrong = decompose_positions(pos, m, right_labelled=False)
    assert wrong.loc["2024-03-09 00:00", "B"] == 1.0


def test_decompose_positions_carries_nan_forward_like_backtest(bars):
    m = _front(bars, name="X")
    idx = pd.date_range("2024-02-01 09:00", "2024-02-01 16:00", freq="1h")
    pos = pd.DataFrame({"X(front)": [np.nan, 1.0, np.nan, np.nan, -2.0, 0.0, np.nan, 1.0]}, index=idx)
    out = decompose_positions(pos, m)["A"]
    expected = [np.nan, 1.0, 1.0, 1.0, -2.0, 0.0, np.nan, 1.0]  # NaN after an explicit 0 stays flat
    np.testing.assert_array_equal(out.to_numpy(), np.array(expected))
    # a NaN that runs into a roll is carried onto the new contract
    idx2 = pd.date_range("2024-03-08 09:00", "2024-03-11 16:00", freq="1h")
    pos2 = pd.DataFrame({"X(front)": np.where(idx2 <= pd.Timestamp("2024-03-08 09:00"), 1.0, np.nan)}, index=idx2)
    out2 = decompose_positions(pos2, m)
    assert out2.loc["2024-03-11 09:00", "B"] == 1.0 and out2.loc["2024-03-08 16:00", "A"] == 1.0


def test_interpolation_weights_use_schedule_not_data_presence(bars):
    # A has no bars on 2024-02-01: weights (and the front) must not change that day.
    gap = bars.filter(~((pl.col("contract") == "A") & (pl.col("time").cast(pl.Date) == dt.date(2024, 2, 1))))
    w = continuous_weights(gap, EXPIRIES, target_days=[0, 60], min_dte=7, name="X")
    day = w.filter(pl.col("date") == dt.date(2024, 2, 1)).sort([pl.col("series"), pl.col("contract")])
    assert day.filter(pl.col("series") == "X(front)")["contract"].to_list() == ["A"]
    sixty = day.filter(pl.col("series") == "X(60d)")
    assert sixty["contract"].to_list() == ["A", "B"]
    np.testing.assert_allclose(sixty["weight"].to_list(), [74 / 91, 17 / 91])
    # front weights are exactly contract_mapping (incl. monotonic-expiration enforcement)
    cm = contract_mapping(gap, EXPIRIES, min_dte=7, name="X").select("date", "series", "contract", "weight", "days_to_expiry")
    assert w.filter(pl.col("series") == "X(front)").sort("date").equals(cm.sort("date"))


def test_decompose_positions_fractional_weights(bars):
    w = continuous_weights(bars, EXPIRIES, target_days=[60], min_dte=7, name="X")
    idx = pd.date_range("2024-02-01 09:00", "2024-02-01 16:00", freq="1h")
    out = decompose_positions(pd.DataFrame({"X(60d)": 2.0}, index=idx), w)
    np.testing.assert_allclose(out["A"].to_numpy(), 2.0 * 74 / 91)
    np.testing.assert_allclose(out["B"].to_numpy(), 2.0 * 17 / 91)


def test_roll_spreads_and_apply_panama_wrappers_match_continuous_prices(bars):
    m = contract_mapping(bars, EXPIRIES, min_dte=7)
    rolls = detect_rolls(m)
    sp = roll_spreads(rolls, bars)
    assert sp["spread"].to_list() == [2.0, 3.0]
    raw = continuous_prices(bars, EXPIRIES, min_dte=7, roll=None)
    adj = apply_panama(raw, sp)
    ref = continuous_prices(bars, EXPIRIES, min_dte=7, roll="instant", panama=True)
    np.testing.assert_allclose(adj.sort("time")["price"].to_numpy(), ref.sort("time")["price"].to_numpy())


def test_blend_join_nearest_matches_exact_on_aligned_bars(bars):
    ex = continuous_prices(bars, EXPIRIES, min_dte=7, blend_join="exact")
    near = continuous_prices(bars, EXPIRIES, min_dte=7, blend_join="nearest")
    np.testing.assert_allclose(near["price"].to_numpy(), ex["price"].to_numpy())


def test_weighted_returns_vol_is_lagged_one_day(bars):
    w = continuous_weights(bars, EXPIRIES, expiry_rank=[1], min_dte=7)
    r = weighted_returns(bars, w, normalize=True, return_vol=True)
    per_day = r.with_columns(pl.col("time").cast(pl.Date).alias("d")).group_by("d").agg(pl.col("vol").n_unique().alias("n"), pl.col("vol").first())
    assert (per_day["n"] == 1).all()  # one vol per day
    # vol used on day D is a function of data through D-1: recomputing from a frame cut at D gives the same value
    d = dt.date(2024, 5, 1)
    cut = weighted_returns(bars.filter(pl.col("time").cast(pl.Date) < d), w, return_vol=True)
    full_vol = r.filter(pl.col("time").cast(pl.Date) == d)["vol"][0]
    prev_vol = cut.filter(pl.col("time").cast(pl.Date) == dt.date(2024, 4, 30))["vol"][0]
    assert full_vol is not None and np.isfinite(full_vol)
    assert full_vol != prev_vol  # consecutive days differ (vol is not constant)
    # and the ratio normalized / raw equals 1 / vol on every row
    raw = weighted_returns(bars, w)
    ok = r.filter(pl.col("vol").is_not_null()).join(raw, on=["time", "series"], suffix="_raw")
    np.testing.assert_allclose(ok["return"].to_numpy() * ok["vol"].to_numpy(), ok["return_raw"].to_numpy(), atol=1e-12)


def test_microprice_expression():
    q = pl.DataFrame({"bid_price": [99.0, 99.0], "ask_price": [101.0, 101.0], "bid_size": [3.0, 0.0], "ask_size": [1.0, 0.0]})
    np.testing.assert_allclose(q.select(microprice()).to_series().to_list(), [99.0 * 0.25 + 101.0 * 0.75, 100.0])


def test_top_n_values_carried_forward_while_live_and_weights_sum_to_one(bars):
    # B's OI row is missing on 02-02; its 02-01 value is carried forward (≤ 5 trading days).
    oi = pl.DataFrame({
        "time": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-05"]) + pd.Timedelta(hours=17),
        "contract": ["A", "B", "A", "A", "B"], "open_interest": [60.0, 40.0, 80.0, 80.0, 20.0],
    })
    w = continuous_weights(bars, EXPIRIES, top_n=2, values=oi, min_dte=7)
    d = w.filter(pl.col("date") == dt.date(2024, 2, 5)).sort("contract")
    assert d["contract"].to_list() == ["A", "B"]
    np.testing.assert_allclose(d["weight"].to_list(), [0.6, 0.4])  # 80/120 capped by the ratchet at 0.6
    np.testing.assert_allclose(w.group_by("date").agg(pl.col("weight").sum())["weight"].to_numpy(), 1.0)

    # a gap longer than the limit drops the contract; the lone front then carries 100% (no cap)
    oi2 = pl.DataFrame({
        "time": pd.to_datetime(["2024-02-01", "2024-02-01"] + ["2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07",
                                                               "2024-02-08", "2024-02-09", "2024-02-12"]) + pd.Timedelta(hours=17),
        "contract": ["A", "B"] + ["A"] * 7, "open_interest": [60.0, 40.0] + [80.0] * 7,
    })
    w2 = continuous_weights(bars, EXPIRIES, top_n=2, values=oi2, min_dte=7, value_ffill_limit=5)
    by_date = {d: sorted(g["contract"].to_list()) for (d,), g in w2.group_by("date")}
    assert by_date[dt.date(2024, 2, 9)] == ["A", "B"]   # from 02-08: 5th carried day
    assert by_date[dt.date(2024, 2, 12)] == ["A"]        # from 02-09: carry limit exceeded
    assert w2.filter(pl.col("date") == dt.date(2024, 2, 12))["weight"][0] == 1.0
    np.testing.assert_allclose(w2.group_by("date").agg(pl.col("weight").sum())["weight"].to_numpy(), 1.0)
    w0 = continuous_weights(bars, EXPIRIES, top_n=2, values=oi2, min_dte=7, value_ffill_limit=0)
    assert sorted(w0.filter(pl.col("date") == dt.date(2024, 2, 5))["contract"].to_list()) == ["A"]


def test_roll_into_contract_with_short_history_raises(bars):
    # B's first print is 3 days before the 03-11 roll: refused by default, allowed with min_history=0
    short = bars.filter(~((pl.col("contract") == "B") & (pl.col("time") < pd.Timestamp("2024-03-08"))))
    with pytest.raises(ValueError, match="min_history"):
        contract_mapping(short, EXPIRIES, min_dte=7)
    with pytest.raises(ValueError, match="min_history"):
        continuous_prices(short, EXPIRIES, min_dte=7)
    with pytest.raises(ValueError, match="min_history"):
        continuous_weights(short, EXPIRIES, target_days=[0], min_dte=7)
    px = continuous_prices(short, EXPIRIES, min_dte=7, min_history=0)
    np.testing.assert_allclose(_seamless_error(px, bars), 0.0, atol=1e-9)
    # contracts present from the first date of the data are exempt
    late_start = bars.filter(pl.col("time") >= pd.Timestamp("2024-03-08"))
    assert detect_rolls(contract_mapping(late_start, EXPIRIES, min_dte=7))["roll_date"].to_list() == [dt.date(2024, 3, 11), dt.date(2024, 6, 10)]
    # no price at all before the roll: the Panama gap cannot be measured, never silently 0
    none_before = bars.filter(~((pl.col("contract") == "B") & (pl.col("time") < pd.Timestamp("2024-03-11"))))
    with pytest.raises(ValueError, match="no price for B before the roll"):
        continuous_prices(none_before, EXPIRIES, min_dte=7, min_history=0)
    assert continuous_prices(none_before, EXPIRIES, min_dte=7, min_history=0, panama=False).height > 0


def test_select_contracts_honours_expiry_column_mapping(bars):
    sched = build_schedule(bars, EXPIRIES).rename({"contract": "symbol", "expiry": "expiration"})
    cols = {"contract": "symbol", "expiry": "expiration"}
    out = select_contracts(sched, min_dte=7, name="X", columns=cols)
    ref = select_contracts(build_schedule(bars, EXPIRIES), min_dte=7, name="X")
    assert out.rename({"symbol": "contract"}).equals(ref)
