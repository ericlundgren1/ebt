"""Streaming outlier detection for trades, quotes and OHLCV bars (Rust kernels).

The detectors are sequential and point-in-time: the decision for a row depends only
on the rows before it. See ``detect_outliers``.
"""

from __future__ import annotations

from typing import Optional, Dict, Literal

import pandas as pd
import numpy as np


Kind = Literal["auto", "bars", "trades", "quotes"]


def detect_kind(df: pd.DataFrame) -> Kind:
    cols = {c.lower() for c in df.columns}
    if {"open", "high", "low", "close"}.issubset(cols):
        return "bars"
    if {"bid", "ask"}.issubset(cols) or {"bid_price", "ask_price"}.issubset(cols):
        return "quotes"
    if "price" in cols:
        return "trades"
    raise ValueError("Could not detect kind. Provide kind='bars'|'trades'|'quotes'.")


_RUST_DEFAULT_COLUMNS = {
    "trades": {"time": None, "price": "price", "volume": "volume"},
    "quotes": {"time": None, "bid": "bid_price", "ask": "ask_price"},
    "bars": {"time": None, "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"},
}


def detect_outliers(df, kind: Kind = "auto", columns: Optional[Dict[str, str]] = None,
                    return_debug: bool = False, **config):
    """Streaming outlier detection with the Rust detectors.

    trades: ensemble of an adaptive level filter, rolling z-score, IQR, price
        jump, time gap and volume checks; rejected when enough methods vote against.
    quotes: mid-price level filter cascaded with a spread upper-bound filter.
    bars: bar consistency checks (non-finite prices, high < low), then the trade
        ensemble on the close.

    Parameters
    ----------
    df : pandas or polars DataFrame, sorted by time. A pandas DatetimeIndex is used
        as the time column when ``columns`` doesn't name one.
    kind : "auto" | "trades" | "quotes" | "bars"
    columns : maps roles to column names, e.g. ``{"time": "ts", "price": "px"}``.
        Roles: trades — time, price, volume; quotes — time, bid, ask;
        bars — time, open, high, low, close, volume. Without a time column the
        frame's single datetime column is used. A missing volume column counts as 0.
    return_debug : return a DataFrame (same library as the input) of
        ``accepted`` plus the detector's internal state per row.
    **config : detector settings.
        trades/bars: ``zscore_threshold``, ``iqr_multiplier``, ``min_votes_to_reject``,
        ``window_size``, ``merge_stat_votes``, ``gap_widen_threshold_sec``,
        ``warmup_threshold``, ``min_scale``, ``min_scale_ticks``, ``n_switch``,
        ``min_switch_seconds``, ``warmup_hard_limit``.
        quotes: ``mid_tau_min``, ``mid_tau_max``, ``mid_n_switch``,
        ``spread_max_consecutive_reject``, ``max_spread``, ``warmup_count``,
        ``allow_crossed``, ``mid_min_scale``, ``mid_min_scale_ticks``,
        ``mid_min_switch_seconds``, ``spread_min_scale``, ``spread_min_scale_ticks``,
        ``warmup_hard_limit``.
        The level filter's acceptance band is the learned variance floored at
        ``min_scale_ticks`` learned ticks (not a fixed percentage); a regime switch needs
        ``n_switch`` confirmations spanning ``min_switch_seconds``; during warm-up a
        price further than ``warmup_hard_limit`` from the last accepted one is rejected
        outright; crossed quotes are invalid unless ``allow_crossed``.

    Returns
    -------
    numpy bool array (True = accepted), or the debug DataFrame. Rows with a missing
    value are not accepted and don't update the detector.
    """
    import ebt_rust

    is_polars = type(df).__module__.startswith("polars")
    pdf = df.to_pandas() if is_polars else df
    if kind == "auto":
        kind = detect_kind(pdf)
        if kind == "quotes" and "bid_price" not in pdf.columns:
            kind_cols = {"bid": "bid", "ask": "ask"}
        else:
            kind_cols = {}
    else:
        kind_cols = {}
    if kind not in _RUST_DEFAULT_COLUMNS:
        raise ValueError(f"Unknown kind: {kind}")
    cols = {**_RUST_DEFAULT_COLUMNS[kind], **kind_cols, **(columns or {})}

    if cols["time"] is not None:
        t = pdf[cols["time"]]
    elif pd.api.types.is_datetime64_any_dtype(pdf.index):
        t = pdf.index.to_series()
    else:
        candidates = [c for c in pdf.columns if pd.api.types.is_datetime64_any_dtype(pdf[c])]
        if len(candidates) != 1:
            raise ValueError(f"Pass columns={{'time': ...}}: found {len(candidates)} datetime columns {candidates}")
        t = pdf[candidates[0]]
    if pd.api.types.is_datetime64_any_dtype(t):
        t = t.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(t.dt, "tz", None) is not None else t
        time_ns = t.to_numpy(dtype="datetime64[ns]").view("int64")
        valid = ~t.isna().to_numpy()
    else:
        valid = ~t.isna().to_numpy()
        time_ns = t.fillna(0).to_numpy(dtype="int64")

    def col(role, default_zero=False):
        name = cols[role]
        if name not in pdf.columns:
            if default_zero:
                return np.zeros(len(pdf))
            raise KeyError(f"Column {name!r} for {role!r} not found; pass columns={{'{role}': ...}}")
        s = pdf[name]
        return s.to_numpy(dtype="float64", na_value=np.nan), ~s.isna().to_numpy()

    def values(roles, zero_ok=()):
        nonlocal valid
        out = []
        for r in roles:
            v = col(r, default_zero=r in zero_ok)
            if isinstance(v, tuple):
                v, ok = v
                valid = valid & ok
            out.append(np.ascontiguousarray(v))
        return out

    if kind == "trades":
        args = values(["price", "volume"], zero_ok=("volume",))
        res = ebt_rust.detect_outliers(time_ns, *args, valid=valid, return_debug=return_debug, **config)
    elif kind == "quotes":
        args = values(["bid", "ask"])
        res = ebt_rust.detect_outliers_quotes(time_ns, *args, valid=valid, return_debug=return_debug, **config)
    else:
        args = values(["open", "high", "low", "close", "volume"], zero_ok=("volume",))
        res = ebt_rust.detect_outliers_ohlcv(time_ns, *args, valid=valid, return_debug=return_debug, **config)

    if not return_debug:
        return res
    if is_polars:
        import polars as pl
        return pl.DataFrame(res)
    return pd.DataFrame(res, index=pdf.index)
