"""Execution simulators.

Contains:
- simulate_vwap_execution: VWAP ramped execution
- smart_execute: spread/price-aware execution timing
- risk_stop_break: suppress trades below a risk threshold
"""

import logging

import numpy as np
import pandas as pd

try:
    from numba import jit
    _HAS_NUMBA = True
except ModuleNotFoundError:
    _HAS_NUMBA = False

import ebt_rust

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Smart execution: numba kernel + pandas wrapper
# ---------------------------------------------------------------------------

def _smart_execute_1d_python(timestamps, desired, spreads, spread_gate,
                             micro_pred, max_wait_ns, price_threshold,
                             use_micro, combine_and):
    """Pure-Python fallback for _smart_execute_1d (same logic, no numba).

    spreads[t] is the aligned+imputed spread at bar t.
    spread_gate[t] is the threshold: trade only when spread < spread_gate.
    """
    T = len(desired)
    realized = np.empty(T, dtype=np.float64)
    pending_since = -1  # index when pending trade started

    for t in range(T):
        desired_t = desired[t]

        # NaN desired → hold previous
        if np.isnan(desired_t):
            realized[t] = realized[t - 1] if t > 0 else np.nan
            continue

        prev = realized[t - 1] if t > 0 else np.nan

        # First trade → execute immediately
        if np.isnan(prev):
            realized[t] = desired_t
            pending_since = -1
            continue

        # No trade needed
        trade = desired_t - prev
        if abs(trade) < 1e-15:
            realized[t] = prev
            pending_since = -1
            continue

        # Track when this pending trade started
        if pending_since < 0:
            pending_since = t

        # Max wait deadline
        if max_wait_ns > 0 and (timestamps[t] - timestamps[pending_since]) >= max_wait_ns:
            realized[t] = desired_t
            pending_since = -1
            continue

        # --- Spread check: current spread < gate ---
        spread_ok = False
        s = spreads[t]
        g = spread_gate[t]
        if not np.isnan(s) and not np.isnan(g):
            spread_ok = s < g

        # --- Price score ---
        price_ok = True
        if use_micro:
            pred = micro_pred[t]
            if not np.isnan(pred):
                sign = 1.0 if trade > 0 else -1.0
                score = sign * pred
                price_ok = score >= price_threshold

        # --- Combine ---
        if combine_and:
            execute = spread_ok and price_ok
        else:
            execute = spread_ok or price_ok

        if execute:
            realized[t] = desired_t
            pending_since = -1
        else:
            realized[t] = prev

    return realized


if _HAS_NUMBA:
    _smart_execute_1d = jit(nopython=True, cache=True)(_smart_execute_1d_python)
else:
    _smart_execute_1d = _smart_execute_1d_python


def smart_execute(desired_pos, prices, spreads, micro_predictions=None,
                  spread_multiplier=3.0, spread_tolerance='10min',
                  spread_fallback_multiplier=10.0,
                  price_threshold=0.0, max_wait='5D', combine='and'):
    """Spread- and price-aware execution timing.

    Decides at each bar whether to execute a pending position change or
    wait for more favorable conditions. Produces a realized position
    that can be fed directly into ``one_pager``.

    Spread handling:
      1. Align spreads to position timestamps via as-of join with a
         tolerance window (spreads older than ``spread_tolerance`` → NaN).
      2. Impute remaining NaN spreads with ``spread_fallback_multiplier``
         × expanding median (conservative, instrument-relative).
      3. Trade only when spread < ``spread_multiplier`` × expanding median.

    Parameters
    ----------
    desired_pos : DataFrame
        Target positions from the signal model (DatetimeIndex, asset cols).
    prices : DataFrame
        Price levels, same structure as desired_pos.
    spreads : DataFrame
        Absolute bid-ask spreads (may be on a different frequency/index).
    micro_predictions : DataFrame, list[DataFrame], or None
        Expected short-term returns (positive = price going up).
        If a list, predictions are averaged. None = spread timing only.
    spread_multiplier : float
        Trade only when spread < this × expanding median. Default 3.0.
    spread_tolerance : str or Timedelta
        Grace period for the as-of spread alignment. Spreads older than
        this become NaN. Default '10min'.
    spread_fallback_multiplier : float
        Multiplier on expanding median for imputing missing spreads.
        Default 10.0.
    price_threshold : float
        Execute when ``sign(trade) * prediction >= price_threshold``.
        0.0 = execute when micro-model agrees with trade direction.
    max_wait : str, Timedelta, or None
        Force execution after this duration. None = no deadline.
    combine : str
        How to combine spread and price conditions: 'and' (both must
        be favorable) or 'or' (either suffices).

    Returns
    -------
    DataFrame
        Realized positions, same shape as desired_pos.

    Examples
    --------
    >>> realized = smart_execute(signal, prices, spreads, spread_multiplier=3.0)
    >>> result = one_pager(realized, prices, shift=0, spreads=spreads)
    """
    from ebt.simulate.tricks import nanagg

    # --- Align columns ---
    common_cols = desired_pos.columns.intersection(prices.columns).intersection(spreads.columns)
    if micro_predictions is not None:
        if isinstance(micro_predictions, list):
            micro_predictions = [mp for mp in micro_predictions if mp is not None]
            if len(micro_predictions) == 0:
                micro_predictions = None
            else:
                for mp in micro_predictions:
                    common_cols = common_cols.intersection(mp.columns)

        elif isinstance(micro_predictions, pd.DataFrame):
            common_cols = common_cols.intersection(micro_predictions.columns)

    desired_aligned = desired_pos[common_cols]

    # --- Align spreads: as-of join with tolerance, then impute ---
    tol = pd.Timedelta(spread_tolerance)
    spreads_col = spreads.reindex(columns=common_cols)
    spreads_aligned = spreads_col.reindex(index=desired_aligned.index, method='ffill', tolerance=tol)

    # Monthly median, shifted forward 1 month (causal: use last month's median)
    monthly_median = spreads_aligned.resample('MS').median()
    spread_baseline = monthly_median.shift(1).fillna(monthly_median).reindex(index=desired_aligned.index, method='ffill')

    # Impute missing with conservative fallback
    if spreads_aligned.isna().any().any():
        spreads_aligned = spreads_aligned.fillna(spread_baseline * spread_fallback_multiplier)

    # Spread gate: trade only when spread < multiplier × baseline
    spread_gate = spread_baseline * spread_multiplier

    # --- Parse time parameters ---
    max_wait_ns = 0
    if max_wait is not None:
        max_wait_ns = int(pd.Timedelta(max_wait).total_seconds() * 1e9)

    # --- Timestamps ---
    timestamps = desired_aligned.index.values.astype('datetime64[ns]').astype(np.int64)

    # --- Micro predictions ---
    use_micro = micro_predictions is not None
    if use_micro:
        if isinstance(micro_predictions, list):
            micro_avg = nanagg(micro_predictions, np.nanmean)
            micro_vals = micro_avg.reindex(columns=common_cols, index=desired_aligned.index).fillna(0.0).values
        else:
            micro_vals = micro_predictions.reindex(columns=common_cols, index=desired_aligned.index).fillna(0.0).values
    else:
        micro_vals = np.zeros((len(desired_aligned), len(common_cols)), dtype=np.float64)

    # --- Combine mode ---
    combine_and = combine == 'and'

    # --- Run kernel per asset ---
    desired_vals = desired_aligned.values
    spread_vals = spreads_aligned.values
    gate_vals = spread_gate.values
    n_rows, n_cols = desired_vals.shape
    result = np.empty_like(desired_vals)

    for j in range(n_cols):
        result[:, j] = _smart_execute_1d(
            timestamps,
            desired_vals[:, j].copy(),
            spread_vals[:, j].copy(),
            gate_vals[:, j].copy(),
            micro_vals[:, j].copy(),
            max_wait_ns,
            price_threshold,
            use_micro,
            combine_and,
        )

    return pd.DataFrame(result, index=desired_aligned.index, columns=common_cols)


# ---------------------------------------------------------------------------
# VWAP execution simulator
# ---------------------------------------------------------------------------


def simulate_vwap_execution(target_positions, volume, vwap_window):
    """Simulate VWAP execution to get realistic positions.

    Instead of assuming instantaneous rebalancing, ramps position
    changes over a VWAP window weighted by volume. When signals
    reverse mid-execution, outstanding orders naturally net,
    reducing realized turnover.

    At each timestep t:
        delta = target[t] - actual[t-1]
        frac  = volume[t] / sum(volume within [t, t + vwap_window))
        actual[t] = actual[t-1] + delta * frac

    Parameters
    ----------
    target_positions : DataFrame
        Signal output (DatetimeIndex, asset columns).
    volume : DataFrame or Series
        Volume per period, typically on the price grid (may be denser
        than target_positions). If Series, broadcast to all asset columns.
        The signal is forward-filled onto the volume grid internally.
    vwap_window : str, timedelta, or pd.Timedelta
        Execution window duration. E.g. '60min', '15min',
        pd.Timedelta(minutes=60).

    Returns
    -------
    DataFrame
        Realized positions after simulated execution.
        Same index and columns as target_positions.

    Notes
    -----
    Shift (reaction delay) and VWAP ramp are independent:

    - shift = delay before execution starts
    - vwap_window = duration of execution after it starts

    Usage::

        actual = simulate_vwap_execution(signal, volume, vwap_window='60min')
        result = one_pager(actual, close_prices, shift=2)
    """
    target = target_positions
    window_td = pd.Timedelta(vwap_window)
    window_ns = int(window_td.total_seconds() * 1_000_000_000)

    if isinstance(volume, pd.Series):
        vol_df = pd.DataFrame(
            {col: volume for col in target.columns},
            index=volume.index,
        )
    else:
        vol_df = volume.reindex(columns=target.columns)

    # Run on the volume grid (typically denser than the signal grid).
    # Forward-fill signal onto volume timestamps so execution ramps at
    # the volume frequency with proper volume weighting.  Signal bars
    # that don't coincide with a volume bar are included via the union
    # so position changes are never lost.
    idx = vol_df.index.union(target.index).sort_values()
    idx = idx[~idx.duplicated(keep='first')]
    tgt = target.reindex(idx).ffill().values.astype(np.float64)
    vol = np.nan_to_num(vol_df.reindex(idx).values.astype(np.float64), nan=0.0)

    n_rows, n_cols = tgt.shape

    # Rust expects column-major (Fortran order) flat arrays
    tgt_flat = np.asfortranarray(tgt).ravel(order='F')
    vol_flat = np.asfortranarray(vol).ravel(order='F')
    # Force nanosecond resolution (pandas 2.x may default to microseconds)
    timestamps = idx.values.astype('datetime64[ns]').astype(np.int64)

    result_flat = ebt_rust.simulate_vwap_execution_core(
        timestamps, tgt_flat, vol_flat, n_rows, n_cols, window_ns,
    )

    # Reshape from column-major flat back to 2D
    result = np.asarray(result_flat).reshape((n_rows, n_cols), order='F')
    return pd.DataFrame(result, index=idx, columns=target.columns)


# ---------------------------------------------------------------------------
# Risk stop-break: suppress trades below a risk threshold
# ---------------------------------------------------------------------------

@jit(nopython=True)
def _risk_stop_break_1d(positions, vol, threshold):
    """Numba kernel: suppress trades whose risk is below threshold.

    positions: 1D array (NaN = no update, value = target position in lots)
    vol: 1D array of per-bar daily volatility (same length as positions)
    threshold: minimum trade risk in daily PnL stds to execute
    """
    T = len(positions)
    result = np.empty(T, dtype=np.float64)
    last_executed_pos = np.nan

    for i in range(T):
        if np.isnan(positions[i]):
            result[i] = np.nan
            continue

        if np.isnan(last_executed_pos):
            last_executed_pos = positions[i]
            result[i] = positions[i]
            continue

        v = vol[i]
        if np.isnan(v) or v == 0:
            # No measured vol: the trade cannot be risk-assessed, so it is not suppressed.
            last_executed_pos = positions[i]
            result[i] = positions[i]
            continue
        trade_risk = abs(positions[i] - last_executed_pos) * abs(v)

        if trade_risk > threshold:
            last_executed_pos = positions[i]
            result[i] = positions[i]
        else:
            result[i] = np.nan

    return result


@jit(nopython=True, nogil=True)
def _risk_stop_break_block(pos_T, vol_T, threshold, j0, j1, out):
    """``_risk_stop_break_1d`` for the assets [j0, j1) of COLUMN-MAJOR (N, T) arrays, written
    into out[j0:j1] — the same per-asset arithmetic; GIL-free so blocks run on threads."""
    T = pos_T.shape[1]
    for j in range(j0, j1):
        last_executed_pos = np.nan
        for i in range(T):
            p = pos_T[j, i]
            if np.isnan(p):
                out[j, i] = np.nan
                continue
            if np.isnan(last_executed_pos):
                last_executed_pos = p
                out[j, i] = p
                continue
            v = vol_T[j, i]
            if np.isnan(v) or v == 0:
                # no measured vol: not suppressed (see _risk_stop_break_1d)
                last_executed_pos = p
                out[j, i] = p
                continue
            trade_risk = abs(p - last_executed_pos) * abs(v)
            if trade_risk > threshold:
                last_executed_pos = p
                out[j, i] = p
            else:
                out[j, i] = np.nan


def _risk_stop_break_2d(pos_T, vol_T, threshold):
    """``_risk_stop_break_1d`` over every asset of COLUMN-MAJOR (N, T) arrays, in contiguous
    blocks on GIL-free threads (each thread streams its own rows: no false sharing, no numba
    parallel layer, so fork- and thread-safe). Returns (N, T)."""
    from ebt.simulate.kf import _run_blocks, _kernel_threads
    N, T = pos_T.shape
    out = np.empty((N, T), dtype=np.float64)
    nb = max(1, min(N, 2 * _kernel_threads(N)))
    bs = -(-N // nb) if N else 1
    bounds = [(j0, min(N, j0 + bs)) for j0 in range(0, N, bs)]
    _run_blocks(lambda j0, j1, b: _risk_stop_break_block(pos_T, vol_T, float(threshold), j0, j1, out),
                bounds, work=N * T)
    return out


def risk_stop_break(pos, prices, threshold=0.001, volatility=None):
    """Suppress position changes whose risk is below threshold daily stds.

    For each asset independently, computes the trade risk as
    ``|trade_lots| * daily_vol``. Trades with risk below ``threshold``
    are suppressed — the position holds at the last executed value
    until a sufficiently large trade accumulates.

    Applied as the last step in the execution chain (after VWAP and
    smart execution).

    Parameters
    ----------
    pos : DataFrame
        Position DataFrame with NaN structure (NaN = no update).
    prices : DataFrame
        Price levels for volatility estimation.
    threshold : float
        Minimum trade risk in daily PnL stds per asset. Default 0.001.
    volatility : DataFrame or None
        Pre-computed per-asset daily volatility (EMA std of daily price
        diffs). If None, estimated from prices via EMA(60) of daily diffs.

    Returns
    -------
    DataFrame
        Filtered positions with small trades suppressed (set to NaN).

    Examples
    --------
    >>> from ebt.simulate.execution import risk_stop_break
    >>> filtered = risk_stop_break(signal, prices, threshold=0.002)
    >>> result = one_pager(filtered, prices, shift=0, spreads=spreads)
    """
    from ebt.simulate.plot import _estimate_asset_vol

    vol = volatility if volatility is not None else _estimate_asset_vol(prices)
    vol = vol.reindex(columns=pos.columns)
    vol = vol.reindex(index=pos.index, method='ffill')

    result_vals = np.empty_like(pos.values)
    for j in range(pos.shape[1]):
        result_vals[:, j] = _risk_stop_break_1d(
            pos.values[:, j].copy(),
            vol.values[:, j].copy(),
            threshold,
        )

    return pd.DataFrame(result_vals, index=pos.index, columns=pos.columns)
