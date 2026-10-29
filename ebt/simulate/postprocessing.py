"""Postprocessing: risk adjustment and lot conversion.

    from ebt.simulate.postprocessing import risk_adjust, risk_to_lots

    pos = risk_adjust(sig, norm_px, shift=1, span=250)
    lots = risk_to_lots(pos, vol, {'A': 10, 'B': 100}, risk_budget=1_000_000)
"""

import logging
import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)


def _resolve_span(span, index):
    """Convert span to an integer number of observations.

    Accepts:
        int/float  — returned as-is (number of observations)
        str        — parsed as pd.Timedelta, then converted using median bar duration
        Timedelta  — converted using median bar duration of the index
    """
    if isinstance(span, (int, float)):
        return int(span)

    td = pd.Timedelta(span) if isinstance(span, str) else span
    if len(index) < 2:
        return 1
    median_bar = np.median(np.diff(index.values).astype('timedelta64[s]').astype(float))
    return max(1, int(round(td.total_seconds() / median_bar)))


def risk_adjust(pos, prices_or_returns, shift=2, span='256D', target_risk=1.0,
                 min_periods='30D', min_active_days=None, act_eps=1e-6):
    """Scale positions ex-post so annualized risk ~ target_risk.

    Uses the realized daily PnL volatility from a trailing EWM window
    to compute a scaling factor, lagged by one day to avoid look-ahead.

    Long span (default 256D) acts as a near-constant scalar, which is
    appropriate for signals that are already vol-normalized at construction
    time. Short spans cause whipsaw scaling that adds noise.

    Args:
        pos: Position DataFrame (DatetimeIndex, asset columns).
        prices_or_returns: Price or return DataFrame (same columns).
        shift: Execution lag in bars (default 2).
        span: EWM span for the volatility estimate. Accepts int (number of
            daily observations), timedelta string ('256D'),
            or pd.Timedelta. Timedelta values are converted to observation
            count using the median bar duration of the daily PnL series.
        target_risk: Target annualized risk (default 1.0).
        min_periods: Minimum observations before producing a vol estimate.
            Accepts int or timedelta string (default '30D'). Positions are
            NaN until this many daily observations are available.
        min_active_days: OPT-IN degenerate-estimate guard. ``min_periods``
            counts calendar days PRESENT, not days that actually traded — a
            window of all-zero PnL days satisfies it, and the resulting
            EWM std is meaningless. Set this to require at least N days on
            which the book ACTUALLY TRADED inside the trailing ``span`` window
            (see ``act_eps``) before the vol
            estimate is trusted; bars failing the test get NaN vol -> NaN scale
            -> NaN position, and drop out of downstream PnL rather than being
            multiplied by a garbage scale factor. ``None`` (default) preserves
            the unguarded behaviour exactly.
        act_eps: A day counts as active when the book holds ``|position| >
            act_eps`` at some point in it. Default 1e-6 — far below any real
            position (positions are contract counts) and far above a dead signal. Applied to POSITIONS, not PnL: a PnL-based test cannot
            work, since a dead stretch still yields tiny NONZERO PnL and any
            near-zero threshold counts those days as active.

    Returns:
        Rescaled position DataFrame.
    """
    from ebt.simulate.kf import backtest

    # `min_count=1` preserves NaN on rows where ALL assets have NaN positions.
    # Default `.sum(axis=1)` (min_count=0) silently converts all-NaN rows to 0,
    # polluting the daily PnL with phantom zeros for any period before the
    # signal has produced values. Those zeros make EWM_std collapse to 0, then
    # the FIRST real PnL day produces a tiny EWM_std and the scale factor
    # explodes, causing single-bar position spikes.
    pnl = backtest(pos, prices_or_returns, shift=shift).sum(axis=1, min_count=1)
    # `min_count=1` keeps days that had observations but happened to PnL=0
    # (e.g. zero-position days for a sparse signal). Replacing them with NaN
    # via .replace(0, NaN).dropna() would discard legitimate-but-zero days
    # from the vol estimate, leaving only the rare non-zero days. With those
    # few days clustered around a similar magnitude, the EWM std collapses
    # toward zero and `target_risk / vol` explodes. Use min_count=1 so
    # only fully-empty days (no intraday data at all) are dropped.
    daily_pnl = pnl.resample('D').sum(min_count=1).dropna()

    span_int = _resolve_span(span, daily_pnl.index)
    min_periods_int = _resolve_span(min_periods, daily_pnl.index)
    vol = daily_pnl.ewm(span=span_int, min_periods=min_periods_int).std() * 252 ** 0.5
    vol = vol.shift(1)

    # Degenerate-estimate guard (opt-in via min_active_days; None = unguarded).
    #
    # The `inf` guard further down catches vol EXACTLY 0. It does NOT catch vol that is
    # tiny-but-finite, which is what a mostly-dead window actually produces: the EWM std
    # tends TOWARD zero, so target_risk/vol becomes a large FINITE number and passes every
    # other check.
    #
    # The root problem is that min_periods counts days PRESENT, not days that traded.
    # Count active days instead, over the same trailing window the EWM actually uses.
    #
    # Causal: `vol` is already shifted one day, and the mask is shifted one day too, so a
    # bar at t is judged only on PnL through t-1.
    # NOTE act_eps is applied to POSITIONS, not to daily PnL. A PnL test cannot work,
    # because a dead stretch produces tiny-but-NONZERO PnL, so any threshold at or near 0
    # still counts those days as "active". Nor can the reference
    # be self-relative — median/max/long-horizon vol all collapse along with the data, so
    # when the whole history so far is dead there is no internal scale left to compare
    # against. Positions do have an absolute scale here: they are contract counts, and a
    # dead signal is numerically negligible next to any real one. Hence an absolute
    # floor on |position|.
    if min_active_days is not None:
        _traded = (pos.abs().max(axis=1) > act_eps)
        _active = _traded.resample('D').max().reindex(daily_pnl.index).fillna(False)
        _n_active = _active.astype(float).rolling(span_int, min_periods=1).sum()
        vol = vol.where(_n_active.shift(1) >= float(min_active_days))

    scale = target_risk / vol
    # Pathological scaling: ewm std can be exactly 0 (e.g. signal is sparse and
    # the entire warmup window has zero PnL) → scale = inf → pos blows up to
    # inf. NaN-out non-finite scales so those bars become NaN positions and
    # are simply dropped from downstream PnL/Sharpe rather than poisoning
    # everything.
    scale = scale.replace([np.inf, -np.inf], np.nan)
    scale = scale.reindex(pos.index, method='ffill')

    return pos.mul(scale, axis=0)


def risk_to_lots(risk_pos, daily_vol, point_values=None, risk_budget=1_000_000):
    """Convert risk positions to contract lots for dollar-denominated trading.

    Produces the number of contracts such that:
        dollar_PnL = lots * price_diff * point_value
                   = risk_budget * risk_pos * normalized_return

    The portfolio's annualized dollar PnL std will be approximately
    ``risk_budget`` (assuming risk_pos has been risk-adjusted to target 1.0).

    Args:
        risk_pos: DataFrame of risk positions (daily stds per asset).
        daily_vol: DataFrame of daily volatility in price-point terms.
            Forward-filled to risk_pos index if indices differ.
        point_values: dict mapping asset name -> USD per price point per
            contract (e.g. {'A': 10, 'B': 100}).  If None, defaults
            to 1.0 for all assets (lots will be in price-point-std units,
            not actual contract counts).
        risk_budget: Target annualized dollar std.

    Returns:
        DataFrame of contract lots (same shape as risk_pos).
    """
    if point_values is None:
        _LOGGER.warning(
            "point_values not specified — defaulting to 1.0 for all assets. "
            "Lot sizes will be in price-point-std units, not actual contracts."
        )
        point_values = {}

    vol = daily_vol.ffill().reindex(risk_pos.index, method='ffill') if not daily_vol.index.equals(risk_pos.index) else daily_vol.ffill()

    # dollar_vol_per_lot = daily_vol * point_value  ($/lot/day)
    dollar_vol = vol.copy()
    for asset in risk_pos.columns:
        pv = point_values.get(asset, 1.0)
        dollar_vol[asset] = vol[asset] * pv

    return risk_pos * risk_budget / dollar_vol
