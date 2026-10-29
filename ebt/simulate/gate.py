"""Position sanity gate — is this book BROKEN, before anyone reads its Sharpe?

A pipeline defect (inf/NaN features, a wrong horizon, an unregularised model, a
units mismatch, a dead data column) still produces positions, and those positions
still produce a Sharpe that looks like a result. ``position_sanity_gate`` inspects the
SHAPE of a position frame and returns the alarm reasons; an experiment runner calls it
on the combined book before every publish and refuses to publish an alarmed book.

An alarm means diagnose the construction. Dropping the alarmed configs is post-hoc
selection and is not what this function is for.

Two artifacts look like alarms and are handled here rather than reported:

* float residue — ``combine_positions`` leaves ~1e-15 gross on dates where every
  constituent is flat, and one_pager's turnover divides by gross, so the headline reads
  1e11 while PnL is fine. Dates whose gross is below ``residue_eps`` x the book's
  99th-percentile gross are zeroed before measuring (count returned as ``n_residue_dates``).
* equity books in SHARES need an explicit ``volatility=`` frame, or turnover, holding and
  lptp are weighted by a price-diff vol that is meaningless after reverse splits. The
  gate cannot detect that; the caller passes it.
"""
import logging

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

GATE_KFS = ['turnover', 'holding(days)', 'lptp']


def position_sanity_gate(book, prices, shift='2m', volatility=None, *, ignore=(),
                         min_traded=0.9, kurtosis_max=100.0, skew_max=20.0,
                         lptp_range=(0.05, 0.95), holding_min=1e-3, turnover_max=10000.0,
                         gross_kurt_max=100.0, gross_max_med=20.0, residue_eps=1e-8,
                         reduce='auto'):
    """Gate ONE book. Returns a dict of the measured figures plus ``reasons`` (list of
    alarm names, empty = clean) and ``alarm`` (bool).

    Args:
        book: position DataFrame (DatetimeIndex x assets). NaN = carry, 0 = flat.
        prices: price DataFrame on the eval grid (same columns), ≤5-min for shift='2m'; or a
            ``ReducedGrid`` built for the book (ebt.simulate.grid), used as-is.
        shift: execution lag passed to one_pager.
        reduce: one_pager's grid mode for a price FRAME ('auto' | True | False); ignored when
            ``prices`` is already a ReducedGrid.
        ignore: names of checks to skip, e.g. ('lptp',) for a LONG-ONLY book (lptp is 1.0 by
            construction there). Skipped checks are still measured and reported; only the
            alarm ignores them (``row['ignored']``).
        volatility: per-asset vol frame. Used for one_pager's turnover/holding weighting AND
            to put the SHAPE statistics (kurtosis, skew, gross) in RISK units: position x vol.
            Pooled raw positions are not comparable across names — shares are dollars / price
            and an equity universe can span many orders of magnitude of back-adjusted price,
            so an ordinary equal-risk book can read an extreme kurtosis. When None, the
            default ``plot._estimate_asset_vol(prices)`` (the risk_adjust unit) is used;
            equity share books should pass a return-based vol instead.
        min_traded: alarm ``traded_days`` when fewer than this share of days since the
            first trade carry any gross. A strategy that is flat BY DESIGN (event windows)
            passes its own lower floor.
        kurtosis_max, skew_max: on the held (forward-filled) non-zero positions.
        lptp_range: long-to-total-position ratio must lie strictly inside.
        holding_min: holding(days) below this = the book flips every bar.
        turnover_max: daily turnover above this = oscillation, inf/NaN features, or
            residue the caller did not remove.
        gross_kurt_max, gross_max_med: blow-up in the daily risk-adjusted gross.
        residue_eps: dates with gross below this fraction of the book's 99th-percentile
            gross are treated as flat (float residue ~1e-15 on O(1) positions), see the
            module docstring.
    """
    from scipy.stats import kurtosis, skew
    from ebt.simulate.plot import one_pager

    reasons = []
    row = {'n_residue_dates': 0}
    if book is None or book.shape[1] == 0 or not np.isfinite(book.to_numpy(dtype='float64')).any():
        row.update(reasons=['never_trades'], alarm=True)
        return row

    from ebt.simulate.plot import _estimate_asset_vol
    from ebt.simulate.grid import ReducedGrid
    _grid = prices if isinstance(prices, ReducedGrid) else None
    if _grid is not None:
        # The grid carries its vol at the kept rows: exact as-of at every book stamp because the
        # first and last row of each day are kept and the vol is constant within a day.
        if volatility is not None:
            raise ValueError('position_sanity_gate: pass volatility through the ReducedGrid, not beside it')
        _vol = _grid.volatility
    else:
        _vol = volatility if volatility is not None else _estimate_asset_vol(prices)
    # AS-OF alignment: the book is on the signal's own stamps (e.g. a few minutes after the open), the vol
    # on the price grid; a plain reindex matches nothing.
    _vol = _vol.reindex(columns=book.columns)
    _vol = _vol.reindex(_vol.index.union(book.index)).ffill().reindex(book.index)
    # A name-bar without a measured vol has no risk unit: NaN, never a stand-in
    _vol = _vol.where(np.isfinite(_vol) & (_vol > 0))
    held = book.ffill()
    held_r = held * _vol                    # RISK units for every shape statistic below
    gross = held_r.abs().sum(axis=1)
    # Residue reference: a high quantile of the positive gross, not the median — on a book
    # that is flat half the time the median itself can be residue.
    ref = gross[gross > 0].quantile(0.99) if (gross > 0).any() else np.nan
    if np.isfinite(ref):
        residue = (gross > 0) & (gross < residue_eps * ref)
        n_res = int(residue.sum())
        if n_res:
            book = book.copy()
            book.loc[residue, :] = 0.0
            held = book.ffill()
            held_r = held * _vol
            gross = held_r.abs().sum(axis=1)
        row['n_residue_dates'] = n_res

    daily = gross.groupby(gross.index.normalize()).max()
    first = daily[daily > 0].index.min() if (daily > 0).any() else None
    if first is None:
        row.update(reasons=['never_trades'], alarm=True)
        return row
    daily = daily.loc[first:]
    traded = daily[daily > 0]

    kf = one_pager(book, prices, shift=shift, plot=False, fast=True, return_subset={'kf'},
                   volatility=None if _grid is not None else volatility, kfs=list(GATE_KFS),
                   reduce=reduce)['kf']
    v = held_r.to_numpy(dtype='float64').ravel()
    v = v[np.isfinite(v) & (v != 0)]
    row.update({
        'turnover': float(kf['turnover']), 'holding': float(kf['holding(days)']),
        'lptp': float(kf['lptp']),
        'kurtosis': float(kurtosis(v)) if len(v) > 10 else np.nan,
        'skew': float(skew(v)) if len(v) > 10 else np.nan,
        'gross_kurt': float(kurtosis(traded)) if len(traded) > 10 else np.nan,
        'gross_max_med': float(traded.max() / traded.median()) if len(traded) else np.nan,
        'traded_days': float((daily > 0).mean()),
        'first_trade': str(first.date()),
    })
    if row['kurtosis'] > kurtosis_max:
        reasons.append('kurtosis')
    if abs(row['skew']) > skew_max:
        reasons.append('skew')
    if not (lptp_range[0] < row['lptp'] < lptp_range[1]):
        reasons.append('lptp')
    if row['holding'] < holding_min:
        reasons.append('holding')
    if row['turnover'] > turnover_max:
        reasons.append('turnover')
    if row['gross_kurt'] > gross_kurt_max:
        reasons.append('gross_kurt')
    if row['gross_max_med'] > gross_max_med:
        reasons.append('gross_blowup')
    if row['traded_days'] < min_traded:
        reasons.append('traded_days')
    ignored = sorted(set(ignore))
    reasons = [r for r in reasons if r not in ignored]
    row.update(reasons=reasons, alarm=bool(reasons), ignored=ignored)
    return row


def format_gate_row(row):
    """One log line for a gate result."""
    if 'turnover' not in row:
        return f"GATE ALARM {row.get('reasons')}"
    head = "GATE ALARM " + "|".join(row['reasons']) if row['alarm'] else "gate clean"
    return (f"{head}: turnover {row['turnover']:.3g} holding {row['holding']:.3g}d "
            f"lptp {row['lptp']:.2f} kurt {row['kurtosis']:.3g} skew {row['skew']:.3g} "
            f"gross_kurt {row['gross_kurt']:.3g} max/med {row['gross_max_med']:.3g} "
            f"traded_days {row['traded_days']:.2f} residue_dates {row['n_residue_dates']}")
