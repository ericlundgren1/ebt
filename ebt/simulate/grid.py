"""Reduced price grid: evaluate a sparse signal on the price rows that matter.

A signal stamped once a day is evaluated against 5-minute prices only so the fill can land at the
first print at-or-after stamp + shift. Every full-size frame one_pager / plot_ensemble build scales
with the price grid, so keeping only the rows the evaluation actually uses cuts memory and compute
by the row ratio without changing the figures.

Kept rows (``reduce_price_grid``). Per column, on its own non-NaN prints:
  a) the FILL row for every stamp: first print at-or-after stamp + shift (both wall-clock mappings
     of the shift, see below);
  b) the print after each fill (where the new position first appears);
  c) the last print of each day and h) the first print of each day (g, the first print ever, is a
     special case of h);
  f) the last print at-or-before every truncation stamp + shift (one_pager cuts the frame there).
Frame-level: d) the first and e) the last row of each day, and the last row at-or-before each
truncation stamp + shift. "Day" is the wall-clock calendar day of the index (kf._index_to_ns).

With these rows the realised position at every kept row, and the daily sums of PnL, equal the full
grid's (the PnL between two kept prints of a name telescopes because the position is constant
there). Statistics that average over ROWS (turnover's mean exposure, lptp, the exbias/bias column
mean) need the forward bar-count weights ``weights()``: the number of full-grid rows each kept row
stands for. The risk-scaled price ``px_v`` (running sum of price change / vol) is built on the FULL
grid and sampled, so a vol that changes between two kept rows is accumulated correctly; the
volatility itself must be constant within each day (asserted).

Figures that are defined per bar (hit rate, decile risk, the EWM holding curve and sharpe_decay,
the lag / trend panels, 'freq pnl') change meaning on a reduced grid and are refused
(``GRID_INCOMPATIBLE_KFS``).
"""
import logging

import numpy as np
import pandas as pd

from ebt.simulate import kf

_LOGGER = logging.getLogger(__name__)

NS_DAY = 86_400_000_000_000

#: key figures that are defined per price bar and cannot be computed on a reduced grid
GRID_INCOMPATIBLE_KFS = frozenset({'hitrate', 'risk_conc', 'tail_risk', 'sharpe_decay',
                                   'sharpe_decay_wc', 'freq pnl'})
#: keep at least this many prints per column so kf.is_price_or_return still reads 'price'
MIN_PRINTS_PER_COLUMN = 20
#: reduce only when the signal is at least this much sparser than the prices (median spacing).
#: Every stamp keeps two rows per column (the fill and the print after it) on top of the
#: first/last rows of each day, so at exactly 4x the grid always keeps more than
#: MAX_KEPT_FRACTION and the build would be paid for nothing.
SPARSITY_RATIO = 5.0
#: ... and only when the kept rows are at most this fraction of the grid
MAX_KEPT_FRACTION = 0.5


class GridNotApplicable(ValueError):
    """The call cannot be evaluated on a reduced grid (see ``applicability``)."""


def _ns(index):
    return kf._index_to_ns(pd.DatetimeIndex(index))


def _to_price_tz(stamps, price_index):
    """Stamps (naive or aware) expressed in the price index's zone, like _apply_eval_window."""
    stamps = pd.DatetimeIndex(stamps)
    ptz = getattr(price_index, 'tz', None)
    stz = getattr(stamps, 'tz', None)
    if ptz is None and stz is None:
        return stamps
    if ptz is not None and stz is None:
        return stamps.tz_localize(ptz)
    if ptz is not None and stz is not None:
        return stamps.tz_convert(ptz)
    raise ValueError('reduce_price_grid: tz-aware stamps against a tz-naive price index')


def shifted_stamps_ns(stamps, shift):
    """Both wall-clock mappings of stamp + shift, as sorted unique ns.

    one_pager shifts the tz-aware signal index in absolute time and strips the tz afterwards
    (wallclock(t + td)); plot_ensemble's stream engine adds the shift to the stripped wall clock
    (wallclock(t) + td). The two agree except across a DST change; the kept set covers both.
    """
    stamps = pd.DatetimeIndex(stamps)
    td = pd.Timedelta(shift)
    a = _ns(stamps) + int(td.value)
    b = _ns(stamps + td)
    return np.unique(np.concatenate([a, b]))


def canonical_trunc_stamps(stamps, extra=()):
    """The truncation stamps every one_pager-built grid uses for a signal: its last stamp and the
    last stamp of each calendar year (per_year_table cuts there), plus ``extra`` (an eval-window
    end). Using the same set everywhere makes a caller's grid for the combined book
    identical to the one a standalone one_pager builds, so the two agree bit for bit."""
    stamps = pd.DatetimeIndex(stamps)
    out = [stamps[-1]]
    years = stamps.year
    for y in np.unique(years):
        out.append(stamps[years == y][-1])
    out.extend(pd.Timestamp(e) for e in extra if e is not None)
    return pd.DatetimeIndex(out).unique()


def sparsity_ratio(signal_index, price_index):
    """median signal spacing / median price spacing (inf for a single stamp)."""
    s = np.diff(_ns(signal_index))
    p = np.diff(_ns(price_index))
    if len(p) == 0 or len(s) == 0:
        return np.inf
    ps = float(np.median(p))
    return np.inf if ps <= 0 else float(np.median(s)) / ps


def applicability(signal_index, price_index, shift, *, resample=True, smart_execution=None,
                  volume=None, is_returns=False, kfs=None):
    """Return (ok, reason) for evaluating on a reduced grid. ``reason`` is '' when ok."""
    int_shift, td = kf._parse_shift(shift)
    if td is None:
        return False, f'integer shift {int_shift} counts price prints, not time'
    if td <= pd.Timedelta(0):
        return False, f'non-positive shift {td}'
    if not resample:
        return False, 'resample=False computes per-row statistics'
    if smart_execution:
        return False, 'smart_execution decides bar by bar'
    if volume is not None:
        return False, 'VWAP execution (volume=) ramps on every bar'
    if is_returns:
        return False, 'input classified as returns'
    if kfs is not None and (set(kfs) & GRID_INCOMPATIBLE_KFS):
        bad = sorted(set(kfs) & GRID_INCOMPATIBLE_KFS)
        return False, f'key figures {bad} are defined per bar'
    ts = _ns(price_index)
    if len(ts) < 2 or not np.all(np.diff(ts) > 0):
        return False, 'price index wall clock is not strictly increasing'
    r = sparsity_ratio(signal_index, price_index)
    if r < SPARSITY_RATIO:
        return False, f'signal spacing is only {r:.1f}x the price spacing'
    return True, ''


class ReducedGrid:
    """Prices, vol, spreads and the risk-scaled price at the kept rows of a full grid.

    ``kept`` are positions into ``full_index``; ``weights()`` are forward bar counts. Slicing and
    truncation keep the two consistent (``slice``/``truncate``). ``stamps_ns`` is the shifted stamp
    set the grid was built for; a signal with stamps outside it must not be evaluated on it
    (``check_covers``).
    """

    def __init__(self, *, kept, full_index, prices, volatility, spreads, px_v, freq_seconds,
                 stamps_ns, trunc_ns, shift, meta):
        self.kept = np.asarray(kept, dtype=np.int64)
        self.full_index = full_index
        self.prices = prices
        self.volatility = volatility
        self.spreads = spreads
        self.px_v = px_v
        self.freq_seconds = float(freq_seconds)
        self.stamps_ns = np.asarray(stamps_ns, dtype=np.int64)
        self.trunc_ns = np.asarray(trunc_ns, dtype=np.int64)
        self.shift = pd.Timedelta(shift)
        self.meta = dict(meta)

    # ------------------------------------------------------------------ basic properties
    @property
    def index(self):
        return self.prices.index

    @property
    def columns(self):
        return self.prices.columns

    @property
    def n_full(self):
        return len(self.full_index)

    @property
    def tz(self):
        return getattr(self.full_index, 'tz', None)

    def __len__(self):
        return len(self.kept)

    def __repr__(self):
        return (f'ReducedGrid({len(self.kept)}/{self.n_full} rows, {self.prices.shape[1]} columns, '
                f'shift={self.shift})')

    # ------------------------------------------------------------------ weights
    def weights(self):
        """Forward bar counts: full-grid rows in [K_k, K_{k+1}), the last up to the frame end."""
        return np.diff(np.r_[self.kept, self.n_full]).astype(np.float64)

    def weights_for(self, start=None, end=None):
        return self.slice(start, end).weights()

    # ------------------------------------------------------------------ views
    def _bounds(self, start, end):
        # compare in wall-clock ns (the index may be datetime64[us]; a ns-precision bound such
        # as "end of Dec 31" must not be forced into its unit)
        ts = _ns(self.full_index)
        n_beg = 0
        n_end = self.n_full
        if start is not None:
            n_beg = int(np.searchsorted(ts, int(_ns(_to_price_tz([start], self.full_index))[0]), side='left'))
        if end is not None:
            n_end = int(np.searchsorted(ts, int(_ns(_to_price_tz([end], self.full_index))[0]), side='right'))
        return n_beg, n_end

    def slice(self, start=None, end=None):
        """Rows of the full grid in [start, end] (inclusive), kept rows re-based.

        Naive bounds are read as wall time in the price zone. A slice that does not begin at a
        kept row leaves the rows before its first kept row unrepresented in the weights (they
        belong to the previous kept row, which is outside the slice); one_pager, per_year_table and
        plot_ensemble always slice at day boundaries or truncation stamps, where a row is kept.
        """
        n_beg, n_end = self._bounds(start, end)
        if n_beg == 0 and n_end == self.n_full:
            return self
        sel = (self.kept >= n_beg) & (self.kept < n_end)
        kept = self.kept[sel] - n_beg
        if len(kept) and kept[0] != 0 and n_beg > 0:
            _LOGGER.warning('ReducedGrid.slice: slice starts %d rows before its first kept row; '
                            'those rows carry no weight', int(kept[0]))
        sub = lambda f: None if f is None else f.iloc[np.flatnonzero(sel)]
        return ReducedGrid(kept=kept, full_index=self.full_index[n_beg:n_end], prices=sub(self.prices),
                           volatility=sub(self.volatility), spreads=sub(self.spreads), px_v=sub(self.px_v),
                           freq_seconds=self.freq_seconds, stamps_ns=self.stamps_ns, trunc_ns=self.trunc_ns,
                           shift=self.shift, meta={**self.meta, 'sliced': (str(start), str(end))})

    def truncate(self, end):
        """The grid cut at ``end`` (inclusive), which must be a truncation stamp the grid was built
        with (or lie at/after the last row): only then does rule (f) guarantee that every column's
        last print before the cut is kept, so the last kept row's weight covers no PnL the grid
        cannot see."""
        end_ns = int(_ns(_to_price_tz([end], self.full_index))[0])
        if end_ns < int(_ns(self.full_index[-1:])[0]) and end_ns not in set(self.trunc_ns.tolist()):
            raise ValueError(f'ReducedGrid.truncate: {end} is not a truncation stamp of this grid; '
                             f'build it with trunc_stamps including the signal\'s last stamp')
        return self.slice(None, end)

    def select_columns(self, cols):
        cols = pd.Index(cols)
        if cols.equals(self.prices.columns):
            return self
        sub = lambda f: None if f is None else f.reindex(columns=cols)
        return ReducedGrid(kept=self.kept, full_index=self.full_index, prices=self.prices[cols],
                           volatility=sub(self.volatility), spreads=sub(self.spreads), px_v=sub(self.px_v),
                           freq_seconds=self.freq_seconds, stamps_ns=self.stamps_ns, trunc_ns=self.trunc_ns,
                           shift=self.shift, meta=self.meta)

    # ------------------------------------------------------------------ validation
    def check_covers(self, stamps, shift=None):
        """Every shifted stamp (both mappings, within the grid) must be one the grid was built for."""
        shift = self.shift if shift is None else pd.Timedelta(shift)
        if shift != self.shift:
            raise ValueError(f'ReducedGrid built for shift {self.shift}, called with {shift}')
        st = shifted_stamps_ns(stamps, shift)
        last = int(_ns(self.full_index[-1:])[0])
        st = st[st <= last]
        missing = ~np.isin(st, self.stamps_ns)
        if missing.any():
            first = pd.Timestamp(int(st[missing][0]))
            raise ValueError(f'ReducedGrid does not cover {int(missing.sum())} of the signal\'s stamps '
                             f'(first missing fill time {first}); build the grid from these stamps')

    def check_rows(self, index):
        """For already-realised positions (shift 0): every row must be a kept row."""
        if not pd.DatetimeIndex(index).isin(self.index).all():
            raise ValueError('positions are not on the reduced grid\'s rows')

    def kept_rows_frame(self):
        return pd.DataFrame({'full_row': self.kept, 'weight': self.weights()}, index=self.index)

    def with_volatility(self, volatility):
        """The same rows with another (day-constant) vol frame on the reduced index and the
        risk-scaled price rebuilt from the reduced prices. Exact because the vol is constant
        within a day and every day boundary is a kept row: the moves between two kept rows are
        divided by one vol on both grids. Used by per_year_table, which estimates the vol on
        each year's own slice when the caller passed none (as it does on the full grid)."""
        vol = volatility.reindex(index=self.index, columns=self.columns)
        pff = self.prices.ffill()
        vv = vol.where(np.isfinite(vol) & (vol > 0))
        px_v = (pff.diff() / vv).fillna(0.0).cumsum().where(pff.notna() & vv.notna())
        return ReducedGrid(kept=self.kept, full_index=self.full_index, prices=self.prices, volatility=vol,
                           spreads=self.spreads, px_v=px_v, freq_seconds=self.freq_seconds,
                           stamps_ns=self.stamps_ns, trunc_ns=self.trunc_ns, shift=self.shift,
                           meta={**self.meta, 'vol_source': 'replaced'})


def _kept_rows(prices, stamps_ns, trunc_ns):
    """The kept-row mask (rules a, b, c, h, f per column; d, e and the truncation rows per frame)."""
    ts = _ns(prices.index)
    n = len(ts)
    day = ts // NS_DAY
    keep = np.zeros(n, dtype=bool)
    change = day[1:] != day[:-1]
    keep[np.flatnonzero(np.r_[True, change])] = True            # d: first row of each day
    keep[np.flatnonzero(np.r_[change, True])] = True            # e: last row of each day
    q = np.searchsorted(ts, trunc_ns, side='right') - 1         # last frame row <= T + td
    keep[q[q >= 0]] = True
    notna = prices.notna().to_numpy()
    n_fill = 0
    for j in range(notna.shape[1]):
        rows = np.flatnonzero(notna[:, j])
        if len(rows) == 0:
            continue
        rt = ts[rows]
        p = np.searchsorted(rt, stamps_ns, side='left')
        pv = p[p < len(rows)]
        keep[rows[pv]] = True                                   # a: fill
        n_fill += len(pv)
        qb = pv + 1
        keep[rows[qb[qb < len(rows)]]] = True                   # b: the print after the fill
        dj = day[rows]
        dchg = dj[1:] != dj[:-1]
        keep[rows[np.flatnonzero(np.r_[dchg, True])]] = True    # c: last print of each day
        keep[rows[np.flatnonzero(np.r_[True, dchg])]] = True    # h (and g): first print of each day
        qf = np.searchsorted(rt, trunc_ns, side='right') - 1    # f: last print <= T + td
        keep[rows[qf[qf >= 0]]] = True
        want = min(MIN_PRINTS_PER_COLUMN, len(rows))
        if keep[rows].sum() < want:                             # classifier guard
            keep[rows[np.linspace(0, len(rows) - 1, want).round().astype(int)]] = True
    return keep, n_fill


def reduce_price_grid(prices, stamps, shift, *, trunc_stamps=(), volatility=None, spreads=None,
                      columns=None, block=256, max_kept_fraction=None):
    """Build the ``ReducedGrid`` for evaluating signals stamped at ``stamps`` on ``prices``.

    stamps: the union of every signal stamp that will be evaluated on the grid (a DatetimeIndex or
        an iterable of them). trunc_stamps: every stamp one_pager will cut the frame at (each
        evaluated signal's last stamp, per-calendar-year last stamps for per_year_table, the
        eval-window end). volatility: the caller's vol frame (as-of reindexed onto the full grid)
        or None for plot._estimate_asset_vol; it must be constant within each day per column.
        spreads: raw spread observations; aligned on the full grid (plot._align_spreads) so the
        monthly-median imputation sees every row, then sampled. columns: restrict the grid.

    Raises GridNotApplicable (a ValueError; ``reduce='auto'`` falls back to the full grid on it)
    when the frame does not classify as prices, its wall clock is not strictly increasing, or the
    vol changes within a day; plain ValueError for an integer shift or a tz mismatch.
    """
    import time as _time
    t0 = _time.time()
    if columns is not None:
        prices = prices[pd.Index(columns)]
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError('reduce_price_grid: prices need a DatetimeIndex')
    int_shift, td = kf._parse_shift(shift)
    if td is None or td <= pd.Timedelta(0):
        raise ValueError(f'reduce_price_grid: shift must be a positive timedelta, got {shift!r}')
    ts = _ns(prices.index)
    if len(ts) < 2 or not np.all(np.diff(ts) > 0):
        raise GridNotApplicable('price index wall clock is not strictly increasing')
    if kf.is_price_or_return(prices) != 'price':
        raise GridNotApplicable('input classified as returns')

    if isinstance(stamps, (list, tuple)):                       # several indexes: union keeps the tz
        _u = pd.DatetimeIndex(stamps[0]).unique()
        for _s in stamps[1:]:
            _u = _u.union(pd.DatetimeIndex(_s))
        stamps = _u.sort_values()
    else:
        stamps = pd.DatetimeIndex(stamps)
    _to_price_tz(stamps[:1], prices.index)                       # tz compatibility check
    st = shifted_stamps_ns(stamps, td)
    st = st[st <= ts[-1]]
    trunc_stamps = pd.DatetimeIndex(list(trunc_stamps)) if len(trunc_stamps) else pd.DatetimeIndex([])
    tr = shifted_stamps_ns(_to_price_tz(trunc_stamps, prices.index), td) if len(trunc_stamps) else np.array([], np.int64)
    tr = np.unique(np.concatenate([tr, [ts[-1]]]))

    keep, n_fill = _kept_rows(prices, st, tr)
    K = np.flatnonzero(keep)
    if max_kept_fraction is not None and len(K) > max_kept_fraction * len(prices):
        # Decided on the kept-row count alone, BEFORE the vol / spreads / risk price passes over
        # the full grid (the bulk of the build): an illiquid universe or intraday stamps can keep
        # more than half the rows, and 'auto' then evaluates on the full grid.
        raise GridNotApplicable(f'kept {len(K)} of {len(prices)} rows')
    reduced = prices.iloc[K]
    if kf.is_price_or_return(reduced) != 'price':
        raise GridNotApplicable('reduced frame no longer classifies as prices')

    # volatility on the FULL grid, then sampled; must be constant within each day per column
    from ebt.simulate.plot import _estimate_asset_vol, _align_spreads
    if volatility is None:
        vol_full = _estimate_asset_vol(prices)
    else:
        vol_full = volatility.reindex(columns=prices.columns).reindex(index=prices.index, method='ffill')
    day = ts // NS_DAY
    same_day = day[1:] == day[:-1]
    n_cols = prices.shape[1]
    for j0 in range(0, n_cols, block):
        v = vol_full.iloc[:, j0:j0 + block].to_numpy(dtype=np.float64)
        eq = (v[1:] == v[:-1]) | (np.isnan(v[1:]) & np.isnan(v[:-1]))
        bad = same_day[:, None] & ~eq
        if bad.any():
            i, jj = np.argwhere(bad)[0]
            raise GridNotApplicable(f'volatility of {prices.columns[j0 + jj]!r} changes within '
                                    f'{prices.index[i + 1].date()}; a reduced grid needs a vol that is '
                                    f'constant within each day (a daily estimate, no intraday fills)')
    vol_red = vol_full.iloc[K]

    spreads_red = None
    if spreads is not None:
        spreads_red = _align_spreads(spreads, prices).fillna(0.0).iloc[K]

    # risk-scaled cumulative price on the FULL grid, in column blocks, sampled at K
    px_v = np.empty((len(K), n_cols), dtype=np.float64)
    for j0 in range(0, n_cols, block):
        p = prices.iloc[:, j0:j0 + block].to_numpy(dtype=np.float64)
        pff = pd.DataFrame(p).ffill().to_numpy()
        dp = np.empty_like(pff)
        dp[0] = np.nan
        dp[1:] = pff[1:] - pff[:-1]
        v = vol_full.iloc[:, j0:j0 + block].to_numpy(dtype=np.float64)
        vv = np.where(np.isfinite(v) & (v > 0), v, np.nan)
        c = np.cumsum(np.nan_to_num(dp / vv), axis=0)
        c[np.isnan(pff) | np.isnan(vv)] = np.nan
        px_v[:, j0:j0 + block] = c[K]
    px_v = pd.DataFrame(px_v, index=reduced.index, columns=prices.columns)

    freq_seconds = float(np.median(np.diff(ts))) / 1e9
    meta = {'rows_full': int(len(ts)), 'rows_kept': int(len(K)), 'ratio': round(len(ts) / max(len(K), 1), 2),
            'kept_per_day': round(len(K) / max(len(np.unique(day)), 1), 2), 'n_stamps': int(len(st)),
            'n_fills': int(n_fill), 'n_cols': int(n_cols), 'shift': str(td), 'build_s': round(_time.time() - t0, 2),
            'vol_source': 'estimated' if volatility is None else 'provided'}
    _LOGGER.info('reduce_price_grid: kept %d of %d rows (%.1fx) for %d stamps, %d columns in %.1fs',
                 len(K), len(ts), meta['ratio'], len(st), n_cols, meta['build_s'])
    return ReducedGrid(kept=K, full_index=prices.index, prices=reduced, volatility=vol_red, spreads=spreads_red,
                       px_v=px_v, freq_seconds=freq_seconds, stamps_ns=st, trunc_ns=tr, shift=td, meta=meta)
