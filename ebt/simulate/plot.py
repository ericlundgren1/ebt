from scipy.stats import kurtosis, skew
from collections.abc import Iterable
from ebt.simulate import kf
from ebt.simulate.timeseries import merge_asof
from ebt.simulate.grid import (ReducedGrid, GridNotApplicable, GRID_INCOMPATIBLE_KFS, MAX_KEPT_FRACTION,
                                applicability, reduce_price_grid, canonical_trunc_stamps)
from ebt.data import LazyDiskDict
from tqdm.auto import tqdm
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns
import pandas as pd
import numpy as np
import logging
import math
import os
import gc
import re

_LOGGER = logging.getLogger(__name__)


def _decile_pnl_matrix(pos_2d, ret_2d, edges, n):
    """(n_rows, n) matrix of per-row PnL contributed by each position decile.

    ``out[i, b] = nansum_j(pos[i, j] * ret[i, j])`` over the cells of row i whose position
    lies in decile b — the quantity the per-decile risk bars are the std of. Equivalent to ten
    masked full-frame passes, up to float rounding.
    """
    pos_2d = np.asarray(pos_2d, dtype=np.float64)
    ret_2d = np.asarray(ret_2d, dtype=np.float64)
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    n_r = pos_2d.shape[0]
    out = np.zeros((n_r, n))
    if n_r == 0 or pos_2d.shape[1] == 0:
        return out
    if kf._decile_pnl_block is None:  # no numba: masked reference path
        nan_mask = np.isnan(pos_2d) | np.isnan(ret_2d)
        bins_2d = np.digitize(np.where(nan_mask, 0, pos_2d), edges)
        for i in range(n):
            pos_bin = np.where((bins_2d == i) & ~nan_mask, pos_2d, 0.0)
            out[:, i] = np.nansum(pos_bin * ret_2d, axis=1)
        return out
    nb = max(1, min(n_r, 4 * kf._kernel_threads(n_r)))
    bs = -(-n_r // nb)
    bounds = [(i0, min(n_r, i0 + bs)) for i0 in range(0, n_r, bs)]
    # pandas hands out column-major blocks; walk them as they lie rather than copying two
    # full frames. Identical sums either way.
    if (pos_2d.flags['F_CONTIGUOUS'] and ret_2d.flags['F_CONTIGUOUS']
            and not pos_2d.flags['C_CONTIGUOUS'] and kf._decile_pnl_block_cm is not None):
        _p, _r, _fn = pos_2d.T, ret_2d.T, kf._decile_pnl_block_cm
    else:
        _p, _r, _fn = np.ascontiguousarray(pos_2d), np.ascontiguousarray(ret_2d), kf._decile_pnl_block
    kf._run_blocks(lambda i0, i1, b: _fn(_p, _r, edges, i0, i1, out), bounds,
                   work=pos_2d.shape[0] * pos_2d.shape[1])
    return out


def _decile_risk(pos, returns, n=10, max_samples=100_000):
    """Compute PnL std per position decile. Returns array of risk shares (sum to 1)."""
    n_rows = len(pos)
    max_rows = max_samples
    if n_rows > max_rows:
        rng = np.random.RandomState(42)
        row_idx = np.sort(rng.choice(n_rows, max_rows, replace=False))
        pos_sub = pos.iloc[row_idx]
        ret_sub = returns.iloc[row_idx] if len(returns) == n_rows else returns.reindex(pos.index[row_idx])
    else:
        pos_sub = pos
        ret_sub = returns if len(returns) == n_rows else returns.reindex(pos.index)

    pos_vals = pos_sub.values.ravel()
    ret_vals = ret_sub.values.ravel()
    mask = ~(np.isnan(pos_vals) | np.isnan(ret_vals))
    pos_flat = pos_vals[mask]

    if len(pos_flat) == 0:
        return np.full(n, np.nan)

    try:
        percentiles = np.percentile(pos_flat, np.linspace(0, 100, n + 1))
    except (ValueError, IndexError):
        return np.full(n, np.nan)

    risk_stds = np.std(_decile_pnl_matrix(pos_sub.values, ret_sub.values, percentiles[1:-1], n), axis=0)

    total = risk_stds.sum()
    return risk_stds / total if total > 0 else np.full(n, 1.0 / n)


plt.set_loglevel('WARNING')


def _format_update_freq(seconds):
    """Format update frequency in human-readable units."""
    if seconds < 60:
        return f'{seconds:.0f}s'
    elif seconds < 3600:
        return f'{seconds/60:.1f}min'
    elif seconds < 86400:
        return f'{seconds/3600:.1f}h'
    elif seconds < 604800:
        return f'{seconds/86400:.1f}d'
    else:
        return f'{seconds/604800:.1f}w'


def plot_kf_table(kf, table_ax=None, kfs_per_column=8):
    '''
    Given a dataframe or series of keyfigures, returns a axes object of the kfs in a table.
    '''

    columns = []
    if kf.shape[0] > kfs_per_column:
        for column in range(math.ceil((kf.shape[0] / kfs_per_column))):
            columns.append((kf[kfs_per_column * column: kfs_per_column * column + kfs_per_column]).reset_index())

        kf = pd.concat(columns, axis=1)

    table_ax = plt if table_ax is None else table_ax
    kf = kf.to_frame().reset_index() if type(kf) == pd.Series else kf
    def _fmt(x):
        if isinstance(x, float):
            if pd.isna(x):
                return ''
            return round(x, 2)
        return x
    kf = kf.copy()
    # Convert time-based KFs to human-readable strings before numeric rounding.
    kf = kf.astype(object)
    for r in range(kf.shape[0]):
        for c in range(kf.shape[1]):
            label = kf.iat[r, c]
            if c + 1 >= kf.shape[1]:
                continue
            v = kf.iat[r, c + 1]
            if not isinstance(v, (int, float)) or pd.isna(v):
                continue
            if label in ('freq signal', 'freq prices', 'freq pnl'):
                kf.iat[r, c + 1] = _format_update_freq(v)
            elif label == 'holding(days)':
                kf.iat[r, c] = 'holding'
                kf.iat[r, c + 1] = _format_update_freq(v * 86400)
    kf = kf.map(_fmt)
    table_ax.axis('off')

    table = table_ax.table(
        cellText=kf.values,
        colLabels=kf.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(kf.columns))))  # Auto adjust column width

    # Scale cell heights so the table stays compact regardless of axes size
    n_rows = kf.shape[0] + 1  # data rows + header
    _target_table_fraction = min(1.0, n_rows * 0.08 + 0.1)  # ~80% for typical 8-row table
    _header_h = _target_table_fraction / n_rows * 1.2
    _data_h = _target_table_fraction / n_rows

    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        if key[0] == 0:  # Header row
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="w", weight="bold")
        elif key[0] > 0 and key[0] % 2 == 0:  # Even rows
            cell.set_facecolor("#f2f2f2")
        else:  # Odd rows
            cell.set_facecolor("#ffffff")
        if key[0] == 0:  # Header row
            cell.set_height(_header_h)
        else:  # Data rows
            cell.set_height(_data_h)
    
    # Add bold borders
    for edge, spine in table.axes.spines.items():
        spine.set_visible(False)  # Hide axis spines

    return table


def plot_equity_curve(pnl, ax=None, color="skyblue", label=None):
    '''
    Returns an axes object of an equity curve
    '''
    _, ax = plt.subplots() if ax is None else (None, ax)

    pnl = pnl.sum(axis=1) if type(pnl) == pd.DataFrame else pnl

    # Daily aggregation with numpy instead of pandas resample
    from ebt.simulate.kf import _index_to_ns
    ts_ns = _index_to_ns(pnl.index)
    day_keys = ts_ns // 86_400_000_000_000
    unique_days, inverse = np.unique(day_keys, return_inverse=True)
    pnl_clean = np.nan_to_num(pnl.values, nan=0.0)
    daily_sum = np.bincount(inverse, weights=pnl_clean, minlength=len(unique_days))
    # Drop zero days, cumsum
    nonzero = daily_sum != 0
    daily_nonzero = daily_sum[nonzero]
    # Create a date index from unique days
    day_dates = pd.to_datetime(unique_days[nonzero] * 86_400_000_000_000)
    pnl_resampled = pd.Series(daily_nonzero.cumsum(), index=day_dates, name='equity curve')

    ax = pnl_resampled.plot(ax=ax, color=color, linewidth=2, alpha=0.8, label=label)
    
    # Styling
    #ax.set_title("Equity Curve", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12, labelpad=10)
    ax.set_ylabel("Cumulative Profit/Loss", fontsize=12, labelpad=10)
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Highlight zero line
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    
    return ax


# Above this many instruments the per-market Sharpe panel is a histogram: the bar labels
# are unreadable anyway and the shape of the distribution is what matters.
SHARPE_BARS_MAX_MARKETS = 30
_SHARPE_HIST_BIN = 0.25   # fixed Sharpe grid so gross and net overlays share bin edges


def _plot_sharpe_hist(sharpes, ax, color, marker_fontsize):
    vals = sharpes.values.astype(float)
    vals = vals[np.isfinite(vals)]
    w = _SHARPE_HIST_BIN
    lo, hi = np.floor(vals.min() / w) * w, np.ceil(vals.max() / w) * w
    if hi <= lo:
        hi = lo + w
    edges = np.arange(lo, hi + w / 2, w)
    ax.hist(vals, bins=edges, color=color, edgecolor='black', alpha=0.6, linewidth=0.5)
    ax.axvline(0.0, color='#666666', linewidth=0.8)
    med = float(np.median(vals))
    ax.axvline(med, color=color, linestyle='--', linewidth=1.2)
    ax.annotate(f'median {med:.2f}', xy=(med, 1.0), xycoords=('data', 'axes fraction'),
                xytext=(3, -2), textcoords='offset points', ha='left', va='top',
                fontsize=marker_fontsize * 0.7, color='#333333')
    ax.text(0.99, 0.98, f'{len(vals)} markets', transform=ax.transAxes, ha='right', va='top',
            fontsize=marker_fontsize * 0.7, color='#666666')
    _decorate_axes(ax, 'Sharpe', 'markets')
    ax.tick_params(axis='x', rotation=0, labelsize=7)
    return ax


def plot_sharpe_bars(sharpes, ax=None, marker_fontsize=10, title="Sharpe", x_title="Instruments", y_title=None, color='skyblue', highlight=None, max_bars=None):
    _, ax = plt.subplots() if ax is None else (None, ax)

    if sharpes.empty:
        return

    vals = sharpes.values.astype(float)
    abs_vals = np.abs(vals[np.isfinite(vals)])
    if len(abs_vals) == 0:
        return ax

    # Histogram mode is opt-in per call site (the per-market panel): a lag or
    # trend-correlation series is ordered and must stay a bar chart however long it is.
    if max_bars is not None and len(sharpes) > max_bars:
        return _plot_sharpe_hist(sharpes, ax, color, marker_fontsize)

    # Build per-bar colors: highlight specific index label in green
    def _bar_colors(base_colors):
        if highlight is None:
            return base_colors
        if isinstance(base_colors, str):
            base_colors = [base_colors] * len(sharpes)
        idx_list = list(sharpes.index)
        return ['#2ecc71' if idx_list[i] == highlight else base_colors[i] for i in range(len(idx_list))]

    # Clip outlier bars: use median absolute value to find outliers,
    # clip at max of the non-outlier values so the small bars are readable
    if len(abs_vals) >= 4:
        median_abs = np.median(abs_vals)
        if median_abs > 0:
            non_outlier_mask = abs_vals < median_abs * 5
            if non_outlier_mask.any() and not non_outlier_mask.all():
                clip_threshold = np.max(abs_vals[non_outlier_mask]) * 1.5
                clipped = sharpes.clip(lower=-clip_threshold, upper=clip_threshold)
                is_clipped = np.array([np.isfinite(v) and abs(v) > clip_threshold for v in vals])
                bar_colors = ['#e74c3c' if c else color for c in is_clipped]
                bar_colors = _bar_colors(bar_colors)
                clipped.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', alpha=0.8)
                for i, (orig, clp) in enumerate(zip(vals, clipped.values)):
                    if np.isfinite(orig) and abs(orig) > clip_threshold:
                        ax.annotate(f'{orig:.1f}', xy=(i, clp), ha='center',
                                    va='bottom' if orig > 0 else 'top',
                                    fontsize=marker_fontsize * 0.9, fontweight='bold',
                                    color='#333333')
                _decorate_axes(ax, x_title, y_title)
                return ax

    sharpes.plot(kind='bar', ax=ax, color=_bar_colors(color), edgecolor='black', alpha=0.8)
    _decorate_axes(ax, x_title, y_title)
    return ax


def plot_quantiles(pos, returns, n=10, ax=None, ax_risk=None, normalize_returns=True, max_samples=100_000):
    _, ax = plt.subplots() if ax is None else (None, ax)

    # Subsample ROWS to keep computation reasonable
    n_rows = len(pos)
    max_rows = max_samples
    if n_rows > max_rows:
        rng = np.random.RandomState(42)
        row_idx = np.sort(rng.choice(n_rows, max_rows, replace=False))
        pos_sub = pos.iloc[row_idx]
        ret_sub = returns.iloc[row_idx] if len(returns) == n_rows else returns.reindex(pos.index[row_idx])
    else:
        pos_sub = pos
        ret_sub = returns if len(returns) == n_rows else returns.reindex(pos.index)

    if normalize_returns:
        ret_std = ret_sub.std()
        ret_std = ret_std.replace(0, np.nan)
        ret_sub = ret_sub / ret_std

    pos_vals = pos_sub.values.ravel()
    ret_vals = ret_sub.values.ravel()

    # Drop pairs where either is NaN
    mask = ~(np.isnan(pos_vals) | np.isnan(ret_vals))
    pos_flat = pos_vals[mask]
    ret_flat = ret_vals[mask]

    if len(pos_flat) == 0:
        return ax

    # Quantile binning with numpy
    try:
        percentiles = np.percentile(pos_flat, np.linspace(0, 100, n + 1))
        bins = np.digitize(pos_flat, percentiles[1:-1])  # 0..n-1
    except (ValueError, IndexError):
        _LOGGER.warning('Could not calculate quantiles for position')
        return ax

    # Collect per-bin mean return, SEM, and range label (one bincount pass per moment;
    # ret_flat has no NaN here, so nanmean/nanstd reduce to mean/std)
    counts = np.bincount(bins, minlength=n).astype(np.float64)
    sums = np.bincount(bins, weights=ret_flat, minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'):
        bin_means = sums / counts
    ss = np.zeros(n)
    _CH = 1 << 25  # bound the two ret_flat-sized temporaries to ~270 MB each
    for _s0 in range(0, len(ret_flat), _CH):
        _sl = slice(_s0, _s0 + _CH)
        ss += np.bincount(bins[_sl], weights=(ret_flat[_sl] - bin_means[bins[_sl]]) ** 2, minlength=n)
    x = []
    means = []
    sems = []
    labels = []
    for i in range(n):
        count = int(counts[i])
        if count < 2:
            continue
        x.append(i + 1)
        means.append(bin_means[i])
        sems.append(np.sqrt(ss[i] / (count - 1)) / np.sqrt(count))
        labels.append(f'{percentiles[i]:.3f}')

    if len(x) < 2:
        return ax

    x = np.array(x)
    means = np.array(means)
    sems = np.array(sems)

    cmap = plt.cm.RdBu
    colors = [cmap(xi / n) for xi in x]

    ax.scatter(x, means, c=colors, s=60, zorder=5, edgecolors='black', linewidths=0.5)
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='#999999', elinewidth=1, capsize=3, capthick=0.8, zorder=4)
    ax.plot(x, means, '-', color='#999999', linewidth=1, alpha=0.5, zorder=3)

    slope, intercept = np.polyfit(x, means, 1)
    ax.plot(x, slope * x + intercept, '--', color='#333333', linewidth=1.2, alpha=0.7)

    # Place labels at bin edges (between bars): x[0]-0.5, x[0]+0.5, x[1]+0.5, ...
    labels.append(f'{percentiles[len(labels)]:.3f}')
    edge_ticks = [x[0] - 0.5] + [xi + 0.5 for xi in x]
    ax.set_xticks(edge_ticks)
    ax.set_xticklabels(labels, fontsize=3)
    ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    _decorate_axes(ax, 'Position range', 'Mean return')

    # Risk / density subplot
    if ax_risk is not None:
        # Risk per decile: backtest positions in each bin and measure PnL std
        # per-row PnL per decile in one pass, then the std of the plotted bins
        _pnl_by_bin = _decile_pnl_matrix(pos_sub.values, ret_sub.values, percentiles[1:-1], n)
        risk_stds = np.array([np.std(_pnl_by_bin[:, xi - 1]) for xi in x])
        total_risk = risk_stds.sum()
        risk_pct = risk_stds / total_risk * 100 if total_risk > 0 else np.zeros_like(risk_stds)

        # Position density: since each decile has equal count,
        # density ∝ 1/bin_width — shows where positions concentrate
        density = []
        for xi in x:
            i = xi - 1
            width = percentiles[i + 1] - percentiles[i]
            density.append(1.0 / width if width > 0 else 0.0)
        density = np.array(density)
        if density.sum() > 0:
            density = density / density.sum() * 100

        # Colored bars for risk, grey filled area for density
        ax_risk.bar(x, risk_pct, width=0.6, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5, zorder=3, label='Risk')
        ax_risk2 = ax_risk.twinx()
        ax_risk2.fill_between(x, 0, density, color='#999999', alpha=0.15, zorder=1)
        ax_risk2.plot(x, density, '-', color='#999999', linewidth=1, alpha=0.5, zorder=2)
        ax_risk2.set_yticks([])
        ax_risk2.spines["top"].set_visible(False)
        ax_risk2.spines["right"].set_visible(False)

        ax_risk.set_xticks(edge_ticks)
        ax_risk.set_xticklabels(labels, fontsize=3)
        ax_risk.set_xlim(x[0] - 0.8, x[-1] + 0.8)
        ax_risk.set_ylabel('Risk %', fontsize=4, labelpad=3)
        ax_risk.tick_params(axis='y', labelsize=5)
        ax_risk.tick_params(axis='x', rotation=0, labelsize=4, pad=1)
        ax_risk.spines["top"].set_visible(False)
        ax_risk.spines["right"].set_visible(False)
        ax_risk.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    return ax


def _format_holding_days(x, pos=None):
    """Format holding period in days to human-readable timedelta."""
    if x <= 0 or not np.isfinite(x):
        return ''
    seconds = x * 86400
    if seconds < 60:
        return f'{seconds:.0f}s'
    minutes = seconds / 60
    if minutes < 60:
        return f'{minutes:.0f}m'
    hours = minutes / 60
    if hours < 24:
        return f'{hours:.1f}h' if hours < 10 else f'{hours:.0f}h'
    if x < 365:
        return f'{x:.1f}d' if x < 10 else f'{x:.0f}d'
    return f'{x / 365:.1f}y'


def _format_holding_with_turnover(x, pos=None):
    """Format holding period with turnover on second line."""
    hold = _format_holding_days(x, pos)
    if not hold:
        return ''
    tv = 1.0 / x if x > 0 else 0
    if tv >= 10:
        tv_str = f'{tv:.0f}/d'
    elif tv >= 1:
        tv_str = f'{tv:.1f}/d'
    else:
        tv_str = f'{tv:.2f}/d'
    return f'{hold}\n{tv_str}'


def _decorate_axes(ax, x_label=None, y_label=None):
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=7, labelpad=5)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=7, labelpad=5)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid lines
    ax.spines["top"].set_visible(False)  # Hide the top spine
    ax.spines["right"].set_visible(False)  # Hide the right spine
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    l = ax.get_legend()
    if l is not None:
        l.remove()


def _align_spreads(spreads, target, tolerance='10min', fallback_multiplier=10.0):
    """Align spreads to a target index using as-of join and impute missing.

    Step 1 — Alignment: reindex spreads onto the target timestamps using
    forward-fill with a tolerance window. A spread observation is only
    carried forward if it's less than ``tolerance`` old. Beyond that it
    becomes NaN (truly missing, not stale).

    Step 2 — Imputation: remaining NaN spreads are filled with
    ``fallback_multiplier`` × the expanding median of the aligned spreads
    (per column, causal). This gives a conservative, instrument-relative
    estimate rather than a hardcoded absolute value.

    Parameters
    ----------
    spreads : DataFrame
        Raw spread observations (may be on a different frequency/index).
    target : DataFrame
        Target DataFrame whose index and columns we align to.
    tolerance : str or Timedelta
        Grace period for the as-of join. Default '10min'.
    fallback_multiplier : float
        Multiplier on expanding median for imputing missing spreads.
    """
    tol = pd.Timedelta(tolerance)

    # Column alignment
    col_aligned = spreads.reindex(columns=target.columns)

    # As-of join: forward-fill within tolerance only
    aligned = col_aligned.reindex(index=target.index, method='ffill', tolerance=tol)

    # Impute: NaN → fallback_multiplier × lagged monthly median (causal)
    if aligned.isna().any().any():
        monthly_median = aligned.resample('MS').median()
        baseline = monthly_median.shift(1).fillna(monthly_median).reindex(index=target.index, method='ffill')
        aligned = aligned.fillna(baseline * fallback_multiplier)

    return aligned


def _estimate_asset_vol(prices, span=60, min_periods=20):
    """Estimate per-asset volatility from daily price diffs with EMA.

    Returns a DataFrame (time x asset) of rolling EMA vol, reindexed to the
    original prices index via forward-fill. Each timestamp uses only past data
    (causal / walk-forward).

    A bar carries a MEASURED vol or NaN: ``min_periods`` completed
    daily bars before the EWM std is trusted, no cross-sectional fill, no 1.0 fill. Everything
    that depends on the vol is NaN where the vol is (one_pager drops those name-bars from its
    risk-unit figures; the stop-break cannot suppress a trade it cannot size). The value is
    constant within each calendar day, which ebt.simulate.grid relies on.
    """
    # label/closed='right': the daily bar stamped T holds prices known at T, so the
    # ffill-reindex below never hands an intraday bar a vol computed from that day's
    # close or weekend prints.
    daily = prices.resample('B', label='right', closed='right').last().dropna(how='all')
    returns = daily.diff()
    vol_daily = returns.ewm(span=span, min_periods=min_periods).std()
    # a zero std is not a measurement either
    vol_daily = vol_daily.replace(0.0, np.nan)
    # Forward-fill to original prices index (causal: each bar sees last known daily vol)
    return vol_daily.reindex(prices.index, method='ffill')


def _compute_holding_curve(pos, prices, shift, spreads, resample=True, volatility=None):
    """Compute sharpe/turnover across EWM smoothing spans without recursive one_pager calls.

    volatility: optional DataFrame of per-asset daily vol (time x asset).
        When provided, holding time is computed on risk-normalized positions
        (smoothed_ff * vol), consistent with one_pager's other KFs.
    """
    ewm_spans_numeric = [0.0] + [2 ** (i / 2) for i in range(2, 24)]
    ewm_spans = [None] + [2 ** (i / 2) for i in range(2, 24)]

    # Rust fast path for the EWM-curve Sharpe ONLY (its holding output is ignored —
    # holding is computed in Python below). Feed gap-aware (ffilled) prices so the
    # Sharpe matches the gap-aware headline; raw prices would drop the reopen-gap PnL
    # and understate the whole curve.
    pos_ff = pos.ffill()
    _prices_ff = _mark_to_last_price(prices)
    rust_results = kf.holding_curve(pos, _prices_ff, shift=shift, ewm_spans=ewm_spans_numeric)
    # Gap-aware returns (single source = _mark_to_last_price) for the Python pnl/cost paths.
    _returns_ff = _prices_ff.diff()

    # Fast path (no spreads, Rust Sharpe available): the holding column needs only the
    # per-bar |d(smoothed*vol)| and |smoothed*vol| sums, which a fused column-major kernel
    # produces per span without materializing 23 smoothed copies of the position frame
    # (the EWM replicates pandas bit for bit).
    _fast_holdings = None
    if spreads is None and rust_results is not None:
        _fast_holdings = kf.holding_curve_fast(pos_ff, volatility, ewm_spans, resample=True)

    rows = []
    for i, span in enumerate(ewm_spans):
        if _fast_holdings is not None:
            rows.append({'sharpe': rust_results[i][0], 'holding(days)': _fast_holdings[i], 'sharpe w costs': np.nan})
            continue
        smoothed_ff = pos_ff if span is None else pos_ff.ewm(span=span).mean()

        # Sharpe: from Rust if available, else Python
        if rust_results is not None:
            sharpe_val = rust_results[i][0]
        else:
            pnl = (smoothed_ff * _returns_ff).dropna(how='all')
            sharpe_val = float(pnl.sum(axis=1).pipe(kf.sharpe, resample=resample))

        # Holding: on risk-normalized positions for consistency
        if volatility is not None:
            smoothed_risk = smoothed_ff * volatility
        else:
            smoothed_risk = smoothed_ff
        holding_val = kf.holding_time(smoothed_risk, resample=True)

        row = {'sharpe': sharpe_val, 'holding(days)': holding_val, 'sharpe w costs': np.nan}

        # Costs
        if spreads is not None:
            pnl = (smoothed_ff * _returns_ff).dropna(how='all')
            trades_smooth = smoothed_ff.diff()
            spreads_aligned = _align_spreads(spreads, trades_smooth)
            costs_smooth = (trades_smooth.abs() * spreads_aligned.abs() * 0.5).dropna(how='all', axis=1)
            pnl_with_costs = pnl - costs_smooth.reindex_like(pnl).fillna(0)
            row['sharpe w costs'] = float(pnl_with_costs.sum(axis=1).pipe(kf.sharpe, resample=resample))

        rows.append(row)

    return pd.DataFrame(rows).set_index('holding(days)')


def _compute_stop_break_curve(pos, prices, spreads, resample=True, volatility=None, max_holding=None):
    """Compute sharpe/holding across stop_break thresholds.

    Same structure as _compute_holding_curve but varies the risk-aware
    stop_break threshold instead of EWM smoothing span.  Thresholds are
    sampled adaptively: start at 0.0005 and double each step, stopping
    once the holding time exceeds *max_holding*.
    """
    from ebt.simulate.execution import _risk_stop_break_1d, _risk_stop_break_2d

    asset_vol = volatility if volatility is not None else _estimate_asset_vol(prices)
    asset_vol = asset_vol.reindex(columns=pos.columns).reindex(index=pos.index, method='ffill')
    vol_vals = asset_vol.values
    # Column-major copies for the blocked stop-break kernel (one transpose, reused for every
    # threshold). Only for float64 frames: the per-column path downcast the filtered result
    # back to the input dtype, so a float32 book must keep its float32 accumulation.
    _float64_pos = np.dtype(getattr(pos.values, 'dtype', np.float64)) == np.float64
    if _float64_pos:
        _pos_T = np.ascontiguousarray(pos.to_numpy(dtype=np.float64).T)
        _vol_T = np.ascontiguousarray(np.asarray(vol_vals, dtype=np.float64).T)

    pos_ff = pos.ffill()
    # Gap-aware returns (single source = _mark_to_last_price): held positions earn
    # the reopen gap; realization/vol above stay on RAW prices for open/closed.
    price_diff = _mark_to_last_price(prices).diff()

    def _eval_threshold(thresh):
        if thresh == 0.0:
            filtered_ff = pos_ff
        else:
            if _float64_pos:
                _filt_T = _risk_stop_break_2d(_pos_T, _vol_T, float(thresh))
                filtered = pd.DataFrame(_filt_T.T, index=pos.index, columns=pos.columns)
            else:
                filtered = pos.copy()
                for j in range(pos.shape[1]):
                    filtered.iloc[:, j] = _risk_stop_break_1d(
                        pos.values[:, j].copy(), vol_vals[:, j].copy(), thresh)
            filtered_ff = filtered.ffill()

        pnl = (filtered_ff * price_diff).dropna(how='all')
        sharpe_val = float(pnl.sum(axis=1).pipe(kf.sharpe, resample=resample))

        _fast = kf.holding_curve_fast(filtered_ff, volatility, [None], resample=True)
        if _fast is not None:
            holding_val = _fast[0]
        else:
            filtered_risk = filtered_ff * volatility if volatility is not None else filtered_ff
            holding_val = kf.holding_time(filtered_risk, resample=True)

        row = {'sharpe': sharpe_val, 'holding(days)': holding_val, 'sharpe w costs': np.nan}

        if spreads is not None:
            trades = filtered_ff.diff()
            spreads_aligned = _align_spreads(spreads, trades)
            costs = (trades.abs() * spreads_aligned.abs() * 0.5).dropna(how='all', axis=1)
            pnl_with_costs = pnl - costs.reindex_like(pnl).fillna(0)
            row['sharpe w costs'] = float(pnl_with_costs.sum(axis=1).pipe(kf.sharpe, resample=resample))

        return row

    # Always include the unfiltered baseline
    rows = [_eval_threshold(0.0)]

    # Adaptive sweep: grow threshold by 25% each step, stop at max_holding
    thresh = 0.005
    while True:
        row = _eval_threshold(thresh)
        rows.append(row)
        if max_holding is not None and row['holding(days)'] >= max_holding:
            break
        thresh *= 1.25
        if thresh > 100:  # safety cap
            break

    return pd.DataFrame(rows).set_index('holding(days)')


def _compute_kfs_direct(pos, prices, keyfigures, spreads=None, resample=True, volatility=None):
    """KFs for a pre-realized sub-portfolio position (shift=0).

    Thin adapter over one_pager — the single KF engine — so plot_ensemble's
    sub-portfolio metrics are identical to the headline one_pager. The numpy
    fast path stays the perf path for per-config metrics; this is only for the
    handful of sub-portfolio aggregates. Correlation keyfigures are produced by
    plot_variable, not here, so they are dropped from the request.
    """
    # Strip tz from ALL timeseries inputs consistently. Stripping pos/prices but
    # leaving spreads/volatility tz-aware makes one_pager's _align_spreads compare
    # tz-aware vs tz-naive indices and raise; a no-op for the tz-naive convention.
    # Inputs in a DIFFERENT tz than prices are converted to the prices tz first:
    # tz_localize(None) keeps wall time, so stripping mixed-tz inputs directly
    # would silently shift them against the price grid (staling every spread).
    if isinstance(prices, ReducedGrid):
        # the grid carries vol/spreads; positions are on its rows, so no tz handling is needed
        assert spreads is None and volatility is None, 'pass spreads/volatility through the ReducedGrid'
        op_kfs = [k for k in keyfigures
                  if k not in ('correlation_within_portfolio', 'correlation_between_portfolios')]
        return one_pager(pos, prices.select_columns(pos.columns), shift=0, kfs=op_kfs, resample=resample,
                         plot=False, return_subset={'kf'})['kf']
    _ref_tz = getattr(getattr(prices, 'index', None), 'tz', None)
    def _strip_tz(df):
        tz = getattr(getattr(df, 'index', None), 'tz', None)
        if df is None or tz is None:
            return df
        if _ref_tz is not None and str(tz) != str(_ref_tz):
            df = df.tz_convert(_ref_tz)
        return df.tz_localize(None)
    pos = _strip_tz(pos)
    prices = _strip_tz(prices)
    spreads = _strip_tz(spreads)
    volatility = _strip_tz(volatility)
    op_kfs = [k for k in keyfigures
              if k not in ('correlation_within_portfolio', 'correlation_between_portfolios')]
    return one_pager(pos, prices, shift=0, spreads=spreads, kfs=op_kfs, resample=resample,
                     plot=False, return_subset={'kf'}, volatility=volatility, reduce=False)['kf']


def _risk_space_inputs(pos_vals, price_diff_vals, price_ff_2d, asset_vol, px_v=None):
    """Inputs that make the variant-Sharpe kernel demean in RISK units.

    ``px_v``: the risk-scaled cumulative price already computed (on the FULL grid, sampled at
    the rows of ``pos_vals``) by ebt.simulate.grid — used as-is instead of the local cumsum,
    which on a reduced grid would divide a multi-bar move by the last bar's vol.

    The kernel subtracts each column's time-mean position and reports the PnL of that mean
    ('sharpe bias') and of the remainder ('sharpe exbias'). Demeaning the raw position
    measures a constant-CONTRACT / constant-SHARE book, which is meaningless once a name's
    unit changes scale inside the sample: after a reverse split the mean SHARE count can
    produce a 'bias' book orders of magnitude larger than the strategy's risk. Convention: the
    static component is the constant mean-RISK position per name, i.e. demean pos*vol. Feeding the kernel pos*vol
    together with a price whose diffs are dP/vol leaves the raw PnL bit-identical
    ((pos*vol)*(dP/vol) == pos*dP), turns its column mean into the mean risk, and keeps the
    long/short legs unchanged (same sign). Names without a usable vol fall back to native
    units (vol = 1). ``vol`` is `volatility` if given, else `_estimate_asset_vol(prices)`
    (per-asset diff-vol, the risk_adjust unit)."""
    if asset_vol is None:
        return pos_vals, price_ff_2d
    # No measured vol -> no risk unit: NaN propagates (never a 1.0 stand-in)
    vol = np.where(np.isfinite(asset_vol) & (asset_vol > 0), asset_vol, np.nan)
    if px_v is not None:
        return pos_vals * vol, np.asarray(px_v, dtype=np.float64)
    dpr = price_diff_vals / vol
    px_risk = np.cumsum(np.where(np.isnan(dpr), 0.0, dpr), axis=0)
    px_risk[np.isnan(price_ff_2d) | np.isnan(vol)] = np.nan
    return pos_vals * vol, px_risk


def _one_pager_numpy_fast(signal_vals, pos_ts, price_ts, price_vals_flat, price_vals_ff,
                          price_diff_vals, spreads_vals, shift,
                          n_rows_signal, n_rows_prices, n_cols,
                          day_keys, unique_days, kfs_set, asset_vol=None,
                          stop_break_threshold=None, weights=None, px_v=None):
    """GIL-free one_pager fast path for plot_ensemble workers.

    All inputs are numpy arrays. Returns (kf_dict, pos_ff_vals, pnl_sum_vals).
    The Rust backtest + all KF computations use numpy/Rust only (no pandas, no GIL).
    asset_vol: optional 2D array (n_rows_prices x n_cols) of per-asset volatilities.
    price_vals_flat: RAW prices (NaN at closed bars) — used for position realization
        so turnover/holding respect open/closed. price_vals_ff: gap-aware (ffilled)
        prices — used for the return-based variant sharpes and holding curve.
    weights, px_v: set when the prices are a reduced grid (ebt.simulate.grid): forward
        bar counts per row for the row-averaged figures (turnover, holding, lptp, exbias,
        bias) and the full-grid risk-scaled price for the variant Sharpes. None = full grid.
    """
    import ebt_rust

    # 1. Rust backtest: signal -> pos (releases GIL)
    signal_flat = np.ascontiguousarray(signal_vals.ravel(order='F'), dtype='float64')

    pos_flat = ebt_rust.backtest_matrix_core(
        pos_ts, signal_flat, price_ts, price_vals_flat,
        shift, 1,  # return_variable=1 means 'pos'
        n_rows_signal, n_rows_prices, n_cols
    )
    pos_vals = pos_flat.reshape(n_rows_prices, n_cols, order='F')

    # 1b. Risk stop-break: suppress small trades in risk terms
    if stop_break_threshold is not None and stop_break_threshold > 0 and asset_vol is not None:
        from ebt.simulate.execution import _risk_stop_break_1d
        for j in range(n_cols):
            pos_vals[:, j] = _risk_stop_break_1d(
                pos_vals[:, j].copy(),
                asset_vol[:, j].copy(),
                stop_break_threshold,
            )

    # 2. pos.ffill() in numpy (vectorized, GIL-free)
    pos_ff = kf._numpy_ffill(pos_vals)

    # 3. pnl_sum via vectorized numpy (GIL-free)
    pnl_2d = pos_ff * price_diff_vals
    any_valid = np.any(~np.isnan(pnl_2d), axis=1)
    pnl_sum_vals = np.nansum(pnl_2d, axis=1)
    pnl_sum_vals[~any_valid] = np.nan
    del pnl_2d

    # Drop NaN rows for pnl_sum (equivalent to .dropna())
    pnl_valid = ~np.isnan(pnl_sum_vals)
    pnl_sum_clean = pnl_sum_vals[pnl_valid]
    day_keys_clean = day_keys[pnl_valid]

    # 4. Compute KFs — all numpy
    kf_dict = {}

    if 'sharpe' in kfs_set:
        kf_dict['sharpe'] = kf._numpy_sharpe(pnl_sum_clean, day_keys_clean, unique_days)

    if 'sharpe w costs' in kfs_set:
        if spreads_vals is not None:
            # Vectorized costs computation (GIL-free)
            trades = np.empty_like(pos_ff)
            trades[0] = np.nan
            trades[1:] = pos_ff[1:] - pos_ff[:-1]
            trades[np.isnan(pos_vals)] = np.nan  # mask where original pos is NaN
            costs_2d = np.abs(trades) * np.abs(spreads_vals) * 0.5
            # Mask costs where PnL is NaN (price gap) — consistent with 2D pandas path
            # PnL is NaN when price_diff is NaN (pos_ff is forward-filled, always non-NaN)
            costs_2d[np.isnan(price_diff_vals)] = np.nan
            costs_arr = np.nansum(costs_2d, axis=1)
            del trades, costs_2d
            costs_clean = costs_arr[pnl_valid]
            kf_dict['sharpe w costs'] = kf._numpy_sharpe(
                pnl_sum_clean - costs_clean, day_keys_clean, unique_days)
        else:
            kf_dict['sharpe w costs'] = kf_dict.get('sharpe',
                kf._numpy_sharpe(pnl_sum_clean, day_keys_clean, unique_days))

    # Any of the four variants triggers the one Rust call that produces all of them; each is
    # then reported only if asked for. Gating on 'sharpe exbias' alone would leave a request
    # for 'sharpe long' with the key MISSING rather than computed.
    if kfs_set & {'sharpe exbias', 'sharpe bias', 'sharpe long', 'sharpe short'}:
        # Variant sharpes via Rust (releases GIL) — pos is already realized (shift applied), so pass shift=0
        if kf._HAS_RUST:
            _v_pos, _v_px = _risk_space_inputs(
                pos_vals, price_diff_vals,
                np.asarray(price_vals_ff).reshape(n_rows_prices, n_cols, order='F'), asset_vol,
                px_v=px_v)
            pos_flat_f = np.ascontiguousarray(_v_pos.ravel(order='F'), dtype='float64')
            px_flat_f = np.ascontiguousarray(_v_px.ravel(order='F'), dtype='float64')
            pos_ts_prices = price_ts  # pos is on prices index after backtest
            result = ebt_rust.variant_sharpes_core(
                pos_ts_prices, pos_flat_f, price_ts, px_flat_f,
                0, True, n_rows_prices, n_rows_prices, n_cols, **kf._rust_flat_kw()
            )
            arr = np.asarray(result)
            if px_v is not None and (kfs_set & {'sharpe exbias', 'sharpe bias'}):
                # reduced grid: the kernel's column mean counts every kept row once; the
                # bar-count weighted mean is what equals the full grid
                _exb, _bias = kf._numpy_bias_sharpes_weighted(
                    pos_vals, asset_vol, _v_px, weights, day_keys, unique_days)
                arr = arr.copy()
                arr[0], arr[1] = _exb, _bias
            if 'sharpe exbias' in kfs_set:
                kf_dict['sharpe exbias'] = float(arr[0])
            if 'sharpe bias' in kfs_set:
                kf_dict['sharpe bias'] = float(arr[1])
            if 'sharpe long' in kfs_set:
                kf_dict['sharpe long'] = float(arr[2])
            if 'sharpe short' in kfs_set:
                kf_dict['sharpe short'] = float(arr[3])

    # Risk-normalized positions for turnover/holding/lptp. Use pos_ff (the actual
    # HELD position, forward-filled across closed bars) to match the pandas path's
    # `_pos_risk = pos_ffilled * _vol`; pos_vals has NaN at closed bars (raw
    # realization) which would mis-count turnover/holding.
    _pos_risk = pos_ff * asset_vol if asset_vol is not None else pos_ff

    # Business-day keys: fold weekend bars into the preceding Friday so turnover/holding
    # match the pandas kf.turnover/kf.holding_time, which resample('B').
    _bday_keys = kf._bday_keys(day_keys)
    _unique_bdays = np.unique(_bday_keys)

    if 'holding(days)' in kfs_set or 'turnover' in kfs_set:
        t = kf._numpy_turnover_mean(_pos_risk, _bday_keys, _unique_bdays, weights=weights)
        if 'turnover' in kfs_set:
            kf_dict['turnover'] = t
        if 'holding(days)' in kfs_set:
            kf_dict['holding(days)'] = kf._numpy_holding_time(_pos_risk, _bday_keys, _unique_bdays, weights=weights)

    if 'lptp' in kfs_set:
        kf_dict['lptp'] = kf._numpy_lptp_mean(_pos_risk, weights=weights)

    if 'up_days' in kfs_set:
        kf_dict['up_days'] = kf._numpy_up_days(pnl_sum_clean, day_keys_clean, unique_days)

    if 'risk' in kfs_set or 'turnover(risk1)' in kfs_set:
        r = kf._numpy_risk(pnl_sum_clean, day_keys_clean, unique_days)
        if 'risk' in kfs_set:
            kf_dict['risk'] = r
        if 'turnover(risk1)' in kfs_set:
            t = kf_dict.get('turnover', kf._numpy_turnover_mean(_pos_risk, day_keys, unique_days, weights=weights))
            kf_dict['turnover(risk1)'] = t / r if r > 0 else np.nan

    if 'sharpe_decay' in kfs_set or 'sharpe_decay_wc' in kfs_set:
        assert px_v is None, 'sharpe_decay is defined per price bar and cannot run on a reduced grid'
        ewm_spans = [0.0] + [2 ** (i / 2) for i in range(2, 24)]
        pos_flat_hc = np.ascontiguousarray(pos_vals.ravel(order='F'), dtype='float64')
        hc_results = ebt_rust.holding_curve_core(
            price_ts, pos_flat_hc, price_ts, price_vals_ff,
            0, [float(s) for s in ewm_spans],
            n_rows_prices, n_rows_prices, n_cols, **kf._rust_flat_kw()
        )
        hc_sharpes = np.array([r[0] for r in hc_results])
        # Holding axis exactly as the full path (_compute_holding_curve -> kf.holding_curve_fast):
        # pandas-identical EWM of the HELD position, vol-scaled, business-day aggregation. The
        # Rust kernel's own holding output (raw units, calendar days) is not used, so the
        # per-signal sharpe_decay equals one_pager's and its span-0 point equals holding(days).
        _vals_T = np.ascontiguousarray(pos_ff.T)
        _vol_T = np.ascontiguousarray(asset_vol.T) if asset_vol is not None else np.ones_like(_vals_T)
        _fast_h = kf._holding_curve_fast_arrays(_vals_T, _vol_T, ewm_spans, _bday_keys, _unique_bdays)
        hc_holdings = np.array(_fast_h) if _fast_h is not None else np.array([r[1] for r in hc_results])

        if 'sharpe_decay' in kfs_set:
            kf_dict['sharpe_decay'] = kf.sharpe_decay(hc_sharpes, hc_holdings)
        if 'sharpe_decay_wc' in kfs_set:
            kf_dict['sharpe_decay_wc'] = np.nan

    # Median update frequencies: signal, prices, returns
    def _median_freq_ns(ts):
        if len(ts) > 1:
            return float(np.median(np.diff(ts).astype(float))) / 1e9
        return np.nan
    if 'freq signal' in kfs_set:
        kf_dict['freq signal'] = _median_freq_ns(pos_ts)
    if 'freq prices' in kfs_set:
        kf_dict['freq prices'] = _median_freq_ns(price_ts)
    if 'freq pnl' in kfs_set:
        kf_dict['freq pnl'] = _median_freq_ns(price_ts[pnl_valid])

    # pos_ff is returned so callers that need the forward-filled grid do not
    # recompute it: kf._numpy_ffill allocates a bool mask + an int64 index grid +
    # the result = ~2.125x the grid, per signal. Same array, so the group sums
    # are bit-identical.
    return kf_dict, pos_vals, pnl_sum_vals, pos_ff


def _mark_to_last_price(prices):
    """Forward-fill prices across market-closed (NaN) bars for backtesting.

    A position HELD across a break (weekend / overnight / trading halt) is marked
    to the last traded price while the market is closed (no PnL move) and realizes
    the FULL gap when the market REOPENS (reopen - last_traded). A plain
    ``prices.diff()`` returns NaN across a closed bar and therefore silently DROPS
    that reopen gap, understating any position held across a break (severe for
    session/overnight signals).

    Forward-filling makes the downstream ``prices.diff()`` gap-aware: 0 while
    closed, full gap at reopen. This is the SINGLE place this gap handling lives;
    every backtest/PnL path in this module derives its returns from the result.
    A no-op when prices are already gapless (e.g. forward-filled by the caller),
    so it is robust to either input.
    """
    return prices.ffill()


def _hitrate(pnl_values):
    """Fraction of profitable bars among bars that actually traded.

    Counts winners over bars with non-zero, non-NaN PnL only — NOT over all bars.
    (Closed/NaN bars and flat 0-PnL bars are excluded; on a grid that includes closed hours most bars are
    NaN, so dividing by all bars would massively understate the hit rate.)
    """
    v = pnl_values
    mask = (v != 0) & ~np.isnan(v)
    return float(np.mean(v[mask] > 0)) if mask.any() else np.nan


def one_pager(signal, prices_or_returns, shift='2min', spreads=None, kfs=None, resample=True, fast=False, figsize=None, plot=True, show_plot=True, return_subset=None, save_memory=False, volume=None, vwap_window='60min', volatility=None, smart_execution=None, stop_break=1e-9, reduce='auto', **kwargs):
    '''
    One pager

    :param signal pd.DataFrame: The signal, what the model wishes it had (unlike a --->position<---, which is the position that the model could've realized and actually held)
    :param returns_in pd.DataFrame: Market returns, or a prebuilt ``ebt.simulate.grid.ReducedGrid``
        (plot_ensemble builds one from every signal's stamps and passes it here).
    :param reduce: 'auto' (default) evaluates on a REDUCED price grid (ebt.simulate.grid) when the
        signal is much sparser than the prices and the call allows it (timedelta shift, resample=True,
        no smart_execution / volume, price input); otherwise the full grid, with a logging.info line.
        True raises when the reduction is not applicable; False never reduces. On the reduced grid
        the PnL-based figures equal the full grid to float precision and the row-weighted ones
        (turnover, holding, lptp, exbias, bias) are bar-count weighted so they do too; figures defined
        per price bar (GRID_INCOMPATIBLE_KFS: hitrate, risk_conc, tail_risk, sharpe_decay(_wc),
        freq pnl) and the lag / trend / holding / stop-break / hour / quantile panels are not
        computed — an explicit ``kfs`` naming one raises. ``pos``/``pnl``/``pnl_sum`` come back on
        the reduced index (their daily sums equal the full grid's).
    :param shift: Execution lag. Integer (number of bars) or timedelta/string ('5m', pd.Timedelta('5m'))
        for wall-clock duration. A timedelta means: the trade decided at t is filled at the FIRST price
        print at-or-after t+shift (the signal index is moved forward by the duration, then lagged one
        print past the as-of join). Across a session gap the fill is therefore the reopen print — the
        overnight gap accrues to the previously-held position, never to a change decided while closed.
    :param kfs list[str]: Which kfs to calculate, defaults to
        ['sharpe', 'sharpe exbias', 'sharpe bias', 'sharpe long', 'sharpe short', 'turnover', 'holding(days)', 'lptp', 'hitrate', 'ncvar']
    :param resample bool: Whether to resample the results to daily, defaults to True
    :param fast bool: When plotting, can optimally skip certain calculations to speed up producing the onepager
    :param plot bool: Whether to plot at all or just return key figures
    :param figsize tuple: Size of onepager
    :param volume: Volume DataFrame or Series for VWAP execution simulation. When set,
        the signal is first run through simulate_vwap_execution to model realistic
        netting of overlapping orders. Prices must be close/open (not VWAP) — an error
        is raised if prices show VWAP-like autocorrelation.
    :param vwap_window: Execution window duration (default '60min'). Only used when volume is set.
    :param smart_execution: Enable spread/price-aware execution timing. When set, the realized
        position is filtered through smart_execute after backtest — trades are delayed until
        spread and/or price conditions are favorable. Accepts True (use defaults), or a dict
        of params passed to smart_execute (e.g. {'spread_lookback': '90D', 'spread_threshold': 0.4}).
        Requires spreads to be set.
    :param stop_break: Risk-aware trade suppression threshold. Trades whose risk
        (|trade_lots| * daily_vol) is below this threshold (in daily PnL stds) are
        suppressed. Applied last in the execution chain (after VWAP and smart execution).
        Accepts a float threshold (default 0.001), True (use default), False/None (disabled).
    '''

    # Parse timedelta shift: pre-shift signal timestamps by the latency, then lag one
    # print past the as-of join so the fill lands on the FIRST print at-or-after t+td
    # (shift=0 would implicitly fill at the LAST print before t+td — untradeable across
    # session gaps; see the matching comment in kf.backtest).
    original_shift = shift
    # A prebuilt reduced grid stands in for the price frame (see ebt.simulate.grid)
    _grid = prices_or_returns if isinstance(prices_or_returns, ReducedGrid) else None
    _signal_stamps = signal.index if isinstance(signal, pd.DataFrame) else None   # unshifted
    int_shift, td_shift = kf._parse_shift(shift)
    if td_shift is not None:
        if signal is not None and not (isinstance(signal, (list, dict))):
            signal = kf._apply_timedelta_shift(signal, td_shift)
        shift = 1
        shift_label = str(td_shift)
    else:
        shift = int_shift
        shift_label = str(shift)

    if _grid is not None:
        _is_returns = False
        prices = _grid.prices
    else:
        _is_returns = kf.is_price_or_return(prices_or_returns) == 'return'
        if _is_returns:
            _LOGGER.warning('onepager received returns instead of prices as input, translating may lead to issues if open hours have already been applied')
            prices = prices_or_returns.fillna(0).cumsum()
        else:
            prices = prices_or_returns

    if type(signal) == list:
        return display_figures_with_tabs([
            one_pager(s, prices_or_returns,
                shift=original_shift,
                spreads=spreads,
                kfs=kfs,
                resample=resample,
                fast=fast,
                figsize=figsize,
                plot=True,
                show_plot=False,
                return_subset=return_subset,
                save_memory=save_memory,
                volume=volume,
                vwap_window=vwap_window,
                volatility=volatility,
                smart_execution=smart_execution,
                stop_break=stop_break,
                reduce=reduce
            )['fig'] for s in signal
        ])

    if type(signal) == dict:
        return display_figures_with_tabs({
            k: one_pager(s, prices_or_returns,
                shift=original_shift,
                spreads=spreads,
                kfs=kfs,
                resample=resample,
                fast=fast,
                figsize=figsize,
                plot=True,
                show_plot=False,
                return_subset=return_subset,
                save_memory=save_memory,
                volume=volume,
                vwap_window=vwap_window,
                volatility=volatility,
                smart_execution=smart_execution,
                stop_break=stop_break,
                reduce=reduce
            )['fig'] for k, s in signal.items()
        })

    if signal is None or signal.empty:
        raise ValueError('one_pager: signal is None or empty (no rows or no columns)')

    if _grid is not None:
        # The grid carries vol / spreads and was built for one shift and one stamp set
        if volatility is not None or spreads is not None or volume is not None or smart_execution:
            raise ValueError('one_pager: with a ReducedGrid pass volatility/spreads through the grid and '
                             'no volume / smart_execution (they need every price bar)')
        if not resample:
            raise ValueError('one_pager: resample=False is a per-row statistic; not on a reduced grid')
        if td_shift is not None:
            _grid.check_covers(_signal_stamps, td_shift)
        elif int_shift == 0:
            _grid.check_rows(_signal_stamps)              # already-realised positions
        else:
            raise ValueError(f'one_pager: integer shift {int_shift} counts price prints; a reduced grid '
                             f'only supports its own timedelta shift or shift=0 for realised positions')

    # Validate that all timeseries inputs have sorted, unique indices
    def _assert_index(df, name):
        if df is None:
            return
        idx = df.index if hasattr(df, 'index') else None
        if idx is None:
            return
        assert idx.is_monotonic_increasing, f"one_pager: {name}.index is not monotonic increasing"
        assert idx.is_unique, f"one_pager: {name}.index has duplicates ({idx.duplicated().sum()} dupes)"

    # Same instants in the price zone before anything strips to wall clock (kf._align_tz:
    # both aware -> signal converted to the price zone; one aware, one naive -> error)
    signal = kf._align_tz(signal, prices, 'signal', 'prices_or_returns')
    _assert_index(signal, 'signal')
    _assert_index(prices, 'prices_or_returns')
    _assert_index(spreads, 'spreads')
    _assert_index(volume, 'volume')
    _assert_index(volatility, 'volatility')

    # Detect VWAP-smoothed prices via lag-1 autocorrelation (meaningless on a reduced grid's
    # irregular rows; only the config-box label and the volume= guard read it)
    _prices_are_vwap = False
    _price_cols = prices.columns.intersection(signal.columns)
    if len(_price_cols) > 0 and _grid is None:
        _p = prices[_price_cols]
        _mid = len(_p) // 2
        _sample = _p.iloc[max(0, _mid - 2500):_mid + 2500].diff().dropna()
        if len(_sample) > 200:
            _acs = _sample.apply(lambda c: c.dropna().autocorr(1))
            _ac = _acs.dropna().median()
            _prices_are_vwap = not np.isnan(_ac) and _ac > 0.15

    # --- VWAP execution simulation ---
    if volume is not None:
        from ebt.simulate.execution import simulate_vwap_execution

        # Guard: prices must not be VWAP-smoothed when using VWAP execution
        if _prices_are_vwap:
            raise ValueError(
                f"Prices appear to be VWAP-smoothed (median lag-1 autocorrelation={_ac:.3f}). "
                f"When using volume-based VWAP execution, prices must be close or open prices, "
                f"not VWAP prices. VWAP netting already accounts for execution; evaluating "
                f"against VWAP prices would double-count the smoothing."
            )

        if isinstance(volume, pd.Series):
            vol = pd.DataFrame({c: volume for c in signal.columns}, index=volume.index)
        else:
            vol = volume.reindex(columns=signal.columns)

        # Check: fraction of price bars covered by volume data
        _vol_coverage = vol.reindex(prices.index).notna().any(axis=1).mean()
        if _vol_coverage < 0.5:
            _LOGGER.warning(
                f"Volume covers {_vol_coverage:.1%} of price timestamps. "
                f"Bars without volume get uniform weighting (TWAP)."
            )

        signal = simulate_vwap_execution(signal, vol, vwap_window=vwap_window)

    if _grid is not None:
        _grid = _grid.truncate(signal.index[-1])   # the same cut, on the grid (weights follow)
        prices = _grid.prices
    else:
        prices = prices[:signal.index[-1]] # Don't evaluate pnl after we stop making predictions


    # Align columns of signal and prices
    new_signal, new_prices = signal.align(prices, axis=1, join='inner')
    if signal.columns.difference(new_signal.columns).shape[0] > 0:
        _LOGGER.warning(f'dropping instruments {signal.columns.difference(new_signal.columns)} due to them not being available both in signals and prices')
    signal = new_signal
    prices = new_prices
    if _grid is not None:
        _grid = _grid.select_columns(prices.columns)

    # Reduced grid (default): when the signal is much sparser than the prices, evaluate on the
    # kept rows only. Built here, on the truncated and column-aligned frame, from this signal's
    # own stamps and the canonical truncation set (its last stamp and each calendar year's last
    # stamp, so per_year_table builds the identical grid).
    if _grid is None and reduce not in (False, None):
        _ok, _why = applicability(_signal_stamps, prices.index, original_shift, resample=resample,
                                  smart_execution=smart_execution, volume=volume, is_returns=_is_returns)
        if _ok and kfs is not None and (set(kfs) & GRID_INCOMPATIBLE_KFS):
            raise ValueError(f'one_pager: key figures {sorted(set(kfs) & GRID_INCOMPATIBLE_KFS)} are defined '
                             f'per price bar and are not available on a reduced grid; pass reduce=False')
        if _ok:
            try:
                _cand = reduce_price_grid(prices, _signal_stamps, td_shift,
                                          trunc_stamps=canonical_trunc_stamps(_signal_stamps),
                                          volatility=volatility, spreads=spreads,
                                          max_kept_fraction=MAX_KEPT_FRACTION)
                if len(_cand) > MAX_KEPT_FRACTION * _cand.n_full:
                    _ok, _why = False, f'kept {len(_cand)} of {_cand.n_full} rows'
                else:
                    _grid = _cand
            except GridNotApplicable as _e:
                _ok, _why = False, str(_e)
        if not _ok:
            if reduce is True:
                raise ValueError(f'one_pager: reduced grid not applicable: {_why}')
            _LOGGER.info(f'one_pager: evaluating on the full grid ({_why})')
    if _grid is not None:
        if kfs is not None and (set(kfs) & GRID_INCOMPATIBLE_KFS):
            raise ValueError(f'one_pager: key figures {sorted(set(kfs) & GRID_INCOMPATIBLE_KFS)} are defined '
                             f'per price bar and are not available on a reduced grid; pass reduce=False')
        prices = _grid.prices
        _weights = _grid.weights()
        _vol_in = _grid.volatility
        _spreads_in = _grid.spreads
        _LOGGER.info(f'one_pager: reduced grid {len(_grid)}/{_grid.n_full} rows')
    else:
        _weights = None
        _vol_in = volatility
        _spreads_in = spreads

    # Gap-aware mark-to-market. `prices` is kept RAW from here on: NaN at market-closed
    # bars is what tells position realization, vol, turnover and holding when the market
    # is open. `_prices_ff` (closed bars carry the last traded price) is the SINGLE source
    # of gap handling and is used ONLY for PnL/return computations, so a held position
    # earns the reopen gap instead of having it dropped by a consecutive diff(). Computed
    # AFTER the truncation/column-alignment above so it always matches `prices` exactly.
    _prices_ff = _mark_to_last_price(prices)

    if fast and kfs is None:
        kfs = ['sharpe', 'sharpe exbias', 'sharpe w costs', 'holding(days)', 'lptp', 'up_days']

    # Resolve stop_break to a threshold float or None
    _sb_threshold = None
    if stop_break is True:
        _sb_threshold = 0.001
    elif isinstance(stop_break, (int, float)) and stop_break > 0:
        _sb_threshold = float(stop_break)

    # Fast numpy path: when we only need KFs + pos + pnl_sum and no plotting,
    # delegate to _one_pager_numpy_fast which is GIL-free and faster.
    _numpy_supported_kfs = {'sharpe', 'sharpe w costs', 'sharpe exbias', 'sharpe bias',
                            'sharpe long', 'sharpe short', 'holding(days)', 'turnover',
                            'lptp', 'up_days', 'risk', 'turnover(risk1)',
                            'freq signal', 'freq prices', 'freq pnl', 'sharpe_decay', 'sharpe_decay_wc'}
    _numpy_supported_results = {'kf', 'pos', 'pnl_sum'}
    _can_use_numpy = (
        fast
        and not plot
        and return_subset is not None
        and return_subset <= _numpy_supported_results
        and kfs is not None
        and set(kfs) <= _numpy_supported_kfs
        and kf._HAS_RUST
        and not smart_execution
        # The numpy path aggregates to days unconditionally (kf._numpy_sharpe/_numpy_risk
        # always resample, and it passed a literal True to the Rust variant kernel), so it
        # can only answer resample=True.
        and resample
    )

    if _can_use_numpy:
        # Pre-compute shared arrays
        price_ts = kf._index_to_ns(prices.index)
        pos_ts = kf._index_to_ns(signal.index)
        # Returns/PnL use gap-aware (ffilled) prices; realization uses RAW prices
        # so the position is only realized on bars where the market is open.
        price_diff_vals = _prices_ff.values.copy()
        price_diff_vals[1:] = price_diff_vals[1:] - price_diff_vals[:-1]
        price_diff_vals[0] = np.nan
        price_vals_flat = np.ascontiguousarray(prices.values.ravel(order='F'), dtype='float64')
        price_vals_ff_flat = np.ascontiguousarray(_prices_ff.values.ravel(order='F'), dtype='float64')
        day_keys = price_ts // 86_400_000_000_000
        unique_days = np.unique(day_keys)

        spreads_vals = None
        if _grid is not None:
            spreads_vals = None if _grid.spreads is None else _grid.spreads.reindex(columns=prices.columns).values
        elif spreads is not None:
            spreads_vals = _align_spreads(spreads, prices).values

        _vol = _vol_in if _vol_in is not None else _estimate_asset_vol(prices)
        # Columns first, then the as-of fill on the index: a single reindex with method='ffill'
        # applies the fill to BOTH axes, which raises on unsorted columns and would fill a missing
        # column from its neighbour.
        _asset_vol = _vol.reindex(columns=prices.columns).reindex(index=prices.index, method='ffill').values
        kf_d, pos_vals, pnl_sum_vals, _ = _one_pager_numpy_fast(
            signal.values, pos_ts, price_ts, price_vals_flat, price_vals_ff_flat,
            price_diff_vals, spreads_vals, shift,
            len(signal), len(prices), len(prices.columns),
            day_keys, unique_days, set(kfs), asset_vol=_asset_vol,
            stop_break_threshold=_sb_threshold, weights=_weights,
            px_v=None if _grid is None else _grid.px_v.reindex(columns=prices.columns).values
        )
        if _grid is not None and 'freq prices' in kf_d:
            kf_d['freq prices'] = _grid.freq_seconds

        # Wrap numpy results in DataFrames
        result = {}
        if 'kf' in return_subset:
            result['kf'] = kf_d
        if 'pos' in return_subset:
            result['pos'] = pd.DataFrame(pos_vals, index=prices.index, columns=prices.columns)
        if 'pnl_sum' in return_subset:
            result['pnl_sum'] = pd.Series(pnl_sum_vals, index=prices.index).dropna()

        if save_memory:
            return LazyDiskDict(result, monitor_dask=kwargs.get('monitor_dask', False))
        return result

    # Full pandas path
    # Create realizeable position based on what prices are available
    _bt = kf._backtest_rust if kf._HAS_RUST else kf._backtest_python
    pos = _bt(signal, prices, shift, 'pos')

    # --- Smart execution: delay trades until spread/price conditions are favorable ---
    if smart_execution:
        from ebt.simulate.execution import smart_execute
        if spreads is None:
            raise ValueError("smart_execution requires spreads to be set")
        _se_kwargs = smart_execution if isinstance(smart_execution, dict) else {}
        pos = smart_execute(pos, prices, spreads, **_se_kwargs)

    # --- Risk stop-break: suppress trades below threshold daily stds ---
    # Compute vol early (reused later for _pos_risk)
    _vol = _vol_in if _vol_in is not None else _estimate_asset_vol(prices)
    if not _vol.index.equals(prices.index) or not _vol.columns.equals(pos.columns):
        _vol = _vol.reindex(columns=pos.columns)
        _vol = _vol.reindex(index=prices.index, method='ffill')

    if _sb_threshold is not None:
        from ebt.simulate.execution import _risk_stop_break_1d
        for j in range(pos.shape[1]):
            pos.iloc[:, j] = _risk_stop_break_1d(
                pos.values[:, j].copy(),
                _vol.values[:, j].copy(),
                _sb_threshold,
            )

    pos_ffilled = pos.ffill()

    # Pre-align spreads (lightweight — just index alignment, no large DataFrame allocation).
    # A reduced grid carries spreads already aligned and imputed on the FULL grid.
    _spreads_aligned = None
    if _grid is not None:
        _spreads_aligned = None if _grid.spreads is None else _grid.spreads.reindex(columns=pos.columns)
    elif spreads is not None:
        _spreads_aligned = _align_spreads(spreads, pos)

    # Determine if we need the full 2D pnl DataFrame.
    # Only hitrate needs it; all other KFs work with pnl_sum (1D).
    _kfs_needing_2d_pnl = {'hitrate'}
    _need_2d_pnl = (
        plot
        or return_subset is None
        or (return_subset is not None and {'pnl', 'pnl_with_costs'} & return_subset)
        or kfs is None
        or bool(_kfs_needing_2d_pnl & set(kfs))
    )

    # Gap-aware returns, computed ONCE and shared by the 2D pnl, the cost loop and
    # the risk-normalized returns (each a full S-sized frame). The last use is
    # unconditional, so eager evaluation
    # holds one frame instead of allocating up to three transients.
    _prices_diff = _prices_ff.diff()

    if _need_2d_pnl:
        # Full 2D pnl — S persistent. Gap-aware returns (_prices_ff) so held
        # positions earn the reopen gap; positions stay realized on RAW prices.
        pnl = (pos_ffilled * _prices_diff).dropna(how='all')
        pnl_sum = pnl.sum(axis=1)
    else:
        # Compute pnl_sum directly — avoids S-sized 2D DataFrame.
        # Column-wise dot product: peak 2S (ffill + diff), 0 persistent.
        _pos_ff = pos_ffilled.values
        _pdiff = _prices_diff.values
        _pnl_vals = np.zeros(len(pos))
        _any_valid = np.zeros(len(pos), dtype=bool)
        for j in range(pos.shape[1]):
            _col = _pos_ff[:, j] * _pdiff[:, j]
            valid = ~np.isnan(_col)
            _any_valid |= valid
            _pnl_vals[valid] += _col[valid]
        del _pos_ff, _pdiff
        _pnl_vals[~_any_valid] = np.nan
        pnl_sum = pd.Series(_pnl_vals, index=pos.index).dropna()
        pnl = None

    # --- Pre-compute costs (single source of truth) ---
    # 1D costs_sum: always computed via numpy column loop (cheap, no 2D intermediate)
    # Must mask costs where pnl is NaN (price gap) to stay consistent with the 2D
    # pandas path where NaN_pnl - cost = NaN → excluded by sum(skipna=True).
    _costs_sum = None
    _pnl_sum_wc = pnl_sum  # PnL after costs (1D) — defaults to no-cost
    if _spreads_aligned is not None:
        _pos_vals = pos_ffilled.values
        _pos_notna = pos.notna().values
        _spreads_vals = _spreads_aligned.values
        _pdiff_vals = _prices_diff.reindex(index=pos.index, columns=pos.columns).values
        _costs_arr = np.zeros(len(pos))
        for j in range(pos.shape[1]):
            _diff_j = np.empty(len(pos))
            _diff_j[0] = np.nan
            _diff_j[1:] = _pos_vals[1:, j] - _pos_vals[:-1, j]
            _diff_j[~_pos_notna[:, j]] = np.nan
            _cost_j = np.abs(_diff_j) * np.abs(_spreads_vals[:, j]) * 0.5
            # Only include costs where PnL is also valid (price exists)
            _pnl_valid_j = ~np.isnan(_pos_vals[:, j] * _pdiff_vals[:, j])
            valid = ~np.isnan(_cost_j) & _pnl_valid_j
            _costs_arr[valid] += _cost_j[valid]
        _costs_sum = pd.Series(_costs_arr, index=pos.index).reindex(index=pnl_sum.index).fillna(0)
        _pnl_sum_wc = pnl_sum - _costs_sum
        del _pos_vals, _pos_notna, _spreads_vals, _pdiff_vals, _costs_arr

    # 2D trades/costs: lazily computed for plot and return dict
    _trades_cache = None
    _costs_cache = None
    def _get_trades_and_costs():
        nonlocal _trades_cache, _costs_cache
        if _trades_cache is None and _spreads_aligned is not None:
            _trades_cache = pos_ffilled.diff().where(pos.notna())
            _costs_cache = (_trades_cache.abs() * _spreads_aligned.abs() * 0.5).dropna(how='all', axis=1)
        return _trades_cache, _costs_cache

    def _get_pnl_with_costs():
        """2D pnl after costs — for equity curve and return dict."""
        _, costs = _get_trades_and_costs()
        if costs is not None and pnl is not None:
            return pnl - costs.reindex_like(pnl).fillna(0)
        return pnl

    # Pre-compute variant sharpes — single Rust call computes all 4 variants
    _variant_kfs = {'sharpe exbias', 'sharpe bias', 'sharpe long', 'sharpe short'}
    _active_variants = _variant_kfs if kfs is None else _variant_kfs & set(kfs)
    _variant_sharpes = {}
    if _active_variants:
        # Demean in RISK units: pos*vol against a vol-normalised price (see _risk_space_inputs).
        # No measured vol -> no risk unit: NaN, never a 1.0 stand-in.
        # _vol is already as-of on the price index; no value fill across a NaN-vol day (a stale
        # vol would score rows the reduced grid and the fast path leave out).
        _vv = _vol.reindex(index=_prices_ff.index, columns=pos.columns)
        _vv = _vv.where(np.isfinite(_vv) & (_vv > 0))
        _pos_v = pos * _vv
        if _grid is not None:
            # built on the FULL grid and sampled: a vol step between two kept rows is honoured
            _px_v = _grid.px_v.reindex(columns=pos.columns)
        else:
            _px_v = (_prices_ff.diff() / _vv).fillna(0.0).cumsum().where(_prices_ff.notna() & _vv.notna())
    if _active_variants and kf._HAS_RUST:
        # pos is already realized (backtest applied shift), so pass shift=0.
        # Gap-aware returns (_prices_ff) for consistency with the headline PnL.
        _variant_sharpes = kf._variant_sharpes_rust(_pos_v, _px_v, 0, resample)
    elif _active_variants:
        # Python fallback: compute individually
        # pos is already realized, so shift=0
        # dropna(how='all') before summing: pandas turns an all-NaN row into 0.0, which
        # injects fake zero-PnL observations at every bar the book was not yet trading and
        # deflates the Sharpe. The Rust path filters those rows out and one_pager's own
        # headline does the same (see the pnl dropna above), so without this the fallback
        # would disagree with both — on all four variants, not just the legs.
        # The column mean is taken over the HELD position (ffill), not the update rows: it is
        # the book's average exposure, so a value held for a week must weigh more than one
        # held for a bar. This also matches the Rust kernel, which averages its merge_asof
        # series; `x.mean()` would weight every update equally and make the two paths
        # disagree on a sparse book.
        _col_mean = _pos_v.ffill().mean()
        pos_exbias = _pos_v.pipe(lambda x: x - _col_mean)
        pos_onlybias = _pos_v.pipe(lambda x: x + _col_mean - x)
        # The long/short legs are the book's two sides, FLAT when the book is not on that
        # side — pos.clip, not pos.mask. mask() writes NaN, and a NaN row means "no new
        # observation, carry the previous position" to the backtest, so masking would hold
        # the last long through every short episode instead of standing the leg down.
        # With clip the legs decompose the book exactly.
        pos_short = _pos_v.clip(upper=0)
        pos_long = _pos_v.clip(lower=0)
        def _variant_sharpe(p):
            pnl = _bt(p, _px_v, 0, 'pnl').dropna(how='all').sum(axis=1)
            if not resample:
                # kf.sharpe only drops flat periods when it resamples. The legs ARE flat
                # whenever the book is not on their side, and counting those zeros drags
                # both moments toward zero, so drop them here to keep the two resample modes
                # measuring the same thing. resample=False means the input is
                # already daily, so a row is a day.
                pnl = pnl.replace(0, np.nan).dropna()
            return float(kf.sharpe(pnl, resample=resample))

        if 'sharpe exbias' in _active_variants:
            _variant_sharpes['exbias'] = _variant_sharpe(pos_exbias)
        if 'sharpe bias' in _active_variants:
            _variant_sharpes['bias'] = _variant_sharpe(pos_onlybias)
        if 'sharpe long' in _active_variants:
            _variant_sharpes['long'] = _variant_sharpe(pos_long)
        if 'sharpe short' in _active_variants:
            _variant_sharpes['short'] = _variant_sharpe(pos_short)
    if _active_variants and _grid is not None and (_active_variants & {'sharpe exbias', 'sharpe bias'}):
        # reduced grid: the column mean must count each kept row for the bars it stands for
        _dk = kf._index_to_ns(pos.index) // 86_400_000_000_000
        _exb, _bias = kf._numpy_bias_sharpes_weighted(
            pos.values, _vol.reindex(index=pos.index, columns=pos.columns).values, _px_v.values,
            _weights, _dk, np.unique(_dk))
        _variant_sharpes['exbias'], _variant_sharpes['bias'] = _exb, _bias

    # Risk-normalized positions and returns (_vol already computed above)
    # Use pos_ffilled (the actual held position) — pos may be sparse after stop_break
    _pos_risk = pos_ffilled * _vol
    _returns_risk = _prices_diff / _vol

    # Cache turnover computation (holding(days) = 1 / mean daily turnover, the same
    # daily series holding_time builds — computed once instead of twice)
    _turnover_mean = None
    def _get_turnover():
        nonlocal _turnover_mean
        if _turnover_mean is None:
            _turnover_mean = kf.turnover(_pos_risk, weights=_weights).mean()
        return _turnover_mean

    def _get_holding():
        m = _get_turnover()
        return 1.0 / m if m > 0 and not np.isnan(m) else np.inf

    # Cache risk computation
    _risk_val = None
    def _get_risk():
        nonlocal _risk_val
        if _risk_val is None:
            _risk_val = float(kf.risk(pnl_sum, resample=resample))
        return _risk_val

    def _compute_sharpe_with_costs():
        return float(kf.sharpe(_pnl_sum_wc, resample=resample))

    # Cache holding curve (used by sharpe_decay kf and the plot)
    # pos is already the realized position (backtest applied the shift), so pass shift=0
    _holding_curve = None
    def _get_holding_curve():
        nonlocal _holding_curve
        if _holding_curve is None:
            _holding_curve = _compute_holding_curve(pos, prices, 0, spreads, resample=resample, volatility=_vol)
        return _holding_curve

    _stop_break_curve = None
    def _get_stop_break_curve():
        nonlocal _stop_break_curve
        if _stop_break_curve is None:
            hc = _get_holding_curve()
            _max_h = hc.index[np.isfinite(hc.index)].max() if len(hc) > 0 else None
            _stop_break_curve = _compute_stop_break_curve(pos, prices, spreads, resample=resample, volatility=_vol, max_holding=_max_h)
        return _stop_break_curve

    def _sharpe_decay(with_costs=False):
        hc = _get_holding_curve()
        col = 'sharpe w costs' if with_costs else 'sharpe'
        vals = hc[col].dropna()
        return kf.sharpe_decay(vals.values, vals.index)

    # Cache decile risk shares
    _decile_risk_shares = None
    def _get_decile_risk():
        nonlocal _decile_risk_shares
        if _decile_risk_shares is None:
            _decile_risk_shares = _decile_risk(_pos_risk, _returns_risk, n=10)
        return _decile_risk_shares

    kf_dict = {
        'sharpe': lambda: float(kf.sharpe(pnl_sum, resample=resample)),
        'sharpe w costs': lambda: _compute_sharpe_with_costs(),
        'sharpe exbias': lambda: _variant_sharpes.get('exbias', np.nan),
        'sharpe bias': lambda: _variant_sharpes.get('bias', np.nan),
        'sharpe long': lambda: _variant_sharpes.get('long', np.nan),
        'sharpe short': lambda: _variant_sharpes.get('short', np.nan),
        'turnover': lambda: _get_turnover(),
        'holding(days)': lambda: _get_holding(),
        'lptp': lambda: kf.lptp(_pos_risk, weights=_weights),
        'hitrate': lambda: _hitrate(pnl.values),  # only called when _need_2d_pnl
        'up_days': lambda: kf.up_days(pnl if pnl is not None else pnl_sum),
        'ncvar': lambda: kf.ncvar(pos, prices),
        'skew': lambda: skew(pnl_sum.resample('D').sum().dropna(), nan_policy='omit'),
        'kurtosis': lambda: kurtosis(pnl_sum.resample('D').sum().dropna(), nan_policy='omit'),
        'risk_conc': lambda: float((np.sum(_get_decile_risk() ** 2) - 0.1) / 0.9) if not np.any(np.isnan(_get_decile_risk())) else np.nan,
        'tail_risk': lambda: float(_get_decile_risk()[:2].sum() + _get_decile_risk()[-2:].sum()) if not np.any(np.isnan(_get_decile_risk())) else np.nan,
        'risk': lambda: _get_risk(),
        'turnover(risk1)': lambda: _get_turnover() / _get_risk() if _get_risk() > 0 else np.nan,
        'freq signal': lambda: float(np.median(np.diff(signal.index.values).astype('timedelta64[s]').astype(float))) if len(signal.index) > 1 else np.nan,
        'freq prices': lambda: _grid.freq_seconds if _grid is not None else (float(np.median(np.diff(prices.index.values).astype('timedelta64[s]').astype(float))) if len(prices.index) > 1 else np.nan),
        'freq pnl': lambda: float(np.median(np.diff(pnl_sum.index.values).astype('timedelta64[s]').astype(float))) if len(pnl_sum.index) > 1 else np.nan,
        'sharpe_decay': lambda: _sharpe_decay(with_costs=False),
        'sharpe_decay_wc': lambda: _sharpe_decay(with_costs=True),
    }

    kf_dict = {k: v() for k, v in kf_dict.items()
               if (kfs is None or k in kfs) and not (_grid is not None and k in GRID_INCOMPATIBLE_KFS)}

    _risk_check = kf_dict.get('risk', _get_risk())
    if not np.isnan(_risk_check) and not (0.8 <= _risk_check <= 1.2):
        _LOGGER.warning(f'Annualized risk is {_risk_check:.3f}, outside [0.8, 1.2]. Consider risk-adjusting positions.')

    kf_series = pd.Series(kf_dict).rename('value').rename_axis('kf')

    fig = None
    axes = None
    if plot:
        pnl_with_costs = _get_pnl_with_costs()

        import matplotlib.gridspec as gridspec

        figsize = (14, 7) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)

        # Single master GridSpec: 2 rows x 3 columns
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[3, 2], hspace=0.25)

        # The lag / trend / holding / stop-break panels count price BARS: not on a reduced grid
        _bar_panels = not fast and _grid is None
        sharpe_per_lag = kf.sharpe_per_lag(pos, prices) if _bar_panels else pd.Series()
        # Batch sharpe per column: resample entire DataFrame once instead of 30 separate resample calls.
        # `.replace(0, NaN)` would treat the gross and with-costs series asymmetrically: for a sparse-
        # position market, gross PnL is exactly 0 on no-trade days (dropped) while with-costs PnL is
        # 0 - cost ≠ 0 whenever any rebalance happens (kept). The two Sharpes then aggregate over
        # different day sets and the with-costs Sharpe can spuriously exceed the gross one. Use
        # min_count=1 + dropna so only fully-empty days (no intraday data) are NaN'd, and the two
        # metrics aggregate over the same days.
        if not fast and resample:
            _pnl_daily    = pnl.resample('D').sum(min_count=1)
            _pnl_wc_daily = pnl_with_costs.resample('D').sum(min_count=1)
            # Use a common day-mask so the two Sharpes are computed over the same set of days.
            _mask = _pnl_daily.notna().any(axis=1) & _pnl_wc_daily.notna().any(axis=1)
            _pnl_daily    = _pnl_daily[_mask]
            _pnl_wc_daily = _pnl_wc_daily[_mask]
            sharpe_per_market          = _pnl_daily.mean()    / _pnl_daily.std()    * 252 ** 0.5
            sharpe_per_market_withcosts = _pnl_wc_daily.mean() / _pnl_wc_daily.std() * 252 ** 0.5
        elif not fast:
            sharpe_per_market = pnl.apply(kf.sharpe, resample=False)
            sharpe_per_market_withcosts = pnl_with_costs.apply(kf.sharpe, resample=False)
        else:
            sharpe_per_market = pd.Series()
            sharpe_per_market_withcosts = pd.Series()
        correlation_to_trend = kf.correlation_to_trend(pos, prices, filter_lengths=None) if _bar_panels else pd.Series()

        sharpe_per_holding = _get_holding_curve() if _bar_panels else None
        sharpe_per_sb = _get_stop_break_curve() if _bar_panels else None

        # --- Bottom row: equity curve [1,0], KF table [1,1], sharpe per market [1,2] ---
        ax_equity = fig.add_subplot(gs[1, 0])
        ax_table = fig.add_subplot(gs[1, 1])
        ax_market = fig.add_subplot(gs[1, 2])

        plot_kf_table(kf_series, ax_table)

        # Config box below KF table
        _config_parts = []
        if _is_returns:
            _input_type = 'returns'
        elif _prices_are_vwap:
            _input_type = 'vwap prices'
        else:
            _input_type = 'prices'
        _config_parts.append(f'input: {_input_type}')
        _config_parts.append(f'shift: {shift_label}')
        _exec_parts = []
        if volume is not None:
            _exec_parts.append(f'VWAP {vwap_window}')
        if smart_execution:
            _se_label = 'smart'
            if isinstance(smart_execution, dict) and 'spread_threshold' in smart_execution:
                _se_label += f'(th={smart_execution["spread_threshold"]})'
            _exec_parts.append(_se_label)
        if _sb_threshold is not None:
            _exec_parts.append(f'sb={_sb_threshold}')
        if _exec_parts:
            _config_parts.append(f'exec: {" + ".join(_exec_parts)}')
        else:
            _config_parts.append('exec: NO VWAP')
        if _spreads_aligned is not None:
            _config_parts.append('costs: yes')
        if not resample:
            _config_parts.append('resample: no')
        if _vol_in is not None:
            _config_parts.append('vol: provided')
        else:
            _config_parts.append('vol: EMA(60) daily')
        if _grid is not None:
            _config_parts.append(f'grid: reduced {len(_grid)}/{_grid.n_full} rows')
        _config_text = ' | '.join(_config_parts)
        # Risk normalization details on second line
        _risk_parts = [f'risk-norm: turnover, holding, lptp, decile, quantile']
        _risk_text = ' | '.join(_risk_parts)
        _full_text = _config_text + '\n' + _risk_text
        ax_table.text(0.5, -0.02, _full_text, transform=ax_table.transAxes,
                      ha='center', va='top', fontsize=7, color='#666666',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                                edgecolor='#cccccc', alpha=0.8))
        if volume is None:
            ax_table.text(0.5, -0.18, 'NO VWAP', transform=ax_table.transAxes,
                          ha='center', va='top', fontsize=9, fontweight='bold',
                          color='red', alpha=0.9)

        plot_equity_curve(pnl, ax_equity, color="skyblue", label="gross")
        if _spreads_aligned is not None and pnl_with_costs is not None:
            plot_equity_curve(pnl_with_costs, ax_equity, color="lightcoral", label="net of costs")
            ax_equity.legend(fontsize=7, loc='upper left')
        plot_sharpe_bars(sharpe_per_market, ax_market, title='Sharpe', x_title=None,
                         max_bars=SHARPE_BARS_MAX_MARKETS)
        plot_sharpe_bars(sharpe_per_market_withcosts, ax_market, title='Sharpe', x_title=None,
                         color='lightcoral', max_bars=SHARPE_BARS_MAX_MARKETS)

        # --- Top-left [0,0]: quantile plot (scatter + risk bar) ---
        if _grid is not None:
            ax_q = fig.add_subplot(gs[0, 0])
            ax_q.text(0.5, 0.5, 'Quantile plot\n(not available on a reduced grid)', ha='center', va='center', fontsize=8, color='#999999', transform=ax_q.transAxes)
            ax_q.set_xticks([])
            ax_q.set_yticks([])
        elif not fast:
            _pos_flat = _pos_risk.values[np.isfinite(_pos_risk.values)]
            _can_bin = len(_pos_flat) > 0 and np.ptp(_pos_flat) > 0

            if _can_bin:
                sub_gs_q = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_q_top = fig.add_subplot(sub_gs_q[0])
                ax_q_bot = fig.add_subplot(sub_gs_q[1])
                plot_quantiles(_pos_risk, _returns_risk, ax=ax_q_top, ax_risk=ax_q_bot, n=10)
                ax_q_top.set_xticklabels([])
                ax_q_top.set_xlabel('')
            else:
                ax_q = fig.add_subplot(gs[0, 0])
                ax_q.text(0.5, 0.5, 'Constant position\n(no decile variation)', ha='center', va='center', fontsize=8, color='#999999', transform=ax_q.transAxes)
                ax_q.set_xticks([])
                ax_q.set_yticks([])

        # --- Top-middle [0,1]: sharpe per lag (top) + calendar Sharpe (bottom, 3 side by side) ---
        sub_gs_mid = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 2], hspace=0.45)
        ax_lag = fig.add_subplot(sub_gs_mid[0])
        _lag_series = (sharpe_per_lag.set_axis(sharpe_per_lag.index + shift)).set_axis((sharpe_per_lag.index + shift).astype(str)) if not sharpe_per_lag.empty else sharpe_per_lag
        plot_sharpe_bars(_lag_series, ax_lag, marker_fontsize=6, x_title=None, highlight=shift_label)

        sub_gs_cal = sub_gs_mid[1].subgridspec(1, 3, wspace=0.5)
        ax_weekday = fig.add_subplot(sub_gs_cal[0])
        ax_dom = fig.add_subplot(sub_gs_cal[1])
        ax_hour = fig.add_subplot(sub_gs_cal[2])

        _pnl_daily_cal = pnl_sum.resample('D').sum().replace(0, np.nan).dropna()
        _pnl_daily_cal.index.name = None
        _pnl_cal = pnl_sum.copy()
        _pnl_cal.index.name = None
        if len(_pnl_daily_cal) > 10:
            def _plot_calendar_dist(pnl_series, groupby_key, ax, ann_factor, labels=None, thin_every=None, y_title=None, title=None, neighbors=0, show_dots=True):
                """Per-year Sharpe distribution: median line + IQR shading + scatter.
                neighbors: pool ±N adjacent buckets for statistics."""
                years = pnl_series.index.year
                buckets = groupby_key(pnl_series.index)
                grouped = pnl_series.groupby([years, buckets])
                sharpe_per_year = (grouped.mean() / grouped.std() * ann_factor ** 0.5).dropna()

                keys = sorted(buckets.unique())
                x = np.arange(len(keys))
                medians = np.full(len(keys), np.nan)
                q25 = np.full(len(keys), np.nan)
                q75 = np.full(len(keys), np.nan)
                all_vals = []
                bucket_levels = sharpe_per_year.index.get_level_values(1)
                for i, k in enumerate(keys):
                    # Pool this bucket + neighbors
                    pool_keys = [keys[j] for j in range(max(0, i - neighbors), min(len(keys), i + neighbors + 1))]
                    vals = np.concatenate([
                        sharpe_per_year.xs(pk, level=1).values
                        for pk in pool_keys if pk in bucket_levels
                    ]) if any(pk in bucket_levels for pk in pool_keys) else np.array([])
                    vals = vals[np.isfinite(vals)]
                    all_vals.append(vals)
                    if len(vals) >= 1:
                        medians[i] = np.median(vals)
                        q25[i] = np.percentile(vals, 25)
                        q75[i] = np.percentile(vals, 75)

                valid = ~np.isnan(medians)
                ax.fill_between(x[valid], q25[valid], q75[valid],
                                alpha=0.2, color='steelblue', zorder=2)
                ax.plot(x[valid], medians[valid], '-', color='steelblue', linewidth=1, zorder=3)
                colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in medians[valid]]
                ax.scatter(x[valid], medians[valid], c=colors, s=20, zorder=6,
                           edgecolors='black', linewidths=0.4)

                # Per-year dots
                if show_dots:
                    rng = np.random.RandomState(42)
                    for i, vals in enumerate(all_vals):
                        if len(vals) > 0:
                            jitter = rng.uniform(-0.1, 0.1, len(vals))
                            ax.scatter(np.full(len(vals), x[i]) + jitter, vals,
                                       s=6, color='#555555', alpha=0.4, zorder=5, linewidths=0)

                ax.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.3)
                tick_labels = labels if labels is not None else [str(k) for k in keys]
                ax.set_xticks(x)
                ax.set_xticklabels(tick_labels)
                if thin_every is not None:
                    for i, label in enumerate(ax.get_xticklabels()):
                        label.set_visible(i % thin_every == 0)
                ax.tick_params(axis='x', rotation=0, labelsize=5, pad=1)
                if title:
                    ax.set_title(title, fontsize=7)
                _decorate_axes(ax, None, y_title)

            # Sharpe by weekday
            _day_labels = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            _wd_keys = sorted(_pnl_daily_cal.index.dayofweek.unique())
            _plot_calendar_dist(_pnl_daily_cal, lambda idx: idx.dayofweek, ax_weekday, ann_factor=252,
                                labels=[_day_labels[k] for k in _wd_keys],
                                y_title='Sharpe', title='Weekday', show_dots=False)

            # Sharpe by day of month
            _plot_calendar_dist(_pnl_daily_cal, lambda idx: idx.day, ax_dom, ann_factor=252,
                                thin_every=5, title='Day of Month', neighbors=1, show_dots=False)

            # Sharpe by hour of day (per price bar: not on a reduced grid)
            if _grid is not None:
                ax_hour.text(0.5, 0.5, 'Hour\n(reduced grid)', ha='center', va='center', fontsize=7, color='#999999', transform=ax_hour.transAxes)
                ax_hour.set_xticks([])
                ax_hour.set_yticks([])
            else:
                _n_hours = _pnl_cal.index.hour.nunique()
                _plot_calendar_dist(_pnl_cal, lambda idx: idx.hour, ax_hour, ann_factor=252 * max(_n_hours, 1),
                                    thin_every=4, title='Hour', neighbors=1, show_dots=False)
        else:
            for _ax in (ax_weekday, ax_dom, ax_hour):
                _ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=7, color='#999999', transform=_ax.transAxes)

        # --- Top-right [0,2]: trend correlation (top) + sharpe per holding (bottom) ---
        sub_gs_tr = gs[0, 2].subgridspec(2, 1)
        ax_top = fig.add_subplot(sub_gs_tr[0])
        ax_bot = fig.add_subplot(sub_gs_tr[1])

        plot_sharpe_bars(correlation_to_trend, ax=ax_top, x_title=None, y_title='Correlation')
        _decorate_axes(ax_top, None, 'Correlation')
        if _grid is not None:
            ax_top.text(0.5, 0.5, 'Trend correlation, lag and holding panels\n(not available on a reduced grid)',
                        ha='center', va='center', fontsize=7, color='#999999', transform=ax_top.transAxes)
        if sharpe_per_holding is not None:
            ax_bot.plot(sharpe_per_holding.index, sharpe_per_holding['sharpe'].values, color='C0', label='EWM')
            ax_bot.plot(sharpe_per_holding.index, sharpe_per_holding['sharpe w costs'].values, color='C1', label='EWM w costs')
            ax_bot.scatter(sharpe_per_holding.index, sharpe_per_holding['sharpe'].values, color='C0', s=12, zorder=5)
            ax_bot.scatter(sharpe_per_holding.index, sharpe_per_holding['sharpe w costs'].values, color='C1', s=12, zorder=5)
        if sharpe_per_sb is not None:
            ax_bot.plot(sharpe_per_sb.index, sharpe_per_sb['sharpe'].values, color='C2', linestyle='--', label='Stop-break')
            ax_bot.plot(sharpe_per_sb.index, sharpe_per_sb['sharpe w costs'].values, color='C3', linestyle='--', label='SB w costs')
            ax_bot.scatter(sharpe_per_sb.index, sharpe_per_sb['sharpe'].values, color='C2', s=12, zorder=5, marker='s')
            ax_bot.scatter(sharpe_per_sb.index, sharpe_per_sb['sharpe w costs'].values, color='C3', s=12, zorder=5, marker='s')
        if sharpe_per_holding is not None or sharpe_per_sb is not None:
            ax_bot.set_xscale('log')
            ax_bot.xaxis.set_major_formatter(ticker.FuncFormatter(_format_holding_with_turnover))
            # Combine indices for tick range
            _all_idx = []
            if sharpe_per_holding is not None:
                _all_idx.append(sharpe_per_holding.index)
            if sharpe_per_sb is not None:
                _all_idx.append(sharpe_per_sb.index)
            _hvals = np.concatenate(_all_idx)
            _hvals = _hvals[np.isfinite(_hvals)]
            if len(_hvals) >= 2:
                _hmin, _hmax = _hvals.min(), _hvals.max()
                _hticks = np.exp(np.linspace(np.log(_hmin), np.log(_hmax), min(5, len(_hvals))))
                ax_bot.xaxis.set_major_locator(ticker.FixedLocator(_hticks))
            ax_bot.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            ax_bot.grid(True, which='major', alpha=0.4)
            ax_bot.grid(True, which='minor', alpha=0.15)
        _decorate_axes(ax_bot, 'Holding / Turnover', 'Sharpe')
        if sharpe_per_holding is not None or sharpe_per_sb is not None:
            ax_bot.legend(fontsize=5, loc='lower right')

        fig.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.06)

        # axes variable for backward compatibility in return dict
        axes = None

        if not show_plot:
            plt.close(fig)

    # Build result dict — only compute what's requested
    if return_subset is not None:
        result = {}
        if 'kf' in return_subset:
            result['kf'] = kf_dict
        if 'pos' in return_subset:
            result['pos'] = pos
        if 'pnl' in return_subset:
            result['pnl'] = pnl
        if 'pnl_sum' in return_subset:
            result['pnl_sum'] = pnl_sum
        if 'fig' in return_subset:
            result['fig'] = fig
        if 'axes' in return_subset:
            result['axes'] = axes
        if {'trades', 'costs', 'pnl_with_costs'} & return_subset:
            _tr, _co = _get_trades_and_costs()
            if 'trades' in return_subset:
                result['trades'] = _tr
            if 'costs' in return_subset:
                result['costs'] = _co
            if 'pnl_with_costs' in return_subset:
                result['pnl_with_costs'] = _get_pnl_with_costs()
    else:
        # Full result dict — compute everything
        _tr, _co = _get_trades_and_costs()
        result = {'kf': kf_dict, 'pnl': pnl, 'pnl_sum': pnl_sum, 'pnl_with_costs': _get_pnl_with_costs(), 'pos': pos, 'trades': _tr, 'costs': _co, 'fig': fig, 'axes': axes}
    if _grid is not None and (return_subset is None or 'reduced_grid' in return_subset):
        result['reduced_grid'] = _grid

    if save_memory:
        return LazyDiskDict(result, monitor_dask=kwargs.get('monitor_dask', False))

    return result


def _natural_sort_key(path):
    """Sort key comparing embedded integers numerically ('2.pq' < '10.pq').

    Signal files are paired with configs_df rows BY POSITION, so directory order
    is load-bearing: plain lexicographic sort puts '10.parquet' before
    '2.parquet' and silently attributes every config to the wrong signal.
    """
    import re
    parts = re.split(r'(\d+)', os.path.basename(str(path)))
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _restore_column_labels(names, pandas_metadata):
    """Arrow field names -> the labels ``pd.read_parquet`` would produce.

    pandas records the column INDEX dtype in ``column_indexes``; anything that
    is not text was stringified to become an Arrow field name and is cast back
    on read. Reproduce that cast so schema-only inspection agrees with an
    actual load. Unknown / text / mixed index dtypes are left as strings, which
    is what pandas does too.
    """
    idx_meta = (pandas_metadata or {}).get('column_indexes') or []
    if len(idx_meta) != 1:            # no metadata, or a MultiIndex on columns
        return list(names)
    numpy_type = idx_meta[0].get('numpy_type')
    if numpy_type in (None, 'object', 'str', 'unicode'):
        return list(names)
    try:
        return list(pd.Index(list(names)).astype(numpy_type))
    except (TypeError, ValueError):
        return list(names)


def _apply_eval_window(frame, start=None, end=None):
    """Restrict a position frame to the evaluation window ``start <= index < end``.

    A book is built (combined, risk-adjusted) on its FULL history so its scale
    estimate is warm, then scored on a common window so every book compared
    against it covers the same days. Rows outside the window are DROPPED, not
    zeroed, so no pre-window stamp reaches turnover or holding.

    Dropping the rows at/after ``end`` is not enough on its own: NaN in a position
    frame means "carry", and kf.backtest joins positions to prices AS-OF, so the
    last in-window position would be held to the end of the price history. A 0.0
    row stamped ``end`` closes every column — the book is flat from the first fill
    at-or-after end + shift. The row keeps the frame's dtype (float32 books stay
    float32). Either bound may be None; both None returns ``frame`` itself.

    Start semantics — "first in-window decision onward": the first position scored is
    the one decided at the first stamp >= ``start``, filled at that stamp + shift. The
    position carried INTO the window from the last pre-window stamp is not kept, so the
    PnL between the start and that first fill is not scored (a daily book stamped at the
    close under shift '2m' scores from the open after its first in-window close). At the
    end, the last in-window position is held until the flat row's fill at end + shift.
    """
    if start is None and end is None:
        return frame
    idx = frame.index

    def _stamp(x):
        # A naive bound against a tz-aware index would raise on comparison; read it
        # as wall time in the index's zone.
        t = pd.Timestamp(x)
        tz = getattr(idx, 'tz', None)
        return t.tz_localize(tz) if (tz is not None and t.tz is None) else t

    keep = np.ones(len(idx), dtype=bool)
    if start is not None:
        keep &= np.asarray(idx >= _stamp(start))
    if end is not None:
        keep &= np.asarray(idx < _stamp(end))
    out = frame.loc[keep]
    if end is not None:
        try:
            dtype = np.result_type(*frame.dtypes) if frame.shape[1] else np.float64
        except TypeError:
            dtype = np.float64
        close = pd.DataFrame(np.zeros((1, frame.shape[1]), dtype=dtype),
                             index=pd.DatetimeIndex([_stamp(end)], name=idx.name),
                             columns=frame.columns)
        out = pd.concat([out, close])
    return out


_EVAL_WINDOW_PRICE_SESSIONS = 5


def _trim_prices_before_window(prices, start, sessions=_EVAL_WINDOW_PRICE_SESSIONS):
    """Drop price rows before the ``sessions``-th distinct date before ``start``.

    Every windowed signal is NaN before ``start``, so the per-config pass of plot_ensemble
    only needs prices from the last print before its first fill. The cut keeps whole
    dates: rows on or after the ``sessions``-th distinct calendar date (of the price index)
    that is earlier than ``start``'s date. Fewer earlier dates, or a tz-aware ``start``
    against a naive index, return ``prices`` unchanged. The caller trims only when a
    ``volatility`` frame is given (the default vol estimate warms up on these rows) and
    no ``spreads`` (their imputation reads monthly medians of the aligned grid).
    """
    idx = prices.index
    t = pd.Timestamp(start)
    tz = getattr(idx, 'tz', None)
    if tz is not None:
        t = t.tz_localize(tz) if t.tz is None else t.tz_convert(tz)
    elif t.tz is not None:
        return prices
    days = idx[idx < t.normalize()].normalize().unique().sort_values()
    if len(days) < sessions:
        return prices
    return prices.loc[days[-sessions]:]


class _PositionSource:
    """Uniform, lazily-loaded access to an ensemble's positions.

    ``plot_ensemble`` accepts positions as any of:
      * a list/tuple of in-RAM DataFrames (the classic case), or
      * a directory containing per-signal ``*.parquet`` files, or
      * an explicit list of ``*.parquet`` paths.

    The disk forms let a sweep of thousands of intraday positions live on
    persistent storage and be streamed one-at-a-time, so plot_ensemble's peak
    memory stays bounded by ``n_workers`` (not by the number of signals).

    ORDER CONTRACT: signal i is paired with configs_df row i. Directory files
    are ordered by natural sort (embedded integers compared numerically), so
    both '0.parquet'..'11.parquet' and zero-padded names order correctly; an
    explicit path list is used in the given order. The files must be written
    by pandas ``to_parquet`` with the DatetimeIndex preserved.
    """

    def __init__(self, positions):
        self._frames = None
        self._paths = None
        if isinstance(positions, (str, os.PathLike)):
            import glob
            d = str(positions)
            self._paths = sorted(glob.glob(os.path.join(d, '*.parquet')),
                                 key=_natural_sort_key)
            if not self._paths:
                raise ValueError(f"_PositionSource: no .parquet files found in {d!r}")
        elif (isinstance(positions, (list, tuple)) and len(positions) > 0
              and isinstance(positions[0], (str, os.PathLike))):
            self._paths = [str(p) for p in positions]
        else:
            self._frames = list(positions)

    @property
    def on_disk(self):
        return self._paths is not None

    def __len__(self):
        return len(self._paths) if self._paths is not None else len(self._frames)

    def load(self, i, window=None):
        """Signal i. ``window=(start, end)`` returns it restricted to the evaluation
        window (see _apply_eval_window); None returns the frame as stored."""
        if self._paths is not None:
            df = pd.read_parquet(self._paths[i])
            if not isinstance(df.index, pd.DatetimeIndex):
                # A file written without pandas index metadata (polars, pyarrow
                # preserve_index=False) comes back with a RangeIndex; downstream
                # that turns into 1970-era timestamps and silent all-NaN KFs.
                raise ValueError(
                    f"_PositionSource: {self._paths[i]!r} loaded with "
                    f"{type(df.index).__name__} instead of DatetimeIndex — signal "
                    f"parquet files must be written by pandas to_parquet with the "
                    f"DatetimeIndex preserved.")
        else:
            df = self._frames[i]
        return df if window is None else _apply_eval_window(df, *window)

    def columns_union(self):
        """Union of data (asset) columns across all signals, without loading the
        full frames when they live on disk (reads parquet schemas only).

        The labels must come back in the SAME dtype ``load()`` produces. Arrow
        field names are always strings, but pandas restores the original column
        dtype from ``column_indexes`` on read — so a book whose assets are
        integer ids round-trips as ``10`` via
        ``load()`` and as ``'10'`` via ``schema.names``. plot_ensemble
        intersects this union with ``prices.columns``; with the two dtypes
        mismatched the intersection is EMPTY, every sub-portfolio becomes a
        0-column frame, and ``one_pager`` raises ValueError on it from
        ``_compute_kfs_direct``, several frames deep and with no hint of the cause.
        """
        cols = set()
        if self._paths is not None:
            import pyarrow.parquet as pq
            for p in self._paths:
                schema = pq.ParquetFile(p).schema_arrow
                md = schema.pandas_metadata
                index_cols = set()
                if md:
                    for c in md.get('index_columns', []):
                        if isinstance(c, str):
                            index_cols.add(c)
                names = [n for n in schema.names if n not in index_cols]
                cols.update(_restore_column_labels(names, md))
        else:
            for f in self._frames:
                cols.update(f.columns)
        try:
            return pd.Index(sorted(cols))
        except TypeError:      # mixed label types — order is not meaningful
            return pd.Index(list(cols))

    def index_union(self):
        """Union of every signal's stamps as a DatetimeIndex (tz preserved). On-disk sources
        read only the index column (``read_parquet(path, columns=[])``); the reduced price grid
        (ebt.simulate.grid) is built from this union. Mixed tz-aware / naive sources raise."""
        idx = None
        for i in range(len(self)):
            if self._paths is not None:
                cur = pd.read_parquet(self._paths[i], columns=[]).index
            else:
                cur = self._frames[i].index
            cur = pd.DatetimeIndex(cur)
            if idx is None:
                idx = cur
            else:
                if (idx.tz is None) != (cur.tz is None):
                    raise ValueError('_PositionSource.index_union: mixed tz-aware and tz-naive signal indexes')
                idx = idx.union(cur)
        return idx if idx is not None else pd.DatetimeIndex([])


def _cgroup_available_bytes():
    """Headroom under the current cgroup memory limit (None if unlimited/unknown).

    Inside a memory-limited container /proc/meminfo reports the
    HOST, but the cgroup limit is what OOM-kills the process — so it must bind.
    """
    try:  # cgroup v2
        with open('/sys/fs/cgroup/memory.max') as f:
            limit = f.read().strip()
        if limit != 'max':
            with open('/sys/fs/cgroup/memory.current') as f:
                used = int(f.read().strip())
            return max(0, int(limit) - used)
    except Exception:
        pass
    try:  # cgroup v1
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
            limit = int(f.read().strip())
        if limit < 1 << 60:  # no-limit sentinel is ~2^63
            with open('/sys/fs/cgroup/memory/memory.usage_in_bytes') as f:
                used = int(f.read().strip())
            return max(0, limit - used)
    except Exception:
        pass
    return None


def _available_bytes():
    """Best-effort available RAM in bytes (None if undetectable).

    min(host MemAvailable, cgroup memory headroom) — whichever binds first.
    """
    host = None
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    host = int(line.split()[1]) * 1024
                    break
    except Exception:
        pass
    if host is None:
        try:
            import psutil
            host = psutil.virtual_memory().available
        except Exception:
            pass
    if host is None:
        try:
            host = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
        except Exception:
            pass
    cg = _cgroup_available_bytes()
    if host is not None and cg is not None:
        return min(host, cg)
    return host if host is not None else cg


def _effective_cpu_count():
    """CPUs actually usable: os.cpu_count() capped by the scheduler affinity mask
    and the cgroup CPU quota — os.cpu_count() reports the HOST inside a
    quota-limited container."""
    n = os.cpu_count() or 8
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:  # cgroup v2
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()
        if quota != 'max':
            n = min(n, max(1, int(int(quota) / int(period))))
    except Exception:
        try:  # cgroup v1
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
                quota = int(f.read().strip())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
                period = int(f.read().strip())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except Exception:
            pass
    return max(1, int(n))


def _auto_n_workers(n_signals, n_rows, n_cols, requested='auto',
                    mem_fraction=0.5, throughput_cap=32, grids_per_worker=8,
                    fixed_grids=0):
    """Pick a thread count for the per-signal pass, from the data + machine.

    The per-signal work is GIL-releasing (Rust backtest + numpy), so threads help
    only until the GIL-bound Python glue saturates, beyond which more workers add
    nothing. Each
    concurrent worker holds ~``grids_per_worker`` full-grid temporaries (numpy
    temporaries plus the Rust backtest internals), so on large grids / small nodes
    memory binds first. We take the min of:
      * n_signals            (no point having more threads than signals),
      * throughput_cap (32)  (GIL plateau — more threads just cost RAM),
      * effective cpu count  (affinity/cgroup-quota aware),
      * memory budget: (mem_fraction * available_RAM - fixed_grids * grid_bytes)
                       / (grids_per_worker * grid_bytes).

    ``fixed_grids`` counts full-grid allocations that exist regardless of the
    worker count (the group-sum buffers — one grid per (variable, group) — and
    the price-grid copies), so the per-worker budget is taken from what is
    actually left. Available RAM is cgroup-aware (see _available_bytes).

    An explicit integer ``requested`` is honored (capped at n_signals); pass 'auto'
    (the default) to let this decide.
    """
    n_signals = max(1, int(n_signals))
    if requested is not None and requested != 'auto':
        return max(1, min(int(requested), n_signals))
    caps = [n_signals, int(throughput_cap), _effective_cpu_count()]
    avail = _available_bytes()
    if avail and n_rows and n_cols:
        grid_bytes = int(n_rows) * int(n_cols) * 8
        per_worker = grids_per_worker * grid_bytes + 64 * 1024**2
        usable = mem_fraction * avail - int(fixed_grids) * grid_bytes
        caps.append(max(1, int(usable / per_worker)))
    return int(max(1, min(caps)))


def _fmt_float_tick(x, pos=None):
    """Picklable FuncFormatter callback: format a tick VALUE as a 3-dp float.

    Module-level (not a lambda) so figures carrying it stay picklable — required
    for save_memory=True / LazyDiskDict.
    """
    return f"{x:.3f}"


class _CategoricalTickFormatter:
    """Picklable FuncFormatter callback: map a tick POSITION to a formatted label.

    Replaces a closure over the box-plot group labels, so figures carrying this
    formatter can be pickled (needed for save_memory=True / LazyDiskDict).
    """
    def __init__(self, labels):
        self._labels = [float(l) for l in labels]

    def __call__(self, x, pos=None):
        if pos is None:
            return ''
        i = int(pos)
        if 0 <= i < len(self._labels):
            return f"{self._labels[i]:.3f}"
        return ''


def _assign_groups(values, categories, n_portfolios):
    """Partition signal ids into sub-portfolio groups from a config variable.

    Mirrors ``split_portfolios``' partitioning EXACTLY (same group keys, same
    order) but returns only the membership ``{group_key: [signal_ids]}`` — no
    positions needed. This lets plot_ensemble decide the grouping up front (from
    configs alone) and accumulate each group's combined position on the fly,
    instead of holding every signal's position to feed ``combine_positions``.
    """
    groups = {}
    if categories:
        for cat in values.value_counts().sort_index().index:
            groups[cat] = list(values[values == cat].index)
    else:
        try:
            pidx = pd.Series(pd.qcut(values, n_portfolios, labels=np.arange(n_portfolios)))
        except ValueError:
            pidx = pd.Series(pd.qcut(values, n_portfolios, duplicates='drop'))
        for ix in pidx.value_counts().sort_index().index:
            pos_ix = pidx[pidx == ix].index
            key = values.loc[pos_ix].mean()
            groups[key] = list(pos_ix)
    return groups


def plot_ensemble(positions,
                  prices_or_returns,
                  configs_df=None,
                  spreads=None,
                  kfs=None,
                  shift=2,
                  figsize=None,
                  n_cols=3,
                  n_portfolios=3,
                  max_unique_values_for_categorical=10,
                  n_draws=100,
                  n_workers='auto',
                  show_tabs=True,
                  add_correlation=True,
                  add_ncvar=True,
                  add_diversification=True,
                  save_memory=True,
                  debug=False,
                  volatility=None,
                  stop_break=1e-9,
                  eval_window=None,
                  reduce='auto'):
    '''
    Plots ensemble keyfigures

    :param reduce: 'auto' (default) scores every signal on ONE reduced price grid
        (ebt.simulate.grid) built from the union of all signals' stamps when they are much
        sparser than the prices; otherwise the full grid, with a logging.info line. True raises
        when not applicable, False never reduces. ``prices_or_returns`` may also be a prebuilt
        ReducedGrid. In reduced mode the default key figures drop 'sharpe_decay' (defined per
        price bar) and an explicit ``kfs`` naming a per-bar figure raises.

    :param positions: list of position DataFrames, OR a directory of per-signal
        ``*.parquet`` files, OR a list of parquet paths. Disk forms are streamed one
        signal at a time so peak memory stays bounded regardless of signal count.
        Signal i pairs with ``configs_df`` row i: directory files are ordered by
        natural sort (embedded integers compared numerically), path lists keep the
        given order, and the signal/config counts must match (ValueError otherwise).
    :param configs_df pd.DataFrame: pd.Dataframe of config values, config variable as columns and ensemble instances for rows
    :param returns pd.DataFrame: pd.DataFrame of market returns
    :param kfs list[str]: Which kfs to calculate, defaults to
        ['sharpe', 'sharpe exbias', 'sharpe bias', 'sharpe long', 'sharpe short', 'turnover', 'holding(days)', 'lptp', 'hitrate', 'ncvar']
    :param n_workers: threads for the per-signal pass. Default 'auto' picks from the
        data + machine (grid size, signal count, available RAM, CPUs); pass an int to
        force it. See _auto_n_workers.
    :param figsize tuple: Size of each variable section, defaults to (20, 3.5)
    :param eval_window: optional (start, end) tuple, either bound may be None. Every
        signal is scored on rows start <= index < end only, closed flat at ``end``
        (_apply_eval_window), so per-config and sub-portfolio figures cover exactly the
        rows of the combined book a caller scores. With an ``end`` the prices
        are cut at end + shift — the cut one_pager applies to a book whose last stamp is
        ``end`` — so the per-config figures equal one_pager on the windowed signal
        instead of also crediting the exit fill after the window. With a ``start``, a
        ``volatility`` frame and no ``spreads``, prices are also cut at the 5th distinct
        date before start (_trim_prices_before_window): the rows before it cannot reach
        any windowed signal's figures and only cost Phase-1 time and buffer memory. The
        window scores from the first in-window decision onward (see _apply_eval_window).
        None (default) leaves signals and prices untouched.

    '''
    from ebt.simulate.tricks import corr_to_pairs

    def prepare_configs_df(df):
        def _prepare_configs_df_column(s):
            if pd.api.types.is_numeric_dtype(s.dtype):
                return s
            if pd.api.types.is_timedelta64_dtype(s.dtype):
                return s.dt.total_seconds() / 24 / 3600
            return s.astype(str)

        return df.apply(_prepare_configs_df_column).reset_index(drop=True)

    _skip_portfolio_kfs = {'risk'}

    _log_scale_kfs = {'holding(days)'}

    def plot_categorical(kfs_configs_df, portfolios, all_portfolio_kfs, k, v, ax):
        ax.set_ylabel(k)
        df = pd.concat([pd.Series(b[k].values, name=a) for a, b in kfs_configs_df[[v, k]].groupby(v)], axis=1)
        if df.columns.dtype == bool:
            # bool column labels break DataFrame.plot.box (the False box is
            # silently dropped — the bool index acts as a mask in its labeling
            # path); stringify so both groups render.
            df.columns = df.columns.map(str)
        df.plot.box(ax=ax)
        if 'float' in df.columns.dtype.name:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_CategoricalTickFormatter(df.columns)))
        if k not in _skip_portfolio_kfs:
            portfolio_kfs = pd.Series({str(portfolio_key): all_portfolio_kfs[portfolio_key][k] for portfolio_key, _ in portfolios.items()})
            ax.scatter(range(1, len(portfolio_kfs.index) + 1), portfolio_kfs.values, color='red', marker='x')
        if k in _log_scale_kfs:
            ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=30, labelsize=7)

    def plot_scatter(kfs_configs_df, portfolios, all_portfolio_kfs, k, v, ax):
        kfs_configs_df.plot.scatter(x=v, y=k, ax=ax)
        if k not in _skip_portfolio_kfs:
            portfolio_kfs = pd.Series({portfolio_key: all_portfolio_kfs[portfolio_key][k] for portfolio_key, _ in portfolios.items()})
            ax.scatter(list(portfolio_kfs.index), portfolio_kfs.values, color='red', marker='x')
            if 'float' in portfolio_kfs.index.dtype.name:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_float_tick))
        if k in _log_scale_kfs:
            ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=30, labelsize=7)


    def _decide_group_plan(plan_configs):
        """Decide, from configs ALONE, which variables get sub-portfolios and how
        signals partition into groups. Same decision as the sub-portfolio phase
        (skip / categorical / numeric-qcut, plus the single 'first_unique' pass)
        exactly, but needs no positions — so grouping is known before the signal
        pass and each group's combined position can be accumulated on the fly.

        Returns {var: {'mode', 'unique_values', 'groups': {key: [signal_ids]}}}.
        """
        plan = {}
        first_unique = True
        for v in list(configs_df.columns):
            col = plan_configs[v]
            unique_values = col.value_counts().shape[0]
            if unique_values == 0:
                plan[v] = {'mode': 'skip', 'unique_values': 0, 'groups': {}}
            elif unique_values == 1:
                if first_unique:
                    first_unique = False
                    plan[v] = {'mode': 'categorical', 'unique_values': 1,
                               'groups': _assign_groups(col, True, n_portfolios)}
                else:
                    plan[v] = {'mode': 'skip', 'unique_values': 1, 'groups': {}}
            elif 1 < unique_values < max_unique_values_for_categorical:
                plan[v] = {'mode': 'categorical', 'unique_values': unique_values,
                           'groups': _assign_groups(col, True, n_portfolios)}
            elif pd.api.types.is_numeric_dtype(plan_configs.dtypes[v]):
                plan[v] = {'mode': 'scatter', 'unique_values': unique_values,
                           'groups': _assign_groups(col, False, n_portfolios)}
            else:
                plan[v] = {'mode': 'skip', 'unique_values': unique_values, 'groups': {}}
        return plan

    def _stream_engine(group_plan):
        """Single streaming pass over the ensemble (GIL-free threads).

        For each signal it computes the per-signal key figures and daily PnL via
        the numpy fast path (identical to the per-strategy one_pager), and
        accumulates that signal's realized position into the running sum of every
        sub-portfolio group it belongs to. Positions are never all held at once:
        peak memory is bounded by n_workers (in-flight signals) plus the group
        buffers (one grid per sub-portfolio, not one per signal).

        Returns (all_kfs_df, pnls_daily, portfolios_by_v) where portfolios_by_v[v]
        maps each group key to its combined-position DataFrame (== combine_positions
        of the group's members, to float precision).
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        # Parse shift: timedelta → pre-shift pos timestamps, then one print of lag so
        # the fill lands on the FIRST print at-or-after t+td (see kf.backtest comment)
        _ens_int_shift, _ens_td_shift = kf._parse_shift(shift)
        _ens_td_ns = int(_ens_td_shift.total_seconds() * 1_000_000_000) if _ens_td_shift is not None else 0
        _ens_shift = 1 if _ens_td_shift is not None else _ens_int_shift

        # Union of signal columns (intersected with prices); single-asset signals
        # get NaN for the columns they don't cover, and the backtest handles that.
        common_cols = prices.columns.intersection(source.columns_union())

        # Gap-aware mark-to-market: RAW prices for realization, ffilled prices for
        # the return/PnL arrays so a held position earns the reopen gap.
        # On a reduced grid every array comes from the grid's kept rows.
        _g = None if _grid_in is None else _grid_in.select_columns(common_cols)
        prices_aligned = prices[common_cols] if _g is None else _g.prices
        _prices_ff_aligned = _mark_to_last_price(prices_aligned)
        price_ts = kf._index_to_ns(prices_aligned.index)
        price_diff_vals = _prices_ff_aligned.values.copy()
        price_diff_vals[1:] = price_diff_vals[1:] - price_diff_vals[:-1]
        price_diff_vals[0] = np.nan
        price_vals_flat = np.ascontiguousarray(prices_aligned.values.ravel(order='F'), dtype='float64')
        price_vals_ff_flat = np.ascontiguousarray(_prices_ff_aligned.values.ravel(order='F'), dtype='float64')
        n_rows_prices = len(prices_aligned)
        n_cols_common = len(common_cols)
        prices_index = prices_aligned.index

        # Day keys for daily resampling (shared inverse maps each bar → day index)
        day_keys = price_ts // 86_400_000_000_000
        unique_days, inverse = np.unique(day_keys, return_inverse=True)
        n_days = len(unique_days)

        if _g is not None:
            spreads_vals = None if _g.spreads is None else _g.spreads.values
            _asset_vol = _g.volatility.values
            _weights = _g.weights()
            _px_v_vals = _g.px_v.values
        else:
            spreads_vals = _align_spreads(spreads, prices_aligned).values if spreads is not None else None
            _vol = volatility if volatility is not None else _estimate_asset_vol(prices_aligned)
            _asset_vol = _vol.reindex(index=prices_aligned.index, columns=common_cols, method='ffill').values
            _weights = None
            _px_v_vals = None
        kfs_set = set(kfs)

        # Resolve stop_break threshold
        _sb_th = None
        if stop_break is True:
            _sb_th = 0.001
        elif isinstance(stop_break, (int, float)) and stop_break > 0:
            _sb_th = float(stop_break)

        N = len(source)

        # Group-sum buffers: one per (variable, group). signal_to_bufs maps each
        # signal to the buffers it contributes to (one per non-skipped variable).
        buf_keys = []
        signal_to_bufs = {}
        for v, info in group_plan.items():
            if info['mode'] == 'skip':
                continue
            for key, ids in info['groups'].items():
                bk = (v, key)
                buf_keys.append(bk)
                for i in ids:
                    signal_to_bufs.setdefault(i, []).append(bk)
        buffers = {bk: np.zeros((n_rows_prices, n_cols_common)) for bk in buf_keys}
        first_valid = {bk: np.full(n_cols_common, n_rows_prices, dtype=np.int64) for bk in buf_keys}
        locks = {bk: threading.Lock() for bk in buf_keys}

        kf_dicts = {}
        daily = {}

        def worker(i):
            try:
                p = source.load(i, window=eval_window)
                sig_cols = p.columns.intersection(common_cols)
                if len(sig_cols) == 0 or len(p) == 0:
                    return i, {}, np.zeros(n_days), np.zeros(n_days, dtype=bool)
                sig = kf._align_tz(p.reindex(columns=common_cols), prices_aligned, f'signal {i}', 'prices')
                pos_ts_i = kf._index_to_ns(sig.index)
                if _ens_td_ns:
                    pos_ts_i = pos_ts_i + _ens_td_ns
                kf_d, pos_vals, pnl_sum_vals, pos_ff = _one_pager_numpy_fast(
                    sig.values, pos_ts_i, price_ts, price_vals_flat, price_vals_ff_flat,
                    price_diff_vals, spreads_vals, _ens_shift,
                    len(sig), n_rows_prices, n_cols_common,
                    day_keys, unique_days, kfs_set,
                    asset_vol=_asset_vol, stop_break_threshold=_sb_th,
                    weights=_weights, px_v=_px_v_vals)
                if _g is not None and 'freq prices' in kf_d:
                    kf_d['freq prices'] = _g.freq_seconds
                _valid_rows = ~np.isnan(pnl_sum_vals)
                d = np.bincount(inverse, weights=np.nan_to_num(pnl_sum_vals, nan=0.0),
                                minlength=n_days)
                h = np.bincount(inverse[_valid_rows], minlength=n_days) > 0
                bufs_i = signal_to_bufs.get(i)
                if bufs_i:
                    # Accumulate this signal's HELD (forward-filled) position into
                    # each group sum. Sum-of-ffilled (+ leading-NaN mask applied at
                    # finalize) reproduces combine_positions on the shared grid.
                    # pos_ff comes back from _one_pager_numpy_fast, which already
                    # computed it — recomputing would cost ~2.125x the grid
                    # per signal in bool-mask + int64-index + result allocations.
                    valid = ~np.isnan(pos_ff)
                    has = valid.any(axis=0)
                    fvr = np.where(has, valid.argmax(axis=0), n_rows_prices).astype(np.int64)
                    # posinf/neginf → 0 as well: a stray inf position (bad upstream
                    # feature) must not poison the group sum with ±1.8e308.
                    pos_ff_z = np.nan_to_num(pos_ff, nan=0.0, posinf=0.0, neginf=0.0)
                    for bk in bufs_i:
                        with locks[bk]:
                            buffers[bk] += pos_ff_z
                            np.minimum(first_valid[bk], fvr, out=first_valid[bk])
                return i, kf_d, d, h
            except Exception as _e:
                # Logged WITH the traceback and remembered: a signal that fails here has
                # no key figures and is missing from every sub-portfolio sum it belongs
                # to, which must not pass as a quiet NaN row.
                _LOGGER.warning(f"plot_ensemble stream worker {i} failed: {_e!r}", exc_info=True)
                failed.append(i)
                return i, {}, np.zeros(n_days), np.zeros(n_days, dtype=bool)

        failed = []
        _n_workers = min(n_workers, max(1, N))
        daily_has = {}
        with ThreadPoolExecutor(max_workers=_n_workers) as ex:
            for i, kf_d, d, h in tqdm(ex.map(worker, range(N)), total=N,
                                      desc=f'Phase 1: per-signal KFs ({_n_workers} threads)'):
                kf_dicts[i] = kf_d
                daily[i] = d
                daily_has[i] = h
        failed.sort()
        if failed:
            if len(failed) == N:
                raise RuntimeError(f'plot_ensemble: all {N} signals failed in the per-signal pass '
                                   f'(tracebacks logged above)')
            _LOGGER.error(f"plot_ensemble: {len(failed)}/{N} signals failed in the per-signal pass and are "
                          f"missing from the key figures and the sub-portfolio sums: {failed[:20]}"
                          f"{' ...' if len(failed) > 20 else ''} (tracebacks logged above)")

        # Finalize group buffers → combined-position DataFrames per variable.
        _row_idx = np.arange(n_rows_prices)[:, None]
        portfolios_by_v = {}
        for v, info in group_plan.items():
            if info['mode'] == 'skip':
                continue
            pf = {}
            for key in info['groups'].keys():
                bk = (v, key)
                buf = buffers[bk]
                buf[_row_idx < first_valid[bk][None, :]] = np.nan  # leading NaN per col
                pf[key] = pd.DataFrame(buf, index=prices_index, columns=common_cols)
            portfolios_by_v[v] = pf

        all_kfs_df = pd.DataFrame(kf_dicts).T.dropna(how='all', axis=1)

        # Daily PnL matrix (days × signals), built from the per-signal daily sums.
        _daily_sums = np.empty((n_days, N))
        _daily_has = np.zeros((n_days, N), dtype=bool)
        for i in range(N):
            _daily_sums[:, i] = daily.get(i, np.zeros(n_days))
            _daily_has[:, i] = daily_has.get(i, np.zeros(n_days, dtype=bool))
        # no-data days and flat days (kf.FLAT_DAY_RTOL) leave the daily sample, as in kf.sharpe
        _daily_sums[kf._flat_mask_2d(_daily_sums, _daily_has)] = np.nan
        _day_dates = pd.to_datetime(unique_days * 86_400_000_000_000)
        pnls_daily = pd.DataFrame(_daily_sums, index=_day_dates,
                                  columns=range(N)).dropna(how='all')

        return all_kfs_df, pnls_daily, portfolios_by_v, failed

    def _plot_variable(v, unique_values, portfolios, portfolios_ix, all_portfolio_kfs):
        """Plot a pre-computed variable (matplotlib, must be sequential)."""
        extra = 0 if len(keyfigures) % n_cols == 0 else 1

        fig, axes = plt.subplots(nrows=len(keyfigures) // n_cols + extra, ncols=n_cols, figsize=figsize)
        fig.set_visible(False)
        fig.patch.set_linewidth(2)
        fig.patch.set_edgecolor('lightgray')
        fig.suptitle(v, fontsize=16, fontweight='bold')
        all_figs[v] = fig

        axes = [ax for rows in axes for ax in (rows if isinstance(rows, Iterable) else [rows])]
        for ax in axes:
            ax.set_visible(False)

        if v == 'diversification':
            axes[0].set_visible(True)
            n_instrumentses = [2**i for i in range(2, 15) if 2**i < pnls_daily.shape[1]]
            if len(n_instrumentses) > 0:
                pnls_vals = pnls_daily.values
                rng = np.random.RandomState(42)
                div_results = {}
                for n_inst in n_instrumentses:
                    sharpes = np.empty(n_draws)
                    for d in range(n_draws):
                        cols = rng.choice(pnls_vals.shape[1], size=n_inst, replace=False)
                        daily_sum = np.nansum(pnls_vals[:, cols], axis=1)
                        mask = (daily_sum != 0) & ~np.isnan(daily_sum)
                        ds = daily_sum[mask]
                        sharpes[d] = ds.mean() / ds.std() * 252**0.5 if len(ds) > 1 else np.nan
                    div_results[n_inst] = sharpes
                pd.DataFrame(div_results).plot.box(ax=axes[0])
        else:
            all_subportfolios[v] = portfolios
            all_subportfolios_ix[v] = portfolios_ix

            for ax, k in zip(axes, keyfigures):
                ax.set_visible(True)
                if k == 'correlation_within_portfolio':
                    ax.tick_params(axis='x', rotation=30, labelsize=7)
                    ax.set_ylabel('correlation')

                    if len(portfolios_ix) > 1:
                        sns.violinplot(pd.concat([pd.Series(corr_to_pairs(pnls_daily[ix].corr()).values, name=str(key)) for key, ix in portfolios_ix.items()], axis=1), ax=ax)
                    else:
                        sns.violinplot(pd.Series(corr_to_pairs(pnls_daily.corr())), ax=ax)
                elif k == 'correlation_between_portfolios':
                    sns.heatmap(
                        pd.DataFrame({k: pnls_daily[v].dropna(how='all').sum(axis=1) for k, v in portfolios_ix.items()}) \
                            .pipe(lambda x: x.rename(columns=lambda s: str(s)[:10]) if 'float' in x.columns.dtype.name else x) \
                            .corr(),
                        annot=True,
                        cmap="RdBu",
                        center=0,
                        vmin=-1, vmax=1,
                        linewidths=0.5,
                        annot_kws={"size": 10},
                        ax=ax
                    )

                    ax.tick_params(axis='x', rotation=30, labelsize=7)
                    ax.tick_params(axis='y', rotation=30, labelsize=7)

                elif unique_values < max_unique_values_for_categorical:
                    plot_categorical(kfs_configs_df, portfolios, all_portfolio_kfs, k, v, ax)
                else:
                    plot_scatter(kfs_configs_df, portfolios, all_portfolio_kfs, k, v, ax)

        for ax in axes:
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)

        fig.set_visible(True)

    # One-time price conversion — avoids repeated is_price_or_return checks in one_pager/sub-portfolio calls
    _grid_in = prices_or_returns if isinstance(prices_or_returns, ReducedGrid) else None
    _was_returns = False
    if _grid_in is not None:
        if volatility is not None or spreads is not None:
            raise ValueError('plot_ensemble: with a ReducedGrid pass volatility/spreads through the grid')
        prices = _grid_in.prices
    elif kf.is_price_or_return(prices_or_returns) == 'return':
        prices = prices_or_returns.fillna(0).cumsum()
        _was_returns = True
    else:
        prices = prices_or_returns

    _win_start = _win_end = None
    if eval_window is not None:
        _win_start, _win_end = eval_window
        if _win_end is not None:
            # one_pager cuts prices at the signal's last stamp + shift; the stream
            # engine below does not, and would credit every windowed signal with the
            # move into its exit fill after `end` plus the exit trade itself.
            _, _win_td = kf._parse_shift(shift)
            _win_cut = pd.Timestamp(_win_end) + (_win_td if _win_td is not None else pd.Timedelta(0))
            if getattr(prices.index, 'tz', None) is not None and _win_cut.tz is None:
                _win_cut = _win_cut.tz_localize(prices.index.tz)
            if _grid_in is not None:
                _grid_in = _grid_in.truncate(_win_cut)
                prices = _grid_in.prices
            else:
                prices = prices.loc[:_win_cut]
        if _win_start is not None and (volatility is not None or _grid_in is not None) and spreads is None:
            # Pre-window rows only size the Phase-1 grid (every windowed signal is NaN
            # there). Realization and the gap-aware marks start at a column's own print
            # at-or-after its first fill, so the cut changes no figure; the vol is read
            # as-of from `volatility`. Without it the default estimate warms up on these
            # rows, so nothing is cut.
            if _grid_in is not None:
                _t = pd.Timestamp(_win_start)
                _tz = getattr(prices.index, 'tz', None)
                if _tz is not None:
                    _t = _t.tz_localize(_tz) if _t.tz is None else _t.tz_convert(_tz)
                _days = prices.index[prices.index < _t.normalize()].normalize().unique().sort_values()
                if len(_days) >= _EVAL_WINDOW_PRICE_SESSIONS:
                    _grid_in = _grid_in.slice(start=_days[-_EVAL_WINDOW_PRICE_SESSIONS])
                    prices = _grid_in.prices
            else:
                prices = _trim_prices_before_window(prices, _win_start)

    # Position source: in-RAM list, a directory of parquet files, or explicit
    # parquet paths. The disk forms are streamed one signal at a time so peak
    # memory stays bounded regardless of the number of signals.
    source = _PositionSource(positions)
    N = len(source)

    if configs_df is None:
        configs_df = pd.DataFrame({'a': 1}, index=range(N))
    elif len(configs_df) != N:
        # Signals pair with config rows BY POSITION. A count mismatch (e.g. a
        # sweep that wrote fewer parquet files than config rows) would silently
        # misattribute configs / corrupt group bins — fail loudly BEFORE the
        # expensive signal pass.
        raise ValueError(
            f"plot_ensemble: {N} positions but {len(configs_df)} configs_df rows — "
            f"signals are paired with config rows by position, so the counts must match")

    configs_df = prepare_configs_df(configs_df)

    figsize = (20, 6) if figsize is None else figsize

    # Reduced grid (default): one grid for every signal, built from the union of their stamps
    # (+ the eval-window close stamp) after the window cuts above, so the fills of every
    # member are kept rows. Built here, not by the caller, so the cut semantics live in
    # one place; a prebuilt grid is honoured as-is.
    _ens_td = kf._parse_shift(shift)[1]
    if _grid_in is None and reduce not in (False, None):
        _stamps = source.index_union()
        _end_stamp = None
        if _win_end is not None:
            _end_stamp = pd.Timestamp(_win_end)
            _tz = getattr(prices.index, 'tz', None)
            if _tz is not None and _end_stamp.tz is None:
                _end_stamp = _end_stamp.tz_localize(_tz)
            _stamps = _stamps.union(pd.DatetimeIndex([_end_stamp]))
        _ok, _why = applicability(_stamps, prices.index, shift, resample=True, is_returns=_was_returns)
        if _ok and kfs is not None and (set(kfs) & GRID_INCOMPATIBLE_KFS):
            raise ValueError(f'plot_ensemble: key figures {sorted(set(kfs) & GRID_INCOMPATIBLE_KFS)} are defined '
                             f'per price bar and are not available on a reduced grid; pass reduce=False')
        if _ok:
            try:
                _cand = reduce_price_grid(prices, _stamps, _ens_td,
                                          trunc_stamps=[] if _end_stamp is None else [_end_stamp],
                                          volatility=volatility, spreads=spreads,
                                          columns=prices.columns.intersection(source.columns_union()),
                                          max_kept_fraction=MAX_KEPT_FRACTION)
                if len(_cand) > MAX_KEPT_FRACTION * _cand.n_full:
                    _ok, _why = False, f'kept {len(_cand)} of {_cand.n_full} rows'
                else:
                    _grid_in = _cand
            except GridNotApplicable as _e:
                _ok, _why = False, str(_e)
        if not _ok:
            if reduce is True:
                raise ValueError(f'plot_ensemble: reduced grid not applicable: {_why}')
            _LOGGER.info(f'plot_ensemble: evaluating on the full grid ({_why})')
    if _grid_in is not None and kfs is not None and (set(kfs) & GRID_INCOMPATIBLE_KFS):
        raise ValueError(f'plot_ensemble: key figures {sorted(set(kfs) & GRID_INCOMPATIBLE_KFS)} are defined '
                         f'per price bar and are not available on a reduced grid; pass reduce=False')

    if kfs is None:
        kfs = ['sharpe', 'sharpe exbias', 'holding(days)', 'sharpe_decay', 'risk', 'lptp', 'up_days']
        if _grid_in is not None:
            kfs = [k for k in kfs if k not in GRID_INCOMPATIBLE_KFS]
    if add_ncvar and 'ncvar' not in kfs:
        kfs += ['ncvar']

    # Decide sub-portfolio grouping from configs alone (before the signal pass),
    # so each group's combined position is accumulated on the fly during the pass.
    _plan_configs = configs_df.replace([np.inf, -np.inf], np.nan)
    group_plan = _decide_group_plan(_plan_configs)

    # Resolve the thread count from the data + machine (n_workers='auto' default);
    # an explicit int is honored. Grid dims estimated from prices (upper bound on
    # the intersected common columns). fixed_grids budgets the allocations that
    # exist regardless of worker count: one group-sum buffer per (variable, group)
    # plus ~5 price-grid copies held by the engine.
    n_buffers = sum(len(info['groups']) for info in group_plan.values()
                    if info['mode'] != 'skip')
    _n_rows_eval = len(prices) if _grid_in is None else len(_grid_in)
    n_workers = _auto_n_workers(N, _n_rows_eval, prices.shape[1], requested=n_workers,
                                fixed_grids=n_buffers + 5)
    _LOGGER.info(f"plot_ensemble: {N} signals, grid {_n_rows_eval}x{prices.shape[1]}"
                 f"{'' if _grid_in is None else f' (reduced from {_grid_in.n_full} rows)'}, "
                 f"{n_buffers} group buffers, n_workers={n_workers}")

    # Single streaming pass: per-signal KFs + daily PnL + group-sum positions.
    all_kfs_df, pnls_daily, portfolios_by_v, failed_signals = _stream_engine(group_plan)

    kfs_configs_df = all_kfs_df.join(configs_df, how='inner')
    kfs_configs_df = kfs_configs_df.replace([np.inf, -np.inf], np.nan)

    variables = list(configs_df.columns)
    keyfigures = [k for k in all_kfs_df.columns if k not in ('freq signal', 'freq prices', 'freq pnl')]
    if add_correlation:
        keyfigures += ['correlation_within_portfolio', 'correlation_between_portfolios']
    if add_diversification:
        variables += ['diversification']

    all_subportfolios = {}
    all_subportfolios_ix = {}
    all_figs = {}

    # Phase 2: sub-portfolio KFs on the already-accumulated group positions.
    # (_compute_kfs_direct is pandas-bound → parallelise across variables. The
    # number of groups is bounded by the config space, not the sweep size, so this
    # phase does not grow with the number of signals.)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    non_div_vars = [v for v in variables if v != 'diversification']
    computed = {}  # v -> (unique_values, portfolios, portfolios_ix, all_portfolio_kfs)
    for v in non_div_vars:
        if group_plan.get(v, {'mode': 'skip'})['mode'] == 'skip':
            computed[v] = ('skip', None, None, None, None)

    def _compute_one_var(v):
        info = group_plan[v]
        portfolios = portfolios_by_v[v]
        portfolios_ix = {key: pd.Index(ids) for key, ids in info['groups'].items()}
        if _grid_in is not None:
            portfolio_kfs = {pk: _compute_kfs_direct(p, _grid_in.select_columns(p.columns), keyfigures)
                             for pk, p in portfolios.items()}
        else:
            portfolio_kfs = {pk: _compute_kfs_direct(p, prices, keyfigures, spreads=spreads, volatility=volatility)
                             for pk, p in portfolios.items()}
        return v, info['unique_values'], portfolios, portfolios_ix, portfolio_kfs

    to_compute = [v for v in non_div_vars if group_plan[v]['mode'] != 'skip']
    if len(to_compute) > 1:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(to_compute))) as executor:
            futures = {executor.submit(_compute_one_var, v): v for v in to_compute}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Phase 2: computing sub-portfolios'):
                v, unique_values, portfolios, portfolios_ix, portfolio_kfs = future.result()
                computed[v] = (unique_values, portfolios, portfolios_ix, portfolio_kfs)
    else:
        for v in tqdm(to_compute, desc='Phase 2: computing sub-portfolios'):
            _, unique_values, portfolios, portfolios_ix, portfolio_kfs = _compute_one_var(v)
            computed[v] = (unique_values, portfolios, portfolios_ix, portfolio_kfs)
    # Phase 3: Plot sequentially (matplotlib is not thread-safe)
    for v in tqdm(variables, desc='Phase 3: plotting variables'):
        if v == 'diversification':
            _plot_variable(v, None, None, None, None)
        elif v in computed and computed[v][0] != 'skip':
            unique_values, portfolios, portfolios_ix, portfolio_kfs = computed[v]
            _plot_variable(v, unique_values, portfolios, portfolios_ix, portfolio_kfs)
        else:
            # skipped variable — no fig needed
            pass
    
    if show_tabs:
        display_figures_with_tabs(all_figs)

    # Build sub_portfolios: {variable_name: DataFrame(portfolio_key × KFs)}
    sub_portfolios = {}
    for v in non_div_vars:
        if v in computed and computed[v][0] != 'skip':
            unique_values, portfolios, portfolios_ix, portfolio_kfs = computed[v]
            sub_portfolios[v] = pd.DataFrame(portfolio_kfs).T
            sub_portfolios[v].index.name = v

    # AI-digestible summary of the whole sweep (JSON-serializable) — built while
    # pnls_daily is still available (diversification/correlation need it).
    try:
        digest = build_ensemble_digest(
            kfs_configs_df, sub_portfolios, pnls_daily, list(configs_df.columns),
            n_draws=n_draws,
            primary='sharpe exbias' if 'sharpe exbias' in kfs_configs_df.columns else 'sharpe',
            include_diversification=add_diversification,
            include_correlation=add_correlation)
    except Exception as _e:
        _LOGGER.warning(f"plot_ensemble: failed to build digest: {_e}")
        digest = None

    # Free large intermediate data
    del pnls_daily, computed, portfolios_by_v

    results = {
        'kfs_df': kfs_configs_df,  # strategies × (config_params + KFs)
        'sub_portfolios': sub_portfolios,  # {variable: DataFrame(portfolio_key × KFs)}
        'figs': all_figs,  # {variable: matplotlib.Figure}
        'digest': digest,  # JSON-serializable AI-readable summary (see build_ensemble_digest)
        'failed_signals': failed_signals,  # signal ids whose per-signal pass raised (see the log)
    }

    if debug:
        results.update({
            'subportfolios_pos': all_subportfolios,
            'subportfolios_ix': all_subportfolios_ix,
        })

    if save_memory:
        return LazyDiskDict(results)

    return results


_TAB_WRAP_CSS = """<style>
.widget-tab .lm-TabBar-content, .widget-tab .p-TabBar-content {
    flex-wrap: wrap !important;
}
.widget-tab .lm-TabBar-tab, .widget-tab .p-TabBar-tab {
    flex: 0 1 auto !important;
}
</style>"""


def display_figures_with_tabs(figures):
    """Creates a Jupyter widget with tabs for each figure."""
    from IPython.display import display, HTML
    import ipywidgets as widgets

    if type(figures) != dict:
        figures = {f'Fig {i}': f for i, f in enumerate(figures)}

    for fig in figures.values():
        plt.close(fig)
        
    tabs = []
    for fig in figures.values():
        output = widgets.Output()
        with output:
            display(fig)
        tabs.append(output)

    tab_widget = widgets.Tab(children=tabs)
    for i, title in enumerate(figures.keys()):
        tab_widget.set_title(i, str(title))

    display(HTML(_TAB_WRAP_CSS))
    display(tab_widget)


def plot_figures_with_tabs(figures):
    return display_figures_with_tabs(figures)


# ---------------------------------------------------------------------------
# AI-digestible ensemble summary
#
# plot_ensemble's figures are for humans. The digest below is the same content
# rendered as a compact, JSON-serializable structure (+ a markdown view) so a
# language model — or any downstream code — can read a sweep's conclusions
# without opening a single PNG.
# ---------------------------------------------------------------------------

def _jsonable(x):
    """Convert numpy/pandas scalars to JSON-native values; NaN/inf → None."""
    if x is None:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        return None if (np.isnan(xf) or np.isinf(xf)) else xf
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if hasattr(x, 'item'):
        try:
            return _jsonable(x.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(x, (int, str)):
        return x
    return str(x)


def _kf_distribution(series):
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return None
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        'count': int(s.shape[0]),
        'mean': _jsonable(s.mean()), 'std': _jsonable(s.std()),
        'min': _jsonable(s.min()), 'p05': _jsonable(q.loc[0.05]),
        'p25': _jsonable(q.loc[0.25]), 'median': _jsonable(q.loc[0.5]),
        'p75': _jsonable(q.loc[0.75]), 'p95': _jsonable(q.loc[0.95]),
        'max': _jsonable(s.max()), 'frac_positive': _jsonable((s > 0).mean()),
    }


def _diversification_digest(pnls_daily, n_draws=100, seed=42):
    vals = pnls_daily.values
    n_cols = vals.shape[1]
    ns = [2 ** i for i in range(2, 15) if 2 ** i < n_cols]
    if not ns:
        return None
    rng = np.random.RandomState(seed)
    out = []
    for n_inst in ns:
        sh = np.empty(n_draws)
        for d in range(n_draws):
            cols = rng.choice(n_cols, size=n_inst, replace=False)
            ds = np.nansum(vals[:, cols], axis=1)
            m = (ds != 0) & ~np.isnan(ds)
            dsm = ds[m]
            sh[d] = dsm.mean() / dsm.std() * 252 ** 0.5 if len(dsm) > 1 else np.nan
        out.append({'n_instruments': int(n_inst),
                    'median_sharpe': _jsonable(np.nanmedian(sh)),
                    'p25_sharpe': _jsonable(np.nanpercentile(sh, 25)),
                    'p75_sharpe': _jsonable(np.nanpercentile(sh, 75))})
    return out


def _correlation_digest(pnls_daily, max_signals=300, seed=42):
    from ebt.simulate.tricks import corr_to_pairs
    df = pnls_daily
    if df.shape[1] > max_signals:
        rng = np.random.RandomState(seed)
        cols = np.sort(rng.choice(df.shape[1], max_signals, replace=False))
        df = df.iloc[:, cols]
    pairs = corr_to_pairs(df.corr()).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) == 0:
        return None
    return {'median_pairwise_corr': _jsonable(pairs.median()),
            'mean_pairwise_corr': _jsonable(pairs.mean()),
            'p95_pairwise_corr': _jsonable(pairs.quantile(0.95)),
            'n_signals_sampled': int(df.shape[1])}


def build_ensemble_digest(kfs_configs_df, sub_portfolios, pnls_daily, config_vars,
                          n_draws=100, primary='sharpe exbias', top_n=10,
                          include_diversification=True, include_correlation=True):
    """Build a JSON-serializable summary of a plot_ensemble sweep.

    Captures, without any figures: per-signal KF distributions, the best configs
    by the primary KF (unbiased Sharpe, anchored with turnover/holding), how each
    config variable moves performance (the sub-portfolio tables), the
    diversification curve, and the cross-signal correlation level. The last two
    recompute bootstraps over the full daily-PnL matrix — ``include_*=False``
    (wired to plot_ensemble's add_diversification/add_correlation) skips them.
    """
    config_vars = list(config_vars)
    kf_cols = [c for c in kfs_configs_df.columns if c not in set(config_vars)]
    prim = primary if primary in kfs_configs_df.columns else (kf_cols[0] if kf_cols else None)
    digest = {
        'n_signals': int(kfs_configs_df.shape[0]),
        'config_variables': config_vars,
        'key_figures': kf_cols,
        'primary_kf': prim,
    }

    digest['kf_distribution'] = {
        k: d for k in kf_cols for d in [_kf_distribution(kfs_configs_df[k])] if d is not None
    }

    if prim is not None:
        anchor = [c for c in ('sharpe', 'sharpe w costs', 'turnover', 'holding(days)')
                  if c in kfs_configs_df.columns]
        show = list(dict.fromkeys([prim] + anchor + config_vars))
        show = [c for c in show if c in kfs_configs_df.columns]
        best = (kfs_configs_df.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[prim]).sort_values(prim, ascending=False).head(top_n))
        digest['best_configs'] = [
            {'signal_id': _jsonable(idx), **{c: _jsonable(row[c]) for c in show}}
            for idx, row in best.iterrows()]

    digest['config_effects'] = {
        v: {'groups': [{'value': _jsonable(idx), **{c: _jsonable(row[c]) for c in df.columns}}
                       for idx, row in df.iterrows()]}
        for v, df in sub_portfolios.items()}

    if include_diversification and pnls_daily is not None and pnls_daily.shape[1] >= 4:
        div = _diversification_digest(pnls_daily, n_draws=n_draws)
        if div is not None:
            digest['diversification'] = div
    if include_correlation and pnls_daily is not None and pnls_daily.shape[1] >= 2:
        corr = _correlation_digest(pnls_daily)
        if corr is not None:
            digest['correlation'] = corr

    return digest


def _fmt_num(x):
    if x is None:
        return ''
    if isinstance(x, str):
        return x
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf):
        return ''
    if xf != 0 and (abs(xf) >= 10000 or abs(xf) < 0.001):
        return f'{xf:.3g}'
    return f'{xf:.3f}'


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    return f'{100 * float(x):.0f}%'


def _md_cell(x):
    """_fmt_num for a markdown table cell — escapes '|' and newlines, which would
    otherwise shift/split the row (config values are arbitrary user strings)."""
    return _fmt_num(x).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')


def digest_to_markdown(digest, headline=None, title='Ensemble summary', notes=None):
    """Render a digest (from build_ensemble_digest) as compact markdown."""
    L = [f'# {title}', '']
    if notes:
        L += ['**Notes:** ' + str(notes).strip(), '']
    L += [f"Signals: **{digest.get('n_signals', '?')}**  |  primary KF: "
          f"**{digest.get('primary_kf')}**", '']

    if headline is not None and len(headline):
        L += ['## Combined portfolio (headline)', '', '| KF | value |', '|---|---|']
        L += [f'| {_md_cell(k)} | {_md_cell(v)} |' for k, v in dict(headline).items()]
        L.append('')

    dist = digest.get('kf_distribution', {})
    if dist:
        L += ['## Per-signal KF distribution', '',
              '| KF | median | mean | p25 | p75 | min | max | % >0 |',
              '|---|---|---|---|---|---|---|---|']
        for k, d in dist.items():
            L.append(f"| {k} | {_fmt_num(d['median'])} | {_fmt_num(d['mean'])} | "
                     f"{_fmt_num(d['p25'])} | {_fmt_num(d['p75'])} | {_fmt_num(d['min'])} | "
                     f"{_fmt_num(d['max'])} | {_fmt_pct(d['frac_positive'])} |")
        L.append('')

    ce = digest.get('config_effects', {})
    if ce:
        L += ['## Config → performance (sub-portfolios)', '']
        for v, info in ce.items():
            groups = info.get('groups', [])
            if not groups:
                continue
            kfcols = [c for c in groups[0].keys() if c != 'value']
            L += [f'### {v}',
                  '| ' + _md_cell(v) + ' | ' + ' | '.join(_md_cell(c) for c in kfcols) + ' |',
                  '|' + '---|' * (len(kfcols) + 1)]
            for g in groups:
                L.append('| ' + _md_cell(g['value']) + ' | '
                         + ' | '.join(_md_cell(g.get(c)) for c in kfcols) + ' |')
            L.append('')

    bc = digest.get('best_configs')
    if bc:
        cols = list(bc[0].keys())
        L += [f"## Top {len(bc)} configs by {digest.get('primary_kf')}", '',
              '| ' + ' | '.join(_md_cell(c) for c in cols) + ' |', '|' + '---|' * len(cols)]
        for r in bc:
            L.append('| ' + ' | '.join(_md_cell(r.get(c)) for c in cols) + ' |')
        L.append('')

    dv = digest.get('diversification')
    if dv:
        L += ['## Diversification (bootstrap Sharpe vs #instruments)', '',
              '| #instruments | median Sharpe | p25 | p75 |', '|---|---|---|---|']
        for r in dv:
            L.append(f"| {r['n_instruments']} | {_fmt_num(r['median_sharpe'])} | "
                     f"{_fmt_num(r['p25_sharpe'])} | {_fmt_num(r['p75_sharpe'])} |")
        L.append('')

    cr = digest.get('correlation')
    if cr:
        L += ['## Cross-signal correlation (daily PnL)', '',
              f"median pairwise **{_fmt_num(cr['median_pairwise_corr'])}**, "
              f"mean {_fmt_num(cr['mean_pairwise_corr'])}, "
              f"p95 {_fmt_num(cr['p95_pairwise_corr'])} "
              f"(sampled {cr['n_signals_sampled']} signals)", '']

    return '\n'.join(L)


def per_year_table(signal, prices, shift=2, volatility=None, kfs=None, logger=None, reduce='auto'):
    """Per-calendar-year key figures of ONE book — the robustness table that
    an experiment package can carry as per_year.csv.

    A summary of one_pager's fast key-figure path per year, NOT a reported
    one-pager: raw sharpe, sharpe exbias (demeaned WITHIN the year — an
    in-sample adjustment that flatters short windows, so read raw first),
    turnover, holding(days), lptp. Each year is scored with 200 days of price
    warm-up ahead of January so the vol estimate is live from the first day.

    Returns a DataFrame (one row per year with a position), or None if the
    table could not be computed (logged as a warning, never raised).
    """
    log = logger or logging.getLogger(__name__)
    kfs = kfs or ['sharpe', 'sharpe exbias', 'turnover', 'holding(days)', 'lptp']
    _grid = prices if isinstance(prices, ReducedGrid) else None
    try:
        if signal is None or signal.shape[1] == 0:
            return None
        rows = []
        for yr in sorted(set(signal.index.year)):
            sub = signal.loc[str(yr)]
            if sub.notna().sum().sum() == 0 or sub.abs().sum().sum() == 0:
                continue
            _y0 = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=200)
            if _grid is not None:
                # the grid slice carries its own vol; one_pager cuts it at the year's last
                # stamp + shift, which the canonical truncation set includes. When the caller
                # passed no vol, estimate it on the year's own slice as the full-grid path does
                # (the daily closes are kept rows, so the estimate is the same).
                g = _grid.slice(_y0, pd.Timestamp(f"{yr}-12-31") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
                if len(g) and _grid.meta.get('vol_source') == 'estimated':
                    g = g.with_volatility(_estimate_asset_vol(g.prices))
                vol = None
            else:
                g = prices.loc[_y0:f"{yr}-12-31"]
                vol = volatility
                if isinstance(vol, pd.DataFrame):
                    vol = vol.reindex(g.index, method='ffill')
            if len(g) == 0:
                continue
            kf = one_pager(sub, g, shift=shift, plot=False, fast=True,
                           return_subset={'kf'}, volatility=vol, kfs=kfs, reduce=reduce)['kf']
            rows.append({'year': int(yr), **{k: float(v) for k, v in kf.items()}})
        if not rows:
            return None
        tab = pd.DataFrame(rows)
        log.info("per-year key figures:\n%s",
                 tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        return tab
    except Exception as e:
        log.warning(f"per-year table failed (per_year.csv will be missing): {e!r}", exc_info=True)
        return None


def combine_figures(figs):
    """
    Combine multiple Matplotlib figures into one PIL Image.
    
    Save to directory using
        img = combine_figures(figs)
        img.save(PATH)
    """
    
    def fig_to_image(fig):
        """Convert a Matplotlib figure to a PIL image."""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        return Image.open(buf)

    # Convert figures to images
    images = [fig_to_image(fig) for fig in figs]

    # Determine final image size (stacked vertically)
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)

    # Create blank canvas
    combined_img = Image.new("RGB", (max_width, total_height))

    # Paste images onto the final canvas
    y_offset = 0
    for img in images:
        combined_img.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return combined_img  # Returns the PIL Image object
