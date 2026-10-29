from ebt.simulate.timeseries import merge_asof
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
from typing import Union, Optional

_LOGGER = logging.getLogger(__name__)

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

try:
    import ebt_rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

_RETURN_VAR_MAP = {'pnl': 0, 'pos': 1, 'market_return': 2}

#: A day whose |PnL| is at most this fraction of the median |daily PnL| (over the days that have
#: data) is FLAT and leaves the Sharpe sample like a non-trading day. A relative tolerance rather
#: than an exact-zero test keeps the day set independent of float rounding: a held book whose
#: price ends the day where it started sums to ~1e-16 when added bar by bar and to exactly 0.0
#: when telescoped over a reduced grid (see ebt.simulate.grid). 0.0 gives the exact-zero rule.
FLAT_DAY_RTOL = 1e-12


def _flat_mask(daily, has_data):
    """Days to drop from a daily PnL vector: no data, or |daily| <= FLAT_DAY_RTOL * median(|daily|
    over the days with data). With FLAT_DAY_RTOL == 0 only exact zeros are dropped (exact-zero rule)."""
    daily = np.asarray(daily, dtype=np.float64)
    has = np.asarray(has_data, dtype=bool)
    a = np.abs(daily)
    # median over the NONZERO days with data: a book that is flat by design most days must still
    # see its rounding-residue days measured against its real days
    nz = has & (a > 0)
    thr = FLAT_DAY_RTOL * float(np.median(a[nz])) if nz.any() else 0.0
    return ~has | (a <= thr)


def _flat_mask_2d(daily, has_data):
    """Column-wise ``_flat_mask`` for a (days x series) matrix."""
    daily = np.asarray(daily, dtype=np.float64)
    has = np.asarray(has_data, dtype=bool)
    out = np.empty(daily.shape, dtype=bool)
    for j in range(daily.shape[1]):
        out[:, j] = _flat_mask(daily[:, j], has[:, j])
    return out


def _rust_flat_kw():
    """The flat-day tolerance as a keyword for the Rust kernels (variant_sharpes_core,
    sharpe_per_lag_core, holding_curve_core)."""
    if not _HAS_RUST:
        return {}
    return {'flat_rtol': FLAT_DAY_RTOL}


def _daily_nonflat(pnl):
    """Daily sums of a PnL Series/DataFrame with the flat days removed (NaN'd, then dropped).
    Like ``resample('D').sum().replace(0, nan).dropna()`` but with the FLAT_DAY_RTOL rule."""
    d = pnl.resample('D').sum(min_count=1)
    if isinstance(d, pd.DataFrame):
        has = d.notna().to_numpy()
        vals = d.fillna(0.0).to_numpy(dtype=np.float64, copy=True)
        vals[_flat_mask_2d(vals, has)] = np.nan
        return pd.DataFrame(vals, index=d.index, columns=d.columns).dropna(how='all')
    has = d.notna().to_numpy()
    vals = d.fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    vals[_flat_mask(vals, has)] = np.nan
    return pd.Series(vals, index=d.index, name=d.name).dropna()


def _align_tz(pos, prices, pos_name='pos', price_name='prices'):
    """Return ``pos`` with an index comparable to ``prices.index`` before both are stripped
    to wall-clock ns by ``_index_to_ns``. Both naive: unchanged. Both tz-aware: ``pos`` is
    converted to the price zone (same instants, same wall clock; stripping each side in its
    own zone would join 14:00 UTC to the 14:00 local-exchange print). One aware, one naive: raise,
    there is no way to tell whether the naive side is UTC or exchange time."""
    ptz = getattr(prices.index, 'tz', None)
    stz = getattr(pos.index, 'tz', None)
    if (ptz is None) != (stz is None):
        raise ValueError(f'{pos_name}.index is tz-{"aware" if stz is not None else "naive"} but '
                         f'{price_name}.index is tz-{"aware" if ptz is not None else "naive"}; '
                         f'put both in the same zone (or both naive) first')
    if ptz is not None and str(stz) != str(ptz):
        pos = pos.tz_convert(ptz)
    return pos


def _check_sorted_index(df, name):
    """Raise unless ``df.index`` is monotonic non-decreasing: the as-of merge (Rust and
    pandas alike) is only defined on sorted timestamps and the Rust path would otherwise
    return wrong numbers without complaint. Cached by pandas, so repeated calls on the
    same frame are free."""
    if not df.index.is_monotonic_increasing:
        raise ValueError(f'{name}.index must be sorted (monotonic non-decreasing) by time')


def _parse_shift(shift):
    """Parse shift into (int_shift, timedelta_or_none).

    Supports integer (bar-based), pd.Timedelta, or string ('5m') shifts.
    """
    if isinstance(shift, pd.Timedelta):
        return 0, shift
    if isinstance(shift, str):
        return 0, pd.Timedelta(shift)
    return int(shift), None


def _apply_timedelta_shift(df, td):
    """Shift a DataFrame/Series index forward by a timedelta."""
    result = df.copy(deep=False)
    result.index = result.index + td
    return result


def _index_to_ns(idx):
    """Convert DatetimeIndex to int64 nanoseconds, handling tz-aware indices."""
    if hasattr(idx, 'tz') and idx.tz is not None:
        idx = idx.tz_localize(None)
    # Fast path: use asi8 (zero-copy int64 view) + scale factor
    resolution = str(idx.dtype)
    if resolution == 'datetime64[ns]':
        return idx.asi8.copy()
    elif resolution == 'datetime64[us]':
        return idx.asi8 * 1000
    elif resolution == 'datetime64[ms]':
        return idx.asi8 * 1_000_000
    elif resolution == 'datetime64[s]':
        return idx.asi8 * 1_000_000_000
    return idx.astype('datetime64[ns]').astype('int64').values

_TIME_COL_PRIORITY = ['ts', 'time', 'timestamp']
_CONTRACT_COL_PRIORITY = ['series', 'symbol', 'contract']
_PRICE_COL_PRIORITY = ['price', 'close', 'return']


def is_price_or_return(df):
    if type(df) == pd.Series:
        df = df.to_frame()

    # One pass over the array (same quantity: per column, the fraction of non-NaN values whose
    # sign equals the previous non-NaN value's sign, averaged over non-empty columns) instead of
    # a Python lambda per column.
    if (_same_sign_fraction is not None and df.shape[0] > 0 and df.shape[1] > 0
            and all(dt.kind in 'fiu' for dt in np.atleast_1d(df.dtypes.values))):
        try:
            arr = np.ascontiguousarray(df.to_numpy(dtype=np.float64).T)
        except (TypeError, ValueError):
            arr = None
        if arr is not None:
            return 'return' if _same_sign_fraction(arr) < 0.9 else 'price'

    same_sign_as_previous_value_mean = df.apply(lambda s: np.sign(s.dropna()).pipe(lambda x: x == x.shift()).mean()).mean()

    if same_sign_as_previous_value_mean < 0.9:
        return 'return'

    return 'price'


def lag(pos, shift=2):
    return pos.apply(lambda s: s.dropna().shift(shift))



def _backtest_rust(pos_df, prices_df, shift, return_variable):
    """Rust-accelerated backtest using matrix API."""
    rv = _RETURN_VAR_MAP.get(return_variable)
    if rv is None:
        raise ValueError(f'Unknown return variable: {return_variable}')

    # Align columns
    common_cols = pos_df.columns.intersection(prices_df.columns)
    if common_cols.empty:
        return pd.DataFrame(0, index=pos_df.index, columns=pos_df.columns)

    pos_aligned = pos_df[common_cols]
    prices_aligned = prices_df[common_cols]

    pos_ts = _index_to_ns(pos_aligned.index)
    price_ts = _index_to_ns(prices_aligned.index)

    # Flatten column-major for cache-friendly column access in Rust
    pos_data = np.ascontiguousarray(pos_aligned.values.ravel(order='F'), dtype='float64')
    price_data = np.ascontiguousarray(prices_aligned.values.ravel(order='F'), dtype='float64')

    result_flat = ebt_rust.backtest_matrix_core(
        pos_ts, pos_data, price_ts, price_data, shift, rv,
        len(pos_aligned), len(prices_aligned), len(common_cols)
    )
    # Result is column-major: reshape as Fortran-order
    result_matrix = result_flat.reshape(len(prices_aligned), len(common_cols), order='F')

    result = pd.DataFrame(result_matrix, index=prices_aligned.index, columns=common_cols)

    # Add missing columns as NaN
    for col in pos_df.columns:
        if col not in result.columns:
            result[col] = np.nan
    return result[pos_df.columns]


def _backtest_python(pos_df, prices_df, shift, return_variable):
    """Pure-Python backtest, used when the ebt_rust extension is unavailable."""
    def _backtest(s_original):
        s = s_original.dropna()
        p = prices_df[s.name].dropna()
        if s.empty or p.empty:
            return pd.Series(0, index=s_original.index)

        if s.index[0] > p.index[-1]:
            return pd.Series(0, index=s_original.index)

        if s.index[-1] < p.index[0]:
            return pd.Series(0, index=s_original.index)

        s = s.pipe(merge_asof, p.index)
        if s.empty:
            return pd.Series(0, index=s_original.index)

        r = p.diff()
        pos = s.shift(shift)
        pnl = pos * r
        if return_variable == 'pnl':
            return pnl
        elif return_variable == 'pos':
            return pos
        elif return_variable == 'market_return':
            return r
        raise ValueError(f'Unknown return variable: {return_variable}')

    return pos_df.apply(_backtest)


def backtest(pos, prices_or_returns, shift=0, resample=False, return_variable='pnl'):
    # Parse timedelta shift: pre-shift signal timestamps by the latency, then lag one
    # print past the as-of join so the fill lands on the FIRST print at-or-after t+td.
    # With shift=0 the new position would earn the full diff INTO its activation print,
    # i.e. an implicit fill at the LAST print before t+td — an untradeable stale price
    # (across a session gap, the prior close: the entire reopen gap would be credited
    # to a position decided while the market was shut).
    int_shift, td_shift = _parse_shift(shift)
    if td_shift is not None:
        pos = _apply_timedelta_shift(pos, td_shift)
        shift = 1
    else:
        shift = int_shift

    if is_price_or_return(prices_or_returns) == 'return':
        _LOGGER.warning('backtest received returns instead of prices as input, translating may lead to issues if open hours have already been applied')
        prices = prices_or_returns.fillna(0).cumsum()
    else:
        prices = prices_or_returns

    if type(prices_or_returns) == pd.Series:
        prices_or_returns = prices_or_returns.to_frame()

    if type(pos) == pd.Series:
        pos = pos.to_frame()

    if type(prices) == pd.Series:
        prices = prices.to_frame()

    pos = _align_tz(pos, prices, 'pos', 'prices_or_returns')
    _check_sorted_index(pos, 'pos')
    _check_sorted_index(prices, 'prices_or_returns')

    if _HAS_RUST:
        pnl = _backtest_rust(pos, prices, shift, return_variable)
    else:
        pnl = _backtest_python(pos, prices, shift, return_variable)

    return pnl.resample('D').sum().replace(0, np.nan) if resample else pnl


def _detect_col(df_columns, priority_list, param_name):
    """Find first matching column from priority list."""
    for c in priority_list:
        if c in df_columns:
            return c
    return None


def _detect_value_col(df_columns, time_col, contract_col, priority_list):
    """Detect value column: try priority list, then fall back to first numeric-looking col."""
    for c in priority_list:
        if c in df_columns:
            return c
    remaining = [c for c in df_columns if c not in (time_col, contract_col)]
    if len(remaining) == 1:
        return remaining[0]
    if remaining:
        return remaining[0]
    return None


def _resolve_columns(df, contract_col, time_col, value_col, value_priority, label):
    """Resolve column names for a stacked DataFrame (Polars or pandas)."""
    if _HAS_POLARS and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        cols = df.columns if isinstance(df, pl.DataFrame) else df.collect_schema().names()
    else:
        cols = list(df.columns)

    if contract_col is None:
        contract_col = _detect_col(cols, _CONTRACT_COL_PRIORITY, 'contract_col')
        if contract_col is None:
            raise ValueError(f"Cannot auto-detect contract column in {label}. "
                             f"Expected one of {_CONTRACT_COL_PRIORITY}. "
                             f"Available: {cols}")

    if time_col is None:
        time_col = _detect_col(cols, _TIME_COL_PRIORITY, 'time_col')
        # For pandas, check if index is datetime
        if time_col is None and isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex):
            time_col = '__index__'
        if time_col is None:
            raise ValueError(f"Cannot auto-detect time column in {label}. "
                             f"Expected one of {_TIME_COL_PRIORITY}. "
                             f"Available: {cols}")

    if value_col is None:
        value_col = _detect_value_col(cols, time_col, contract_col, value_priority)
        if value_col is None:
            raise ValueError(f"Cannot auto-detect value column in {label}. "
                             f"Available: {cols}")

    return contract_col, time_col, value_col


def _is_price_or_return_stacked(values):
    """Check if a 1-d array of values looks like prices or returns."""
    if len(values) < 10:
        return 'price'
    signs = np.sign(values[~np.isnan(values)])
    if len(signs) < 2:
        return 'price'
    same_sign = (signs[1:] == signs[:-1]).mean()
    return 'return' if same_sign < 0.9 else 'price'


def _extract_stacked_arrays(df, contract_col, time_col, value_col):
    """Extract flat (names, contract_ids_i32, timestamps_i64, values_f64) from Polars or pandas.

    Encodes contract strings as integer IDs for efficient Rust processing.
    Returns (unique_names_list, ids_array_i32, ts_array_i64, vals_array_f64).
    """
    if _HAS_POLARS and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        # Encode contracts as integers (same factorize as the pandas path; independent of the
        # polars Categorical implementation, which changed in polars 2.0)
        codes, uniques = pd.factorize(df[contract_col].cast(pl.String).to_numpy(), sort=True)
        names = uniques.tolist()
        ids = codes.astype('int32')
        ts = df[time_col].cast(pl.Datetime('ns')).to_physical().to_numpy().astype('int64')
        vals = df[value_col].to_numpy().astype('float64')
    elif time_col == '__index__':
        contracts = df[contract_col].values
        # Factorize: encode string contracts as integers
        codes, uniques = pd.factorize(contracts, sort=True)
        names = uniques.tolist()
        ids = codes.astype('int32')
        ts = df.index.astype('datetime64[ns]').astype('int64').values
        vals = df[value_col].values.astype('float64')
    else:
        contracts = df[contract_col].values
        codes, uniques = pd.factorize(contracts, sort=True)
        names = uniques.tolist()
        ids = codes.astype('int32')
        ts = df[time_col].values.astype('datetime64[ns]').astype('int64')
        vals = df[value_col].values.astype('float64')

    # Drop NaN values
    mask = ~np.isnan(vals)
    if not mask.all():
        ids = ids[mask]
        ts = ts[mask]
        vals = vals[mask]

    return names, ids, ts, vals


def _extract_per_contract(df, contract_col, time_col, value_col):
    """Extract per-contract (name, timestamps_i64, values_f64) from Polars or pandas.

    Used by the Python fallback path and tests.
    """
    groups = {}
    if _HAS_POLARS and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        for part in df.partition_by(contract_col, maintain_order=True):
            name = str(part[contract_col][0])
            ts = part[time_col].cast(pl.Datetime('ns')).to_physical().to_numpy().astype('int64')
            vals = part[value_col].to_numpy().astype('float64')
            mask = ~np.isnan(vals)
            groups[name] = (ts[mask], vals[mask])
    else:
        if time_col == '__index__':
            for name, group in df.groupby(contract_col):
                ts = group.index.astype('datetime64[ns]').astype('int64').values
                vals = group[value_col].values.astype('float64')
                mask = ~np.isnan(vals)
                groups[str(name)] = (ts[mask], vals[mask])
        else:
            for name, group in df.groupby(contract_col):
                ts = group[time_col].values.astype('datetime64[ns]').astype('int64')
                vals = group[value_col].values.astype('float64')
                mask = ~np.isnan(vals)
                groups[str(name)] = (ts[mask], vals[mask])
    return groups


def _detect_returns_stacked(price_vals, contract_ids):
    """Check if stacked price values are returns. Samples first contract."""
    if len(price_vals) < 10:
        return False
    first_id = contract_ids[0]
    mask = contract_ids == first_id
    sample = price_vals[mask][:200]
    if len(sample) < 10:
        return False
    return _is_price_or_return_stacked(sample) == 'return'


def backtest_stacked(
    pos,
    prices_or_returns,
    shift=0,
    resample=False,
    return_variable='pnl',
    contract_col=None,
    time_col=None,
    pos_value_col=None,
    price_value_col=None,
    sum_contracts=True,
) -> Union[pd.Series, pd.DataFrame]:
    """Backtest from stacked/long-format DataFrames (Polars or pandas).

    Unlike ``backtest()`` which requires wide DataFrames (columns=instruments),
    this accepts stacked DataFrames with a contract identifier column —
    avoiding wide sparse matrix allocation for futures rolling.

    Args:
        pos: Stacked positions with contract, time, and value columns.
        prices_or_returns: Stacked prices/returns with contract, time, and value columns.
        shift: Execution lag in bars (default 0).
        resample: If True, resample PnL to daily.
        return_variable: 'pnl', 'pos', or 'market_return'.
        contract_col: Column identifying contracts. Auto-detected from
            series > symbol > contract.
        time_col: Timestamp column. Auto-detected from ts > time > timestamp
            > DatetimeIndex.
        pos_value_col: Value column in pos. Auto-detected.
        price_value_col: Value column in prices. Auto-detected from
            price > close > return.
        sum_contracts: If True (default), return summed pd.Series across all
            contracts. If False, return stacked pd.DataFrame per contract.

    Returns:
        pd.Series (sum_contracts=True) or pd.DataFrame (sum_contracts=False).
    """
    # Parse timedelta shift: add timedelta to pos timestamps, then lag one print past
    # the as-of join so the fill lands on the FIRST print at-or-after t+td (see the
    # matching comment in backtest(); shift=0 would fill at the stale previous print).
    int_shift, td_shift = _parse_shift(shift)
    if td_shift is not None:
        shift = 1
    else:
        shift = int_shift

    rv = _RETURN_VAR_MAP.get(return_variable)
    if rv is None:
        raise ValueError(f'Unknown return variable: {return_variable}')

    # Resolve columns
    pos_contract_col, pos_time_col, pos_val_col = _resolve_columns(
        pos, contract_col, time_col, pos_value_col,
        ['value', 'position', 'pos', 'weight'], 'pos')
    price_contract_col, price_time_col, price_val_col = _resolve_columns(
        prices_or_returns, contract_col, time_col, price_value_col,
        _PRICE_COL_PRIORITY, 'prices_or_returns')

    # Extract flat arrays — no Python-side groupby
    pos_names, pos_ids, pos_ts, pos_vals = _extract_stacked_arrays(
        pos, pos_contract_col, pos_time_col, pos_val_col)
    # Apply timedelta shift to position timestamps
    if td_shift is not None:
        td_ns = int(td_shift.total_seconds() * 1_000_000_000)
        pos_ts = pos_ts + td_ns
    price_names, price_ids, price_ts, price_vals = _extract_stacked_arrays(
        prices_or_returns, price_contract_col, price_time_col, price_val_col)

    if len(pos_ts) == 0 or len(price_ts) == 0:
        return pd.Series(dtype='float64') if sum_contracts else pd.DataFrame()

    # Build unified name mapping (union of both sides, sorted)
    all_names = sorted(set(pos_names) | set(price_names))
    name_to_id = {n: i for i, n in enumerate(all_names)}
    # Remap IDs to unified encoding
    pos_id_map = np.array([name_to_id[n] for n in pos_names], dtype='int32')
    price_id_map = np.array([name_to_id[n] for n in price_names], dtype='int32')
    pos_ids_unified = pos_id_map[pos_ids]
    price_ids_unified = price_id_map[price_ids]

    # Detect returns → cumsum per contract
    if _detect_returns_stacked(price_vals, price_ids_unified):
        _LOGGER.warning('backtest_stacked received returns, converting to prices via cumsum')
        order = np.lexsort((price_ts, price_ids_unified))
        sorted_ids = price_ids_unified[order]
        sorted_ts = price_ts[order]
        sorted_vals = price_vals[order]
        cumvals = np.empty_like(sorted_vals)
        cur_id = sorted_ids[0]
        cur_sum = 0.0
        for i in range(len(sorted_vals)):
            if sorted_ids[i] != cur_id:
                cur_id = sorted_ids[i]
                cur_sum = 0.0
            cur_sum += sorted_vals[i]
            cumvals[i] = cur_sum
        price_ids_unified = sorted_ids
        price_ts = sorted_ts
        price_vals = cumvals

    if _HAS_RUST:
        if sum_contracts:
            out_ts, out_vals = ebt_rust.backtest_stacked_summed_core(
                all_names, pos_ids_unified, pos_ts, pos_vals,
                price_ids_unified, price_ts, price_vals,
                shift, rv,
            )
            out_ts = np.asarray(out_ts)
            out_vals = np.asarray(out_vals)
            if len(out_ts) == 0:
                return pd.Series(dtype='float64')
            result = pd.Series(out_vals, index=pd.to_datetime(out_ts, unit='ns'))
            if resample:
                result = result.resample('D').sum().replace(0, np.nan)
            return result
        else:
            result_dict = ebt_rust.backtest_stacked_core(
                all_names, pos_ids_unified, pos_ts, pos_vals,
                price_ids_unified, price_ts, price_vals,
                shift, rv,
            )
            parts = []
            for name, (ts, vals) in result_dict.items():
                ts_arr = np.asarray(ts)
                vals_arr = np.asarray(vals)
                mask = ~np.isnan(vals_arr)
                if mask.any():
                    parts.append(pd.DataFrame({
                        'time': pd.to_datetime(ts_arr[mask], unit='ns'),
                        'contract': name,
                        return_variable: vals_arr[mask],
                    }))
            if not parts:
                return pd.DataFrame()
            return pd.concat(parts, ignore_index=True).sort_values('time')
    else:
        return _backtest_stacked_python_fallback(
            all_names, pos_ids_unified, pos_ts, pos_vals,
            price_ids_unified, price_ts, price_vals,
            shift, return_variable, sum_contracts, resample,
        )


def _backtest_stacked_python_fallback(
    names, pos_ids, pos_ts, pos_vals,
    price_ids, price_ts, price_vals,
    shift, return_variable, sum_contracts, resample,
):
    """Python fallback: partition + merge_asof + shift + diff per contract."""
    # Partition into per-contract groups using integer IDs
    pos_groups = {}
    for i in range(len(pos_ids)):
        cid = pos_ids[i]
        if cid not in pos_groups:
            pos_groups[cid] = ([], [])
        pos_groups[cid][0].append(pos_ts[i])
        pos_groups[cid][1].append(pos_vals[i])

    price_groups = {}
    for i in range(len(price_ids)):
        cid = price_ids[i]
        if cid not in price_groups:
            price_groups[cid] = ([], [])
        price_groups[cid][0].append(price_ts[i])
        price_groups[cid][1].append(price_vals[i])

    common = set(pos_groups.keys()) & set(price_groups.keys())
    result_dict = {}
    for cid in sorted(common):
        name = names[cid]
        pts = np.array(pos_groups[cid][0], dtype='int64')
        pvals = np.array(pos_groups[cid][1], dtype='float64')
        prts = np.array(price_groups[cid][0], dtype='int64')
        prvals = np.array(price_groups[cid][1], dtype='float64')

        if len(pts) == 0 or len(prts) == 0:
            continue

        pos_idx = pd.to_datetime(pts, unit='ns')
        price_idx = pd.to_datetime(prts, unit='ns')

        s = pd.Series(pvals, index=pos_idx)
        p = pd.Series(prvals, index=price_idx)

        if s.index[0] > p.index[-1] or s.index[-1] < p.index[0]:
            continue

        s = merge_asof(s, p.index)
        if s.empty:
            continue

        r = p.diff()
        pos_shifted = s.shift(shift)
        if return_variable == 'pnl':
            out = pos_shifted * r
        elif return_variable == 'pos':
            out = pos_shifted
        elif return_variable == 'market_return':
            out = r
        else:
            raise ValueError(f'Unknown return variable: {return_variable}')

        out = out.dropna()
        if not out.empty:
            # .astype('datetime64[ns]') first: a us-resolution index (the pandas
            # 3.0 default) astype'd straight to int64 yields MICROseconds, which
            # to_datetime(..., unit='ns') below then reads as nanoseconds and
            # collapses every date to ~1970. Same guard as in _extract_stacked_arrays.
            result_dict[name] = (
                out.index.astype('datetime64[ns]').astype('int64').values,
                out.values,
            )

    if not result_dict:
        return pd.Series(dtype='float64') if sum_contracts else pd.DataFrame()

    if sum_contracts:
        combined_ts = np.concatenate([r[0] for r in result_dict.values()])
        combined_vals = np.concatenate([r[1] for r in result_dict.values()])
        order = np.argsort(combined_ts, kind='mergesort')
        sorted_ts = combined_ts[order]
        sorted_vals = combined_vals[order]
        change = np.empty(len(sorted_ts), dtype=bool)
        change[0] = True
        change[1:] = sorted_ts[1:] != sorted_ts[:-1]
        unique_ts = sorted_ts[change]
        group_starts = np.flatnonzero(change)
        summed_vals = np.add.reduceat(sorted_vals, group_starts)
        result = pd.Series(summed_vals, index=pd.to_datetime(unique_ts, unit='ns'))
        if resample:
            result = result.resample('D').sum().replace(0, np.nan)
        return result
    else:
        parts = []
        for name, (ts, vals) in result_dict.items():
            parts.append(pd.DataFrame({
                'time': pd.to_datetime(ts, unit='ns'),
                'contract': name,
                return_variable: vals,
            }))
        return pd.concat(parts, ignore_index=True).sort_values('time')


def sharpe_per_lag_stacked(
    pos,
    prices_or_returns,
    lookbacks=None,
    resample=True,
    contract_col=None,
    time_col=None,
    pos_value_col=None,
    price_value_col=None,
):
    """Sharpe per execution lag from stacked/long-format DataFrames.

    Uses ``backtest_stacked_multi_shift_summed_core`` — partitioning,
    merge_asof, multi-shift, and cross-contract summation all in Rust.

    Args:
        pos: Stacked positions DataFrame (Polars or pandas).
        prices_or_returns: Stacked prices/returns DataFrame.
        lookbacks: List of shift values to test. Default: standard range.
        resample: If True, resample PnL to daily before computing Sharpe.
        contract_col, time_col, pos_value_col, price_value_col:
            Column names (auto-detected if None).

    Returns:
        pd.Series indexed by shift value, values are Sharpe ratios.
    """
    default_lookbacks = (list(range(-20, -5, 3)) + list(range(-5, 6))
                         + list(range(7, 30, 3)) + list(range(30, 100, 10))
                         + list(range(100, 250, 25)))
    lookbacks = lookbacks if lookbacks is not None else default_lookbacks

    # Resolve columns
    pos_contract_col, pos_time_col, pos_val_col = _resolve_columns(
        pos, contract_col, time_col, pos_value_col,
        ['value', 'position', 'pos', 'weight'], 'pos')
    price_contract_col, price_time_col, price_val_col = _resolve_columns(
        prices_or_returns, contract_col, time_col, price_value_col,
        _PRICE_COL_PRIORITY, 'prices_or_returns')

    # Extract flat arrays
    pos_names, pos_ids, pos_ts, pos_vals = _extract_stacked_arrays(
        pos, pos_contract_col, pos_time_col, pos_val_col)
    price_names, price_ids, price_ts, price_vals = _extract_stacked_arrays(
        prices_or_returns, price_contract_col, price_time_col, price_val_col)

    if len(pos_ts) == 0 or len(price_ts) == 0:
        return pd.Series({i: np.nan for i in lookbacks})

    # Unified name mapping
    all_names = sorted(set(pos_names) | set(price_names))
    name_to_id = {n: i for i, n in enumerate(all_names)}
    pos_id_map = np.array([name_to_id[n] for n in pos_names], dtype='int32')
    price_id_map = np.array([name_to_id[n] for n in price_names], dtype='int32')
    pos_ids_unified = pos_id_map[pos_ids]
    price_ids_unified = price_id_map[price_ids]

    # Detect returns → cumsum per contract
    if _detect_returns_stacked(price_vals, price_ids_unified):
        order = np.lexsort((price_ts, price_ids_unified))
        sorted_ids = price_ids_unified[order]
        sorted_ts = price_ts[order]
        sorted_vals = price_vals[order]
        cumvals = np.empty_like(sorted_vals)
        cur_id = sorted_ids[0]
        cur_sum = 0.0
        for i in range(len(sorted_vals)):
            if sorted_ids[i] != cur_id:
                cur_id = sorted_ids[i]
                cur_sum = 0.0
            cur_sum += sorted_vals[i]
            cumvals[i] = cur_sum
        price_ids_unified = sorted_ids
        price_ts = sorted_ts
        price_vals = cumvals

    if _HAS_RUST:
        result = ebt_rust.backtest_stacked_multi_shift_summed_core(
            all_names, pos_ids_unified, pos_ts, pos_vals,
            price_ids_unified, price_ts, price_vals,
            lookbacks,
        )
        sharpes = {}
        for shift_val, (out_ts, out_vals) in result.items():
            out_ts = np.asarray(out_ts)
            out_vals = np.asarray(out_vals)
            if len(out_ts) == 0:
                sharpes[shift_val] = np.nan
            else:
                pnl = pd.Series(out_vals, index=pd.to_datetime(out_ts, unit='ns'))
                sharpes[shift_val] = sharpe(pnl, resample=resample)
        return pd.Series(sharpes).reindex(lookbacks)

    # Python fallback
    return pd.Series({
        shift_val: backtest_stacked(
            pos, prices_or_returns, shift=shift_val,
            contract_col=contract_col, time_col=time_col,
            pos_value_col=pos_value_col, price_value_col=price_value_col,
        ).pipe(sharpe, resample=resample)
        for shift_val in lookbacks
    })


def up_days(pnl, resample=True):
    if isinstance(pnl, pd.DataFrame):
        pnl_daily = pnl.dropna(how='all').sum(axis=1)
    else:
        pnl_daily = pnl.dropna()
    if resample:
        pnl_daily = _daily_nonflat(pnl_daily)

    vc = np.sign(pnl_daily).value_counts().pipe(lambda x: x / x.sum())
    return float(vc.get(1, 0.0))


def sharpe(pnl, resample=False):
    if resample:
        pnl = _daily_nonflat(pnl)
    return pnl.mean() / pnl.std() * 252 ** 0.5


def risk(pnl, resample=True):
    if resample:
        pnl = _daily_nonflat(pnl)
    return pnl.std() * 252 ** 0.5


def _sharpe_per_col_fast(pnl_df, resample=True):
    """Compute sharpe for each column using numpy daily aggregation — avoids N separate resample calls."""
    if not resample:
        means = np.nanmean(pnl_df.values, axis=0)
        stds = np.nanstd(pnl_df.values, axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sharpes = means / stds * 252 ** 0.5
        return pd.Series(sharpes, index=pnl_df.columns)

    # Daily aggregation with numpy
    ts_ns = _index_to_ns(pnl_df.index)
    day_keys = ts_ns // 86_400_000_000_000
    n_cols = pnl_df.shape[1]

    vals = pnl_df.values
    has_cell = ~np.isnan(vals)
    vals = np.where(has_cell, vals, 0.0)

    # Sum per day per column. Rows are in time order, so the days are contiguous segments and
    # add.reduceat does it in one pass; add.at (unbuffered, per element) only for an unsorted index.
    if len(day_keys) and np.all(day_keys[1:] >= day_keys[:-1]):
        starts = np.flatnonzero(np.r_[True, day_keys[1:] != day_keys[:-1]])
        daily = np.add.reduceat(vals, starts, axis=0)
        has = np.add.reduceat(has_cell.astype(np.int32), starts, axis=0)
    else:
        unique_days, inverse = np.unique(day_keys, return_inverse=True)
        daily = np.zeros((len(unique_days), n_cols), dtype=np.float64)
        np.add.at(daily, inverse, vals)
        has = np.zeros((len(unique_days), n_cols), dtype=np.int64)
        np.add.at(has, inverse, has_cell.astype(np.int64))

    # Drop non-trading and flat days (see FLAT_DAY_RTOL)
    daily[_flat_mask_2d(daily, has > 0)] = np.nan

    means = np.nanmean(daily, axis=0)
    stds = np.nanstd(daily, axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpes = means / stds * 252 ** 0.5
    return pd.Series(sharpes, index=pnl_df.columns)


def turnover(p, resample=True, weights=None):
    """Daily turnover series whose mean is the pooled turnover of the sample.

    Per day d: TV_d = sum(|Δpos|) over the day's bars and names, E_d = mean(|pos|) over the
    day's bars (summed over names). The series is TV_d / mean_d(E_d), so its mean equals
    sum(TV_d) / sum(E_d): a ratio of sums over the whole sample. A mean of per-day ratios
    TV_d / E_d would let a day on which a book of few names sits near zero (a sign change)
    dominate the figure; pooling makes the result frequency-independent and stable there.

    weights: forward bar counts of a reduced price grid (ebt.simulate.grid), one per row.
        They weight the EXPOSURE mean only — a trade is an event and counts once whatever
        the row stands for. None (the default) is the unweighted computation, unchanged.
    """
    vals = p.values
    diff_abs = np.abs(np.diff(vals, axis=0, prepend=np.full((1, vals.shape[1]), np.nan)))
    p_abs = np.abs(vals)
    num_per_bar = np.nansum(diff_abs, axis=1)
    den_per_bar = np.nansum(p_abs, axis=1)
    if weights is not None and not resample:
        raise ValueError('turnover: weights need resample=True')
    if resample:
        num_s = pd.Series(np.nan_to_num(num_per_bar, nan=0.0), index=p.index)
        den_s = pd.Series(np.nan_to_num(den_per_bar, nan=0.0), index=p.index)
        daily_tv = num_s.resample('B').sum()
        if weights is None:
            daily_mean_exposure = den_s.resample('B').mean()
        else:
            w = np.asarray(weights, dtype=np.float64)
            w_s = pd.Series(w, index=p.index)
            daily_mean_exposure = (den_s * w).resample('B').sum() / w_s.resample('B').sum()
        valid = daily_mean_exposure > 0
        result = pd.Series(np.nan, index=daily_tv.index)
        result[valid] = daily_tv[valid] / daily_mean_exposure[valid].mean()
        return result.dropna()
    else:
        ok = den_per_bar > 0
        mean_exposure = den_per_bar[ok].mean() if ok.any() else np.nan
        t_vals = np.where(ok, num_per_bar / mean_exposure, np.nan)
        return pd.Series(t_vals, index=p.index)


def lptp(signal, weights=None):
    """Long-position-to-total-position ratio: the fraction of total (risk-weighted)
    exposure held long, pooled over every row and name:

        0.5 + sum(pos) / (2 * sum(|pos|))

    1.0 for a long-only book, 0.0 for a short-only book, 0.5 for a dollar-neutral one.
    Pooled rather than a mean of per-row ratios: a per-row ratio weights an almost-empty
    row as much as a fully invested one, and a flat row whose only content is float residue
    (e.g. +-1e-15) would score an arbitrary ~0.5. Returns a float.
    weights: forward bar counts of a reduced grid (see turnover); None = unweighted.
    """
    return _numpy_lptp_mean(np.asarray(signal, dtype='float64'), weights=weights)


try:
    from numba import jit as _numba_jit

    @_numba_jit(nopython=True, cache=True)
    def _ncvar_prep(pos, returns):
        """Per-day normalized positions u_k = s_k / std_k (zero off the min_periods mask),
        |u_k|^2, |s_k|^2 and c2_k, with the diagonal/std kept on the ORIGINAL element
        recursion (bit-identical std to the N x N version)."""
        T, N = pos.shape
        post_a = 1 - 1 / 130
        post_b = 1 - post_a
        min_periods = 60
        start_index_returns = np.argmax(returns != 0, axis=0)
        first = start_index_returns.min()
        diag = np.ones(N)
        c2 = 1.0
        U = np.zeros((T, N))
        usq = np.zeros(T)
        ssum = np.zeros(T)
        c2s = np.zeros(T)
        valid = np.zeros(T, dtype=np.bool_)
        for k in range(first, T):
            r = returns[k, :]
            for i in range(N):
                diag[i] = post_b * (r[i] * r[i]) + post_a * diag[i]
            c2 = post_b * c2
            c2s[k] = c2
            n_used = 0
            for i in range(N):
                # only assets with at least min_periods of history by row k
                if start_index_returns[i] + min_periods <= k:
                    std_i = np.sqrt((diag[i] + c2) / 2) + 1e-8
                    ui = pos[k, i] / std_i
                    U[k, i] = ui
                    usq[k] += ui * ui
                    ssum[k] += pos[k, i] * pos[k, i]
                    n_used += 1
            valid[k] = n_used > 0
        return U, usq, ssum, c2s, valid, first

    @_numba_jit(nopython=True, cache=True)
    def _ncvar_finish(G, usq, ssum, c2s, valid, first):
        """ncvar_k from G[k, m] = u_k . r_m: u' covar u = post_a^n |u|^2 + post_b *
        sum_{m<=k} post_a^(k-m) G[k,m]^2, risk = (u' covar u + c2 |u|^2) / 2."""
        T = G.shape[0]
        post_a = 1 - 1 / 130
        post_b = 1 - post_a
        ncvars = np.full(T, np.nan)
        for k in range(first, T):
            if not valid[k]:
                continue
            s = 0.0
            w = 1.0
            for m in range(k, first - 1, -1):
                g = G[k, m]
                s += w * g * g
                w *= post_a
            quad = post_a ** (k - first + 1) * usq[k] + post_b * s
            risk = 0.5 * quad + 0.5 * c2s[k] * usq[k]
            if risk > 0:
                ncvars[k] = np.sqrt(ssum[k] / risk)
        return np.nanmean(ncvars)

    @_numba_jit(nopython=True, cache=True)
    def _ncvar_kernel_dense(pos, returns):
        """The original N x N recursion, kept for inputs with non-finite values (where the
        original's nan_to_num(C) step defines the result and the factored form cannot)."""
        covar = np.eye(pos.shape[1])
        covar_2 = covar.copy()
        ncvars = np.full(pos.shape[0], np.nan)

        post_a = 1 - 1 / 130
        post_b = 1 - post_a
        min_periods = 60

        start_index_returns = np.argmax(returns != 0, axis=0)

        for k in range(pos.shape[0]):
            if np.any(start_index_returns <= k):
                covar = post_b * np.outer(returns[k, :], returns[k, :]) + post_a * covar
                covar_2 = post_b * covar_2
                C = (covar + covar_2) / 2
                stds = np.sqrt(np.diag(C)) + 1e-8
                C = C / np.outer(stds, stds)
                C = np.nan_to_num(C)

                mask = start_index_returns + min_periods <= k
                s = pos[k, mask]
                risk = np.dot(np.dot(s, C[mask, :][:, mask]), s)
                if risk > 0:
                    ncvars[k] = np.sqrt(np.sum(s ** 2) / risk)

        return np.nanmean(ncvars)

    def _ncvar_kernel(pos, returns):
        """Normalized covariance ratio of the book (EWMA correlation, span ~130 days).

        Same quantity as the original N x N recursion — C = ((covar + c2*I)/2) / (std std^T),
        std = sqrt(diag) + 1e-8, risk = s' C[mask,mask] s — but never materializes covar:
        after n updates covar = post_a^n I + post_b * sum_m post_a^(k-m) r_m r_m', so with
        u = s/std the quadratic form only needs the day x day inner products G = U R'
        (one BLAS matrix product) plus O(T^2) scalar work.

        Accuracy: agrees with the recursion to float rounding on well-conditioned inputs. The
        one regime where the two genuinely part company is a DEGENERATE covariance — duplicated
        return series held in exactly offsetting sizes over a long history — because there the
        recursion's risk term is the residue of a cancelling sum (its value is float noise, and
        ncvar comes out in the hundreds or thousands rather than near 1) while this form sums
        non-negative terms; the factored value is the better-conditioned of the two. Duplicate
        price columns are a data-quality fault,
        not a strategy property — if ncvar comes back orders of magnitude above sqrt(N),
        check the universe for duplicated series rather than trusting either number.
        """
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        returns = np.ascontiguousarray(returns, dtype=np.float64)
        if not _ncvar_prefer_factored(pos.shape[0], pos.shape[1]):
            # Long history, few assets: the recursion's N x N is the small object and the
            # day-by-day matrix would be the big one.
            return _ncvar_kernel_dense(pos, returns)
        if pos.shape[0] == 0 or pos.shape[1] == 0:
            # N == 0 matches the original (nanmean of an all-NaN vector); T == 0 raised
            # ZeroDivisionError inside the original kernel, which kf.ncvar never reaches
            # (it raises IndexError first).
            return np.nan
        if not (np.isfinite(returns).all() and np.isfinite(pos).all()
                and np.abs(returns).max() < 1e150):
            return _ncvar_kernel_dense(pos, returns)
        U, usq, ssum, c2s, valid, first = _ncvar_prep(pos, returns)
        G = U @ returns.T
        return _ncvar_finish(G, usq, ssum, c2s, valid, int(first))

    @_numba_jit(nopython=True, cache=True)
    def _same_sign_fraction(vals_T):
        """mean over non-empty columns of mean_i[sign(v_i) == sign(v_{i-1})] over the column's
        non-NaN values (the first one compares False), NaN if every column is empty —
        exactly ``df.apply(lambda s: (np.sign(s.dropna()) == ...shift()).mean()).mean()``."""
        N, T = vals_T.shape
        total = 0.0
        ncols = 0
        for j in range(N):
            prev = 0.0
            eq = 0
            cnt = 0
            for i in range(T):
                v = vals_T[j, i]
                if v != v:
                    continue
                sg = np.sign(v)
                if cnt > 0 and sg == prev:
                    eq += 1
                prev = sg
                cnt += 1
            if cnt > 0:
                total += eq / cnt
                ncols += 1
        return total / ncols if ncols > 0 else np.nan

    @_numba_jit(nopython=True, nogil=True, cache=True)
    def _decile_pnl_block(pos_2d, ret_2d, edges, i0, i1, out):
        """out[i, b] += pos*ret for every cell of rows [i0, i1) whose position falls in
        decile b, skipping cells whose product is NaN (nansum semantics, which is also what
        makes a NaN position or a 0*inf product contribute nothing). ``edges`` are the n-1
        interior quantile boundaries, so b is ``np.digitize(pos, edges)``. One pass instead
        of ten masked passes over the frame plus a materialized bin array."""
        ncols = pos_2d.shape[1]
        nedge = edges.shape[0]
        for i in range(i0, i1):
            for j in range(ncols):
                x = pos_2d[i, j] * ret_2d[i, j]
                if x != x:
                    continue
                p = pos_2d[i, j]
                b = 0
                while b < nedge and edges[b] <= p:
                    b += 1
                out[i, b] += x

    @_numba_jit(nopython=True, nogil=True, cache=True)
    def _decile_pnl_block_cm(pos_cm, ret_cm, edges, i0, i1, out):
        """``_decile_pnl_block`` for COLUMN-MAJOR frames, given the (ncols, nrows) transposed
        view: column j of rows [i0, i1) is contiguous, so this walks memory in order instead
        of forcing a full C-order copy of the frame. Cells are added to out[i, b] in the same
        j-ascending order per row, so the sums are bit-identical to the row-major kernel."""
        ncols = pos_cm.shape[0]
        nedge = edges.shape[0]
        for j in range(ncols):
            for i in range(i0, i1):
                x = pos_cm[j, i] * ret_cm[j, i]
                if x != x:
                    continue
                p = pos_cm[j, i]
                b = 0
                while b < nedge and edges[b] <= p:
                    b += 1
                out[i, b] += x

    @_numba_jit(nopython=True, nogil=True, cache=True)
    def _ewm_pandas_col(vals, alpha, out):
        """pandas ``ewm(span, adjust=True, ignore_na=False, min_periods=0).mean()`` on one
        contiguous column, operation for operation (bit-identical to pandas 3.0). Like
        pandas' window prep, +/-inf counts as a missing value."""
        old_wt_factor = 1.0 - alpha
        new_wt = 1.0
        weighted = vals[0]
        if not np.isfinite(weighted):
            weighted = np.nan
        nobs = 1 if weighted == weighted else 0
        out[0] = weighted if nobs >= 1 else np.nan
        old_wt = 1.0
        for i in range(1, vals.shape[0]):
            cur = vals[i]
            is_obs = np.isfinite(cur)
            if is_obs:
                nobs += 1
            if weighted == weighted:
                old_wt *= old_wt_factor
                if is_obs:
                    if weighted != cur:
                        weighted = old_wt * weighted + new_wt * cur
                        weighted = weighted / (old_wt + new_wt)
                    old_wt += new_wt
            elif is_obs:
                weighted = cur
            out[i] = weighted if nobs >= 1 else np.nan

    @_numba_jit(nopython=True, nogil=True, cache=True)
    def _ewm_holding_block(vals_T, vol_T, span, j0, j1, num_out, den_out):
        """Accumulate per-bar nansum|d(ewm(pos)*vol)| and nansum|ewm(pos)*vol| over the
        columns [j0, j1) of COLUMN-MAJOR (N, T) inputs into num_out/den_out (length T),
        without materializing the smoothed frame. span <= 0 = no smoothing. GIL-free, so
        blocks run on plain threads (no numba parallel layer: fork-safe, thread-safe)."""
        T = vals_T.shape[1]
        alpha = 2.0 / (span + 1.0) if span > 0 else 0.0
        sm = np.empty(T)
        for j in range(j0, j1):
            if span > 0:
                _ewm_pandas_col(vals_T[j], alpha, sm)
            else:
                for i in range(T):
                    sm[i] = vals_T[j, i]
            prev = np.nan
            for i in range(T):
                x = sm[i] * vol_T[j, i]
                if x == x:
                    den_out[i] += abs(x)
                    if prev == prev:
                        d = abs(x - prev)
                        if d == d:
                            num_out[i] += d
                prev = x
except ImportError:
    _ncvar_kernel = None
    _ewm_holding_block = None
    _same_sign_fraction = None
    _decile_pnl_block = None
    _decile_pnl_block_cm = None


def ncvar(pos, returns):
    if _ncvar_kernel is None:
        raise ImportError('numba is required for ncvar')

    # Fast path: skip pandas align if indices already match
    if not (pos.index.equals(returns.index) and pos.columns.equals(returns.columns)):
        # align columns too — row-only alignment would pair asset i's
        # positions with asset j's prices on reordered columns
        pos, returns = pos.align(returns, join='inner')

    # Prices arrive raw (NaN at closed bars). ffill before diff — fillna(0)+diff
    # would fabricate ±price-level jumps at every session gap.
    ret_vals = _numpy_ffill(returns.values.astype(np.float64, copy=True))
    ret_diff = np.diff(ret_vals, axis=0, prepend=0)
    ret_diff[0] = 0
    ret_diff = np.nan_to_num(ret_diff)

    # Subsample to daily resolution — ncvar is a low-freq diversification metric. Chosen by
    # whether the frame is INTRADAY (some day has more than one row), not by row count:
    # a reduced grid (ebt.simulate.grid) of a long history can have few rows and would otherwise
    # fall to the per-row EWMA.
    ts_ns = _index_to_ns(pos.index)
    day_keys = ts_ns // 86_400_000_000_000
    if len(pos) > 1 and np.any(day_keys[1:] == day_keys[:-1]):
        pos_vals = pos.ffill().fillna(0).values

        if len(day_keys) > 1 and np.all(day_keys[1:] >= day_keys[:-1]):
            # sorted index -> each day is one contiguous block: per-day sums by reduceat
            # (pairwise within a block rather than add.at's strict row order, so ~1e-13
            # relative apart) and the day's LAST position row, which is what the
            # repeated-index assignment kept.
            starts = np.flatnonzero(np.concatenate(([True], day_keys[1:] != day_keys[:-1])))
            ret_daily = np.add.reduceat(ret_diff, starts, axis=0)
            last = np.concatenate((starts[1:] - 1, [len(day_keys) - 1]))
            pos_daily = pos_vals[last]
        else:
            unique_days, inverse = np.unique(day_keys, return_inverse=True)
            n_days = len(unique_days)
            n_cols = pos_vals.shape[1]

            ret_daily = np.zeros((n_days, n_cols), dtype=np.float64)
            np.add.at(ret_daily, inverse, ret_diff)

            pos_daily = np.empty((n_days, n_cols), dtype=np.float64)
            pos_daily[inverse] = pos_vals

        return _ncvar_kernel(pos_daily, ret_daily)

    # small-data branch: same diffed returns as the daily branch (not raw
    # price LEVELS), so the two branches agree
    pos = pos.ffill().fillna(0)
    return _ncvar_kernel(pos.astype(float).values, ret_diff)


# Approximate per-unit costs (s): the factored ncvar form runs one
# T x T x N BLAS product plus a T x T scan, the recursion runs T rank-1 updates of an
# N x N matrix. Which is cheaper depends entirely on the book's shape.
_NCVAR_C_GEMM = 1.1e-11
_NCVAR_C_SCAN = 2.3e-9
_NCVAR_C_DENSE = 6.0e-9
_NCVAR_MAX_G_BYTES = 2 << 30


def _ncvar_prefer_factored(n_rows, n_cols):
    """Whether the factored (BLAS) ncvar form beats the N x N recursion for this shape."""
    if n_rows == 0 or n_cols == 0:
        return True  # empty: the wrapper answers directly
    if n_rows * n_rows * 8 > _NCVAR_MAX_G_BYTES:
        return False
    factored = n_rows * n_rows * (n_cols * _NCVAR_C_GEMM + _NCVAR_C_SCAN)
    dense = n_rows * n_cols * n_cols * _NCVAR_C_DENSE
    return factored < dense


def _numpy_ffill(arr):
    """Forward-fill NaN values along axis=0. Vectorized numpy (GIL-free)."""
    mask = np.isnan(arr)
    # Build index array: for each cell, store the row index of the last non-NaN value
    idx = np.where(~mask, np.arange(arr.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return arr[idx, np.arange(arr.shape[1])]


def _numpy_daily_sum(vals, day_keys, unique_days):
    """Sum values per day using bincount. Days without data and flat days (FLAT_DAY_RTOL) are NaN."""
    inverse = np.searchsorted(unique_days, day_keys)
    daily = np.bincount(inverse, weights=vals, minlength=len(unique_days)).astype(np.float64)
    has = np.bincount(inverse, minlength=len(unique_days)) > 0
    daily[_flat_mask(daily, has)] = np.nan
    return daily


def _numpy_sharpe(vals, day_keys, unique_days, annualization=252 ** 0.5):
    """Sharpe ratio with daily resampling, pure numpy."""
    daily = _numpy_daily_sum(vals, day_keys, unique_days)
    valid = ~np.isnan(daily)
    daily = daily[valid]
    if len(daily) < 2:
        return np.nan
    m = np.mean(daily)
    s = np.std(daily, ddof=1)
    if s == 0:
        return np.nan
    return float(m / s * annualization)


def _numpy_risk(vals, day_keys, unique_days, annualization=252 ** 0.5):
    """Annualized risk (std) with daily resampling, pure numpy."""
    daily = _numpy_daily_sum(vals, day_keys, unique_days)
    valid = ~np.isnan(daily)
    daily = daily[valid]
    if len(daily) < 2:
        return np.nan
    return float(np.std(daily, ddof=1) * annualization)


def _daily_tv_exposure(pos_vals, day_keys, unique_days, weights=None):
    """Per-day traded volume TV_d = sum(|Δpos|) and mean exposure E_d = weighted mean(|pos|)
    over the day's rows, pure numpy. Returns ``(tv, exposure)`` over the days with exposure
    (both empty when none has). ``weights`` are forward bar counts (ebt.simulate.grid)
    weighting the exposure mean and the bar count only; None = every row counts once.
    """
    diff_abs = np.abs(np.diff(pos_vals, axis=0, prepend=np.full((1, pos_vals.shape[1]), np.nan)))
    p_abs = np.abs(pos_vals)
    num = np.nansum(diff_abs, axis=1)  # total |diff| per bar across columns
    den = np.nansum(p_abs, axis=1)     # total |pos| per bar across columns
    inverse = np.searchsorted(unique_days, day_keys)
    w = np.ones(len(day_keys)) if weights is None else np.asarray(weights, dtype=np.float64)
    daily_tv = np.bincount(inverse, weights=np.nan_to_num(num, nan=0.0), minlength=len(unique_days)).astype(np.float64)
    daily_pos_sum = np.bincount(inverse, weights=np.nan_to_num(den, nan=0.0) * w, minlength=len(unique_days)).astype(np.float64)
    daily_n_bars = np.bincount(inverse, weights=w, minlength=len(unique_days)).astype(np.float64)
    valid = daily_n_bars > 0
    daily_mean_exposure = np.zeros_like(daily_pos_sum)
    daily_mean_exposure[valid] = daily_pos_sum[valid] / daily_n_bars[valid]
    valid2 = daily_mean_exposure > 0
    return daily_tv[valid2], daily_mean_exposure[valid2]


def _numpy_turnover_mean(pos_vals, day_keys, unique_days, weights=None):
    """Pooled turnover sum(TV_d) / sum(E_d) over business days, pure numpy (= ``turnover(...).mean()``)."""
    tv, exposure = _daily_tv_exposure(pos_vals, day_keys, unique_days, weights)
    if len(tv) == 0:
        return np.nan
    return float(tv.sum() / exposure.sum())


def _numpy_holding_time(pos_vals, day_keys, unique_days, weights=None):
    """Holding time in days: 1 / pooled turnover (see _daily_tv_exposure, ``holding_time``)."""
    tv, exposure = _daily_tv_exposure(pos_vals, day_keys, unique_days, weights)
    if len(tv) == 0:
        return np.inf
    mean_t = float(tv.sum() / exposure.sum())
    return 1.0 / mean_t if mean_t > 0 else np.inf


def holding_time(pos, resample=True, weights=None):
    """Holding time in days: 1 / pooled turnover = sum_d(E_d) / sum_d(TV_d), with
    TV_d the day's traded volume sum(|Δpos|) and E_d its mean exposure mean(|pos|) (see
    ``turnover``). A ratio of sums over the sample, so days on which the book sits near
    zero do not dominate.
    weights: forward bar counts of a reduced grid (see turnover); None = unweighted.
    """
    vals = pos.values
    diff_abs = np.abs(np.diff(vals, axis=0, prepend=np.full((1, vals.shape[1]), np.nan)))
    p_abs = np.abs(vals)
    num_per_bar = np.nansum(diff_abs, axis=1)
    den_per_bar = np.nansum(p_abs, axis=1)
    return _holding_from_bars(num_per_bar, den_per_bar, pos.index, resample, weights=weights)


def _holding_from_bars(num_per_bar, den_per_bar, index, resample=True, weights=None):
    """holding_time from the per-bar total-variation and exposure sums."""
    if weights is not None and not resample:
        raise ValueError('holding_time: weights need resample=True')
    if resample:
        num_s = pd.Series(np.nan_to_num(num_per_bar, nan=0.0), index=index)
        den_s = pd.Series(np.nan_to_num(den_per_bar, nan=0.0), index=index)
        # Total variation per day
        daily_tv = num_s.resample('B').sum()
        # Mean exposure per day (mean of per-bar |pos|, NOT sum; bar-count weighted on a reduced grid)
        if weights is None:
            daily_mean_exposure = den_s.resample('B').mean()
        else:
            w = np.asarray(weights, dtype=np.float64)
            daily_mean_exposure = (den_s * w).resample('B').sum() / pd.Series(w, index=index).resample('B').sum()
        valid = daily_mean_exposure > 0
        if not valid.any():
            return np.inf
        mean_t = daily_tv[valid].sum() / daily_mean_exposure[valid].sum()
    else:
        ok = den_per_bar > 0
        mean_t = np.nan_to_num(num_per_bar[ok], nan=0.0).sum() / den_per_bar[ok].sum() if ok.any() else np.nan
    return 1.0 / mean_t if mean_t > 0 and not np.isnan(mean_t) else np.inf


_POOL_MIN_WORK = 2_000_000  # cells; below this the thread pool costs more than the kernel


def _kernel_threads(n_items):
    """Threads for a block-dispatched GIL-free kernel: min(32, cores) by default
    (EBT_KERNEL_THREADS overrides; a malformed value is ignored). Memory-bound kernels gain
    nothing past a few dozen threads and lose badly to oversubscription on a shared machine."""
    try:
        cap = int(os.environ.get('EBT_KERNEL_THREADS', '') or 0)
    except ValueError:
        _LOGGER.warning('Ignoring malformed EBT_KERNEL_THREADS=%r',
                        os.environ.get('EBT_KERNEL_THREADS'))
        cap = 0
    if cap <= 0:
        cap = min(32, os.cpu_count() or 1)
    return max(1, min(cap, n_items))


def _run_blocks(fn, bounds, *args, work=None):
    """fn(j0, j1, b, *args) for each (j0, j1) block, on GIL-free threads.

    ``work`` is the total element count; below _POOL_MIN_WORK the blocks run inline, since
    spinning up a pool costs more than the kernel does on a small frame.
    """
    nthreads = 1 if (work is not None and work < _POOL_MIN_WORK) else _kernel_threads(len(bounds))
    if nthreads <= 1:
        for b, (j0, j1) in enumerate(bounds):
            fn(j0, j1, b, *args)
        return
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(lambda t: fn(t[1][0], t[1][1], t[0], *args), enumerate(bounds)))


def _bday_keys(day_keys):
    """Business-day keys: weekend bars fold into the preceding Friday, matching the pandas
    ``resample('B')`` in turnover/holding_time. day_keys count days since 1970-01-01 (a
    Thursday), so (day_keys + 3) % 7 is 0=Mon..6=Sun."""
    dow = (day_keys + 3) % 7
    return day_keys - np.where(dow >= 5, dow - 4, 0)


def _holding_from_bars_numpy(num_per_bar, den_per_bar, day_keys, unique_days):
    """``_holding_from_bars(..., resample=True)`` on day keys instead of an index: per day
    TV_d = sum|d pos| and E_d = mean|pos|, holding = sum(E_d) / sum(TV_d) over the days with exposure."""
    inverse = np.searchsorted(unique_days, day_keys)
    n = len(unique_days)
    daily_tv = np.bincount(inverse, weights=np.nan_to_num(num_per_bar, nan=0.0), minlength=n)
    daily_pos_sum = np.bincount(inverse, weights=np.nan_to_num(den_per_bar, nan=0.0), minlength=n)
    daily_n_bars = np.bincount(inverse, minlength=n).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_mean_exposure = np.where(daily_n_bars > 0, daily_pos_sum / daily_n_bars, 0.0)
    valid = daily_mean_exposure > 0
    if not valid.any():
        return np.inf
    mean_t = float(daily_tv[valid].sum() / daily_mean_exposure[valid].sum())
    return 1.0 / mean_t if mean_t > 0 else np.inf


def _holding_curve_fast_arrays(vals_T, vol_T, spans, day_keys, unique_days, nblocks=64):
    """holding_curve_fast on arrays: ``vals_T``/``vol_T`` are column-major (N, T) float64,
    ``day_keys`` business-day keys per bar (see _bday_keys). Returns a list of holding times,
    or None without numba. Same fused kernel and the same day aggregation as the pandas
    path, so one_pager's numpy fast path and its full path put the holding curve on the
    same axis."""
    if _ewm_holding_block is None:
        return None
    N, T = vals_T.shape
    if N == 0 or T == 0:
        return None
    nblocks = int(max(1, min(nblocks, N)))
    bs = -(-N // nblocks)
    bounds = [(j0, min(N, j0 + bs)) for j0 in range(0, N, bs)]
    out = []
    for span in spans:
        sp = 0.0 if span is None else float(span)
        num_p = np.zeros((len(bounds), T))
        den_p = np.zeros((len(bounds), T))

        def _block(j0, j1, b, sp=sp, num_p=num_p, den_p=den_p):
            _ewm_holding_block(vals_T, vol_T, sp, j0, j1, num_p[b], den_p[b])

        _run_blocks(_block, bounds, work=N * T)
        out.append(_holding_from_bars_numpy(num_p.sum(axis=0), den_p.sum(axis=0), day_keys, unique_days))
    return out


def holding_curve_fast(pos_ff, volatility, spans, resample=True, nblocks=64):
    """holding_time of ``pos_ff.ewm(span).mean() * volatility`` for every span, each in one
    fused column-major pass (no smoothed frame materialized). ``spans`` entries <= 0 or None
    mean no smoothing. Returns a list of holding times; None when numba is unavailable or
    the input is empty (callers fall back to the pandas path). The block count is fixed so
    the per-bar sums are the same regardless of the thread count."""
    if _ewm_holding_block is None:
        return None
    # float32/float16 positions: pandas would run the EWM in the input's own precision, so
    # keep those on the pandas path rather than silently returning a float64 answer.
    if any(dt != np.float64 for dt in np.atleast_1d(pos_ff.dtypes.values)):
        return None
    if volatility is not None and not (volatility.index.equals(pos_ff.index)
                                       and volatility.columns.equals(pos_ff.columns)):
        # The pandas path multiplies by LABEL (outer join). Labels present only in
        # ``volatility`` would add all-NaN rows to the product and change the daily
        # exposure mean, so only a subset can be realigned; anything else -> slow path.
        # pandas' aligned product SORTS a non-identical index, which the positional kernel
        # cannot reproduce, so a non-monotonic index also falls back.
        if not (pos_ff.index.is_unique and pos_ff.columns.is_unique
                and pos_ff.index.is_monotonic_increasing
                and volatility.index.is_unique and volatility.columns.is_unique
                and volatility.index.isin(pos_ff.index).all()
                and volatility.columns.isin(pos_ff.columns).all()):
            return None
        volatility = volatility.reindex(index=pos_ff.index, columns=pos_ff.columns)
    vals_T = np.ascontiguousarray(pos_ff.to_numpy(dtype=np.float64).T)
    N, T = vals_T.shape
    if N == 0 or T == 0:
        return None
    if volatility is None:
        vol_T = np.ones_like(vals_T)
    else:
        vol_T = np.ascontiguousarray(volatility.to_numpy(dtype=np.float64).T)
    nblocks = int(max(1, min(nblocks, N)))
    bs = -(-N // nblocks)
    bounds = [(j0, min(N, j0 + bs)) for j0 in range(0, N, bs)]
    out = []
    for span in spans:
        sp = 0.0 if span is None else float(span)
        num_p = np.zeros((len(bounds), T))
        den_p = np.zeros((len(bounds), T))

        def _block(j0, j1, b, sp=sp, num_p=num_p, den_p=den_p):
            _ewm_holding_block(vals_T, vol_T, sp, j0, j1, num_p[b], den_p[b])

        _run_blocks(_block, bounds, work=N * T)
        out.append(_holding_from_bars(num_p.sum(axis=0), den_p.sum(axis=0), pos_ff.index, resample))
    return out


def _numpy_lptp_mean(pos_vals, weights=None):
    """Pooled lptp (see ``lptp``), pure numpy: 0.5 + sum(pos) / (2 * sum(|pos|)) over all
    cells; NaN when the book has no exposure at all. ``weights`` (forward bar counts of a
    reduced grid) weight each row's contribution; None = every row once."""
    if weights is None:
        tot = float(np.nansum(pos_vals))
        abs_tot = float(np.nansum(np.abs(pos_vals)))
    else:
        w = np.asarray(weights, dtype=np.float64)
        tot = float(np.nansum(np.nansum(pos_vals, axis=1) * w))
        abs_tot = float(np.nansum(np.nansum(np.abs(pos_vals), axis=1) * w))
    if abs_tot == 0:
        return np.nan
    return 0.5 + tot / (2.0 * abs_tot)


def _numpy_bias_sharpes_weighted(pos_vals, asset_vol, px_v, weights, day_keys, unique_days, block=256):
    """'sharpe exbias' and 'sharpe bias' of a realised position on a (reduced) price grid.

    Mirrors ebt_rust variant_sharpes_matrix with shift=0 — the risk position pos*vol carried
    as-of from the trade rows, demeaned per column over the rows from the column's first realised
    row, PnL against the risk-scaled price ``px_v`` (cumsum of dP/vol; on a reduced grid built on
    the FULL grid by ebt.simulate.grid), daily sums, the flat-day rule — except that the column
    mean is weighted by the forward bar counts ``weights`` (None = every row once, which equals
    the Rust kernel on a full grid). Returns (exbias, bias). NaN vol propagates (no 1.0 stand-in).
    """
    pos_vals = np.asarray(pos_vals, dtype=np.float64)
    asset_vol = np.asarray(asset_vol, dtype=np.float64)
    px_v = np.asarray(px_v, dtype=np.float64)
    n, n_cols = pos_vals.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    rows_idx = np.arange(n)
    # Column blocks: the routine is column-separable up to the per-row sums, and its ~12
    # (rows x cols) temporaries would otherwise peak at >10x the frame inside every
    # parallel worker. The daily aggregation only needs the row sums.
    row_ex = np.zeros(n)
    row_bias = np.zeros(n)
    fin_ex = np.zeros(n, dtype=bool)
    fin_bias = np.zeros(n, dtype=bool)
    for j0 in range(0, n_cols, block):
        j1 = min(j0 + block, n_cols)
        vol = asset_vol[:, j0:j1]
        vv = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
        pv = pos_vals[:, j0:j1] * vv
        px = px_v[:, j0:j1]
        asof = _numpy_ffill(pv)                               # held risk position, as-of the trade rows
        has_any = ~np.isnan(pv).all(axis=0)
        first = np.where(has_any, np.argmax(~np.isnan(pv), axis=0), n)
        active = (rows_idx[:, None] >= first[None, :]) & ~np.isnan(px)
        a = np.where(active, asof, np.nan)
        ok = ~np.isnan(a)                                     # active rows with a held risk position
        wsum = (w[:, None] * ok).sum(axis=0)
        m = np.where(wsum > 0, (np.where(ok, a, 0.0) * w[:, None]).sum(axis=0)
                     / np.where(wsum > 0, wsum, 1.0), 0.0)
        # return into each row from the column's previous non-NaN risk price (the kernel's
        # per-column diff over its non-NaN timeline, incl. the return into the first realised row)
        pxf = _numpy_ffill(px)
        prev = np.empty_like(pxf)
        prev[0] = np.nan
        prev[1:] = pxf[:-1]
        ret = np.where(active, px - prev, np.nan)
        pnl_bias = m[None, :] * ret
        pnl_ex = a * ret - pnl_bias
        for p2, rs, fin in ((pnl_ex, row_ex, fin_ex), (pnl_bias, row_bias, fin_bias)):
            finite = ~np.isnan(p2)
            fin |= finite.any(axis=1)
            rs += np.where(finite, p2, 0.0).sum(axis=1)
    inverse = np.searchsorted(unique_days, day_keys)

    def _sharpe(rs, rows):
        d = np.bincount(inverse[rows], weights=rs[rows], minlength=len(unique_days)).astype(np.float64)
        has = np.bincount(inverse[rows], minlength=len(unique_days)) > 0
        d = d[~_flat_mask(d, has)]
        if len(d) < 2:
            return np.nan
        s = np.std(d, ddof=1)
        return float(np.mean(d) / s * 252 ** 0.5) if s > 0 else np.nan

    return _sharpe(row_ex, fin_ex), _sharpe(row_bias, fin_bias)


def _numpy_up_days(pnl_sum_vals, day_keys, unique_days):
    """Fraction of positive days, pure numpy."""
    daily = _numpy_daily_sum(pnl_sum_vals, day_keys, unique_days)
    valid = ~np.isnan(daily)
    daily = daily[valid]
    if len(daily) == 0:
        return np.nan
    signs = np.sign(daily)
    n_pos = np.sum(signs > 0)
    return float(n_pos / len(daily))


def _variant_sharpes_rust(pos, prices, shift, resample):
    """Compute exbias/bias/long/short sharpes in one Rust call."""
    pos_df = pos if type(pos) == pd.DataFrame else pos.to_frame()
    prices_df = prices if type(prices) == pd.DataFrame else prices.to_frame()
    pos_df = _align_tz(pos_df, prices_df)

    common_cols = pos_df.columns.intersection(prices_df.columns)
    if common_cols.empty:
        return {'exbias': np.nan, 'bias': np.nan, 'long': np.nan, 'short': np.nan}

    pos_aligned = pos_df[common_cols]
    prices_aligned = prices_df[common_cols]

    pos_ts = _index_to_ns(pos_aligned.index)
    price_ts = _index_to_ns(prices_aligned.index)
    pos_data = np.ascontiguousarray(pos_aligned.values.ravel(order='F'), dtype='float64')
    price_data = np.ascontiguousarray(prices_aligned.values.ravel(order='F'), dtype='float64')

    result = ebt_rust.variant_sharpes_core(
        pos_ts, pos_data, price_ts, price_data, shift, bool(resample),
        len(pos_aligned), len(prices_aligned), len(common_cols), **_rust_flat_kw()
    )
    arr = np.asarray(result)
    return {'exbias': float(arr[0]), 'bias': float(arr[1]), 'long': float(arr[2]), 'short': float(arr[3])}


def _sharpe_per_lag_rust(pos, prices, lookbacks, resample):
    """Optimized sharpe_per_lag — computes Sharpe per shift entirely in Rust."""
    pos_df = pos if type(pos) == pd.DataFrame else pos.to_frame()
    prices_df = prices if type(prices) == pd.DataFrame else prices.to_frame()
    pos_df = _align_tz(pos_df, prices_df)

    common_cols = pos_df.columns.intersection(prices_df.columns)
    if common_cols.empty:
        return pd.Series({i: np.nan for i in lookbacks})

    pos_aligned = pos_df[common_cols]
    prices_aligned = prices_df[common_cols]

    pos_ts = _index_to_ns(pos_aligned.index)
    price_ts = _index_to_ns(prices_aligned.index)
    pos_data = np.ascontiguousarray(pos_aligned.values.ravel(order='F'), dtype='float64')
    price_data = np.ascontiguousarray(prices_aligned.values.ravel(order='F'), dtype='float64')

    sharpe_arr = ebt_rust.sharpe_per_lag_core(
        pos_ts, pos_data, price_ts, price_data, lookbacks, bool(resample),
        len(pos_aligned), len(prices_aligned), len(common_cols), **_rust_flat_kw()
    )

    return pd.Series(np.asarray(sharpe_arr), index=lookbacks)


def sharpe_per_lag(pos, returns, lookbacks=None, resample=True):
    default_lookbacks = list(range(-20, -5, 3)) + list(range(-5, 6)) + list(range(7, 30, 3)) + list(range(30, 100, 10)) + list(range(100, 250, 25))
    lookbacks = lookbacks if lookbacks is not None else default_lookbacks

    if is_price_or_return(returns) == 'return':
        prices = returns.fillna(0).cumsum()
    else:
        prices = returns

    if _HAS_RUST:
        return _sharpe_per_lag_rust(pos, prices, lookbacks, resample)

    sharpe_per_lag_result = pd.Series({i: backtest(pos, prices, shift=i).sum(axis=1).pipe(sharpe, resample=resample) for i in lookbacks})
    return sharpe_per_lag_result


def holding_curve(pos, prices, shift=2, ewm_spans=None):
    """Compute holding curve using Rust acceleration.

    Returns list of (sharpe, holding_days) tuples, one per EWM span.
    """
    if ewm_spans is None:
        ewm_spans = [0.0] + [2 ** (i / 2) for i in range(2, 24)]

    if not _HAS_RUST:
        return None  # Caller should fall back to Python

    pos_df = pos if type(pos) == pd.DataFrame else pos.to_frame()
    prices_df = prices if type(prices) == pd.DataFrame else prices.to_frame()
    pos_df = _align_tz(pos_df, prices_df)

    common_cols = pos_df.columns.intersection(prices_df.columns)
    if common_cols.empty:
        return [(np.nan, np.nan) for _ in ewm_spans]

    pos_aligned = pos_df[common_cols]
    # Gap-aware marks, as in backtest(): the Rust kernel diffs the price column on the FULL
    # grid and would otherwise drop the reopen-gap PnL at every NaN (closed) bar and
    # understate the whole curve. ffill is idempotent, so callers that already pass
    # forward-filled prices (one_pager) are unchanged.
    prices_aligned = prices_df[common_cols].ffill()

    pos_ts = _index_to_ns(pos_aligned.index)
    price_ts = _index_to_ns(prices_aligned.index)
    pos_data = np.ascontiguousarray(pos_aligned.values.ravel(order='F'), dtype='float64')
    price_data = np.ascontiguousarray(prices_aligned.values.ravel(order='F'), dtype='float64')

    results = ebt_rust.holding_curve_core(
        pos_ts, pos_data, price_ts, price_data,
        shift, [float(s) for s in ewm_spans],
        len(pos_aligned), len(prices_aligned), len(common_cols), **_rust_flat_kw()
    )

    return [(r[0], r[1]) for r in results]


def sharpe_decay(sharpes, holdings):
    """Log-log slope of Sharpe vs holding time.

    Measures how fast Sharpe decays as holding period increases.
    Values near 0 mean signal persists well; -0.5 is noise-like decay.

    Accepts any array-like inputs (numpy arrays, pd.Series, lists).
    """
    s = np.asarray(sharpes, dtype=float)
    h = np.asarray(holdings, dtype=float)
    mask = (s > 0) & (h > 0) & np.isfinite(s) & np.isfinite(h)
    if mask.sum() < 3:
        return np.nan
    log_h = np.log(h[mask])
    log_s = np.log(s[mask])
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    slope, _ = np.linalg.lstsq(A, log_s, rcond=None)[0]
    return float(-slope)


def _correlation_to_trend_rust(pos, prices, filter_lengths):
    """Rust-accelerated correlation_to_trend."""
    pos_df = pos if type(pos) == pd.DataFrame else pos.to_frame()
    prices_df = prices if type(prices) == pd.DataFrame else prices.to_frame()
    pos_df = _align_tz(pos_df, prices_df)

    common_cols = pos_df.columns.intersection(prices_df.columns)
    if common_cols.empty:
        return pd.Series({fl: np.nan for fl in filter_lengths})

    pos_aligned = pos_df[common_cols]
    prices_aligned = prices_df[common_cols]

    pos_ts = _index_to_ns(pos_aligned.index)
    price_ts = _index_to_ns(prices_aligned.index)
    pos_data = np.ascontiguousarray(pos_aligned.values.ravel(order='F'), dtype='float64')
    price_data = np.ascontiguousarray(prices_aligned.values.ravel(order='F'), dtype='float64')

    result = ebt_rust.correlation_to_trend_core(
        pos_ts, pos_data, price_ts, price_data,
        0,  # shift=0, pos is already realized
        [float(fl) for fl in filter_lengths],
        len(pos_aligned), len(prices_aligned), len(common_cols),
    )
    result = np.asarray(result)
    return pd.Series(result, index=filter_lengths)


def correlation_to_trend(pos, returns, filter_lengths=None):
    '''
    Returns correlation of returns of a portfolio to univariate trend following of different lengths
    '''
    if is_price_or_return(returns) == 'return':
        prices = returns.fillna(0).cumsum()
    else:
        prices = returns

    filter_lengths = [2 ** i for i in range(2, 15, 2)] if filter_lengths is None else filter_lengths

    if _HAS_RUST:
        return _correlation_to_trend_rust(pos, prices, filter_lengths)

    # pos is a realised position: shift=0. The trend leg is the EWM of each column's price
    # diffs (adjust=True, the Rust kernel's definition); the EWM through print t contains the
    # diff INTO t, so it is tradable from the next print only: shift=1. (shift=0 would earn
    # alpha * ret^2 on every bar, a look-ahead with Sharpe > 10 on a random walk.)
    pnl = backtest(pos, prices, resample=True).dropna(how='all').sum(axis=1)
    trend_returns = pd.DataFrame({
        l: backtest(prices.apply(lambda s: s.dropna().diff().ewm(span=l, min_periods=min(l // 2, 32)).mean()),
                    prices, shift=1, resample=True).dropna(how='all').sum(axis=1)
        for l in filter_lengths
    })

    return trend_returns.corrwith(pnl)
