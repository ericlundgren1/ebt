from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os


def identity(x, *args, **kwargs):
    return x


def progress(*args, progress=True, **kwargs):
    if progress:
        return tqdm(*args, **kwargs)
    
    return identity


def sort_columns(df):
    return df.reindex(columns=df.columns.sort_values())


def corr_to_pairs(corr):
    '''
    Given a correlation matrix, returns correlations of unique pairs excluding diagonal
    '''
    return corr.stack().pipe(lambda x: x[x.index.get_level_values(0) < x.index.get_level_values(1)])


def nanagg(dfs, agg_f=np.nanmean):

    if len(dfs) == 0:
        return pd.DataFrame()

    if agg_f.__name__ == 'nansum':
        '''
        Memory optimized — incremental .add() instead of concat+groupby
        '''
        from functools import reduce
        df = reduce(lambda a, b: a.add(b, fill_value=0), dfs)
        df = df.where(df != 0)
        df.dropna(how='all', inplace=True)
        return df
    
    if agg_f.__name__ == 'nanmean':
        from functools import reduce
        sum_df = reduce(lambda a, b: a.add(b, fill_value=0), dfs)
        count_df = reduce(lambda a, b: a.add(b, fill_value=0),
                          [df.notna().astype(int) for df in dfs])
        result = sum_df / count_df
        result = result.where(count_df > 0)
        result.dropna(how='all', inplace=True)
        return result

    import warnings
    index = pd.concat([pd.Series(np.nan, index=df.index) for df in dfs], axis=1, sort=True).index
    columns = sorted({c for df in dfs for c in list(df)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return pd.DataFrame(agg_f(np.array([df.reindex(index=index, columns=columns).values for df in dfs]), axis=0), index=index, columns=columns)


def split_portfolios(positions, values, n_portfolios=3, categories=False):
    '''
    From a bunch of positions and metadata, combines the positions based on some key into portfolios.

    :param positions dict: Dictionary of ids and pd.DataFrames
    :param values pd.Series: Indices must match keys in positions dictionary and the values should be something which is used to combine portfolios from,
        for instance lookback values or network depth.
    :param n_portfolios int: Number of portfolios to create, does not apply if dividing by category.
    :param categories bool: If true, uses the values as categorical variables. Useful if values are strings (for instance two different activation functions).
    '''
    
    portfolios = {}
    portfolios_ix = {}

    if categories:
        for category in values.value_counts().sort_index().index:
            pos_ix = values[values == category].index
            portfolio = combine_positions([positions[i] for i in pos_ix])
            portfolios[category] = portfolio
            portfolios_ix[category] = pos_ix
    else:
        try:
            portfolio_index = pd.Series(pd.qcut(values, n_portfolios, labels=np.arange(n_portfolios))) # Split into n_portfolios based on range of values
        except ValueError:  # duplicate edges drop some portfolios, so labels cannot be fixed in advance
            portfolio_index = pd.Series(pd.qcut(values, n_portfolios, duplicates='drop')) # Split into n_portfolios based on range of values

        for ix in portfolio_index.value_counts().sort_index().index:
            pos_ix = portfolio_index[portfolio_index == ix].index
            key = values.loc[pos_ix].mean()
            value = combine_positions([positions[k] for k in pos_ix])
            portfolios[key] = value
            portfolios_ix[key] = pos_ix
    
    return portfolios, portfolios_ix


def union_dataframes(dfs, init_value=np.nan):
    '''
    Returns a dataframe which has the shape of the union of all input dataframes
    '''
    if len(dfs) == 0:
        return pd.DataFrame()

    df = dfs[0]
    columns = df.columns
    index = df.index
    for df in dfs[1:]:
        columns = columns.union(df.columns)
        index = index.union(df.index)
    
    return pd.DataFrame(init_value, columns=columns, index=index)


def equal_weight_portfolios(ps, use_progress=True):
    progress = tqdm if use_progress else lambda x: x
    pos = union_dataframes(ps, init_value=0)
    for p in progress(ps):
        pos = pos.add(p.reindex_like(pos)[:p.index[-1]].ffill(), fill_value=0)
    
    return pos


def dask_client_muted(**client_kwargs):
    '''
    Returns a dask client with quieter logging. Close by using client.close().
    '''
    import logging
    from dask.distributed import Client
    
    logging.getLogger("distributed").setLevel(logging.ERROR)
    logging.getLogger("distributed.node").setLevel(logging.ERROR)
    
    
    # Start a local client
    client = Client(**client_kwargs)

    return client


# Moved to postprocessing.py; re-exported here for backward compatibility.
from ebt.simulate.postprocessing import risk_adjust  # noqa: F401


def combine_positions(dfs, agg='sum'):
    """
    Combine position DataFrames that update at different frequencies.

    Each input is forward-filled to a common timeline (union of all input
    timestamps) before aggregation, so slow-updating positions carry forward.

    Algorithm: compute trades (position deltas) from each df, scatter them
    onto the target index, then cumsum to recover combined positions. This
    touches only O(sum of input rows) rather than O(n_dfs * target_rows).

    Uses a Rust backend (ebt_rust.combine_positions_core) for the fused
    ffill → diff → scatter-add → cumsum pipeline.

    Args:
        dfs: list of DataFrames (DatetimeIndex, asset columns). Different frequencies and columns OK.
        agg: 'sum' (default) or 'mean'.

    Returns:
        Single DataFrame with combined positions.
    """
    if len(dfs) == 0:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0].ffill()

    # Build target index: union of all input timestamps
    target_idx = dfs[0].index
    for df in dfs[1:]:
        target_idx = target_idx.union(df.index)
    target_idx = target_idx.sort_values()

    # Build union of all columns
    all_columns = dfs[0].columns
    for df in dfs[1:]:
        all_columns = all_columns.union(df.columns)
    all_columns = all_columns.sort_values()

    n_cols = len(all_columns)
    col_to_idx = {c: i for i, c in enumerate(all_columns)}

    is_mean = (agg == 'mean')

    # Convert target timestamps to i64 nanoseconds. _index_to_ns normalizes the
    # resolution (ns/us/ms/s) — a raw .view(int64) would misalign mixed-resolution
    # inputs by factors of 1000, scattering trades to row 0.
    from ebt.simulate.kf import _index_to_ns
    target_ts = _index_to_ns(target_idx)

    # Prepare per-df arrays for Rust
    df_timestamps_list = []
    df_values_list = []
    df_n_rows_list = []
    df_n_cols_list = []
    df_col_indices_list = []

    for df in dfs:
        col_indices = np.array([col_to_idx[c] for c in df.columns], dtype=np.uint32)
        ts = _index_to_ns(df.index)
        vals = df.values.astype(np.float64, copy=False).ravel(order='F')
        n_rows, n_src_cols = df.shape

        df_timestamps_list.append(ts)
        df_values_list.append(vals)
        df_n_rows_list.append(n_rows)
        df_n_cols_list.append(n_src_cols)
        df_col_indices_list.append(col_indices)

    import ebt_rust
    result_flat, first_valid_row, count_starts_flat = ebt_rust.combine_positions_core(
        target_ts,
        df_timestamps_list,
        df_values_list,
        df_n_rows_list,
        df_n_cols_list,
        df_col_indices_list,
        n_cols,
        is_mean,
    )

    result = result_flat.reshape(len(target_idx), n_cols)
    return pd.DataFrame(result, index=target_idx, columns=all_columns)


def is_dask_process():
    '''
    Returns true if called from within a dask worker, false otherwise.
    '''
    try:
        from distributed import get_worker
        _ = get_worker()
        return True
    except (ImportError, ValueError):
        return False