import pandas as pd
import numpy as np


def make_ohlc(s_or_df, freq='d', label='right', dropna=True, ffill=True, cut_latest=True):
    '''
    Makes OHLC bins of a series or dataframe.

    Timestamp refers to time of CLOSE: with the default ``label='right'`` the bar stamped T holds
    the prints in (T - freq, T], so a print stamped exactly T (the price known at T) is that
    bar's close, as in make_daily_from_intraday. ``label='left'`` gives [T, T + freq) stamped T.
    The bin is closed on the labelled side in every input form (Series, plain DataFrame, OHLC
    DataFrame), so re-aggregating an OHLC frame equals aggregating the prints directly.

    :param s_or_df pd.DataFrame: Series or Dataframe of prices
    :param freq str: Sampling frequency
    :param label str: Passed into s_or_df.resample(label=label, closed=label), determines if the name of the bar is the timestamp at the start or end of the bar.
    :param dropna bool: If true, drops all columns with only nan values
    :param ffill bool: If true, fills missing values with the latest available close price
    :param cut_latest bool: If true, does not return the last bar. This is because it may not yet be complete.
    '''
    if type(s_or_df) == pd.Series:
        if type(s_or_df.index) == pd.MultiIndex:
            if s_or_df.index.get_level_values(0).nunique() == 1:
                s_or_df = s_or_df.droplevel(0, axis=0)
            else:
                raise ValueError(f'Cannot deal with multiindices, found {s_or_df.index.get_level_values(0).nunique()} unique values in level 0')

        resampler = s_or_df.resample(freq, label=label, closed=label)
        df = pd.DataFrame({
            'open': resampler.first(),
            'high': resampler.max(),
            'low': resampler.min(),
            'close': resampler.last()
        }).pipe(lambda x: x.dropna(how='all') if dropna else x)

        if ffill:
            df['close'] = df['close'].ffill()
            df['open'] = df['open'].fillna(df['close'].shift())
            df['high'] = df['high'].fillna(df['close'])
            df['low'] = df['low'].fillna(df['close'])

        return df[:-1] if cut_latest else df

    elif type(s_or_df) == pd.DataFrame:
        # If the DataFrame already has OHLC columns, resample those columns directly
        if type(s_or_df.columns) == pd.MultiIndex:
            if {'open', 'high', 'low', 'close'}.issubset(s_or_df.columns.get_level_values(0)):
                columns = s_or_df.columns.get_level_values(1).drop_duplicates()
                return pd.concat({c: make_ohlc(s_or_df.xs(c, axis=1, level=1), freq=freq, label=label, dropna=dropna, ffill=ffill, cut_latest=cut_latest)
                                  for c in columns}, axis=1, sort=True).swaplevel(0, 1, axis=1)
            else:
                raise ValueError('MultiIndex columns must have open/high/low/close as the top level')

        elif {'open', 'high', 'low', 'close'}.issubset(s_or_df.columns):

            resampled = pd.DataFrame({
                col: s_or_df[col].resample(freq, label=label, closed=label).agg(agg)
                for col, agg in [('open', 'first'), ('high', 'max'), ('low', 'min'), ('close', 'last')]
            }).pipe(lambda x: x.dropna(how='all') if dropna else x)

            if ffill:
                resampled['close'] = resampled['close'].ffill()
                resampled['open'] = resampled['open'].fillna(resampled['close'].shift())
                resampled['high'] = resampled['high'].fillna(resampled['close'])
                resampled['low'] = resampled['low'].fillna(resampled['close'])

            return resampled[:-1] if cut_latest else resampled

        # Otherwise, treat as a multi-column DataFrame of prices
        return pd.concat({c: make_ohlc(s_or_df[c], freq, label=label, dropna=dropna, ffill=ffill, cut_latest=cut_latest) for c in s_or_df}, axis=1, sort=True) \
                .swaplevel(0, 1, axis=1).sort_index(axis=1) \
                .pipe(lambda x: x.dropna(how='all') if dropna else x)

    raise ValueError(f"Unsupported type {type(s_or_df)}, supported types are pd.Series and pd.DataFrame")


def make_forward_labels(s_or_df, smoothing_length=5, prediction_offset=2, dropna=True, mean=True):
    '''
    At timestep N, represents the mean of returns [N+prediction_offset : N+prediction_offset+smoothing_length],
    scaled by sqrt(smoothing_length).
    '''

    if type(s_or_df) == pd.Series:
        min_periods = 1
        if dropna:
            s_or_df = s_or_df.dropna()
            min_periods = smoothing_length

        # PREFIX-STABLE forward window: a reversed-rolling
        # (s[::-1].rolling(...)[::-1]) accumulates from the series END, so the
        # floating-point rounding of every historical label would depend on the
        # series end date (~1e-13), and downstream models can amplify that wobble.
        # Forward rolling streams from the START: the value at N depends bit-exactly
        # only on data up to N+smoothing_length-1. The last smoothing_length-1
        # positions (windows truncated by the series end) are patched with a
        # reversed-rolling on just the series tail.
        agg = 'mean' if mean else 'sum'
        h = smoothing_length
        fwd = getattr(s_or_df.rolling(h, min_periods=min_periods), agg)().shift(-(h - 1))
        if h > 1 and len(s_or_df):
            tail_src = s_or_df.iloc[-min(len(s_or_df), 2 * h):]
            tail = getattr(tail_src[::-1].rolling(h, min_periods=min_periods), agg)()[::-1]
            k = min(h - 1, len(s_or_df))
            fwd.iloc[-k:] = tail.iloc[-k:].values
        return (fwd * h ** 0.5).shift(-prediction_offset)

    elif type(s_or_df) == pd.DataFrame:
        return pd.concat({c: make_forward_labels(s_or_df[c], smoothing_length, prediction_offset, dropna, mean) for c in s_or_df}, axis=1, sort=True)

    raise ValueError(f"Unsupported type {type(s_or_df)}, supported types are pd.Series and pd.DataFrame")


def apply_filters(s_or_df, lengths=None, dropna=True, rolling_rescale=True, rolling_rescale_min_periods=64, rolling_demean=False, rolling_demean_min_periods=64):
    '''
    Applies emas of different lengths to a series or dataframe.
    '''
    if type(s_or_df) == pd.Series:
        if dropna:
            s_or_df = s_or_df.dropna()
        lengths = [32, 64, 128, 256, 512, 1024, 2048] if lengths is None else lengths
        result = pd.concat({f'ema_{l}': s_or_df.ewm(span=l, min_periods=min(l // 2, 64)).mean() * l ** 0.5 for l in lengths}, axis=1).replace(0, np.nan)
        if dropna:
            result = result.dropna(how='all')

        if rolling_rescale == True:
            rolling_rescale = max(lengths) * 10

        if rolling_rescale > 0:
            std = result.ewm(span=rolling_rescale, min_periods=rolling_rescale_min_periods).std()
            result = result / std

        if rolling_demean == True:
            rolling_demean = max(lengths) * 10

        if rolling_demean > 0:
            mean = result.ewm(span=rolling_demean, min_periods=rolling_demean_min_periods).mean()
            result = result - mean

        return result

    elif type(s_or_df) == pd.DataFrame:
        lengths = [32, 64, 128, 256, 512, 1024, 2048] if lengths is None else lengths

        # Compute all EMAs across all columns at once (vectorized)
        parts = {}
        for l in lengths:
            ema = s_or_df.ewm(span=l, min_periods=min(l // 2, 64)).mean() * l ** 0.5
            for c in s_or_df.columns:
                parts[(c, f'ema_{l}')] = ema[c]
        result = pd.DataFrame(parts).replace(0, np.nan)
        if dropna:
            result = result.dropna(how='all')

        if rolling_rescale == True:
            rolling_rescale = max(lengths) * 10

        if rolling_rescale > 0:
            std = result.ewm(span=rolling_rescale, min_periods=rolling_rescale_min_periods).std()
            result = result / std

        if rolling_demean == True:
            rolling_demean = max(lengths) * 10

        if rolling_demean > 0:
            mean = result.ewm(span=rolling_demean, min_periods=rolling_demean_min_periods).mean()
            result = result - mean

        # Column order: (col, ema_len) grouped by col
        col_order = []
        for c in s_or_df.columns:
            for l in lengths:
                key = (c, f'ema_{l}')
                if key in result.columns:
                    col_order.append(key)
        return result[col_order]


def normalized_rank(df):
    """Cross-sectional rank normalization, mapped to [-1, 1].

    Ranks columns at each row, then linearly maps the rank to [-1, 1].
    Useful as a feature transform that removes scale differences between
    features while preserving relative ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Features to rank. Each row is ranked independently across columns.

    Returns
    -------
    pd.DataFrame
        Same shape as input, values in [-1, 1].
    """
    return df.rank(axis=1).pipe(lambda x: x - 1).pipe(lambda x: x.divide(x.max(axis=1), axis=0) * 2 - 1)


def ffill_timeseries(df, time_to_forwardfill='1d'):
    '''
    Forward fills a timeseries based on time rather than number of observations
    '''
    if type(time_to_forwardfill) == str:
        time_to_forwardfill = pd.Timedelta(time_to_forwardfill)

    smallest_interval = pd.Series(df.index.values).diff().min()
    units_to_forwardfill = time_to_forwardfill // smallest_interval
    return df.resample(smallest_interval).last().ffill(limit=units_to_forwardfill).loc[df.index]


def sample_every(timedelta: pd.Timedelta):
    """
    Returns a list of datetime.time objects, spaced at timedelta intervals from 00:00:00 up to but not including 24:00:00.
    """
    if type(timedelta) == str:
        timedelta = pd.Timedelta(timedelta)

    import datetime
    times = []
    t = datetime.datetime.combine(datetime.date.today(), datetime.time(0, 0))
    end = t + datetime.timedelta(days=1)
    while t < end:
        times.append(t.time())
        t += timedelta
    return times


def make_daily_from_intraday(returns_or_prices, interval='10min', strip_time_component=True, strip_timezone=True):
    '''
    Takes a price or return intraday timeseries and makes it into multiple daily timeseries with some interval.

    For instance, if interval is 12h, two daily timeseries will be returned, one with prices (or returns) from
    midnight to midnight the next day and one which is from noon until noon the next day.
    '''
    from ebt.simulate.kf import is_price_or_return
    prices = returns_or_prices
    got_returns = is_price_or_return(returns_or_prices) == 'return'
    if got_returns:
        prices = returns_or_prices.cumsum()

    results = {}
    intervals = sample_every(interval)
    for t in intervals:
        # label/closed='right': the bar stamped T holds the last price at or before T,
        # never a price from (T, T+interval) — same convention as make_ohlc.
        results[str(t)] = prices.resample(interval, label='right', closed='right').last().ffill().between_time(t, t)

    if strip_time_component:
        results = {k: v.resample('D').last().dropna(how='all') for k, v in results.items()}

    if strip_timezone:
        results = {k: v.tz_localize(None) for k, v in results.items()}

    if got_returns:
        results = {k: v.apply(lambda s: s.dropna().diff()) for k, v in results.items()}

    return results
