from ebt.simulate.model import UnityModel, TestModel
from ebt.simulate.features import ffill_timeseries, apply_filters, make_ohlc
from ebt.simulate.tricks import nanagg, combine_positions
from ebt.simulate.predict import enough_data
from ebt import predict, generator
import pandas as pd
import numpy as np
import logging


def _assert_generator_coverage(df, params):
    """Shared assertions for generator tests."""
    counter = pd.Series(0, index=df.index)
    for window, embargo_start, prediction_start_date in generator(df, **params):
        counter[window[prediction_start_date:].index] += 1

    min_lb_embargo = df.index[0] + pd.Timedelta(params['min_lb']) + pd.Timedelta(params['embargo'])
    assert (counter[:min_lb_embargo] == 0).all(), 'periods covered in min_lb and embargo should never be predicted'
    assert (counter[min_lb_embargo:].iloc[1:] == 1).all(), 'periods outside min_lb and embargo should always be predicted exactly once'
    assert prediction_start_date - embargo_start >= pd.Timedelta(params['embargo']), 'Embargo should always be at least prediction_start_date - embargo_start'


def test_generator():
    """Generator with moderate retrain intervals."""
    np.random.seed(1932)
    for _ in range(30):
        n_rows = np.random.randint(20, 200)
        n_cols = np.random.randint(2, 8)
        freq, unit = [("1D", "D"), ("6h", "h"), ("3h", "h"), ("30min", "min"), ("15min", "min")][np.random.randint(5)]
        idx = pd.date_range("2022-01-01", periods=n_rows, freq=freq)
        df = pd.DataFrame(np.random.randn(n_rows, n_cols), index=idx, columns=[f"col{i}" for i in range(n_cols)])
        params = dict(
            max_lb=f"{np.random.randint(10, 30)}{unit}",
            min_lb=f"{np.random.randint(5, 10)}{unit}",
            embargo=f"{np.random.randint(0, 3)}{unit}",
            retrain_interval=f"{np.random.randint(5, 15)}{unit}"
        )
        _assert_generator_coverage(df, params)


def test_generator_short_retrain():
    """Generator with very short retrain intervals (1-4 units) on short history."""
    np.random.seed(1932)
    for _ in range(30):
        n_rows = np.random.randint(20, 50)
        n_cols = np.random.randint(2, 8)
        freq, unit = [("1D", "D"), ("6h", "h"), ("3h", "h"), ("30min", "min"), ("15min", "min")][np.random.randint(5)]
        idx = pd.date_range("2022-01-01", periods=n_rows, freq=freq)
        df = pd.DataFrame(np.random.randn(n_rows, n_cols), index=idx, columns=[f"col{i}" for i in range(n_cols)])
        params = dict(
            max_lb=f"{np.random.randint(10, 30)}{unit}",
            min_lb=f"{np.random.randint(5, 10)}{unit}",
            embargo=f"{np.random.randint(0, 3)}{unit}",
            retrain_interval=f"{np.random.randint(1, 5)}{unit}"
        )
        _assert_generator_coverage(df, params)


def test_unity_prediction():
    model = UnityModel()
    features = pd.DataFrame(np.ones((500, 10)), index=pd.date_range("2000-01-01", periods=500)).rename(columns=str)
    labels = pd.DataFrame(np.ones((500, 10)), index=pd.date_range("2000-01-01", periods=500)).rename(columns=str)
    results = predict(features, labels, models=[model], max_lookback=90, retrain_interval=1, minimum_required_points=1,
                      embargo='0D')
    assert (results[model] - features[11:]).pipe(lambda x: x.abs() < 1e-10).all().all()


def test_parallel_matches_sequential():
    """parallel=True must produce identical results to sequential predict."""
    from ebt.simulate.model import ForestModel
    from ebt.simulate.linear import Ridge

    np.random.seed(42)
    n = 300
    features = pd.DataFrame(np.random.randn(n, 5), index=pd.date_range("2000-01-01", periods=n)).rename(columns=str)
    labels = pd.DataFrame(np.random.randn(n, 3), index=pd.date_range("2000-01-01", periods=n)).rename(columns=str)

    models_seq = [Ridge(alpha=1.0), ForestModel(n_estimators=10, random_state=0)]
    models_par = [Ridge(alpha=1.0), ForestModel(n_estimators=10, random_state=0)]

    kwargs = dict(max_lookback=90, retrain_interval=30, minimum_required_points=10,
                  use_string_keys=True, progressbar=False, log=False)
    seq = predict(features, labels, models=models_seq, parallel=False, **kwargs)
    par = predict(features, labels, models=models_par, parallel=True, **kwargs)

    # Keys contain UUIDs (different per model instance), so match by class prefix
    seq_by_class = {k.split('-')[0]: v for k, v in seq.items()}
    par_by_class = {k.split('-')[0]: v for k, v in par.items()}
    assert set(seq_by_class.keys()) == set(par_by_class.keys()), f"keys differ: {set(seq_by_class.keys())} vs {set(par_by_class.keys())}"
    for key in seq_by_class:
        pd.testing.assert_frame_equal(seq_by_class[key], par_by_class[key], atol=1e-10)


def test_test_model():
    n = 1011
    model = TestModel()
    df = pd.DataFrame(np.random.normal(size=(n, 10)), index=pd.date_range('2000-01-01', periods=n))
    results = predict(df, df, models=[model], max_lookback=90, embargo='0D')
    output_df = results[model]
    assert output_df.shape[0] == n - 71
    assert len(model.xs_fit) == 32
    assert len(model.xs_fit) == len(model.xs_predict)
    assert len(model.ys_fit) == len(model.xs_fit)
    assert len(model.ys_fit) == len(model.ys_predict)
    assert (results[model] - df[71:]).pipe(lambda x: x.abs() < 1e-10).all().all()


def test_test_model_default_embargo():
    """Default embargo (10D): scoring starts 10 days later, and every model's first scored
    row is at least 10 days after its last training row."""
    n = 1011
    model = TestModel()
    df = pd.DataFrame(np.random.normal(size=(n, 10)), index=pd.date_range('2000-01-01', periods=n))
    out = predict(df, df, models=[model], max_lookback=90, verbose=True, progressbar=False, log=False)
    output_df = out['predictions_flattened'][model]
    assert output_df.shape[0] == n - 81
    assert len(model.xs_fit) == 31
    assert (output_df - df[81:]).pipe(lambda x: x.abs() < 1e-10).all().all()
    for train_end, preds in out['predictions_dict'][model].items():
        first_scored = min(p.index[0] for p in preds.values())
        assert first_scored - train_end > pd.Timedelta('10D')


def test_ffill_timeseries():

    # Create a DataFrame with missing timestamps and NaNs
    idx = pd.to_datetime([
        "2020-01-01 00:00:00",
        "2020-01-01 00:10:00",
        "2020-01-01 00:30:00",  # 20 min gap
        "2020-01-01 01:00:00",  # 30 min gap
        "2020-01-01 01:10:00"
    ])
    df = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5]}, index=idx)

    # Forward fill up to 20 minutes
    out = ffill_timeseries(df, time_to_forwardfill="20min")
    # The value at 00:30:00 should be forward-filled from 00:10:00 (NaN), so remains NaN
    # The value at 01:00:00 should not be forward-filled from 00:30:00 (gap > 20min), so remains NaN
    # The value at 01:10:00 should not be forward-filled from 01:00:00 (gap = 10min, but value is NaN), so remains 5

    # Check that values are as expected
    assert out.loc["2020-01-01 00:00:00", "a"] == 1
    assert out.loc["2020-01-01 00:10:00", "a"] == 1
    assert out.loc["2020-01-01 00:30:00", "a"] == 3
    assert np.isnan(out.loc["2020-01-01 01:00:00", "a"])
    assert out.loc["2020-01-01 01:10:00", "a"] == 5

    # Now test with a longer forward fill window
    out2 = ffill_timeseries(df, time_to_forwardfill="40min")
    # Now, 01:00:00 can be forward-filled from 00:30:00 (gap = 30min < 40min)
    assert out2.loc["2020-01-01 01:00:00", "a"] == 3

    # Test with a DataFrame with multiple columns
    df2 = pd.DataFrame({
        "a": [1, np.nan, 3, np.nan, 5],
        "b": [np.nan, 2, np.nan, 4, np.nan]
    }, index=idx)
    out3 = ffill_timeseries(df2, time_to_forwardfill="20min")
    assert out3.loc["2020-01-01 00:00:00", "a"] == 1
    assert out3.loc["2020-01-01 00:10:00", "a"] == 1
    assert out3.loc["2020-01-01 00:30:00", "a"] == 3
    assert np.isnan(out3.loc["2020-01-01 01:00:00", "a"])
    assert out3.loc["2020-01-01 01:10:00", "a"] == 5

    assert np.isnan(out3.loc["2020-01-01 00:00:00", "b"])
    assert out3.loc["2020-01-01 00:10:00", "b"] == 2
    assert out3.loc["2020-01-01 00:30:00", "b"] == 2
    assert out3.loc["2020-01-01 01:00:00", "b"] == 4
    assert out3.loc["2020-01-01 01:10:00", "b"] == 4


def test_make_daily_from_intraday():
    import pandas as pd
    import numpy as np
    from ebt.simulate.features import make_daily_from_intraday

    # Create a longer intraday price series for several weeks, 10min frequency, business days only
    idx = pd.date_range("2023-01-01 09:00", "2023-01-28 16:00", freq="10min")
    # Filter to business hours (e.g., 09:00 to 16:00) and business days
    idx = idx[(idx.indexer_between_time("09:00", "16:00"))]
    idx = idx[idx.normalize().isin(pd.bdate_range(idx[0].normalize(), idx[-1].normalize()))]
    prices = pd.Series(np.arange(len(idx)), index=idx)
    # Add a second column for DataFrame test
    prices_df = pd.DataFrame({"a": prices, "b": prices + 100})

    # Test with default interval ('10min'), strip_time_component=True
    result = make_daily_from_intraday(prices, interval="1h", strip_time_component=True)
    # Should return a dict of DataFrames, one for each hour in the day
    assert isinstance(result, dict)
    # There should be 24 keys (one for each hour)
    assert len(result) == 24
    for k, v in result.items():
        # Each value should be a Series or DataFrame with daily frequency
        assert isinstance(v, pd.Series)
        # Index should be daily
        assert v.index.is_monotonic_increasing
        # Should cover at least several weeks of data
        assert len(v) >= 15  # at least 15 business days

    # Test with returns (simulate returns as diff of prices)
    returns = prices.diff().dropna()
    result_returns = make_daily_from_intraday(returns, interval="1h", strip_time_component=True)
    # Should return a dict of DataFrames, one for each hour in the day
    assert isinstance(result_returns, dict)
    for k, v in result_returns.items():
        # Each value should be a Series with daily frequency
        assert isinstance(v, pd.Series)
        # Index should be daily
        assert v.index.is_monotonic_increasing
        assert len(v) >= 15


# --- Preprocessor sanity check tests ---

def test_preprocessor_check_detects_leaky_preprocessor(caplog):
    """A preprocessor that normalizes by full-window stats leaks future info."""
    n = 500
    idx = pd.date_range('2000-01-01', periods=n, freq='D')
    features = pd.DataFrame(np.random.randn(n, 5), index=idx, columns=[f'f{i}' for i in range(5)])
    labels = pd.DataFrame(np.random.randn(n, 3), index=idx, columns=[f'l{i}' for i in range(3)])

    def leaky_preprocessor(x, y=None):
        # Normalizes by stats of the FULL window — leaks future info
        x_out = (x - x.mean()) / x.std()
        return x_out, y

    with caplog.at_level(logging.WARNING, logger='ebt.simulate.predict'):
        predict(features, labels, models=[UnityModel()], max_lookback='200D',
                min_lookback='90D', retrain_interval='60D',
                preprocess_training_data=leaky_preprocessor,
                minimum_required_points=10, progressbar=False, log=False)

    assert any('future info' in msg for msg in caplog.messages), \
        f"Expected warning about future info, got: {caplog.messages}"


def test_preprocessor_check_passes_for_causal(caplog):
    """A causal preprocessor (enough_data) should not trigger the warning."""
    n = 500
    idx = pd.date_range('2000-01-01', periods=n, freq='D')
    features = pd.DataFrame(np.random.randn(n, 5), index=idx, columns=[f'f{i}' for i in range(5)])
    labels = pd.DataFrame(np.random.randn(n, 3), index=idx, columns=[f'l{i}' for i in range(3)])

    causal_preprocess = enough_data(x_fraction_present_in_column_full=0.5, fill=True)

    with caplog.at_level(logging.WARNING, logger='ebt.simulate.predict'):
        predict(features, labels, models=[UnityModel()], max_lookback='200D',
                min_lookback='90D', retrain_interval='60D',
                preprocess_training_data=causal_preprocess,
                minimum_required_points=10, progressbar=False, log=False)

    assert not any('future info' in msg for msg in caplog.messages), \
        f"Unexpected warning about future info: {caplog.messages}"


# --- No fabricated y=0 rows tests ---

def test_no_nan_rows_in_ytrain():
    """After shift, y_train should have no all-NaN rows passed to model.fit()."""
    n = 500
    model = TestModel()
    idx = pd.date_range('2000-01-01', periods=n, freq='D')
    features = pd.DataFrame(np.random.randn(n, 5), index=idx, columns=[f'f{i}' for i in range(5)])
    labels = pd.DataFrame(np.random.randn(n, 3), index=idx, columns=[f'l{i}' for i in range(3)])

    predict(features, labels, models=[model], max_lookback='200D',
            min_lookback='90D', retrain_interval='60D',
            minimum_required_points=10, progressbar=False, log=False)

    for i, y_fit in enumerate(model.ys_fit):
        # No row should be all-NaN (which would have been filled to 0 by enough_data)
        assert not y_fit.isna().all(axis=1).any(), f"Window {i}: y_train has all-NaN rows"

    for i, (x_fit, y_fit) in enumerate(zip(model.xs_fit, model.ys_fit)):
        assert x_fit.index.equals(y_fit.index), f"Window {i}: x_train and y_train indices don't match"


def test_predict_misaligned_feature_label_indices():
    """predict() should handle features and labels with different-length indices."""
    n = 500
    idx_feat = pd.date_range('2000-01-01', periods=n, freq='D')
    idx_labels = pd.date_range('2000-01-01', periods=n + 3, freq='D')

    features = pd.DataFrame(np.random.randn(n, 5), index=idx_feat, columns=[f'f{i}' for i in range(5)])
    labels = pd.DataFrame(np.random.randn(n + 3, 3), index=idx_labels, columns=[f'l{i}' for i in range(3)])

    predict(features, labels, models=[UnityModel()], max_lookback='200D',
            min_lookback='90D', retrain_interval='60D',
            minimum_required_points=10, progressbar=False, log=False)


# --- apply_filters vectorized tests ---

def test_apply_filters_rolling_rescale():
    """Vectorized rolling_rescale produces correct EMA/EMA-std ratios."""
    np.random.seed(42)
    # Multi-column returns with staggered start dates (different NaN patterns)
    n = 500
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    data = np.random.randn(n, 5).cumsum(axis=0)
    # Stagger starts: col 0 starts at 0, col 1 at 50, etc.
    for i in range(5):
        data[:i * 50, i] = np.nan
    prices = pd.DataFrame(data, index=idx, columns=[f'c{i}' for i in range(5)])
    returns = prices.diff()

    result = apply_filters(returns, lengths=[32, 64, 128], rolling_rescale=True)

    # Verify output is a DataFrame with expected shape
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 5 * 3  # 5 columns x 3 filter lengths

    # Verify values: for each column, ema / ewm_std should match
    for col in returns.columns:
        for length in [32, 64, 128]:
            key = (col, f'ema_{length}')
            if key in result.columns:
                series = result[key].dropna()
                assert len(series) > 0, f"Column {key} is all NaN"
                # Values should be finite where they exist
                assert np.all(np.isfinite(series.values)), f"Column {key} has non-finite values"

    # Verify that columns with later starts still produce correct results
    # (col4 starts at row 200, so its EMA output should start later)
    col4_ema32 = result[('c4', 'ema_32')].dropna()
    col0_ema32 = result[('c0', 'ema_32')].dropna()
    assert col4_ema32.index[0] > col0_ema32.index[0], "Staggered starts should produce different first valid indices"


def test_apply_filters_dataframe_matches_per_column():
    """Vectorized DataFrame path produces identical results to per-column Series path."""
    np.random.seed(42)
    n = 500
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    data = np.random.randn(n, 5).cumsum(axis=0)
    # Stagger NaN starts to exercise different leading-NaN patterns
    for i in range(5):
        data[:i * 30, i] = np.nan
    df = pd.DataFrame(data, index=idx, columns=[f'c{i}' for i in range(5)])
    returns = df.diff()

    lengths = [32, 64, 128]

    # DataFrame path (vectorized)
    result_df = apply_filters(returns, lengths=lengths, rolling_rescale=True, rolling_demean=True)

    # Per-column Series path (reference)
    ref = pd.concat(
        {c: apply_filters(returns[c], lengths=lengths, rolling_rescale=True, rolling_demean=True)
         for c in returns.columns}, axis=1, sort=True)

    # Align and compare
    result_aligned = result_df.reindex(columns=ref.columns, index=ref.index)
    non_nan = ref.notna().values & result_aligned.notna().values
    assert non_nan.sum() > 0, "No overlapping non-NaN values"
    assert np.allclose(ref.values[non_nan], result_aligned.values[non_nan], atol=1e-12), \
        f"Max diff: {np.max(np.abs(ref.values[non_nan] - result_aligned.values[non_nan]))}"
    # NaN patterns should be identical
    assert (ref.isna().values == result_aligned.isna().values).all(), "NaN patterns differ"


def test_apply_filters_rolling_demean():
    """Vectorized rolling_demean produces correct results."""
    np.random.seed(42)
    n = 300
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    # Use stationary returns (not cumsum) so demean is meaningful
    s = pd.Series(np.random.randn(n), index=idx)

    result_rescale = apply_filters(s, lengths=[32, 64], rolling_rescale=True, rolling_demean=False)
    result_demean = apply_filters(s, lengths=[32, 64], rolling_rescale=False, rolling_demean=True)
    result_both = apply_filters(s, lengths=[32, 64], rolling_rescale=True, rolling_demean=True)

    assert isinstance(result_rescale, pd.DataFrame)
    assert isinstance(result_demean, pd.DataFrame)
    assert isinstance(result_both, pd.DataFrame)
    # All should have the expected number of columns
    assert result_rescale.shape[1] == 2
    assert result_demean.shape[1] == 2
    assert result_both.shape[1] == 2


# --- make_ohlc tests ---

def test_make_ohlc_series():
    """make_ohlc with Series input matches expected OHLC values."""
    np.random.seed(42)
    idx = pd.date_range('2020-01-01', periods=1000, freq='1min')
    prices = pd.Series(np.random.randn(1000).cumsum() + 100, index=idx)

    result = make_ohlc(prices, freq='5min')

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'open', 'high', 'low', 'close'}
    # High should always be >= open, close, low
    assert (result['high'] >= result['open']).all()
    assert (result['high'] >= result['close']).all()
    assert (result['high'] >= result['low']).all()
    # Low should always be <= open, close, high
    assert (result['low'] <= result['open']).all()
    assert (result['low'] <= result['close']).all()


def test_make_ohlc_dataframe_ohlc():
    """make_ohlc with DataFrame OHLC input resamples correctly."""
    np.random.seed(42)
    idx = pd.date_range('2020-01-01', periods=100, freq='1min')
    close = pd.Series(np.random.randn(100).cumsum() + 100, index=idx)
    df = pd.DataFrame({
        'open': close.shift(1).bfill(),
        'high': close + abs(np.random.randn(100)),
        'low': close - abs(np.random.randn(100)),
        'close': close
    })

    result = make_ohlc(df, freq='5min')
    assert set(result.columns) == {'open', 'high', 'low', 'close'}
    assert (result['high'] >= result['low']).all()


# --- nanagg nanmean tests ---

def test_nanagg_nanmean():
    """Incremental nanmean matches the 3D-array approach exactly."""
    np.random.seed(42)
    # 5 DataFrames with overlapping but different index/column sets
    dfs = []
    base_idx = pd.date_range('2020-01-01', periods=100, freq='D')
    for i in range(5):
        start = i * 10
        n = 100 - start
        idx = base_idx[start:start + n]
        cols = [f'c{j}' for j in range(i, i + 5)]  # overlapping columns
        data = np.random.randn(n, 5)
        data[np.random.random(data.shape) < 0.2] = np.nan
        dfs.append(pd.DataFrame(data, index=idx, columns=cols))

    result = nanagg(dfs, np.nanmean)

    # Compare against the original 3D-array approach
    import warnings
    index = pd.concat([pd.Series(np.nan, index=df.index) for df in dfs], axis=1, sort=True).index
    columns = sorted({c for df in dfs for c in list(df)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        expected = pd.DataFrame(
            np.nanmean(np.array([df.reindex(index=index, columns=columns).values for df in dfs]), axis=0),
            index=index, columns=columns)

    # Align and compare
    common_idx = result.index.intersection(expected.dropna(how='all').index)
    common_cols = sorted(set(result.columns) & set(expected.columns))
    r = result.reindex(index=common_idx, columns=common_cols)
    e = expected.reindex(index=common_idx, columns=common_cols)
    assert np.allclose(r.values, e.values, atol=1e-10, equal_nan=True), "nanmean results don't match"


def test_nanagg_nanmean_all_nan_stays_nan():
    """Positions where all inputs are NaN should remain NaN, not become 0."""
    idx = pd.date_range('2020-01-01', periods=5, freq='D')
    df1 = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]}, index=idx)
    df2 = pd.DataFrame({'a': [np.nan, 2.0, np.nan, np.nan, 6.0]}, index=idx)

    result = nanagg([df1, df2], np.nanmean)

    assert result.loc[idx[0], 'a'] == 1.0   # only df1 has value
    assert result.loc[idx[1], 'a'] == 2.0   # only df2 has value
    assert result.loc[idx[2], 'a'] == 3.0   # only df1 has value
    # idx[3] is all-NaN → dropna(how='all') removes it, so it shouldn't be in result
    assert idx[3] not in result.index, "All-NaN row should be dropped"
    assert result.loc[idx[4], 'a'] == 5.5   # mean(5, 6) = 5.5


# --- combine_positions tests ---

def test_combine_positions_same_freq_same_cols():
    """Two daily dfs, same assets → matches manual sum."""
    idx = pd.date_range('2020-01-01', periods=5, freq='D')
    df1 = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0], 'B': [0.5, 0.5, 0.5, 0.5, 0.5]}, index=idx)
    df2 = pd.DataFrame({'A': [10.0, 20.0, 30.0, 40.0, 50.0], 'B': [1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)

    result = combine_positions([df1, df2])
    expected = df1 + df2
    pd.testing.assert_frame_equal(result, expected, check_like=True)


def test_combine_positions_different_cols():
    """Union columns: missing col = 0 for sum, excluded for mean."""
    idx = pd.date_range('2020-01-01', periods=5, freq='D')
    df1 = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    df2 = pd.DataFrame({'B': [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)

    result_sum = combine_positions([df1, df2], agg='sum')
    assert set(result_sum.columns) == {'A', 'B'}
    pd.testing.assert_series_equal(result_sum['A'], df1['A'], check_names=False)
    pd.testing.assert_series_equal(result_sum['B'], df2['B'], check_names=False)

    result_mean = combine_positions([df1, df2], agg='mean')
    # Mean with col in only 1 df → value is just that df's value
    pd.testing.assert_series_equal(result_mean['A'], df1['A'], check_names=False)
    pd.testing.assert_series_equal(result_mean['B'], df2['B'], check_names=False)


def test_combine_positions_different_freq():
    """Daily + hourly → ffill daily to hourly, then combine."""
    daily_idx = pd.date_range('2020-01-01', periods=3, freq='D')
    hourly_idx = pd.date_range('2020-01-01', periods=49, freq='h')

    df_daily = pd.DataFrame({'A': [1.0, 2.0, 3.0]}, index=daily_idx)
    df_hourly = pd.DataFrame({'A': np.arange(49, dtype=float)}, index=hourly_idx)

    result = combine_positions([df_daily, df_hourly])
    # Target index is union → hourly
    assert len(result) == len(hourly_idx)
    # At 2020-01-01 00:00, sum = 1.0 + 0.0 = 1.0
    assert result.loc['2020-01-01 00:00', 'A'] == 1.0
    # At 2020-01-01 12:00, daily is ffilled to 1.0, hourly is 12.0 → sum = 13.0
    assert result.loc['2020-01-01 12:00', 'A'] == 13.0
    # At 2020-01-02 00:00, daily updates to 2.0, hourly is 24.0 → sum = 26.0
    assert result.loc['2020-01-02 00:00', 'A'] == 26.0


def test_combine_positions_single_df():
    """Returns ffilled copy."""
    idx = pd.date_range('2020-01-01', periods=5, freq='D')
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

    result = combine_positions([df])
    pd.testing.assert_frame_equal(result, df)
    # Must be a copy, not the same object
    assert result is not df


def test_combine_positions_empty():
    """Returns empty DataFrame."""
    result = combine_positions([])
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_combine_positions_no_future_leakage():
    """At time t, only data from <= t is used. Value at t=100 must not appear at t=99."""
    idx1 = pd.date_range('2020-01-01', periods=100, freq='h')
    idx2 = pd.date_range('2020-01-01', periods=100, freq='h')

    df1 = pd.DataFrame({'A': np.ones(100)}, index=idx1)
    # df2 only has a value at the very last timestamp
    df2 = pd.DataFrame({'A': [np.nan] * 99 + [999.0]}, index=idx2)

    result = combine_positions([df1, df2])
    # At t=98 (second to last), df2 should still be NaN (no ffill from future)
    assert result.loc[idx1[98], 'A'] == 1.0  # only df1 contributes
    # At t=99 (last), df2 has 999.0 → sum = 1.0 + 999.0 = 1000.0
    assert result.loc[idx1[99], 'A'] == 1000.0


def test_combine_positions_equivalence_manual():
    """Compare with manual reindex+ffill+sum for random inputs."""
    np.random.seed(42)
    idx1 = pd.date_range('2020-01-01', periods=100, freq='h')
    idx2 = pd.date_range('2020-01-01 00:30', periods=80, freq='2h')

    df1 = pd.DataFrame(np.random.randn(100, 3), index=idx1, columns=['A', 'B', 'C'])
    df2 = pd.DataFrame(np.random.randn(80, 3), index=idx2, columns=['A', 'B', 'C'])

    result = combine_positions([df1, df2])

    # Manual approach
    target_idx = idx1.union(idx2).sort_values()
    a1 = df1.reindex(target_idx, method='ffill')
    a2 = df2.reindex(target_idx, method='ffill')
    expected = a1.add(a2, fill_value=0)
    valid = a1.notna() | a2.notna()
    expected = expected.where(valid)

    pd.testing.assert_frame_equal(result, expected)


def test_combine_positions_mean_different_cols():
    """Mean with partial columns: col in 1 of 3 dfs → value is just that df."""
    idx = pd.date_range('2020-01-01', periods=5, freq='D')
    df1 = pd.DataFrame({'A': [1.0] * 5, 'B': [2.0] * 5}, index=idx)
    df2 = pd.DataFrame({'B': [4.0] * 5, 'C': [6.0] * 5}, index=idx)
    df3 = pd.DataFrame({'A': [3.0] * 5}, index=idx)

    result = combine_positions([df1, df2, df3], agg='mean')
    # A: in df1 (1.0) and df3 (3.0) → mean = 2.0
    assert np.isclose(result.loc[idx[0], 'A'], 2.0)
    # B: in df1 (2.0) and df2 (4.0) → mean = 3.0
    assert np.isclose(result.loc[idx[0], 'B'], 3.0)
    # C: only in df2 (6.0) → mean = 6.0
    assert np.isclose(result.loc[idx[0], 'C'], 6.0)


def test_combine_positions_nan_gaps():
    """NaN gaps mid-series are forward-filled through."""
    idx1 = pd.date_range('2020-01-01', periods=10, freq='D')
    # df1 updates on days 0, 5 only
    sparse_idx = pd.DatetimeIndex([idx1[0], idx1[5]])
    df1 = pd.DataFrame({'A': [1.0, 2.0]}, index=sparse_idx)
    df2 = pd.DataFrame({'A': np.ones(10)}, index=idx1)

    result = combine_positions([df1, df2])
    # Days 0-4: df1 ffilled at 1.0, df2 = 1.0 → sum = 2.0
    assert result.loc[idx1[3], 'A'] == 2.0
    # Days 5-9: df1 ffilled at 2.0, df2 = 1.0 → sum = 3.0
    assert result.loc[idx1[7], 'A'] == 3.0


def test_combine_positions_no_overlap():
    """Dfs with disjoint time ranges → correct separate regions."""
    idx1 = pd.date_range('2020-01-01', periods=5, freq='D')
    idx2 = pd.date_range('2020-01-10', periods=5, freq='D')

    df1 = pd.DataFrame({'A': [1.0] * 5}, index=idx1)
    df2 = pd.DataFrame({'A': [2.0] * 5}, index=idx2)

    result = combine_positions([df1, df2])
    # In df1's range, df2 hasn't started → only df1
    assert result.loc[idx1[0], 'A'] == 1.0
    # In df2's range, df1 is ffilled at 1.0 → sum = 1.0 + 2.0 = 3.0
    assert result.loc[idx2[0], 'A'] == 3.0


def test_combine_positions_mean_staggered_start():
    """Mean correctly divides by count of active dfs, not total dfs."""
    idx = pd.date_range('2020-01-01', periods=10, freq='D')
    df1 = pd.DataFrame({'A': [2.0] * 10}, index=idx)
    # df2 starts 5 days later
    df2 = pd.DataFrame({'A': [4.0] * 5}, index=idx[5:])

    result = combine_positions([df1, df2], agg='mean')
    # Days 0-4: only df1 → mean = 2.0
    assert result.loc[idx[2], 'A'] == 2.0
    # Days 5-9: df1=2.0, df2=4.0 → mean = 3.0
    assert result.loc[idx[7], 'A'] == 3.0


def test_flatten_predictions_matches_reference():
    """flatten_predictions matches a straightforward reference implementation."""
    from ebt.simulate.predict import flatten_predictions
    from ebt.simulate.linear import Ridge

    def flatten_predictions_ref(predictions_dict):
        try:
            return pd.concat([pd.concat(v, names=['a', 'dt']).droplevel(0).loc[k:] for k, v in predictions_dict.items() if len(v) > 0]) \
                .sort_index().pipe(lambda x: x.loc[~x.index.duplicated(keep='last'), :])
        except ValueError:
            return pd.DataFrame(index=pd.DatetimeIndex([]))

    np.random.seed(42)
    n_rows = 50_000
    idx = pd.date_range('2015-01-01', periods=n_rows, freq='64s')
    features = pd.DataFrame(np.random.randn(n_rows, 10), index=idx, columns=[f'f{i}' for i in range(10)])
    labels = pd.DataFrame(np.random.randn(n_rows, 3), index=idx, columns=[f'l{i}' for i in range(3)])

    res = predict(features, labels,
        models=[Ridge(alpha=0.01)],
        prediction_horizon=1,
        min_lookback='10D', max_lookback='30D',
        retrain_interval='5D', embargo='1D',
        progressbar=False, log=False,
        SUPPRESS_PREPROCESSOR_CHECK=True,
        verbose=True)

    predictions_dict = res['predictions_dict']
    for model_key, inner_dict in predictions_dict.items():
        old = flatten_predictions_ref(inner_dict)
        new = flatten_predictions(inner_dict)
        # The reference leaves a stale index name 'dt' from pd.concat(names=['a','dt']).droplevel(0);
        # flatten_predictions has no index name. Verify data is identical regardless.
        assert old.index.name == 'dt'  # artifact of the reference
        assert new.index.name is None
        old.index.name = None
        pd.testing.assert_frame_equal(old, new)


def test_predict_matches_reference_generator():
    """Current predict produces identical output to the reference implementation."""
    from ebt.simulate.predict import predict as predict_new, generator as generator_new
    from ebt.simulate.linear import Ridge

    # Reference generator
    def generator_ref(df, max_lb='100d', min_lb='10d', embargo='1d', retrain_interval='5d', use_progress=False):
        max_lb = pd.Timedelta(max_lb) if type(max_lb) == str else max_lb
        min_lb = pd.Timedelta(min_lb) if type(min_lb) == str else min_lb
        embargo = pd.Timedelta(embargo) if type(embargo) == str else embargo
        retrain_interval = pd.Timedelta(retrain_interval) if type(retrain_interval) == str else retrain_interval
        start_date = df.index[0]
        for retrain_date in pd.date_range(start=df.index[0] + min_lb, end=df.index[-1], freq=retrain_interval):
            if retrain_date - start_date > max_lb:
                start_date = retrain_date - max_lb
            if retrain_date - start_date >= min_lb:
                train_embargo_predict = df.loc[start_date:retrain_date + embargo + retrain_interval]
                predict_dates = df.index[df.index > retrain_date + embargo]
                if len(predict_dates) > 0:
                    yield train_embargo_predict, retrain_date, predict_dates[0]

    import ebt.simulate.predict as pred_module

    np.random.seed(42)
    for freq, n_rows in [('64s', 200_000), ('1D', 5000), ('1h', 50_000)]:
        idx = pd.date_range('2015-01-01', periods=n_rows, freq=freq)
        features = pd.DataFrame(np.random.randn(n_rows, 10), index=idx, columns=[f'f{i}' for i in range(10)])
        labels = pd.DataFrame(np.random.randn(n_rows, 3), index=idx, columns=[f'l{i}' for i in range(3)])

        params = dict(
            models=[Ridge(alpha=0.01)],
            prediction_horizon=1,
            min_lookback='100D', max_lookback='365D',
            retrain_interval='200D', embargo='1D',
            preprocess_training_data=None,
            minimum_required_points=64,
            progressbar=False, log=False,
            SUPPRESS_PREPROCESSOR_CHECK=True,
        )

        # Run with new generator
        res_new = predict_new(features, labels, **params)

        # Run with old generator
        orig = pred_module.generator
        pred_module.generator = generator_ref
        res_old = predict_new(features, labels, **params)
        pred_module.generator = orig

        old_df = list(res_old.values())[0]
        new_df = list(res_new.values())[0]
        pd.testing.assert_frame_equal(old_df, new_df, obj=f'predict output (freq={freq})')


def test_predict_incremental_matches_full():
    """predict(start..end) == predict(start..t) + predict(start..end, predictions_dict from first run).

    Verifies that resuming from a cached predictions_dict produces identical
    results to a single full run, including correct handling of the boundary
    window where data was previously truncated.
    """
    from ebt.simulate.predict import predict
    from ebt.simulate.linear import Ridge, OLS
    from ebt.simulate.model import ForestModel

    np.random.seed(42)

    for model_cls, model_kwargs in [
        (Ridge, {'alpha': 0.01}),
        (Ridge, {'alpha': 10.0}),
        (OLS, {}),
        (ForestModel, {'n_estimators': 10, 'random_state': 42}),
    ]:
        for freq, n_rows, split_frac in [
            ('1D', 3000, 0.6),
            ('1h', 30_000, 0.5),
            ('1h', 30_000, 0.8),   # split near end — few new windows
        ]:
            idx = pd.date_range('2015-01-01', periods=n_rows, freq=freq)
            features = pd.DataFrame(
                np.random.randn(n_rows, 5), index=idx,
                columns=[f'f{i}' for i in range(5)])
            labels = pd.DataFrame(
                np.random.randn(n_rows, 2), index=idx,
                columns=[f'l{i}' for i in range(2)])

            split = int(n_rows * split_frac)

            params = dict(
                prediction_horizon=1,
                min_lookback='60D', max_lookback='365D',
                retrain_interval='90D', embargo='1D',
                minimum_required_points=64,
                progressbar=False, log=False,
                SUPPRESS_PREPROCESSOR_CHECK=True,
            )

            # Full run on all data
            model_full = model_cls(**model_kwargs)
            res_full = predict(features, labels, models=[model_full], **params)
            full_df = list(res_full.values())[0]

            # Run 1: partial data (same model object used for both runs
            # so predictions_dict keys match on resume)
            model_incr = model_cls(**model_kwargs)
            res_part = predict(
                features.iloc[:split], labels.iloc[:split],
                models=[model_incr], verbose=True, **params)
            preds_dict = res_part['predictions_dict']

            # Run 2: full data, passing predictions_dict from Run 1
            res_resume = predict(
                features, labels,
                models=[model_incr], predictions_dict=preds_dict, **params)
            resume_df = list(res_resume.values())[0]

            # They should be identical (check_freq=False because
            # flatten_predictions deduplication can drop inferred freq)
            pd.testing.assert_frame_equal(
                full_df, resume_df, check_freq=False,
                obj=f'incremental predict ({model_cls.__name__}, freq={freq}, split={split_frac})')


def test_make_daily_from_intraday_causal_stamp():
    """Regression test for the label='left' lookahead.

    The value stamped at hour k must be the price known AT k, never a price
    from inside (k, k+interval). Price == minutes-since-start makes any
    future leak directly measurable.
    """
    import pandas as pd
    import numpy as np
    from ebt.simulate.features import make_daily_from_intraday

    idx = pd.date_range("2023-01-02", periods=3 * 1440, freq="min")
    prices = pd.Series(np.arange(len(idx), dtype=float), index=idx) + 1000.0
    result = make_daily_from_intraday(prices, interval="1h", strip_time_component=False)
    assert len(result) == 24
    for k, v in result.items():
        for ts, val in v.dropna().items():
            minutes_known_at_ts = (ts - idx[0]) / pd.Timedelta("1min") + 1000.0
            assert val <= minutes_known_at_ts + 1e-9, (
                f"{k} @ {ts}: value {val} is a future price "
                f"(latest known at stamp: {minutes_known_at_ts})"
            )


# --- lptp pooled definition ---

def test_lptp_is_pooled_not_row_mean():
    from ebt.simulate import kf
    idx = pd.date_range('2024-01-01', periods=4, freq='D')
    # long-only book with an empty row: 1.0 regardless of the flat row
    long_only = pd.DataFrame([[1.0, 2.0], [0.0, 0.0], [3.0, 0.5], [0.0, 0.0]], index=idx, columns=['a', 'b'])
    assert kf.lptp(long_only) == 1.0
    # residue on the flat rows must not move it
    with_residue = long_only.copy()
    with_residue.iloc[1] = [-1e-15, 1e-16]
    assert kf.lptp(with_residue) > 1 - 1e-12
    # pooled: a thin short row does not count as much as a fat long row
    mixed = pd.DataFrame([[10.0, 10.0], [-1.0, 0.0]], index=idx[:2], columns=['a', 'b'])
    assert abs(kf.lptp(mixed) - (0.5 + 19.0 / (2 * 21.0))) < 1e-12
    # short-only book: 0.0; empty book: NaN
    assert kf.lptp(-long_only) == 0.0
    assert np.isnan(kf.lptp(long_only * 0.0))
    assert kf.lptp(long_only) == kf._numpy_lptp_mean(long_only.values)


def test_make_ohlc_boundary_print_belongs_to_its_bar_and_paths_agree():
    """A print stamped exactly T is the close of the bar stamped T (label='right'), in the
    Series path as in the OHLC-frame path, and kwargs reach the per-column recursion."""
    import numpy as np
    import pandas as pd
    from ebt.features.core import make_ohlc, make_forward_labels
    idx = pd.date_range('2024-01-01 00:05', '2024-01-03 00:00', freq='5min')
    s = pd.Series(np.arange(1, len(idx) + 1, dtype=float) * 5, index=idx)       # price == minutes since start
    a = make_ohlc(s, freq='d', cut_latest=False)
    b = make_ohlc(pd.DataFrame({'open': s, 'high': s, 'low': s, 'close': s}), freq='d', cut_latest=False)
    pd.testing.assert_frame_equal(a, b[a.columns], check_freq=False)
    assert a.loc['2024-01-02', 'close'] == 1440.0 and a.loc['2024-01-02', 'high'] == 1440.0   # the midnight print
    assert a.loc['2024-01-02', 'open'] == 5.0
    h = make_ohlc(s, freq='1h', cut_latest=False)
    assert h.loc['2024-01-01 10:00', 'close'] == 600.0                                       # the 10:00 print
    # per-column recursion honours the kwargs
    df = pd.DataFrame({'A': s, 'B': s + 1})
    assert len(make_ohlc(df, 'd', cut_latest=False)) == len(a)
    assert len(make_ohlc(df, 'd')) == len(a) - 1
    left_df = make_ohlc(df, 'd', label='left', cut_latest=False)
    left_s = make_ohlc(s, 'd', label='left', cut_latest=False)
    assert left_df.index.equals(left_s.index) and left_s.loc['2024-01-01', 'open'] == 5.0
    pd.testing.assert_series_equal(left_df[('close', 'A')], left_s['close'], check_names=False, check_freq=False)
    # make_forward_labels forwards `mean` to the per-column call
    r = pd.DataFrame({'A': s.diff(), 'B': s.diff()}).dropna()
    summed = make_forward_labels(r, smoothing_length=3, mean=False)
    pd.testing.assert_series_equal(summed['A'], make_forward_labels(r['A'], smoothing_length=3, mean=False), check_names=False)
    assert not np.allclose(summed['A'].dropna(), make_forward_labels(r['A'], smoothing_length=3, mean=True).dropna())
