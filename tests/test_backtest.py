import pytest
import pandas as pd
import numpy as np
import time

from ebt.simulate.kf import (
    backtest, sharpe, sharpe_per_lag, is_price_or_return,
    _backtest_python, _backtest_rust, _HAS_RUST,
)


def _make_prices(n_cols, n_rows, freq='1min', start='2024-01-02 09:00', seed=42):
    """Generate random price-like DataFrames."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=n_rows, freq=freq)
    returns = rng.randn(n_rows, n_cols) * 0.01
    prices = pd.DataFrame(
        100 + np.cumsum(returns, axis=0),
        index=idx,
        columns=[f'col_{i}' for i in range(n_cols)],
    )
    return prices


def _make_positions(prices, density=0.3, seed=123):
    """Generate random positions (sparse, some NaNs)."""
    rng = np.random.RandomState(seed)
    mask = rng.rand(*prices.shape) < density
    vals = rng.randn(*prices.shape)
    pos = pd.DataFrame(
        np.where(mask, vals, np.nan),
        index=prices.index,
        columns=prices.columns,
    )
    return pos


@pytest.fixture
def simple_data():
    """Simple deterministic test case."""
    idx = pd.date_range('2024-01-02', periods=10, freq='h')
    prices = pd.DataFrame({
        'A': np.arange(100, 110, dtype=float),
        'B': np.arange(200, 210, dtype=float),
    }, index=idx)
    pos = pd.DataFrame({
        'A': [1.0, 1.0, np.nan, 2.0, 2.0, 2.0, np.nan, 3.0, 3.0, 3.0],
        'B': [0.5, 0.5, 0.5, 0.5, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0],
    }, index=idx)
    return pos, prices


class TestBacktestCorrectness:

    @pytest.mark.skipif(not _HAS_RUST, reason='ebt_rust not available')
    def test_rust_matches_python_simple(self, simple_data):
        pos, prices = simple_data
        for rv in ['pnl', 'pos', 'market_return']:
            py_result = _backtest_python(pos, prices, shift=0, return_variable=rv)
            rs_result = _backtest_rust(pos, prices, shift=0, return_variable=rv)
            py_vals = py_result.stack().dropna()
            rs_vals = rs_result.stack().dropna()
            common = py_vals.index.intersection(rs_vals.index)
            assert len(common) > 0, f"No common non-NaN values for {rv}"
            np.testing.assert_allclose(
                rs_vals.loc[common].values,
                py_vals.loc[common].values,
                atol=1e-10,
                err_msg=f'Mismatch for return_variable={rv}',
            )

    @pytest.mark.skipif(not _HAS_RUST, reason='ebt_rust not available')
    def test_rust_matches_python_with_shift(self, simple_data):
        pos, prices = simple_data
        for shift in [0, 1, 2, -1, 5]:
            py_result = _backtest_python(pos, prices, shift=shift, return_variable='pnl')
            rs_result = _backtest_rust(pos, prices, shift=shift, return_variable='pnl')
            py_vals = py_result.stack().dropna()
            rs_vals = rs_result.stack().dropna()
            common = py_vals.index.intersection(rs_vals.index)
            if len(common) > 0:
                np.testing.assert_allclose(
                    rs_vals.loc[common].values,
                    py_vals.loc[common].values,
                    atol=1e-10,
                    err_msg=f'Mismatch for shift={shift}',
                )

    @pytest.mark.skipif(not _HAS_RUST, reason='ebt_rust not available')
    def test_rust_matches_python_multicol(self):
        prices = _make_prices(15, 5000)
        pos = _make_positions(prices)
        for rv in ['pnl', 'pos', 'market_return']:
            py_result = _backtest_python(pos, prices, shift=2, return_variable=rv)
            rs_result = _backtest_rust(pos, prices, shift=2, return_variable=rv)
            py_vals = py_result.stack().dropna()
            rs_vals = rs_result.stack().dropna()
            common = py_vals.index.intersection(rs_vals.index)
            assert len(common) > 0
            np.testing.assert_allclose(
                rs_vals.loc[common].values,
                py_vals.loc[common].values,
                atol=1e-10,
                err_msg=f'Multi-col mismatch for {rv}',
            )

    def test_backtest_empty_dataframe(self):
        idx = pd.date_range('2024-01-02', periods=5, freq='h')
        prices = pd.DataFrame({'A': [100.0, 101, 102, 103, 104]}, index=idx)
        pos = pd.DataFrame({'A': [np.nan] * 5}, index=idx)
        result = backtest(pos, prices)
        assert not result.empty

    def test_backtest_non_overlapping(self):
        idx1 = pd.date_range('2024-01-01', periods=5, freq='h')
        idx2 = pd.date_range('2024-06-01', periods=5, freq='h')
        prices = pd.DataFrame({'A': [100.0, 101, 102, 103, 104]}, index=idx2)
        pos = pd.DataFrame({'A': [1.0, 1, 1, 1, 1]}, index=idx1)
        result = backtest(pos, prices)
        assert not result.empty

    def test_backtest_single_row(self):
        idx = pd.date_range('2024-01-02', periods=1, freq='h')
        prices = pd.DataFrame({'A': [100.0]}, index=idx)
        pos = pd.DataFrame({'A': [1.0]}, index=idx)
        result = backtest(pos, prices)
        assert not result.empty

    def test_backtest_returns_input(self):
        """Test auto-detection of returns and cumsum conversion."""
        idx = pd.date_range('2024-01-02', periods=100, freq='h')
        rng = np.random.RandomState(42)
        returns = pd.DataFrame({
            'A': rng.randn(100) * 0.01,
            'B': rng.randn(100) * 0.01,
        }, index=idx)
        pos = pd.DataFrame({
            'A': np.ones(100),
            'B': np.ones(100) * 0.5,
        }, index=idx)
        assert is_price_or_return(returns) == 'return'
        result = backtest(pos, returns)
        assert not result.empty
        assert result.shape[1] == 2


class TestFuzz:

    @pytest.mark.skipif(not _HAS_RUST, reason='ebt_rust not available')
    def test_fuzz_rust_vs_python(self):
        """100 random DataFrames, verify Rust matches Python."""
        rng = np.random.RandomState(0)
        n_tests = 100
        for i in range(n_tests):
            n_cols = rng.randint(1, 8)
            n_rows = rng.randint(10, 500)
            shift = rng.randint(-3, 4)

            prices = _make_prices(n_cols, n_rows, seed=rng.randint(0, 10000))
            pos = _make_positions(prices, density=rng.uniform(0.1, 0.8), seed=rng.randint(0, 10000))

            py_result = _backtest_python(pos, prices, shift=shift, return_variable='pnl')
            rs_result = _backtest_rust(pos, prices, shift=shift, return_variable='pnl')

            py_vals = py_result.stack().dropna()
            rs_vals = rs_result.stack().dropna()
            common = py_vals.index.intersection(rs_vals.index)
            if len(common) > 0:
                np.testing.assert_allclose(
                    rs_vals.loc[common].values,
                    py_vals.loc[common].values,
                    atol=1e-10,
                    err_msg=f'Fuzz test {i} failed (n_cols={n_cols}, n_rows={n_rows}, shift={shift})',
                )


class TestPerformance:

    @pytest.mark.skipif(not _HAS_RUST, reason='ebt_rust not available')
    def test_benchmark_rust_vs_python(self):
        """Benchmark Rust vs Python on 30 columns x 1M rows."""
        prices = _make_prices(30, 1_000_000)
        pos = _make_positions(prices, density=0.3)

        # Warmup
        _backtest_rust(pos.iloc[:1000], prices.iloc[:1000], shift=2, return_variable='pnl')

        # Python
        t0 = time.time()
        py_result = _backtest_python(pos, prices, shift=2, return_variable='pnl')
        py_time = time.time() - t0

        # Rust
        t0 = time.time()
        rs_result = _backtest_rust(pos, prices, shift=2, return_variable='pnl')
        rs_time = time.time() - t0

        print(f'\nPython: {py_time:.3f}s, Rust: {rs_time:.3f}s, speedup: {py_time/rs_time:.1f}x')

        # Verify correctness on this large dataset
        py_vals = py_result.stack().dropna()
        rs_vals = rs_result.stack().dropna()
        common = py_vals.index.intersection(rs_vals.index)
        np.testing.assert_allclose(
            rs_vals.loc[common].values,
            py_vals.loc[common].values,
            atol=1e-10,
        )


class TestSharpePerLag:

    @pytest.mark.skipif(not _HAS_RUST, reason='ebt_rust not available')
    def test_sharpe_per_lag_equivalence(self):
        """Verify batch multi-shift matches individual backtests."""
        prices = _make_prices(5, 10000)
        pos = _make_positions(prices, density=0.5)
        lookbacks = [-2, -1, 0, 1, 2, 5, 10]

        # Use the public sharpe_per_lag function (uses Rust)
        result = sharpe_per_lag(pos, prices, lookbacks=lookbacks, resample=True)

        # Compare with individual backtests
        for lb in lookbacks:
            individual = backtest(pos, prices, shift=lb).sum(axis=1).pipe(sharpe, resample=True)
            if not np.isnan(individual) and not np.isnan(result[lb]):
                np.testing.assert_allclose(
                    result[lb], individual, atol=1e-6,
                    err_msg=f'sharpe_per_lag mismatch at lookback={lb}',
                )


# ── Stacked backtest tests ────────────────────────────────────────────────────

from ebt.simulate.kf import (
    backtest_stacked, sharpe_per_lag_stacked,
    _detect_col, _resolve_columns, _is_price_or_return_stacked,
)


def _make_wide_stacked(n_timestamps=200, n_contracts=3, seed=42, overlap=True):
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range('2024-01-01', periods=n_timestamps, freq='1s')
    if overlap:
        pos_data = {f'contract_{i}': rng.randn(n_timestamps).cumsum() * 0.01 for i in range(n_contracts)}
        price_data = {f'contract_{i}': 100 + rng.randn(n_timestamps).cumsum() * 0.1 for i in range(n_contracts)}
    else:
        chunk = n_timestamps // n_contracts
        pos_data, price_data = {}, {}
        for i in range(n_contracts):
            name = f'contract_{i}'
            vp, vx = np.full(n_timestamps, np.nan), np.full(n_timestamps, np.nan)
            s, e = i * chunk, min((i + 1) * chunk, n_timestamps)
            vp[s:e] = rng.randn(e - s).cumsum() * 0.01
            vx[s:e] = 100 + rng.randn(e - s).cumsum() * 0.1
            pos_data[name], price_data[name] = vp, vx
    return pd.DataFrame(pos_data, index=timestamps), pd.DataFrame(price_data, index=timestamps)


def _wide_to_stacked_pd(df, value_col='value'):
    df = df.copy()
    df.index.name = 'ts'
    return (df.reset_index().melt(id_vars='ts', var_name='series', value_name=value_col)
            .dropna(subset=[value_col]).sort_values(['ts', 'series']).reset_index(drop=True))


def _wide_to_stacked_pl(df, value_col='value'):
    import polars as pl
    return pl.from_pandas(_wide_to_stacked_pd(df, value_col))


class TestStackedColumnDetection:
    def test_detect_time_col(self):
        assert _detect_col(['ts', 'foo'], ['ts', 'time'], 't') == 'ts'
        assert _detect_col(['time', 'x'], ['ts', 'time'], 't') == 'time'
        assert _detect_col(['foo', 'bar'], ['ts', 'time'], 't') is None

    def test_detect_contract_col(self):
        assert _detect_col(['series', 'x'], ['series', 'symbol', 'contract'], 'c') == 'series'

    def test_resolve_columns_polars(self):
        pl = pytest.importorskip('polars')
        df = pl.DataFrame({'ts': [1, 2], 'series': ['a', 'b'], 'price': [100.0, 101.0]})
        c, t, v = _resolve_columns(df, None, None, None, ['price', 'close'], 'test')
        assert (c, t, v) == ('series', 'ts', 'price')

    def test_resolve_columns_pandas_datetimeindex(self):
        df = pd.DataFrame({'series': ['a', 'b'], 'value': [1.0, 2.0]},
                          index=pd.to_datetime(['2024-01-01', '2024-01-02']))
        c, t, v = _resolve_columns(df, None, None, None, ['value'], 'test')
        assert (c, t) == ('series', '__index__')


class TestIsPriceOrReturnStacked:
    def test_prices(self):
        assert _is_price_or_return_stacked(np.arange(100, 120, dtype=float)) == 'price'

    def test_returns(self):
        assert _is_price_or_return_stacked(np.random.randn(100)) == 'return'


class TestBacktestStackedVsWide:
    @pytest.mark.parametrize("shift", [0, 1, 2])
    def test_overlapping_pandas(self, shift):
        pos_w, px_w = _make_wide_stacked(200, 3, overlap=True)
        expected = backtest(pos_w, px_w, shift=shift).sum(axis=1)
        result = backtest_stacked(_wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(px_w, 'price'),
                                  shift=shift, pos_value_col='value', price_value_col='price')
        expected, result = expected.align(result, join='inner')
        mask = expected != 0
        np.testing.assert_allclose(result[mask].values, expected[mask].values, atol=1e-10)

    @pytest.mark.parametrize("shift", [0, 2])
    def test_overlapping_polars(self, shift):
        pl = pytest.importorskip('polars')
        pos_w, px_w = _make_wide_stacked(200, 3, overlap=True)
        expected = backtest(pos_w, px_w, shift=shift).sum(axis=1)
        result = backtest_stacked(_wide_to_stacked_pl(pos_w, 'value'), _wide_to_stacked_pl(px_w, 'price'),
                                  shift=shift, pos_value_col='value', price_value_col='price')
        expected, result = expected.align(result, join='inner')
        mask = expected != 0
        np.testing.assert_allclose(result[mask].values, expected[mask].values, atol=1e-10)

    def test_non_overlapping(self):
        pos_w, px_w = _make_wide_stacked(300, 3, overlap=False)
        expected = backtest(pos_w, px_w, shift=1).sum(axis=1)
        expected = expected[expected != 0].dropna()
        result = backtest_stacked(_wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(px_w, 'price'),
                                  shift=1, pos_value_col='value', price_value_col='price')
        expected, result = expected.align(result, join='inner')
        mask = expected != 0
        np.testing.assert_allclose(result[mask].values, expected[mask].values, atol=1e-10)


class TestStackedReturnVariable:
    @pytest.mark.parametrize("rv", ['pnl', 'pos', 'market_return'])
    def test_all_return_variables(self, rv):
        pos_w, px_w = _make_wide_stacked(100, 2, overlap=True)
        expected = backtest(pos_w, px_w, shift=1, return_variable=rv).sum(axis=1)
        result = backtest_stacked(_wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(px_w, 'price'),
                                  shift=1, return_variable=rv, pos_value_col='value', price_value_col='price')
        expected, result = expected.align(result, join='inner')
        mask = expected != 0
        if mask.sum() > 0:
            np.testing.assert_allclose(result[mask].values, expected[mask].values, atol=1e-10)


class TestStackedSumContractsFalse:
    def test_per_contract_sums_to_summed(self):
        pos_w, px_w = _make_wide_stacked(100, 2, overlap=True)
        args = dict(pos_value_col='value', price_value_col='price', shift=1)
        pos_s, px_s = _wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(px_w, 'price')
        summed = backtest_stacked(pos_s, px_s, sum_contracts=True, **args)
        per = backtest_stacked(pos_s, px_s, sum_contracts=False, **args)
        assert isinstance(per, pd.DataFrame) and 'contract' in per.columns
        per_summed = per.groupby('time')['pnl'].sum().sort_index()
        summed, per_summed = summed.align(per_summed, join='inner')
        np.testing.assert_allclose(per_summed.values, summed.values, atol=1e-10)


class TestStackedReturnsInput:
    def test_auto_detect_returns(self):
        pos_w, px_w = _make_wide_stacked(200, 2, overlap=True)
        ret_w = px_w.diff()
        ret_w.iloc[0] = 0
        expected = backtest(pos_w, ret_w, shift=1).sum(axis=1)
        result = backtest_stacked(_wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(ret_w, 'return'),
                                  shift=1, pos_value_col='value', price_value_col='return')
        expected, result = expected.align(result, join='inner')
        mask = expected != 0
        if mask.sum() > 0:
            np.testing.assert_allclose(result[mask].values, expected[mask].values, atol=1e-10)


class TestStackedAutoDetect:
    def test_default_column_autodetect(self):
        pl = pytest.importorskip('polars')
        pos_w, px_w = _make_wide_stacked(100, 2, overlap=True)
        result = backtest_stacked(_wide_to_stacked_pl(pos_w, 'weight'), _wide_to_stacked_pl(px_w, 'price'), shift=1)
        assert isinstance(result, pd.Series) and len(result) > 0

    def test_symbol_fallback(self):
        pos_w, px_w = _make_wide_stacked(50, 2, overlap=True)
        pos_s = _wide_to_stacked_pd(pos_w, 'value').rename(columns={'series': 'symbol'})
        px_s = _wide_to_stacked_pd(px_w, 'price').rename(columns={'series': 'symbol'})
        result = backtest_stacked(pos_s, px_s, pos_value_col='value', price_value_col='price')
        assert len(result) > 0


class TestStackedEdgeCases:
    def test_empty_input(self):
        empty = pd.DataFrame({'ts': pd.Series(dtype='datetime64[ns]'),
                              'series': pd.Series(dtype=str), 'value': pd.Series(dtype=float)})
        result = backtest_stacked(empty, empty, pos_value_col='value', price_value_col='value')
        assert isinstance(result, pd.Series) and len(result) == 0

    def test_single_contract(self):
        pos_w, px_w = _make_wide_stacked(100, 1, overlap=True)
        expected = backtest(pos_w, px_w, shift=1).sum(axis=1)
        result = backtest_stacked(_wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(px_w, 'price'),
                                  shift=1, pos_value_col='value', price_value_col='price')
        expected, result = expected.align(result, join='inner')
        mask = expected != 0
        np.testing.assert_allclose(result[mask].values, expected[mask].values, atol=1e-10)

    def test_no_contract_overlap(self):
        pos = pd.DataFrame({'ts': pd.date_range('2024-01-01', periods=10, freq='1s'),
                            'series': 'A', 'value': np.ones(10)})
        px = pd.DataFrame({'ts': pd.date_range('2024-01-01', periods=10, freq='1s'),
                           'series': 'B', 'price': np.arange(100, 110, dtype=float)})
        assert len(backtest_stacked(pos, px, pos_value_col='value', price_value_col='price')) == 0

    def test_resample(self):
        pos_w, px_w = _make_wide_stacked(200, 2, overlap=True)
        result = backtest_stacked(_wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(px_w, 'price'),
                                  shift=1, resample=True, pos_value_col='value', price_value_col='price')
        assert isinstance(result, pd.Series)


class TestSharpePerLagStacked:
    def test_basic(self):
        pos_w, px_w = _make_wide_stacked(200, 2, overlap=True)
        result = sharpe_per_lag_stacked(
            _wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(px_w, 'price'),
            lookbacks=[0, 1, 2, 3], resample=False, pos_value_col='value', price_value_col='price')
        assert isinstance(result, pd.Series) and len(result) == 4 and not result.isna().all()


class TestStackedFuzz:
    @pytest.mark.skipif(not _HAS_RUST, reason='ebt_rust not available')
    def test_rust_vs_wide(self):
        for seed in [42, 123, 777]:
            pos_w, px_w = _make_wide_stacked(150, 4, seed=seed, overlap=True)
            pos_s, px_s = _wide_to_stacked_pd(pos_w, 'value'), _wide_to_stacked_pd(px_w, 'price')
            for shift in [0, 1, 3]:
                expected = backtest(pos_w, px_w, shift=shift).sum(axis=1)
                result = backtest_stacked(pos_s, px_s, shift=shift, pos_value_col='value', price_value_col='price')
                expected, result = expected.align(result, join='inner')
                mask = expected != 0
                np.testing.assert_allclose(result[mask].values, expected[mask].values, atol=1e-10,
                                          err_msg=f'Mismatch seed={seed} shift={shift}')


class TestTimedeltaShift:
    """Tests for timedelta-based shift in backtest / one_pager."""

    def test_timedelta_matches_integer_no_gaps(self):
        """Timedelta shifts fill at the FIRST print at-or-after t+td: on gapless
        5-min bars a signal at print k with shift='5m' fills at print k+1 and earns
        from the following increment == integer shift=2; '10m' == shift=3."""
        idx = pd.date_range('2024-01-02 09:00', periods=100, freq='5min')
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({
            'A': 100 + np.cumsum(rng.randn(100) * 0.01),
            'B': 200 + np.cumsum(rng.randn(100) * 0.01),
        }, index=idx)
        pos = pd.DataFrame({
            'A': rng.randn(100),
            'B': rng.randn(100),
        }, index=idx)

        # shift='5m' == shift=2 on 5-min bars (fill at print k+1, earn from k+2)
        pnl_td = backtest(pos, prices, shift=pd.Timedelta('5m'))
        pnl_int = backtest(pos, prices, shift=2)
        np.testing.assert_allclose(pnl_td.values, pnl_int.values, atol=1e-10)

        # shift='10m' == shift=3
        pnl_td2 = backtest(pos, prices, shift='10min')
        pnl_int2 = backtest(pos, prices, shift=3)
        np.testing.assert_allclose(pnl_td2.values, pnl_int2.values, atol=1e-10)

    def test_timedelta_with_overnight_gap(self):
        """Bars with overnight gap. shift='5m' and shift=2 produce same result
        (stamps sit on the print grid, so t+5m snaps exactly to the next print
        even across the gap)."""
        # 5-min bars: last bars of day 1 + first bars of day 2
        idx_day1 = pd.date_range('2024-01-02 15:30', periods=10, freq='5min')
        idx_day2 = pd.date_range('2024-01-03 09:30', periods=10, freq='5min')
        idx = idx_day1.append(idx_day2)

        rng = np.random.RandomState(42)
        prices = pd.DataFrame({
            'A': 100 + np.cumsum(rng.randn(20) * 0.01),
        }, index=idx)
        pos = pd.DataFrame({
            'A': rng.randn(20),
        }, index=idx)

        pnl_td = backtest(pos, prices, shift=pd.Timedelta('5m'))
        pnl_int = backtest(pos, prices, shift=2)
        np.testing.assert_allclose(pnl_td.values, pnl_int.values, atol=1e-10)

    def test_timedelta_string_and_object(self):
        """Both shift='5m' and shift=pd.Timedelta('5m') produce identical results."""
        idx = pd.date_range('2024-01-02 09:00', periods=50, freq='5min')
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({'A': 100 + np.cumsum(rng.randn(50) * 0.01)}, index=idx)
        pos = pd.DataFrame({'A': rng.randn(50)}, index=idx)

        pnl_str = backtest(pos, prices, shift='5min')
        pnl_td = backtest(pos, prices, shift=pd.Timedelta('5m'))
        np.testing.assert_allclose(pnl_str.values, pnl_td.values, atol=1e-10)

    def test_timedelta_shift_pos_return_variable(self):
        """backtest with timedelta shift and return_variable='pos' returns correctly shifted positions."""
        idx = pd.date_range('2024-01-02 09:00', periods=50, freq='10min')
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({'A': 100 + np.cumsum(rng.randn(50) * 0.01)}, index=idx)
        pos = pd.DataFrame({'A': rng.randn(50)}, index=idx)

        pos_td = backtest(pos, prices, shift='10min', return_variable='pos')
        pos_int = backtest(pos, prices, shift=2, return_variable='pos')
        np.testing.assert_allclose(pos_td.values, pos_int.values, atol=1e-10)

    def test_timedelta_zero(self):
        """shift='0s' fills at the signal's own print and earns from the next
        increment == integer shift=1 (integer shift=0 remains the pre-realized
        "caller owns the lag" convention where pos[t] earns the diff INTO t)."""
        idx = pd.date_range('2024-01-02 09:00', periods=50, freq='5min')
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({'A': 100 + np.cumsum(rng.randn(50) * 0.01)}, index=idx)
        pos = pd.DataFrame({'A': rng.randn(50)}, index=idx)

        pnl_zero_td = backtest(pos, prices, shift='0s')
        pnl_zero_int = backtest(pos, prices, shift=1)
        np.testing.assert_allclose(pnl_zero_td.values, pnl_zero_int.values, atol=1e-10)

    def test_timedelta_backtest_stacked(self):
        """Stacked format backtest with timedelta shift matches wide format."""
        pos_w, px_w = _make_wide_stacked(200, 3, overlap=True)
        pos_s = _wide_to_stacked_pd(pos_w, 'value')
        px_s = _wide_to_stacked_pd(px_w, 'price')

        # Wide format with timedelta shift
        expected = backtest(pos_w, px_w, shift='1s').sum(axis=1)
        # Stacked format with timedelta shift
        result = backtest_stacked(pos_s, px_s, shift='1s',
                                   pos_value_col='value', price_value_col='price')
        expected, result = expected.align(result, join='inner')
        mask = expected != 0
        if mask.sum() > 0:
            np.testing.assert_allclose(result[mask].values, expected[mask].values, atol=1e-10)
