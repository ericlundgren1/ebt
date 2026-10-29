"""Kernel-level consistency and input-validation tests (Rust extension + numpy paths)."""
import gc
import logging
import os

import numpy as np
import pandas as pd
import pytest

import ebt_rust
from ebt.simulate import kf
from ebt.simulate.plot import one_pager
from ebt.data import LazyDiskDict


def _rw(n, k, seed, start='2020-01-01', freq='D'):
    """Random-walk price frame + the RandomState that made it."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    prices = pd.DataFrame(100 + np.cumsum(rng.randn(n, k), axis=0), index=idx,
                          columns=[f'c{i}' for i in range(k)])
    return prices, rng


# ── correlation_to_trend ─────────────────────────────────────────────────────

class TestCorrelationToTrend:
    def test_rust_matches_python_fallback(self):
        prices, rng = _rw(1500, 3, 0)
        pos = pd.DataFrame(rng.randn(1500, 3), index=prices.index, columns=prices.columns)
        fls = [4, 16, 64]
        rust = kf.correlation_to_trend(pos, prices, fls)
        had = kf._HAS_RUST
        kf._HAS_RUST = False
        try:
            py = kf.correlation_to_trend(pos, prices, fls)
        finally:
            kf._HAS_RUST = had
        np.testing.assert_allclose(rust.values, py.values, atol=1e-8)

    def test_own_filter_length_correlates_one(self):
        """A book that IS the tradable trend signal (EWM of diffs through t-1) correlates
        1.0 with the matching filter length: pins the EWM definition (adjust=True,
        min_periods=fl//2) and the one-print lag of the trend leg."""
        prices, _ = _rw(800, 2, 2)
        fl = 16
        trend = prices.diff().ewm(span=fl, min_periods=fl // 2).mean().shift(1)
        r = kf.correlation_to_trend(trend, prices, [fl, 4])
        assert abs(r[fl] - 1.0) < 1e-9, r
        assert r[4] < 1 - 1e-6, r

    def test_trend_leg_is_not_lookahead(self):
        """Flat price with one jump: a lagged trend leg is flat (signal is 0 before the
        jump, price is flat after), so the correlation is 0.0. A shift=0 trend leg would
        earn alpha*jump^2 on the jump day and report 1.0 against a long book."""
        idx = pd.date_range('2021-01-01', periods=60, freq='D')
        prices = pd.DataFrame({'a': np.where(np.arange(60) < 30, 100.0, 101.0)}, index=idx)
        pos = pd.DataFrame({'a': 1.0}, index=idx)
        r = kf.correlation_to_trend(pos, prices, [4, 16])
        assert (r == 0.0).all(), r


# ── input validation: sorted indices, shapes ─────────────────────────────────

class TestInputValidation:
    def test_backtest_rejects_unsorted_index(self):
        prices, rng = _rw(50, 1, 3)
        pos = pd.DataFrame(rng.randn(50, 1), index=prices.index, columns=prices.columns)
        shuffled = pos.sample(frac=1, random_state=0)
        with pytest.raises(ValueError, match='sorted'):
            kf.backtest(shuffled, prices, shift=1)
        with pytest.raises(ValueError, match='sorted'):
            kf.backtest(pos, prices.sample(frac=1, random_state=0), shift=1)
        # sorted input is unchanged
        assert kf.backtest(shuffled.sort_index(), prices, shift=1).equals(kf.backtest(pos, prices, shift=1))

    def test_backtest_stacked_rejects_unsorted_contract_rows(self):
        idx = pd.date_range('2024-01-01', periods=40, freq='1s')
        rng = np.random.RandomState(4)
        pos_s = pd.DataFrame({'ts': idx, 'series': 'A', 'value': rng.randn(40)})
        px_s = pd.DataFrame({'ts': idx, 'series': 'A', 'price': 100 + np.cumsum(rng.randn(40))})
        kw = dict(shift=1, pos_value_col='value', price_value_col='price')
        ok = kf.backtest_stacked(pos_s, px_s, **kw)
        assert len(ok) > 0
        with pytest.raises(ValueError, match='sorted'):
            kf.backtest_stacked(pos_s.sample(frac=1, random_state=2), px_s, **kw)
        with pytest.raises(ValueError, match='sorted'):
            kf.sharpe_per_lag_stacked(pos_s, px_s.sample(frac=1, random_state=2), lookbacks=[0, 1],
                                      pos_value_col='value', price_value_col='price')
        # a frame grouped by contract (not globally time-sorted) is fine
        two = pd.concat([pos_s, pos_s.assign(series='B')], ignore_index=True)
        two_px = pd.concat([px_s, px_s.assign(series='B')], ignore_index=True)
        assert len(kf.backtest_stacked(two, two_px, **kw)) > 0

    def test_matrix_kernels_raise_value_error_not_panic(self):
        ts = np.arange(5, dtype=np.int64)
        ones = np.ones(5)
        # length mismatch (n_cols=2 but one column of data): used to be a rayon panic
        with pytest.raises(ValueError, match='expected length'):
            ebt_rust.backtest_matrix_core(ts, ones, ts, ones, 0, 0, 5, 5, 2)
        for fn in (ebt_rust.sharpe_per_lag_core, ebt_rust.backtest_matrix_multi_shift_core):
            with pytest.raises(ValueError):
                fn(ts, ones, ts, ones, [0, 1], *(() if fn is ebt_rust.backtest_matrix_multi_shift_core else (True,)), 5, 5, 2)
        with pytest.raises(ValueError):
            ebt_rust.variant_sharpes_core(ts, ones, ts, ones, 0, True, 5, 4, 1)
        with pytest.raises(ValueError):
            ebt_rust.holding_curve_core(ts, ones, ts, ones, 0, [0.0], 5, 5, 2)
        with pytest.raises(ValueError):
            ebt_rust.correlation_to_trend_core(ts, ones, ts, ones, 0, [4.0], 5, 5, 2)
        # unsorted timestamps
        bad = np.array([0, 2, 1, 3, 4], dtype=np.int64)
        with pytest.raises(ValueError, match='sorted'):
            ebt_rust.backtest_matrix_core(bad, ones, ts, ones, 0, 0, 5, 5, 1)
        with pytest.raises(ValueError, match='sorted'):
            ebt_rust.backtest_matrix_core(ts, ones, bad, ones, 0, 0, 5, 5, 1)
        # per-column API
        with pytest.raises(ValueError):
            ebt_rust.backtest_core(['a'], [bad], [ones], [ts], [ones], 0, 0)
        with pytest.raises(ValueError):
            ebt_rust.backtest_core(['a'], [ts], [ones[:4]], [ts], [ones], 0, 0)

    def test_combine_positions_core_validation(self):
        target = np.arange(5, dtype=np.int64)
        ok = ebt_rust.combine_positions_core(target, [target[1:4]], [np.ones(3)], [3], [1],
                                             [np.zeros(1, dtype=np.uint32)], 1, False)
        np.testing.assert_allclose(ok[0], [np.nan, 1, 1, 1, 1])
        with pytest.raises(ValueError, match='not in target'):
            ebt_rust.combine_positions_core(target, [np.array([1, 7], dtype=np.int64)], [np.ones(2)], [2], [1],
                                            [np.zeros(1, dtype=np.uint32)], 1, False)
        with pytest.raises(ValueError, match='n_target_cols'):
            ebt_rust.combine_positions_core(target, [target], [np.ones(5)], [5], [1],
                                            [np.array([3], dtype=np.uint32)], 1, False)
        with pytest.raises(ValueError, match='expected length'):
            ebt_rust.combine_positions_core(target, [target], [np.ones(4)], [5], [1],
                                            [np.zeros(1, dtype=np.uint32)], 1, False)

    def test_vwap_core_validation(self):
        ns = 60_000_000_000
        t = np.array([0, ns, 2 * ns], dtype=np.int64)
        with pytest.raises(ValueError, match='expected length'):
            ebt_rust.simulate_vwap_execution_core(t, np.ones(3), np.ones(2), 3, 1, 3 * ns)
        with pytest.raises(ValueError, match='sorted'):
            ebt_rust.simulate_vwap_execution_core(t[::-1].copy(), np.ones(3), np.ones(3), 3, 1, 3 * ns)


# ── VWAP kernel ──────────────────────────────────────────────────────────────

def test_vwap_nan_target_mid_window_still_reaches_target():
    ns = 60_000_000_000
    t = np.array([0, ns, 2 * ns], dtype=np.int64)
    out = ebt_rust.simulate_vwap_execution_core(t, np.array([1.0, np.nan, 1.0]), np.full(3, 100.0), 3, 1, 3 * ns)
    np.testing.assert_allclose(out, [1 / 3, 1 / 3, 1.0])


# ── kernel consistency ───────────────────────────────────────────────────────

class TestKernelConsistency:
    def test_rust_and_python_backtest_agree_including_nan_pattern(self):
        prices, rng = _rw(300, 4, 8, freq='h')
        pos = pd.DataFrame(np.where(rng.rand(300, 4) < 0.3, rng.randn(300, 4), np.nan),
                           index=prices.index, columns=prices.columns)
        for shift in (0, 1, 3):
            rs = kf._backtest_rust(pos, prices, shift, 'pnl')
            py = kf._backtest_python(pos, prices, shift, 'pnl')
            pd.testing.assert_frame_equal(rs, py, check_names=False, atol=1e-10)

    def test_holding_curve_span0_sharpe_equals_backtest_sharpe_on_gapless_prices(self):
        prices, rng = _rw(600, 2, 7, freq='h')
        sig = pd.DataFrame(rng.randn(600, 2), index=prices.index, columns=prices.columns)
        pos = kf.backtest(sig, prices, shift=1, return_variable='pos')
        ref = kf.sharpe(kf.backtest(pos, prices, shift=0).sum(axis=1), resample=True)
        hc = kf.holding_curve(pos, prices, shift=0, ewm_spans=[0.0])
        assert abs(hc[0][0] - ref) < 1e-9

    def test_variant_sharpes_decompose_the_book(self):
        prices, rng = _rw(400, 2, 5)
        pos = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        v = kf._variant_sharpes_rust(pos, prices, 0, True)
        s = kf.sharpe(kf.backtest(pos, prices, shift=0).sum(axis=1), resample=True)
        assert abs(v['bias'] - s) < 1e-9 and abs(v['long'] - s) < 1e-9
        assert np.isnan(v['exbias']) and np.isnan(v['short'])     # no timing component, no short leg

    def test_numpy_turnover_and_holding_match_pandas(self):
        prices, rng = _rw(500, 3, 6, freq='h')                      # spans weekends
        pos = pd.DataFrame(rng.randn(500, 3), index=prices.index, columns=prices.columns)
        ts = kf._index_to_ns(pos.index)
        dk = ts // 86_400_000_000_000
        dow = (dk + 3) % 7
        bk = dk - np.where(dow >= 5, dow - 4, 0)
        assert np.isclose(kf.turnover(pos).mean(), kf._numpy_turnover_mean(pos.values, bk, np.unique(bk)))
        assert np.isclose(kf.holding_time(pos), kf._numpy_holding_time(pos.values, bk, np.unique(bk)))


# ── one_pager / plot_ensemble error handling ─────────────────────────────────

def test_one_pager_empty_signal_raises():
    prices, _ = _rw(100, 2, 4, freq='h')
    with pytest.raises(ValueError, match='empty'):
        one_pager(pd.DataFrame(index=prices.index[:5]), prices, shift='2min', plot=False)


def test_plot_ensemble_reports_failed_signals(monkeypatch, caplog):
    import matplotlib
    matplotlib.use('Agg')
    import ebt.simulate.plot as plot_mod
    prices, rng = _rw(24 * 60, 2, 9, freq='h')
    daily = prices.index[::24]
    sigs = [pd.DataFrame(rng.randn(len(daily), 2) * (1000.0 if i == 1 else 1.0), index=daily, columns=prices.columns)
            for i in range(3)]
    orig = plot_mod._one_pager_numpy_fast

    def boom(signal_vals, *a, **k):
        if np.nanmax(np.abs(signal_vals)) > 100:
            raise RuntimeError('synthetic worker failure')
        return orig(signal_vals, *a, **k)

    monkeypatch.setattr(plot_mod, '_one_pager_numpy_fast', boom)
    with caplog.at_level(logging.WARNING, logger='ebt.simulate.plot'):
        res = plot_mod.plot_ensemble(sigs, prices, configs_df=pd.DataFrame({'p': [1, 2, 3]}), shift='2min',
                                     show_tabs=False, save_memory=False, add_correlation=False,
                                     add_diversification=False, add_ncvar=False, reduce=False, n_workers=1)
    assert res['failed_signals'] == [1]
    assert any('synthetic worker failure' in r.getMessage() and r.exc_info for r in caplog.records)
    assert any('1/3 signals failed' in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(plot_mod, '_one_pager_numpy_fast', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('all fail')))
    with pytest.raises(RuntimeError, match='all 3 signals failed'):
        plot_mod.plot_ensemble(sigs, prices, configs_df=pd.DataFrame({'p': [1, 2, 3]}), shift='2min',
                               show_tabs=False, save_memory=False, add_correlation=False,
                               add_diversification=False, add_ncvar=False, reduce=False, n_workers=1)


# ── LazyDiskDict ─────────────────────────────────────────────────────────────

def test_lazy_disk_dict_safe_keys_private_dir_and_cleanup():
    d = LazyDiskDict({'a/b': 1, 'a%2Fb': 2, 3: [1, 2]})
    assert d['a/b'] == 1 and d['a%2Fb'] == 2 and d[3] == [1, 2]
    path = d._dir
    files = os.listdir(path)
    assert len(files) == 3 and all(os.sep not in f for f in files)
    e = LazyDiskDict({'a/b': 'other'})
    assert e._dir != path and d['a/b'] == 1 and e['a/b'] == 'other'
    del d['a/b']
    assert 'a/b' not in d and len(os.listdir(path)) == 2
    del d
    gc.collect()
    assert not os.path.exists(path)
    assert os.path.exists(e._dir)


# ── tz handling ──────────────────────────────────────────────────────────────

class TestTimezones:
    def test_both_aware_different_zones_are_the_same_instants(self):
        prices, rng = _rw(200, 1, 10, freq='h')
        prices_utc = prices.tz_localize('UTC')
        pos_utc = pd.DataFrame(rng.randn(200, 1), index=prices_utc.index, columns=prices.columns)
        pos_ny = pos_utc.tz_convert('America/New_York')
        a = kf.backtest(pos_utc, prices_utc, shift=1)
        b = kf.backtest(pos_ny, prices_utc, shift=1)
        pd.testing.assert_frame_equal(a, b)
        assert kf.sharpe_per_lag(pos_utc, prices_utc, lookbacks=[0, 1]).equals(
            kf.sharpe_per_lag(pos_ny, prices_utc, lookbacks=[0, 1]))
        assert kf._variant_sharpes_rust(pos_utc, prices_utc, 0, True) == kf._variant_sharpes_rust(pos_ny, prices_utc, 0, True)
        assert kf.holding_curve(pos_utc, prices_utc, 0, [0.0]) == kf.holding_curve(pos_ny, prices_utc, 0, [0.0])

    def test_mixed_aware_and_naive_raise(self):
        prices, rng = _rw(200, 1, 11, freq='h')
        pos = pd.DataFrame(rng.randn(200, 1), index=prices.index, columns=prices.columns)
        with pytest.raises(ValueError, match='tz-aware'):
            kf.backtest(pos.tz_localize('UTC'), prices, shift=1)
        with pytest.raises(ValueError, match='tz-naive'):
            kf.backtest(pos, prices.tz_localize('UTC'), shift=1)
        with pytest.raises(ValueError, match='tz-'):
            one_pager(pos.tz_localize('UTC'), prices, shift='2min', plot=False, fast=True,
                      return_subset={'kf'}, kfs=['sharpe'], reduce=False)
        # both naive: unchanged
        kf.backtest(pos, prices, shift=1)


# ── holding_curve on raw (gappy) prices ──────────────────────────────────────

def test_holding_curve_on_raw_prices_matches_backtest_sharpe():
    prices, rng = _rw(400, 2, 12, start='2024-01-01 09:00', freq='h')
    prices.loc[(prices.index.hour < 9) | (prices.index.hour > 16), :] = np.nan   # closed bars
    sig = pd.DataFrame(rng.randn(400, 2), index=prices.index, columns=prices.columns)
    pos = kf.backtest(sig, prices, shift=1, return_variable='pos')
    ref = kf.sharpe(kf.backtest(pos, prices, shift=0).sum(axis=1), resample=True)
    assert abs(kf.holding_curve(pos, prices, shift=0, ewm_spans=[0.0])[0][0] - ref) < 1e-9


# ── flat-row rule, resample=False ────────────────────────────────────────────

def test_sharpe_per_lag_resample_false_applies_flat_rtol_like_variant_kernel():
    day = 86_400_000_000_000
    ts = np.arange(7, dtype=np.int64) * day
    pos = np.ones(7)
    price = np.array([100.0, 101.0, 101.0 + 1e-10, 101.0 + 1e-10, 101.0 + 2e-10, 103.0, 102.0])
    for rtol in (0.0, 1e-9):
        a = ebt_rust.sharpe_per_lag_core(ts, pos, ts, price, [0], False, 7, 7, 1, flat_rtol=rtol)[0]
        b = ebt_rust.variant_sharpes_core(ts, pos, ts, price, 0, False, 7, 7, 1, flat_rtol=rtol)[1]
        assert abs(a - b) < 1e-12, (rtol, a, b)


# ── neural-net model wiring ──────────────────────────────────────────────────

class TestNeuralNetModel:
    @staticmethod
    def _xy(n=80, seed=13):
        rng = np.random.RandomState(seed)
        idx = pd.date_range('2020-01-01', periods=n, freq='D')
        x = pd.DataFrame(rng.randn(n, 3), index=idx, columns=list('xyz'))
        y = pd.DataFrame(rng.randn(n, 2), index=idx, columns=list('ab'))
        return x, y

    def test_feedforward_fit_predict_and_progress_flag(self):
        import torch
        from ebt.simulate.nn import NeuralNetModel, FeedForwardNet
        x, y = self._xy()
        m = NeuralNetModel(create_net=FeedForwardNet, n_hidden=[4], dropout_p=[0.0, 0.0],
                           active_f=[torch.nn.GELU(), torch.nn.Identity()], n_epochs=2, lr=1e-3,
                           factor=0.5, patience=3, progress=True, progress_batch=True)
        m.fit(x, y)
        assert m.scheduler.factor == 0.5 and m.scheduler.patience == 3      # forwarded by signature
        assert m.optim.param_groups[0]['lr'] == 1e-3
        p = m.predict(x, y)
        assert p.shape == y.shape and np.isfinite(p.values).all()

    def test_unknown_hyperparameter_is_reported(self, caplog):
        from ebt.simulate.nn import NeuralNetModel, FeedForwardNet, MSELoss
        with caplog.at_level(logging.WARNING, logger='ebt.simulate.nn.model'):
            NeuralNetModel(create_net=FeedForwardNet, n_epochs=1, progressbar=False)   # loss factory has **kwargs
        assert any("['progressbar']" in r.getMessage() for r in caplog.records)
        with pytest.raises(ValueError, match="unknown hyperparameters \\['typo'\\]"):
            NeuralNetModel(create_net=FeedForwardNet, create_loss_f=lambda: MSELoss(), n_epochs=1, typo=1)

    def test_lstm_fit_predict(self):
        from ebt.simulate.nn.model import LSTMNeuralNetModel
        x, y = self._xy(n=60)
        m = LSTMNeuralNetModel(n_epochs=1, hidden_size=2, sequence_length=5, dropout_lstm=0.0, dropout_linear=0.0)
        m.fit(x, y)
        p = m.predict(x, y)
        assert p.shape == (60 - 5, 2) and p.index[0] == x.index[5] and np.isfinite(p.values).all()


# ── holding curve: fast path == full path ────────────────────────────────────

def test_sharpe_decay_and_holding_axis_agree_between_fast_and_full_path():
    rng = np.random.RandomState(7)
    n = 288 * 120
    idx = pd.date_range('2022-01-03', periods=n, freq='5min')
    prices = pd.DataFrame(100 + np.cumsum(rng.randn(n, 3) * 0.05 + 0.004, axis=0), index=idx, columns=list('ABC'))
    prices.loc[(idx.hour < 8) | (idx.hour >= 17), :] = np.nan
    sig = pd.DataFrame(1.0 + 0.3 * rng.randn(n, 3), index=idx, columns=list('ABC')).iloc[::48]
    kfs = ['sharpe', 'sharpe_decay', 'holding(days)']
    fast = one_pager(sig, prices, shift='2min', kfs=kfs, fast=True, plot=False, return_subset={'kf'}, reduce=False)['kf']
    full = one_pager(sig, prices, shift='2min', kfs=kfs, fast=False, plot=False, return_subset={'kf'}, reduce=False)['kf']
    for k in kfs:
        assert np.isclose(fast[k], full[k], rtol=1e-9, atol=1e-12), (k, fast[k], full[k])
    # the span-0 point of the holding axis is the holding(days) key figure itself, on both paths
    import ebt.simulate.plot as plot_mod
    r = one_pager(sig, prices, shift='2min', kfs=['holding(days)'], fast=True, plot=False,
                  return_subset={'kf', 'pos'}, reduce=False)
    pos, holding = r['pos'], r['kf']['holding(days)']
    prices_t = prices.loc[pos.index]                 # one_pager cuts prices at the signal's last stamp
    vol = plot_mod._estimate_asset_vol(prices_t).reindex(columns=prices_t.columns).reindex(index=prices_t.index, method='ffill')
    hc = plot_mod._compute_holding_curve(pos, prices_t, 0, None, resample=True, volatility=vol)
    assert np.isclose(hc.index[0], holding, rtol=1e-9)
    dk = kf._bday_keys(kf._index_to_ns(prices_t.index) // 86_400_000_000_000)
    h = kf._holding_curve_fast_arrays(np.ascontiguousarray(kf._numpy_ffill(pos.values).T),
                                      np.ascontiguousarray(vol.values.T), [0.0, 2.0], dk, np.unique(dk))   # spans 0 and 2**(2/2)
    assert np.isclose(h[0], holding, rtol=1e-9) and np.isclose(h[1], hc.index[1], rtol=1e-9)
