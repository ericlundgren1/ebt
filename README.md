# ebt

Backtesting and evaluation toolkit for position-based trading strategies.

You bring a price frame (DatetimeIndex × assets) and a position frame on the same assets.
ebt fits models walk-forward, turns predictions into positions, backtests them with an
execution lag, reports key figures, and sweeps strategy variants into sub-portfolios.
The hot loops (backtest, execution simulation, position combination, outlier detection)
are Rust kernels built into the package as `ebt_rust`.

## Features

- **Walk-forward prediction** (`ebt.predict`): retrain any `fit`/`predict` model on a
  rolling window with an embargo gap, for sklearn-style linear models and PyTorch nets.
- **Feature and label helpers** (`ebt.features`): multi-scale EMA filter banks, forward
  labels, OHLC resampling with causal timestamps, cross-sectional ranks.
- **Backtest and key figures** (`ebt.simulate.kf`): PnL from positions and prices with an
  integer or timedelta execution lag, Sharpe decomposition, turnover, holding time,
  Sharpe-by-lag, trend correlation. Wide (one column per asset) and stacked/long input.
- **One-pager and ensemble sweeps** (`ebt.simulate.plot`): `one_pager` evaluates one book
  and draws the standard figure; `plot_ensemble` evaluates a list of variants, splits them
  into sub-portfolios by config variable and summarises the sweep.
- **Risk sizing and sanity checks**: `risk_adjust` to a target annualised risk,
  `risk_to_lots`, cost-aware sizing (`ebt.simulate.costaware`), and a
  `position_sanity_gate` that flags broken books before anyone reads their Sharpe.
- **Execution models**: VWAP-ramped fills, spread-aware timing, trade suppression below
  a risk threshold; effective-spread estimators from daily OHLC.
- **Futures tooling** (`ebt.futures`): contract selection, rolls, Panama adjustment,
  blended rolls, interpolated and open-interest-weighted continuous series, position
  decomposition back into contracts.
- **Streaming outlier detection** (`ebt.data.outlier`): point-in-time detectors for
  trades, quotes and OHLCV bars.
- **Parallel map** (`ebt.parallel`): fork or lean-subprocess workers with zero-copy
  shared frames.

## Install

Requires Python 3.9+ and a Rust toolchain (https://rustup.rs) to build the extension.

```bash
pip install .
# editable install (the top-level ebt_rust extension needs compat mode):
pip install -e . --config-settings editable_mode=compat
```

Extras: `ebt[notebook]` (IPython widgets for the tabbed sweep views), `ebt[optimize]`
(cvxpy for the allocation optimiser), `ebt[test]`.

## Quickstart

```python
import numpy as np, pandas as pd
import ebt
from ebt.simulate.linear import Ridge
from ebt.simulate.plot import one_pager, plot_ensemble
from ebt.simulate.tricks import risk_adjust

# synthetic 5-minute prices for four assets
rng = np.random.default_rng(0)
idx = pd.date_range('2022-01-03', '2023-12-29', freq='5min')
rets = 1e-3 * rng.standard_normal((len(idx), 4))
prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=list('ABCD'))

# daily features and forward labels; a bar's timestamp is when its price is known
daily_ret = prices.resample('1D', label='right').last().dropna().pct_change()
features = ebt.apply_filters(daily_ret, lengths=[4, 16, 64]).replace([np.inf, -np.inf], np.nan).dropna()
labels = ebt.make_forward_labels(daily_ret, smoothing_length=5).reindex(features.index)

# walk-forward fit: retrain every 60 days on an expanding window, 10-day embargo
def walk_forward(model):
    out = ebt.predict(features, labels, models=[model], min_lookback='120D',
                      retrain_interval='60D', embargo='10D', progressbar=False)
    return risk_adjust(out[model], prices, shift='2min')   # scale to annualised risk 1.0

pos = walk_forward(Ridge(alpha=1.0))

# evaluate against the raw prices with a 2-minute execution lag
res = one_pager(pos, prices, shift='2min')
print(res['kf'])            # sharpe, sharpe exbias, turnover, holding(days), lptp, ...

# sweep variants and compare sub-portfolios per config variable
configs = [{'alpha': a} for a in (0.1, 1.0, 10.0)]
positions = [walk_forward(Ridge(alpha=c['alpha'])) for c in configs]
ens = plot_ensemble(positions, prices, configs_df=pd.DataFrame(configs), shift='2min')
print(ens['kfs_df'])        # one row per variant: config columns + key figures
```

Neural-net models work the same way: `ebt.simulate.nn.NeuralNetModel(create_net=FeedForwardNet,
n_hidden=[16], ...)` trains with a masked MSE loss by default; pass `create_loss_f` for your own
objective.

## Conventions

- **Timestamps mark when a value is known.** Resample prices with `label='right'`; the bar
  stamped `T` holds prints in `(T - freq, T]`. `make_ohlc` and `make_forward_labels` follow this.
- **Positions** are frames on the same assets as the prices. `NaN` means "carry the previous
  position", `0` means flat. After `risk_adjust` a book has annualised risk 1.0, so key figures
  across books are comparable.
- **Execution lag** `shift` is an integer number of price bars or a timedelta (`'2min'`,
  `pd.Timedelta('5m')`). A timedelta fills at the first price print at or after
  `stamp + shift`, so pass raw prices at a resolution finer than the signal.
- **Stacked (long) input**: `backtest_stacked` and friends accept one row per `(time, contract)`
  in pandas or Polars. Column names are auto-detected (`ts` / `time` / `timestamp` for time,
  `series` / `symbol` / `contract` for the contract, `price` / `close` / `return` for the value)
  and can be given explicitly with `time_col=`, `contract_col=`, `pos_value_col=`, `price_value_col=`.
- **Futures tools** take a `columns={...}` mapping (`time`, `contract`, `expiry`, `price`, `weight`)
  so any source schema works without renaming; outputs add a `series` column naming the
  continuous series (`'FUT(front)'`, `'FUT(30d)'`, `'FUT(top3)'`, ...).

## Key figures

Computed by `one_pager` and `plot_ensemble` ("kf" = key figure). Sharpe and risk use daily PnL.

| Key figure | Meaning |
|---|---|
| `sharpe` | Annualised Sharpe of the book's daily PnL. |
| `sharpe exbias` / `sharpe bias` | Sharpe of the timing component and of the static component. Each asset's position (in risk units) is split into its time-mean and the remainder; `bias` is the PnL of the constant mean position, `exbias` the rest. Slow books can earn most of their raw Sharpe from `bias`. |
| `sharpe long` / `sharpe short` | Sharpe of the long and short legs. |
| `sharpe w costs` | Sharpe after spread costs when `spreads=` is given. |
| `turnover` | Daily `sum(|Δpos|) / mean(|pos|)`, frequency-independent. |
| `holding(days)` | Average holding period implied by the turnover. |
| `lptp` | Long-position-to-total-position: `0.5 + sum(pos) / (2 sum(|pos|))`. 1 long-only, 0 short-only, 0.5 dollar-neutral. |
| `up_days` | Fraction of days with positive PnL. |
| `risk` | Annualised std of daily PnL (1.0 after `risk_adjust`). |
| `ncvar` | Normalised covariance ratio: the book's variance under an EWMA correlation model divided by its variance if the assets were uncorrelated. 1 for uncorrelated positions, above 1 when positions load on correlated assets, below 1 when they hedge each other. |
| `sharpe_decay` | Log-log slope of Sharpe against holding time across smoothed versions of the book; near 0 the signal persists, around -0.5 it decays like noise. |
| `skew`, `kurtosis` | Of the daily PnL. |

## Package map

| Module | Contents |
|---|---|
| `ebt.features.core` | `apply_filters`, `make_forward_labels`, `make_ohlc`, `make_daily_from_intraday`, `normalized_rank`, `ffill_timeseries` |
| `ebt.simulate.predict` | `predict` walk-forward driver, `generator`, `enough_data` |
| `ebt.simulate.model`, `.linear`, `.nn` | `Model` base class, sklearn wrappers (`Ridge`, `LassoLars`, `SparseMarkowitz`, ...), `NeuralNetModel` / `LSTMNeuralNetModel` |
| `ebt.simulate.kf` | `backtest`, `backtest_stacked`, `sharpe`, `turnover`, `holding_time`, `lptp`, `ncvar`, `sharpe_per_lag`, `holding_curve`, `sharpe_decay`, `correlation_to_trend` |
| `ebt.simulate.plot` | `one_pager`, `plot_ensemble`, `per_year_table`, figure building blocks |
| `ebt.simulate.postprocessing`, `.tricks` | `risk_adjust`, `risk_to_lots`, `combine_positions`, `split_portfolios`, `nanagg` |
| `ebt.simulate.gate` | `position_sanity_gate` |
| `ebt.simulate.grid` | `ReducedGrid`: evaluate sparse signals on the price rows that matter |
| `ebt.simulate.execution`, `.spread`, `.costaware` | execution simulators, spread estimators, cost-aware sizing and turnover budgets |
| `ebt.simulate.optimize` | ensemble weight optimisation |
| `ebt.futures` | contract schedules, rolls, continuous prices and weights, position decomposition |
| `ebt.data.outlier` | `detect_outliers` for trades, quotes and bars |
| `ebt.parallel` | `pmap` |
| `ebt_rust` | the compiled kernels (not a public API) |

## Tests

```bash
pip install -e . --config-settings editable_mode=compat
python -m pytest tests
cd ebt_rust && cargo test --profile perf
```

## License

MIT, see `LICENSE`.
