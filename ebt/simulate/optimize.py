from ebt.simulate.linear import SparseMarkowitz, SparseMarkowitzCV
from ebt.simulate.kf import backtest, turnover
from ebt.simulate.predict import predict
from ebt.simulate.tricks import nanagg, combine_positions
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model as lm
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


def optimize(list_of_positions, returns, shift=2, pnls=None, model_f=None, param_grid=None, retrain_interval=30, min_lookback=300, max_lookback=None, progressbar=True):
    '''
    Optimizes a list of position dataframes, returns a weighted position plus a bunch of debug info.

    Example

        # Optimizes using LassoLarsFixedFeatures to keep the number of non zero weights constant
        optimized = ebt.optimize(
            ps, 
            returns, 
            model_f=lambda: ebt.simulate.LM_LassoLarsFixedFeatures(), 
            retrain_interval=300,
            max_lookback=[None]
        )

        # Optimizes using a gridsearch of the supplied parameter grid
        optimized = ebt.optimize(
            list(rs_series), 
            returns, 
            param_grid={'alpha': np.logspace(start=2, stop=15, base=2, num=15)}, 
            retrain_interval=300,
            max_lookback=[None]
        )

        
    '''
    progress = tqdm if progressbar else lambda x, *_, **__: x
    if pnls is None:
        pnls = pd.concat([p.dropna(how='all').sum(axis=1) for p in [backtest(p, returns, shift=shift) for p in progress(list_of_positions, desc='Calculate Pnls')]], axis=1, sort=True)

    if type(max_lookback) != list:
        max_lookback = [max_lookback]

    if model_f is None:
        model_f = lambda: lm.Ridge(fit_intercept=False, positive=True)

    if param_grid is not None:

        models = [
            SparseMarkowitzCV(
                model_or_constructor=GridSearchCV, 
                apply_default_arguments=False, 
                estimator=model_f(), 
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=5,
            )
            for _ in range(len(max_lookback))
        ]
    else: # No grid means run model as is
        models = [SparseMarkowitz(model_or_constructor=model_f()) for _ in range(len(max_lookback))]

    rs = [
        predict(features=pnls.fillna(0), 
                labels=pd.DataFrame().reindex_like(pnls).fillna(1), 
                models=[m], 
                prediction_horizon=2, 
                min_lookback=min_lookback, 
                max_lookback=l,
                retrain_interval=retrain_interval,
                progressbar=progressbar,
                log=False)
        for m, l in progress(zip(models, max_lookback), desc='Lookbacks', total=len(max_lookback))
    ]

    individual_results = []
    for l, r in zip(max_lookback, rs):
        for m in r.keys():
            weights = r[m]
            # Not `reindex_like(p)`: it fails for multi-asset positions
            # because Series.reindex_like to a multi-column DataFrame is
            # ambiguous. Reindex on the time axis only (preserves all asset
            # columns), then ffill across time.
            # combine_positions, not nanagg(nansum): nanagg treats "no print" as 0
            # (dropping a signal's held position from the sum) and converts exact-zero
            # sums to NaN, which backtest ffills into a phantom hold
            weighted_positions = combine_positions([p.mul(weights[i].shift(2).reindex(p.index, method='ffill'), axis=0) for i, p in enumerate(list_of_positions)], agg='sum')
            individual_results.append({
                'pos': weighted_positions,
                'weights': weights,
                'model': m,
                'lookback': l,
                'r': r,
            })

    total_pos = combine_positions([r['pos'] for r in individual_results], agg='sum')
    total_weight = combine_positions([r['weights'] for r in individual_results], agg='sum')
    total_weight = total_weight.div(total_weight.sum(axis=1), axis=0)

    return {
        'pos': total_pos,
        'weights': total_weight,
        'results': individual_results,
        'pnls': pnls, # For debug purposes
    }


def allocate(positions, prices=None, shift=2, lookback='500D', rebalance_interval='30D',
             risk_aversion=1.0, turnover_penalty=0.0, diversification=0.1,
             stability=0.1, max_weight=None, min_lookback='250D',
             shrinkage=0.5, max_signals=500, pnls=None, turnovers=None,
             persistence_bonus=0.0, consistency_bonus=0.0,
             tail_penalty=0.0, concentration_penalty=0.0,
             min_turnover=0.1,
             progressbar=True):
    """
    Multi-objective walk-forward portfolio allocation across trading signals.

    At each rebalance date, solves:

        maximize  w'mu  - (risk_aversion/2) w'Cov w
                        - (diversification/2) ||w||^2
                        - (stability/2) ||w - w_prev||^2
        s.t.  w >= 0,  sum(w) = 1,  w_i <= max_weight

    where mu_adj is the adjusted expected return per signal:

        mu_adj = sharpe
                 - turnover_penalty * turnover
                 + persistence_bonus * clip(variance_ratio - 1, -1, 1)
                 + consistency_bonus * consistency
                 - tail_penalty * max(0, -skewness)
                 - concentration_penalty * sharpe_cv

    Cov is the shrunk covariance of daily PnLs (annualized), and
    w_prev are the weights from the previous rebalance.

    - Covariance term deduplicates correlated signals.
    - Diversification term pushes toward equal weights (more phenomena).
    - Stability term keeps weights from jumping between rebalances.
    - Turnover penalty discounts signals that rely on frequent trading.
    - Persistence bonus rewards signals whose alpha survives smoothing.
    - Consistency bonus rewards signals that are positive in most sub-periods.
    - Tail penalty punishes signals with negative PnL skewness.
    - Concentration penalty punishes signals whose alpha is clustered in time.

    Args:
        positions: list of position DataFrames (roughly equal annualized risk)
        prices: price DataFrame (required if pnls not provided)
        shift: execution lag for backtesting (default 2)
        lookback: rolling window for metrics (default '500D')
        rebalance_interval: weight update frequency (default '30D')
        risk_aversion: covariance penalty, higher = more diversified (default 1.0)
        turnover_penalty: Sharpe discount per unit daily turnover (default 0.0)
        diversification: L2 regularization toward equal weights (default 0.1)
        stability: penalty for deviating from previous weights (default 0.1)
        max_weight: per-signal weight cap (default None = uncapped)
        min_lookback: minimum history before first allocation (default '250D')
        shrinkage: covariance shrinkage, 0=sample, 1=diagonal (default 0.5)
        max_signals: pre-filter to top-N by adjusted Sharpe (default 500)
        pnls: pre-computed daily PnL DataFrame (skips backtest if provided)
        turnovers: pre-computed daily turnover DataFrame (skips turnover calc if provided)
        persistence_bonus: reward per unit of (variance_ratio - 1); VR>1 = persistent
            alpha, VR<1 = noise-like decay. Uses 5-day variance ratio. (default 0.0)
        consistency_bonus: reward per unit of consistency (0-1); fraction of 60-day
            sub-windows with positive Sharpe within the lookback. (default 0.0)
        tail_penalty: penalty per unit of negative skewness; only applied when
            skewness < 0 (fat left tail). (default 0.0)
        concentration_penalty: penalty per unit of rolling Sharpe CV; high CV means
            alpha is clustered in time (regime-dependent). (default 0.0)
        min_turnover: minimum mean daily turnover to keep a signal. Signals below
            this threshold are filtered out before optimization — they are
            degenerate (constant positions). (default 0.1)
        progressbar: show progress bars (default True)

    Returns:
        dict:
            'pos':     combined weighted position DataFrame
            'weights': weight DataFrame (rebalance dates x signals)
            'pnls':    daily PnL DataFrame per signal
            'kfs':     list of per-rebalance diagnostic dicts
    """
    from scipy.optimize import minimize as _scipy_minimize
    from ebt.simulate.stats import shrink_cov as _shrink_cov

    progress_f = tqdm if progressbar else lambda x, **kw: x
    n = len(positions)
    if n < 2:
        raise ValueError(f"Need at least 2 signals, got {n}")

    lookback_td = pd.Timedelta(lookback)
    min_lookback_td = pd.Timedelta(min_lookback)
    rebalance_td = pd.Timedelta(rebalance_interval)
    min_obs = max(int(min_lookback_td.days * 0.5), 30)

    # ---- 1. Daily PnLs ----
    if pnls is not None:
        daily_pnls = pnls.copy()
        daily_pnls.columns = range(n)
    else:
        if prices is None:
            raise ValueError("prices required when pnls not provided")
        pnl_list = []
        for pos in progress_f(positions, desc='Backtest'):
            pnl = backtest(pos, prices, shift=shift).sum(axis=1)
            pnl_list.append(pnl.resample('D').sum().replace(0, np.nan))
        daily_pnls = pd.concat(pnl_list, axis=1, sort=True)
        daily_pnls.columns = range(n)

    # ---- 2. Validate roughly equal risk ----
    ann_risk = daily_pnls.std() * 252 ** 0.5
    valid_risk = ann_risk[(ann_risk > 0) & ann_risk.notna()]
    if len(valid_risk) > 1:
        ratio = valid_risk.max() / valid_risk.min()
        assert ratio < 3.0, (
            f"Risk ratio {ratio:.2f} exceeds 3.0x. Risk-adjust signals first "
            f"(max={valid_risk.max():.3f}, min={valid_risk.min():.3f})."
        )

    # ---- 3. Daily turnovers ----
    if turnovers is not None:
        daily_tos = turnovers.copy()
        daily_tos.columns = range(n)
    else:
        to_list = []
        for pos in progress_f(positions, desc='Turnover'):
            to_list.append(turnover(pos, resample=True))
        daily_tos = pd.concat(to_list, axis=1, sort=True)
        daily_tos.columns = range(n)

    # ---- 3b. Filter low-turnover (degenerate) signals ----
    if min_turnover > 0:
        mean_tos_all = daily_tos.mean()
        active_mask = mean_tos_all >= min_turnover
        n_filtered = (~active_mask).sum()
        if n_filtered > 0:
            _LOGGER.warning("allocate: filtered %d/%d signals with turnover < %.3f",
                            n_filtered, n, min_turnover)
            keep_idx = active_mask[active_mask].index.tolist()
            daily_pnls = daily_pnls[keep_idx]
            daily_tos = daily_tos[keep_idx]
            positions = [positions[i] for i in keep_idx]
            n = len(positions)
            daily_pnls.columns = range(n)
            daily_tos.columns = range(n)

    # ---- 4. Rebalance schedule ----
    all_dates = daily_pnls.dropna(how='all').index
    first_eligible = all_dates[0] + min_lookback_td
    eligible = all_dates[all_dates >= first_eligible]
    if len(eligible) == 0:
        raise ValueError(
            f"Insufficient data: {len(all_dates)} days, need min_lookback={min_lookback}"
        )
    rebalance_dates = [eligible[0]]
    for d in eligible[1:]:
        if d - rebalance_dates[-1] >= rebalance_td:
            rebalance_dates.append(d)

    _LOGGER.info("allocate: %d signals, %d rebalances, lookback=%s",
                 n, len(rebalance_dates), lookback)

    # ---- 5. Walk-forward optimization ----
    import time as _time
    prev_w = np.ones(n) / n
    weight_records = []
    kf_records = []
    _opt_start = _time.perf_counter()
    _n_rebalances = len(rebalance_dates)

    for _reb_i, date in enumerate(progress_f(rebalance_dates, desc='Allocate')):
        window_start = date - lookback_td
        pnl_window = daily_pnls.loc[window_start:date]
        to_window = daily_tos.loc[window_start:date]

        # Active signals: enough observations in window
        obs_count = pnl_window.notna().sum()
        active_mask = obs_count >= min_obs
        active_idx = active_mask[active_mask].index.tolist()

        if len(active_idx) < 2:
            weight_records.append(prev_w.copy())
            kf_records.append({'date': date, 'n_active': len(active_idx),
                               'effective_n': 1.0})
            continue

        pnl_active = pnl_window[active_idx]

        # Key figures
        sharpes = (pnl_active.mean() / pnl_active.std() * 252 ** 0.5).fillna(0).values
        mean_tos = to_window[active_idx].mean().fillna(0).values
        mu_adj = sharpes - turnover_penalty * mean_tos

        # Signal quality metrics (computed from daily PnLs in lookback window)
        has_quality = (persistence_bonus != 0 or consistency_bonus != 0
                       or tail_penalty != 0 or concentration_penalty != 0)
        if has_quality:
            vr, consist, skew, s_cv = _signal_quality(pnl_active)
            if persistence_bonus != 0:
                mu_adj = mu_adj + persistence_bonus * np.clip(vr - 1.0, -1.0, 1.0)
            if consistency_bonus != 0:
                mu_adj = mu_adj + consistency_bonus * consist
            if tail_penalty != 0:
                mu_adj = mu_adj - tail_penalty * np.maximum(0.0, -skew)
            if concentration_penalty != 0:
                mu_adj = mu_adj - concentration_penalty * np.clip(s_cv, 0.0, 5.0)

        n_active = len(active_idx)

        # Correlation-based pruning + hard cap
        if max_signals and n_active > max_signals:
            pnl_vals = pnl_active.fillna(0)
            corr_mat = pnl_vals.corr().values.copy()
            np.fill_diagonal(corr_mat, 1.0)
            abs_corr = np.abs(np.clip(corr_mat, -1, 1))

            # Hierarchical clustering: group signals with |corr| > corr_threshold
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            dist = 1 - abs_corr
            np.fill_diagonal(dist, 0)
            dist = (dist + dist.T) / 2
            Z = linkage(squareform(dist, checks=False), method='average')
            labels = fcluster(Z, t=0.2, criterion='distance')  # |corr| > 0.8

            # Per-cluster: keep fewer from tighter clusters (retention = 1 - mean_corr)
            keep_mask = np.zeros(n_active, dtype=bool)
            order = np.argsort(mu_adj)[::-1]  # best-adjusted-Sharpe first
            for cid in set(labels):
                members = [j for j in range(n_active) if labels[j] == cid]
                if len(members) == 1:
                    keep_mask[members[0]] = True
                    continue
                member_corrs = abs_corr[np.ix_(members, members)]
                upper = member_corrs[np.triu_indices(len(members), k=1)]
                mean_corr = upper.mean() if len(upper) > 0 else 0.0
                retention = max(1 - mean_corr, 0.05)
                n_keep = max(1, int(np.ceil(len(members) * retention)))
                # Pick best by mu_adj within cluster
                ranked = sorted(members, key=lambda j: mu_adj[j], reverse=True)
                for j in ranked[:n_keep]:
                    keep_mask[j] = True

            keep_idx = np.where(keep_mask)[0]

            # Hard cap: top max_signals by mu_adj from survivors
            if len(keep_idx) > max_signals:
                top_k = keep_idx[np.argsort(mu_adj[keep_idx])[-max_signals:]]
                keep_idx = top_k

            active_idx = [active_idx[i] for i in keep_idx]
            mu_adj = mu_adj[keep_idx]
            sharpes = sharpes[keep_idx]
            mean_tos = mean_tos[keep_idx]
            pnl_active = pnl_active[active_idx]
            n_active = len(active_idx)

        # Covariance (annualized, shrunk, positive-definite)
        cov_raw = pnl_active.fillna(0).cov().values * 252
        cov = _shrink_cov(cov_raw, shrinkage)
        eigmin = np.linalg.eigvalsh(cov).min()
        if eigmin < 1e-10:
            cov += np.eye(n_active) * (1e-10 - eigmin)

        # Previous weights mapped to active set
        w_prev_active = np.array([prev_w[i] for i in active_idx])
        s = w_prev_active.sum()
        w_prev_norm = w_prev_active / s if s > 0 else np.ones(n_active) / n_active

        # Solve QP
        w_opt = _solve_allocation_qp(
            mu_adj, cov, risk_aversion, diversification, stability,
            max_weight, w_prev_norm, n_active,
        )

        # Map to full weight vector
        w_full = np.zeros(n)
        for j, idx in enumerate(active_idx):
            w_full[idx] = w_opt[j]

        weight_records.append(w_full)
        prev_w = w_full

        kf_records.append({
            'date': date,
            'n_active': n_active,
            'effective_n': _effective_n(w_opt),
            'sharpes': pd.Series(sharpes, index=active_idx),
            'turnovers': pd.Series(mean_tos, index=active_idx),
            'mu_adj': pd.Series(mu_adj, index=active_idx),
        })

        if (_reb_i + 1) % 10 == 0 or _reb_i == _n_rebalances - 1:
            _elapsed = _time.perf_counter() - _opt_start
            _per_window = _elapsed / (_reb_i + 1)
            _eta = _per_window * (_n_rebalances - _reb_i - 1)
            _LOGGER.debug(
                f"allocate: {_reb_i+1}/{_n_rebalances} windows, "
                f"n_active={n_active}, eff_n={_effective_n(w_opt):.1f}, "
                f"{_elapsed:.1f}s elapsed, {_per_window:.2f}s/window, ETA {_eta:.0f}s")

    # ---- 6. Weight DataFrame ----
    weights_df = pd.DataFrame(
        weight_records,
        index=pd.DatetimeIndex(rebalance_dates),
        columns=range(n),
    )

    # ---- 7. Apply weights and combine positions ----
    # Shift weights forward by 1 day for causality: weight computed at end of
    # day T uses PnLs through T, so it can only be applied starting T+1.
    weights_shifted = weights_df.copy()
    # 2-day lag: weights computed at end of day T are first applied to
    # positions on day T+2 (no-lookahead with execution-buffer).
    weights_shifted.index = weights_shifted.index + pd.Timedelta('2D')

    weighted = []
    rebalance_idx = weights_shifted.index
    for i, pos in enumerate(positions):
        if weights_shifted[i].abs().max() < 1e-10:
            continue
        w_series = weights_shifted[i]
        # Union indices so weight changes take effect at rebalance dates
        idx = pos.index.union(rebalance_idx).sort_values()
        pos_ext = pos.reindex(idx).ffill()
        w_ext = w_series.reindex(idx, method='ffill')
        wp = pos_ext.mul(w_ext, axis=0).dropna(how='all')
        if not wp.empty:
            weighted.append(wp)

    if len(weighted) == 0:
        final_pos = pd.DataFrame()
    elif len(weighted) == 1:
        final_pos = weighted[0]
    else:
        final_pos = combine_positions(weighted)

    return {
        'pos': final_pos,
        'weights': weights_df,
        'pnls': daily_pnls,
        'kfs': kf_records,
    }


def _solve_allocation_qp(mu, cov, risk_aversion, diversification, stability,
                          max_weight, w_prev, n):
    """Solve the allocation QP via cvxpy OSQP.

    minimize  (1/2) w' H w  -  f' w
    s.t.      w >= 0,  sum(w) = 1,  w <= max_weight

    H = risk_aversion * Cov + (diversification + stability) * I
    f = mu + stability * w_prev
    """
    import cvxpy as cp
    import time as _t

    _t0 = _t.perf_counter()
    H = risk_aversion * cov + (diversification + stability) * np.eye(n)
    f = mu + stability * w_prev

    w = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(H)) - f @ w)

    ub = max_weight if max_weight else 1.0
    constraints = [w >= 0, w <= ub, cp.sum(w) == 1]

    prob = cp.Problem(objective, constraints)
    _t1 = _t.perf_counter()
    prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6,
               max_iter=20000, verbose=False)
    _t2 = _t.perf_counter()

    _LOGGER.debug(
        f"[QP] n={n} build={(_t1-_t0)*1000:.0f}ms solve={(_t2-_t1)*1000:.0f}ms status={prob.status}")

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        prob.solve(solver=cp.SCS, max_iters=10000, verbose=False)
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            return np.ones(n) / n

    w_val = np.maximum(w.value, 0.0)
    s = w_val.sum()
    return w_val / s if s > 0 else np.ones(n) / n


def _effective_n(w):
    """Effective number of signals (exponential Shannon entropy)."""
    w_pos = w[w > 1e-10]
    if len(w_pos) == 0:
        return 0.0
    w_norm = w_pos / w_pos.sum()
    return float(np.exp(-np.sum(w_norm * np.log(w_norm))))


def _signal_quality(pnl_df, vr_period=5, sub_window=60):
    """Compute signal quality metrics from daily PnL DataFrame.

    Args:
        pnl_df: DataFrame of daily PnLs (rows=days, cols=signals)
        vr_period: period for variance ratio (default 5 days)
        sub_window: sub-window size for consistency and concentration (default 60 days)

    Returns:
        variance_ratio: ndarray, >1 = persistent alpha, <1 = noise
        consistency: ndarray, fraction of sub-windows with positive Sharpe (0-1)
        skewness: ndarray, negative = fat left tail
        sharpe_cv: ndarray, CV of rolling sub-window Sharpe (high = lumpy alpha)
    """
    import warnings
    vals = pnl_df.values  # (n_obs, n_signals)
    n_obs, n_sig = vals.shape

    # --- Variance ratio (vr_period-day) ---
    var_1 = np.nanvar(vals, axis=0, ddof=1)
    # Rolling sum via cumsum
    cs = np.nancumsum(np.nan_to_num(vals, nan=0.0), axis=0)
    roll_sum = np.full_like(vals, np.nan)
    roll_sum[vr_period:] = cs[vr_period:] - cs[:-vr_period]
    var_n = np.nanvar(roll_sum, axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        vr = np.where(var_1 > 0, var_n / (vr_period * var_1), 1.0)
    vr = np.nan_to_num(vr, nan=1.0)

    # --- Consistency: fraction of non-overlapping sub-windows with positive Sharpe ---
    sub_w = max(sub_window, 30)
    n_windows = n_obs // sub_w
    if n_windows < 2:
        consist = np.full(n_sig, 0.5)
    else:
        positive = np.zeros(n_sig)
        total = np.zeros(n_sig)
        for i in range(n_windows):
            chunk = vals[i * sub_w:(i + 1) * sub_w]
            with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
                warnings.simplefilter('ignore')
                m = np.nanmean(chunk, axis=0)
                s = np.nanstd(chunk, axis=0, ddof=1)
                chunk_sharpe = np.where(s > 0, m / s, 0.0)
            valid = np.isfinite(chunk_sharpe)
            total += valid
            positive += valid & (chunk_sharpe > 0)
        consist = np.where(total > 0, positive / total, 0.5)

    # --- Skewness ---
    m = np.nanmean(vals, axis=0)
    s = np.nanstd(vals, axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(s > 0, (vals - m) / s, 0.0)
    n_valid = np.sum(~np.isnan(vals), axis=0).astype(float)
    skew = np.nanmean(z ** 3, axis=0) * n_valid / np.maximum(n_valid - 2, 1)
    skew = np.nan_to_num(skew, nan=0.0)

    # --- Sharpe CV: coefficient of variation of rolling sub-window Sharpe ---
    if n_windows < 3:
        s_cv = np.ones(n_sig)
    else:
        window_sharpes = np.zeros((n_windows, n_sig))
        for i in range(n_windows):
            chunk = vals[i * sub_w:(i + 1) * sub_w]
            with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
                warnings.simplefilter('ignore')
                cm = np.nanmean(chunk, axis=0)
                cs_ = np.nanstd(chunk, axis=0, ddof=1)
                window_sharpes[i] = np.where(cs_ > 0, cm / cs_, 0.0)
        ws_mean = np.mean(window_sharpes, axis=0)
        ws_std = np.std(window_sharpes, axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s_cv = np.where(np.abs(ws_mean) > 1e-10, ws_std / np.abs(ws_mean), 5.0)
        s_cv = np.nan_to_num(s_cv, nan=1.0)

    return vr, consist, skew, s_cv
