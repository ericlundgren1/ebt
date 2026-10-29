"""Low-frequency effective-spread estimators for daily OHLC panels.

Daily bars carry no quotes, so a trading-cost model built on them has to infer
the spread from the bar itself. Three estimators are implemented, all of them
purely trailing (the estimate stamped at date ``t`` uses bars up to and
including ``t`` and nothing after), so they compose with a causal backtest
without further care:

``corwin_schultz``
    Corwin & Schultz (2012, JF 67:2). The daily high-low range contains both
    volatility and spread; volatility scales with the observation interval and
    the spread does not, so a one-day range and the two-day range spanning the
    same days identify the two separately.

``abdi_ranaldo``
    Abdi & Ranaldo (2017, RFS 30:12) close-high-low. Uses the close as a proxy
    for the traded price and the mid-range as a proxy for the efficient price:
    ``S^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})]``. Fewer negative estimates
    than CS and no overnight-gap correction to get wrong.

``roll``
    Roll (1984). Bid-ask bounce induces negative first-order autocovariance in
    price changes: ``S = 2 sqrt(-cov)``. Noisy on daily data and undefined
    whenever the sample covariance is positive, which on daily data is
    roughly half the time. Kept as a cross-check, never as a primary.

``amihud``
    Amihud (2002) illiquidity, ``mean(|r| / value traded)``. Not a spread; a
    price-impact proxy on a different scale. ``fit_amihud_map`` calibrates a
    cross-sectional log-log map from it onto whichever spread estimate is
    taken as primary.

All estimators return **proportional full spreads** (ask minus bid over the
mid), on the same scale as a quoted-spread tier table -- halve them for a
one-way cost.

Conventions, all of them deliberate and all of them reversible by keyword:

* Two-day estimates are stamped at the SECOND of the two days, so a rolling
  window ending at ``t`` contains only bars at or before ``t``.
* Negative estimates are averaged in with their sign and the WINDOW MEAN is
  floored at zero (``floor="window"``, the convention of both source papers).
  Flooring each observation instead (``floor="obs"``) throws away the negative
  half of the sampling error and biases the level upward; it is available for
  a sensitivity read, not as a default.
* Zero-range bars (``high == low``) are dropped from the window average by
  default (``drop_degenerate=True``). A bar with no range is a bar in which no
  trade crossed a spread; it is a missing observation, not an observation of a
  zero spread. Keeping them (``drop_degenerate=False``) drags stale, illiquid
  names toward zero, which is the wrong direction for a cost model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "corwin_schultz", "abdi_ranaldo", "roll", "amihud",
    "cs_alpha", "ar_s2", "degenerate_mask", "rolling_estimate",
    "fit_amihud_map", "tier_spreads", "assumed_tier_panel",
    "pooled_spread", "roll_positive_share",
]

#: 3 - 2*sqrt(2), the Corwin-Schultz denominator.
_CS_DEN = 3.0 - 2.0 * np.sqrt(2.0)


def _pos_log(df: pd.DataFrame) -> pd.DataFrame:
    """log of a price panel, NaN where the price is not strictly positive."""
    x = df.astype("float64")
    return np.log(x.where(x > 0))


def degenerate_mask(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """True where the bar has no range (``high == low``) or is unusable.

    A zero-range bar is not an observation of a zero spread: both estimators
    map it to a hard zero contribution (the CS beta term collapses, the AR
    mid-range equals the close), so leaving it in drags a stale name toward
    zero. ``drop_degenerate=True`` blanks these cells instead.
    """
    h, l = high.astype("float64"), low.astype("float64")
    return ~((h > 0) & (l > 0) & (h > l))


def _valid_bar(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """True where the bar is arithmetically usable (``high >= low > 0``)."""
    h, l = high.astype("float64"), low.astype("float64")
    return (h > 0) & (l > 0) & (h >= l)


def cs_alpha(high: pd.DataFrame, low: pd.DataFrame,
             adjust_overnight: bool = True) -> pd.DataFrame:
    """Corwin-Schultz alpha for every two-day window, stamped at day 2.

    ``adjust_overnight`` applies the paper's correction: when today's range
    lies entirely above (below) yesterday's, the two-day range is inflated by
    an overnight jump that no spread was ever quoted across, so today's bar is
    shifted down (up) by the gap before the ranges are combined.
    """
    ok = _valid_bar(high, low)
    h = high.astype("float64").where(ok)
    l = low.astype("float64").where(ok)
    hp, lp = h.shift(1), l.shift(1)
    if adjust_overnight:
        gap = (l - hp).clip(lower=0) - (lp - h).clip(lower=0)
        h, l = h - gap, l - gap
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h / l) ** 2 + np.log(hp / lp) ** 2
        gamma = np.log(np.maximum(h, hp) / np.minimum(l, lp)) ** 2
        return ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DEN
                - np.sqrt(gamma / _CS_DEN))


def ar_s2(close: pd.DataFrame, high: pd.DataFrame,
          low: pd.DataFrame) -> pd.DataFrame:
    """Abdi-Ranaldo squared-spread estimate per two-day window, at day 2.

    ``S^2_t = 4 (c_{t-1} - eta_{t-1}) (c_{t-1} - eta_t)`` with
    ``eta = (log high + log low) / 2``. Stamped at ``t`` so the window is
    trailing; the paper stamps the same quantity at ``t-1``.
    """
    ok = _valid_bar(high, low)
    c = _pos_log(close).where(ok)
    eta = (_pos_log(high).where(ok) + _pos_log(low).where(ok)) / 2.0
    cp, etap = c.shift(1), eta.shift(1)
    return 4.0 * (cp - etap) * (cp - eta)


def rolling_estimate(x: pd.DataFrame, window: int, min_periods: int,
                     mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trailing mean of ``x`` over ``window`` rows, NaN-skipping.

    ``mask`` (True = drop) blanks observations before averaging;
    ``min_periods`` counts surviving observations, so a name with too few
    usable bars in the window returns NaN rather than a one-bar estimate.
    """
    if mask is not None:
        x = x.where(~mask)
    return x.rolling(window, min_periods=min_periods).mean()


def _finish(mean_est: pd.DataFrame, floor: str, square_root: bool,
            per_obs_floored: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the negative-estimate convention to a window mean.

    ``floor='none'`` keeps the sign so the treated fraction can be counted;
    for a squared estimator it returns the SIGNED root, which is the only
    sign-preserving quantity on the spread scale.
    """
    if floor not in ("window", "obs", "none"):
        raise ValueError(f"floor must be window|obs|none, got {floor!r}")
    if floor == "obs":
        if per_obs_floored is None:
            raise ValueError("floor='obs' needs the per-observation panel")
        mean_est = per_obs_floored
    if floor == "none":
        return (np.sign(mean_est) * np.sqrt(mean_est.abs()) if square_root
                else mean_est)
    return np.sqrt(mean_est.clip(lower=0.0)) if square_root \
        else mean_est.clip(lower=0.0)


def corwin_schultz(high: pd.DataFrame, low: pd.DataFrame, window: int = 63,
                   min_periods: int = 21, adjust_overnight: bool = True,
                   drop_degenerate: bool = True,
                   floor: str = "window") -> pd.DataFrame:
    """Trailing Corwin-Schultz proportional spread panel.

    Returns the full spread ``2 (e^a - 1) / (1 + e^a)`` averaged over the
    trailing window. See the module docstring for the ``floor`` and
    ``drop_degenerate`` conventions.
    """
    a = cs_alpha(high, low, adjust_overnight=adjust_overnight)
    s = 2.0 * (np.exp(a) - 1.0) / (1.0 + np.exp(a))
    bad = degenerate_mask(high, low)
    bad = (bad | bad.shift(1, fill_value=True)) if drop_degenerate else None
    m = rolling_estimate(s, window, min_periods, mask=bad)
    obs = (rolling_estimate(s.clip(lower=0.0), window, min_periods, mask=bad)
           if floor == "obs" else None)
    return _finish(m, floor, square_root=False, per_obs_floored=obs)


def abdi_ranaldo(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                 window: int = 63, min_periods: int = 21,
                 drop_degenerate: bool = True,
                 floor: str = "window") -> pd.DataFrame:
    """Trailing Abdi-Ranaldo close-high-low proportional spread panel."""
    s2 = ar_s2(close, high, low)
    bad = degenerate_mask(high, low)
    bad = (bad | bad.shift(1, fill_value=True)) if drop_degenerate else None
    m = rolling_estimate(s2, window, min_periods, mask=bad)
    obs = (rolling_estimate(s2.clip(lower=0.0), window, min_periods, mask=bad)
           if floor == "obs" else None)
    return _finish(m, floor, square_root=True, per_obs_floored=obs)


def roll(close: pd.DataFrame, window: int = 63, min_periods: int = 21,
         floor: str = "window") -> pd.DataFrame:
    """Trailing Roll (1984) proportional spread panel.

    ``S = 2 sqrt(-cov(dp_t, dp_{t-1}))``, zero where the covariance is
    non-negative (which is not rare -- see ``roll_positive_share``).
    """
    dp = _pos_log(close).diff()
    cov = dp.rolling(window, min_periods=min_periods).cov(dp.shift(1))
    if floor == "none":
        return 2.0 * np.sqrt(cov.abs()) * -np.sign(cov)
    return 2.0 * np.sqrt((-cov).clip(lower=0.0))


def roll_positive_share(close: pd.DataFrame, window: int = 63,
                        min_periods: int = 21) -> pd.DataFrame:
    """True where the Roll covariance is non-negative (estimator undefined)."""
    dp = _pos_log(close).diff()
    cov = dp.rolling(window, min_periods=min_periods).cov(dp.shift(1))
    return cov >= 0


def amihud(returns: pd.DataFrame, value_traded: pd.DataFrame,
           window: int = 63, min_periods: int = 21,
           scale: float = 1e6) -> pd.DataFrame:
    """Trailing Amihud (2002) illiquidity, ``mean(|r| / value)`` * ``scale``.

    ``value_traded`` must be in the same currency for every column
    or the cross-section is not comparable. ``scale=1e6`` reports the price
    move per million of value traded.
    """
    v = value_traded.astype("float64")
    x = returns.abs() / v.where(v > 0)
    return rolling_estimate(x, window, min_periods) * scale


def fit_amihud_map(illiq: pd.DataFrame, target: pd.DataFrame,
                   by_year: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate ``log spread = a + b log illiq`` against ``target``.

    Returns ``(fitted_spread_panel, coefficients)``. A descriptive in-sample
    map -- it is fitted on the same panel it predicts, so it cannot be used as
    independent evidence for the level, only to show whether the two liquidity
    measures order the cross-section the same way (``coefficients`` carries
    the per-year Pearson r of the logs and the observation count).
    """
    x = np.log(illiq.where(illiq > 0))
    y = np.log(target.where(target > 0))
    keys = (x.index.year if by_year else pd.Series(0, index=x.index))
    fits, rows = [], []
    for k, idx in pd.Series(keys, index=x.index).groupby(keys).groups.items():
        xi, yi = x.loc[idx].stack(), y.loc[idx].stack()
        j = xi.index.intersection(yi.index)
        xi, yi = xi.loc[j], yi.loc[j]
        good = np.isfinite(xi) & np.isfinite(yi)
        xi, yi = xi[good], yi[good]
        if len(xi) < 100:
            rows.append(dict(key=k, n=len(xi), a=np.nan, b=np.nan, r=np.nan))
            fits.append(pd.DataFrame(np.nan, index=x.loc[idx].index,
                                     columns=x.columns))
            continue
        b, a = np.polyfit(xi.values, yi.values, 1)
        r = float(np.corrcoef(xi.values, yi.values)[0, 1])
        rows.append(dict(key=k, n=len(xi), a=float(a), b=float(b), r=r))
        fits.append(np.exp(a + b * x.loc[idx]))
    return pd.concat(fits).sort_index(), pd.DataFrame(rows)


def pooled_spread(s2: pd.DataFrame, mask: pd.DataFrame | None = None,
                  weights: pd.DataFrame | None = None) -> float:
    """Pooled Abdi-Ranaldo spread: sqrt of the mean squared estimate.

    ``abdi_ranaldo`` floors each NAME's window mean at zero before taking the
    square root, and both operations are convex, so the per-name panel carries
    a positive small-sample bias that grows with volatility and shrinks with
    the window. Pooling the two-day squared estimates over many cells FIRST
    drives the sampling error of the mean to ~0, so a single flooring at the
    end costs essentially nothing. Use this for a group-level level estimate
    (a tier, a sector, a year); use ``abdi_ranaldo`` when a per-name-day
    panel is required and accept the documented floor.
    """
    x = s2.where(~mask) if mask is not None else s2
    if weights is None:
        m = float(np.nanmean(x.to_numpy(dtype="float64")))
    else:
        v = x.to_numpy(dtype="float64")
        w = np.asarray(weights.reindex_like(x), dtype="float64")
        good = np.isfinite(v) & np.isfinite(w)
        m = float((v[good] * w[good]).sum() / w[good].sum())
    return float(np.sqrt(max(m, 0.0)))


def assumed_tier_panel(adv: pd.DataFrame, tiers) -> pd.DataFrame:
    """Step-function spread panel from ``(min value traded, spread)`` tiers."""
    out = np.full(adv.shape, max(t[1] for t in tiers))
    for floor, s in sorted(tiers, key=lambda t: t[0]):
        out = np.where(adv.values >= floor, s, out)
    return pd.DataFrame(out, index=adv.index, columns=adv.columns)


def tier_spreads(measured: pd.DataFrame, assumed: pd.DataFrame,
                 mask: pd.DataFrame | None = None,
                 weights: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assumed-vs-measured table, one row per distinct assumed tier.

    ``mask`` restricts to eligible (name, day) cells; ``weights`` (e.g. traded
    notional) produces the weighted mean beside the unweighted one, which is
    the number a cost model actually responds to.
    """
    m = measured.where(mask) if mask is not None else measured
    a = assumed.where(mask) if mask is not None else assumed
    w = (weights.where(mask) if (weights is not None and mask is not None)
         else weights)
    ms, as_ = m.stack(), a.stack()
    j = ms.notna() & as_.notna()
    ms, as_ = ms[j], as_[j]
    ws = w.stack().reindex(ms.index).fillna(0.0) if w is not None \
        else None
    rows = []
    for tier, g in ms.groupby(as_):
        row = dict(assumed_bps=round(float(tier) * 1e4, 2), n_cells=int(len(g)),
                   measured_bps_mean=float(g.mean()) * 1e4,
                   measured_bps_median=float(g.median()) * 1e4,
                   measured_bps_p25=float(g.quantile(0.25)) * 1e4,
                   measured_bps_p75=float(g.quantile(0.75)) * 1e4)
        if ws is not None:
            wg = ws.loc[g.index]
            row["measured_bps_wmean"] = (float((g * wg).sum() / wg.sum()) * 1e4
                                         if wg.sum() > 0 else np.nan)
        row["ratio_mean"] = row["measured_bps_mean"] / row["assumed_bps"]
        row["ratio_median"] = row["measured_bps_median"] / row["assumed_bps"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("assumed_bps").reset_index(drop=True)
