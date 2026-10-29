"""Futures contract tools: contract selection, rolls, continuous series and position decomposition.

Everything here operates on data already in memory. Inputs are long-format
frames with one row per (time, contract); a ``columns`` mapping says which
input column plays which role, so data from any source can be used without
renaming first:

    ``time``      timestamp column of the bars/ticks          (default ``'time'``)
    ``contract``  contract identifier                          (default ``'contract'``)
    ``expiry``    expiry date column in the expiries frame     (default ``'expiry'``)
    ``price``     price used for continuous prices / spreads   (default ``'close'``)
    ``weight``    value used by top-N weighting (OI, volume)   (default ``'open_interest'``)

Outputs keep the caller's ``time`` and ``contract`` column names and add
``series`` (e.g. ``'FUT(front)'``, ``'FUT(30d)'``, ``'FUT(rank1)'``, ``'FUT(top3)'``
when ``name='FUT'``; without a name just ``'front'``, ``'30d'``, ...).

Typical use::

    cols = {'time': 'ts', 'contract': 'symbol', 'expiry': 'expiration'}
    px = continuous_prices(bars, expiries, expiry_rank=[1], roll='blend', panama=True, columns=cols)
    w = continuous_weights(bars, expiries, target_days=[30, 60], columns=cols)
    mapping = contract_mapping(bars, expiries, expiry_rank=[1], columns=cols)
    pos_by_contract = decompose_positions(positions, mapping, columns=cols)

Days to expiry are calendar days from the trading date to the expiry date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, Mapping, Optional, Union

import numpy as np
import polars as pl

_LOGGER = logging.getLogger(__name__)

# Internal canonical column names
_T, _C, _E, _D = "time", "contract", "expiry", "_date"
_SYN, _DTE = "series", "days_to_expiry"


@dataclass(frozen=True)
class Columns:
    """Which input columns play which role."""
    time: str = "time"
    contract: str = "contract"
    expiry: str = "expiry"
    price: str = "close"
    weight: str = "open_interest"

    @classmethod
    def make(cls, columns: Union["Columns", Mapping[str, str], None]) -> "Columns":
        if columns is None:
            return cls()
        if isinstance(columns, Columns):
            return columns
        unknown = set(columns) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown column roles {sorted(unknown)}; valid: {sorted(cls.__dataclass_fields__)}")
        return cls(**columns)


# ── helpers ──────────────────────────────────────────────────────────────

def _series_name(name: Optional[str], tag: str) -> str:
    return f"{name}({tag})" if name else tag


def _to_polars(df) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return pl.from_pandas(df)


def _date_expr(time_col: str, right_labelled: bool) -> pl.Expr:
    """Trading date of a stamp: its calendar date in the stamp's own time zone (not UTC).

    For right-labelled bars (stamped at the end of their interval) a stamp exactly at
    midnight closes the previous day's last bar, so it belongs to the previous date;
    any stamp after midnight is on its own date. No bar length or time resolution is
    assumed; the rule is exact whenever midnight is a bar boundary.
    """
    t = pl.col(time_col)
    date = t.dt.date()
    if right_labelled:
        date = pl.when(t.dt.time() == pl.time(0)).then(date.dt.offset_by("-1d")).otherwise(date)
    return date.alias(_D)


def _to_date(x):
    if x is None:
        return None
    import pandas as pd
    return pd.Timestamp(x).date()


def _prepare_bars(bars, cols: Columns, right_labelled: bool, keep: tuple[str, ...] = ()) -> pl.DataFrame:
    """Rename the caller's columns to canonical names and add the trading date."""
    df = _to_polars(bars)
    rename = {cols.time: _T, cols.contract: _C}
    for k in keep:
        src = getattr(cols, k)
        if src in df.columns and src not in rename:
            rename[src] = k
    missing = [c for c in (cols.time, cols.contract) if c not in df.columns]
    if missing:
        raise KeyError(f"bars is missing columns {missing}; pass columns= to map them")
    return df.rename(rename).with_columns(_date_expr(_T, right_labelled))


def _prepare_expiries(expiries, cols: Columns) -> pl.DataFrame:
    if isinstance(expiries, Mapping):
        df = pl.DataFrame({_C: list(expiries.keys()), _E: list(expiries.values())})
    else:
        df = _to_polars(expiries).rename({cols.contract: _C, cols.expiry: _E}).select(_C, _E)
    return df.with_columns(pl.col(_E).cast(pl.Date)).unique(subset=[_C], keep="first")


def _restore(df: pl.DataFrame, cols: Columns) -> pl.DataFrame:
    rename = {k: v for k, v in ((_T, cols.time), (_C, cols.contract)) if k in df.columns and k != v}
    return df.rename(rename)


def microprice(bid_price="bid_price", ask_price="ask_price", bid_size="bid_size", ask_size="ask_size") -> pl.Expr:
    """Size-weighted mid of a quote; falls back to the plain mid when both sizes are zero."""
    total = pl.col(bid_size) + pl.col(ask_size)
    return (
        pl.when(total > 0)
        .then((pl.col(bid_price) * pl.col(ask_size) + pl.col(ask_price) * pl.col(bid_size)) / total)
        .otherwise((pl.col(bid_price) + pl.col(ask_price)) / 2.0)
    )


# ── schedule and contract selection ─────────────────────────────────────

def build_schedule(bars, expiries, columns=None, right_labelled: bool = False) -> pl.DataFrame:
    """Daily contract schedule: every trading date on which each contract is live.

    The calendar is the union of dates present in ``bars``. A contract is live on
    every calendar date from its first observation through its expiry, whether or
    not it has data that day, so data gaps can't cause flip-backs in selection.

    Returns columns ``date, contract, expiry, days_to_expiry`` (caller's contract name).
    """
    cols = Columns.make(columns)
    df = _prepare_bars(bars, cols, right_labelled)
    exp = _prepare_expiries(expiries, cols)

    first = df.group_by(_C).agg(pl.col(_D).min().alias("_first"))
    calendar = df.select(pl.col(_D).alias("date")).unique()
    info = first.join(exp, on=_C, how="inner")
    unknown = set(first[_C].to_list()) - set(info[_C].to_list())
    if unknown:
        _LOGGER.warning(f"build_schedule: no expiry for {len(unknown)} contracts, dropped: {sorted(unknown)[:5]}")

    sched = (
        calendar.join(info, how="cross")
        .filter((pl.col("date") >= pl.col("_first")) & (pl.col("date") <= pl.col(_E)))
        .drop("_first")
        .with_columns((pl.col(_E) - pl.col("date")).dt.total_days().cast(pl.Int32).alias(_DTE))
        .sort(["date", _DTE])
        .select("date", _C, _E, _DTE)
    )
    return _restore(sched, cols)


def _enforce_monotonic_expiration(mapping: pl.DataFrame, filtered: pl.DataFrame, min_dte: int) -> pl.DataFrame:
    """Selected contract's expiry never decreases over time, per series.

    When a late-appearing contract would cause a backward roll, keep the previous
    contract if it still has at least ``min_dte`` days left that day; otherwise
    accept the new one.
    """
    exp_map = dict(filtered.select(_C, _E).unique(subset=[_C]).iter_rows())
    mapping_cids = set(mapping[_C].unique().to_list())
    dte_index = {
        (d, c): dte
        for d, c, dte in filtered.filter(pl.col(_C).is_in(mapping_cids)).select("date", _C, _DTE).iter_rows()
    }

    parts = []
    for syn in mapping[_SYN].unique(maintain_order=True).to_list():
        sub = mapping.filter(pl.col(_SYN) == syn).sort(_D)
        dates, cids, dtes = sub[_D].to_list(), sub[_C].to_list(), sub[_DTE].to_list()
        max_exp = max_cid = None
        changed = False
        for i in range(len(dates)):
            exp = exp_map.get(cids[i])
            if max_exp is not None and exp is not None and exp < max_exp:
                prev_dte = dte_index.get((dates[i], max_cid))
                if prev_dte is not None and prev_dte >= min_dte:
                    cids[i], dtes[i] = max_cid, prev_dte
                    changed = True
                else:
                    max_exp, max_cid = exp, cids[i]
            elif exp is not None and (max_exp is None or exp > max_exp):
                max_exp, max_cid = exp, cids[i]
        if changed:
            parts.append(pl.DataFrame({_D: dates, _C: cids, _SYN: [syn] * len(dates), _DTE: dtes},
                                      schema={_D: pl.Date, _C: sub.schema[_C], _SYN: pl.Utf8, _DTE: pl.Int32}))
        else:
            parts.append(sub)
    return pl.concat(parts) if parts else mapping


def _check_roll_history(mapping: pl.DataFrame, schedule: pl.DataFrame, min_history: int) -> None:
    """Refuse to roll into a contract first observed fewer than ``min_history`` calendar days
    before the roll: a contract that appears in the data just as it becomes the selected one
    almost always means its history is missing from the feed. Contracts present from the
    first date of the data are exempt (their earlier history simply isn't in the frame)."""
    if min_history <= 0:
        return
    rolls = _detect_rolls(mapping)
    if rolls.is_empty():
        return
    first = schedule.group_by(_C).agg(pl.col("date").min().alias("_first"))
    data_start = schedule["date"].min()
    bad = (
        rolls.join(first, left_on="new_contract", right_on=_C, how="left")
        .with_columns((pl.col("roll_date") - pl.col("_first")).dt.total_days().alias("_hist"))
        .filter((pl.col("_first") > data_start) & (pl.col("_hist") < min_history))
    )
    if not bad.is_empty():
        rows = "; ".join(f"{syn}: {old}->{new} on {rd}, {new} first seen {h} days earlier"
                         for syn, rd, old, new, h in bad.select(_SYN, "roll_date", "old_contract", "new_contract", "_hist").iter_rows())
        raise ValueError(f"roll into a contract with less than min_history={min_history} days of history "
                         f"(missing data? pass min_history=0 to allow): {rows}")


def _select(schedule: pl.DataFrame, target_days, expiry_rank, min_dte, start, end, name,
            min_history: int = 30) -> pl.DataFrame:
    """Canonical-name core of ``select_contracts``. ``schedule`` has date, contract, expiry, days_to_expiry."""
    empty = pl.DataFrame(schema={_D: pl.Date, _C: schedule.schema.get(_C, pl.Utf8), _SYN: pl.Utf8, _DTE: pl.Int32})
    if target_days is not None and expiry_rank is not None:
        raise ValueError("target_days and expiry_rank are mutually exclusive")
    use_rank = expiry_rank is not None
    if not use_rank and target_days is None:
        target_days = [0]

    lf = schedule.lazy().filter(pl.col(_DTE) >= min_dte)
    if start is not None:
        lf = lf.filter(pl.col("date") >= _to_date(start))
    if end is not None:
        lf = lf.filter(pl.col("date") < _to_date(end))
    filtered = lf.collect()
    if filtered.is_empty():
        return empty

    def pick(ranked: pl.LazyFrame, rank: int, syn: str) -> pl.DataFrame:
        return (
            ranked.filter(pl.col("_rank") == rank)
            .select(pl.col("date").alias(_D), _C, pl.lit(syn).alias(_SYN), _DTE)
            .collect()
        )

    by_dte = filtered.lazy().sort(["date", _DTE]).with_columns(pl.col(_C).cum_count().over("date").alias("_rank"))
    mappings = []
    if use_rank:
        for r in expiry_rank:
            mappings.append(pick(by_dte, r, _series_name(name, f"rank{r}")))
    else:
        for target in target_days:
            if target == 0:
                mappings.append(pick(by_dte, 1, _series_name(name, "front")))
            else:
                ranked = (
                    filtered.lazy()
                    .with_columns((pl.col(_DTE) - target).abs().alias("_dist"))
                    .sort(["date", "_dist", _DTE])
                    .with_columns(pl.col(_C).cum_count().over("date").alias("_rank"))
                )
                mappings.append(pick(ranked, 1, _series_name(name, f"{target}d")))
    mapping = _enforce_monotonic_expiration(pl.concat(mappings), filtered, min_dte)
    _check_roll_history(mapping, schedule, min_history)
    return mapping


def select_contracts(schedule, target_days: Optional[list[int]] = None, expiry_rank: Optional[list[int]] = None,
                     min_dte: int = 7, start=None, end=None, name: Optional[str] = None,
                     columns=None, min_history: int = 30) -> pl.DataFrame:
    """Pick one contract per trading date for each requested continuous series.

    Args:
        schedule: output of ``build_schedule``.
        target_days: target days-to-expiry; per date the contract closest to each
            target (ties to the nearer expiry). ``0`` (or None) means front month.
        expiry_rank: alternative to target_days; 1 = nearest expiry, 2 = next, ...
        min_dte: contracts with fewer days to expiry are excluded, so the front month is
            the nearest contract that still has at least ``min_dte`` calendar days left.
        start, end: optional date range [start, end).
        name: prefix for synthetic ids, e.g. ``'FUT'`` → ``'FUT(front)'``.
        min_history: raise if a series rolls into a contract first observed fewer than
            this many calendar days before the roll (contracts present from the first
            date of the data are exempt). 0 disables the check.

    Returns:
        columns ``date, contract, series, days_to_expiry``. The selected
        contract's expiry never moves backwards within a series.
    """
    cols = Columns.make(columns)
    sched = _to_polars(schedule)
    sched = sched.rename({k: v for k, v in ((cols.contract, _C), (cols.expiry, _E)) if k != v and k in sched.columns})
    out = _select(sched, target_days, expiry_rank, min_dte, start, end, name, min_history)
    return _restore(out.rename({_D: "date"}), cols)


def detect_rolls(mapping, columns=None) -> pl.DataFrame:
    """Roll dates of each continuous series: days on which the selected contract changes.

    Args:
        mapping: output of ``select_contracts`` (date, contract, series).

    Returns:
        columns ``series, roll_date, old_contract, new_contract``.
    """
    cols = Columns.make(columns)
    m = _to_polars(mapping).rename({cols.contract: _C, "date": _D} if cols.contract != _C else {"date": _D})
    return _detect_rolls(m)


def _detect_rolls(mapping: pl.DataFrame) -> pl.DataFrame:
    return (
        mapping.lazy()
        .sort([_SYN, _D])
        .with_columns(pl.col(_C).shift(1).over(_SYN).alias("_prev"))
        .filter(pl.col("_prev").is_not_null() & (pl.col(_C) != pl.col("_prev")))
        .select(_SYN, pl.col(_D).alias("roll_date"), pl.col("_prev").alias("old_contract"),
                pl.col(_C).alias("new_contract"))
        .collect()
    )


# ── roll adjustment ─────────────────────────────────────────────────────

def _daily_last(df: pl.DataFrame, value: str) -> pl.DataFrame:
    """Last value per (date, contract)."""
    return df.lazy().sort(_T).group_by([_D, _C]).agg(pl.col(value).last()).collect()


def _roll_spreads(rolls: pl.DataFrame, daily_close: pl.DataFrame) -> pl.DataFrame:
    """new − old daily close, each on its last date before the roll. Raises if either is missing."""
    schema = {_SYN: pl.Utf8, "roll_date": pl.Date, "spread": pl.Float64}
    if rolls.is_empty():
        return pl.DataFrame(schema=schema)
    rows = []
    for syn, roll_date, old, new in rolls.select(_SYN, "roll_date", "old_contract", "new_contract").iter_rows():
        prev = roll_date - timedelta(days=1)

        def last_close(cid):
            hit = (daily_close.filter((pl.col(_C) == cid) & (pl.col(_D) <= prev))
                   .sort(_D, descending=True).head(1))
            return None if hit.is_empty() else hit["close"][0]

        o, n = last_close(old), last_close(new)
        if o is None or n is None:
            raise ValueError(f"{syn}: no price for {old if o is None else new} before the roll on {roll_date}; "
                             f"the Panama gap cannot be measured")
        rows.append({_SYN: syn, "roll_date": roll_date, "spread": float(n - o)})
    return pl.DataFrame(rows, schema=schema)


def roll_spreads(rolls, bars, columns=None, right_labelled: bool = False) -> pl.DataFrame:
    """Price gap between the new and old contract at each roll.

    The gap is ``new − old`` using each contract's last ``price`` on the latest date
    strictly before the roll date. Raises if either contract has no price by then.

    Args:
        rolls: output of ``detect_rolls``.
        bars: per-contract prices covering the old and new contracts before each roll.

    Returns:
        columns ``series, roll_date, spread``.
    """
    cols = Columns.make(columns)
    df = _prepare_bars(bars, cols, right_labelled, keep=("price",))
    daily = _daily_last(df, "price").rename({"price": "close"})
    return _roll_spreads(_to_polars(rolls), daily)


def apply_panama(prices, spreads, columns=None, right_labelled: bool = False) -> pl.DataFrame:
    """Panama (back) adjustment: add to each price the sum of all roll gaps after it.

    The adjustment for a roll applies to dates strictly before its roll date, so
    the most recent segment is unchanged and history is shifted to line up.

    Args:
        prices: continuous prices with ``time``, ``series`` and ``price`` columns.
        spreads: output of ``roll_spreads``.
    """
    cols = Columns.make(columns)
    df = _to_polars(prices).rename({cols.time: _T}) if cols.time != _T else _to_polars(prices)
    added = _D not in df.columns
    if added:
        df = df.with_columns(_date_expr(_T, right_labelled))
    out = _apply_panama(df, _to_polars(spreads))
    if added:
        out = out.drop(_D)
    return _restore(out, cols)


def _apply_panama(df: pl.DataFrame, spreads: pl.DataFrame) -> pl.DataFrame:
    if spreads.is_empty():
        return df
    adj = (
        spreads.lazy()
        .sort([_SYN, "roll_date"], descending=[False, True])
        .with_columns(pl.col("spread").cum_sum().over(_SYN).alias("_cum_adj"))
        .sort([_SYN, "roll_date"])
        .collect()
    )
    parts = []
    for syn in df[_SYN].unique(maintain_order=True).to_list():
        syn_df = df.filter(pl.col(_SYN) == syn)
        syn_adj = adj.filter(pl.col(_SYN) == syn)
        if syn_adj.is_empty():
            parts.append(syn_df)
            continue
        # Shift each roll date back a day so the forward as-of join applies a roll's
        # gap only to dates strictly before it.
        lookup = syn_adj.select((pl.col("roll_date") - pl.duration(days=1)).alias("_roll"), "_cum_adj").sort("_roll")
        parts.append(
            syn_df.sort(_D)
            .join_asof(lookup, left_on=_D, right_on="_roll", strategy="forward")
            .with_columns((pl.col("price") + pl.col("_cum_adj").fill_null(0.0)).alias("price"))
            .drop("_cum_adj", "_roll")
        )
    return pl.concat(parts).sort([_T, _SYN])


def _blend_roll_days(data: pl.DataFrame, mapped: pl.DataFrame, rolls: pl.DataFrame,
                     blend_join: Literal["exact", "nearest"], spreads: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Replace each roll day of a series with a linear blend from the old to the new contract.

    Over the roll day's common timestamps the weight on the new contract goes 0 → 1.
    If either contract has no data that day the series switches instantly.

    ``spreads`` (output of ``_roll_spreads``) must be given when the result is Panama
    adjusted afterwards: the old leg is lifted by that roll's gap, so the blend starts
    at the level the adjusted history ends on and finishes on the new contract's price.
    Without it the roll day would open ``spread`` below the previous close and drift
    back over the day.
    """
    cols_out = [_T, _SYN, "price", _D]
    if rolls.is_empty():
        return mapped.select(cols_out)
    spread_of = {} if spreads is None else {
        (syn, rd): s for syn, rd, s in spreads.select(_SYN, "roll_date", "spread").iter_rows()}
    roll_keys = rolls.select(_SYN, pl.col("roll_date").alias(_D)).with_columns(pl.lit(True).alias("_is_roll"))
    non_roll = (mapped.join(roll_keys, on=[_SYN, _D], how="left")
                .filter(pl.col("_is_roll").is_null()).select(cols_out))

    blended = []
    for syn, roll_date, old, new in rolls.select(_SYN, "roll_date", "old_contract", "new_contract").iter_rows():
        lift = spread_of.get((syn, roll_date), 0.0)
        old_p = data.filter((pl.col(_D) == roll_date) & (pl.col(_C) == old)).select(_T, (pl.col("price") + lift).alias("_old"))
        new_p = data.filter((pl.col(_D) == roll_date) & (pl.col(_C) == new)).select(_T, pl.col("price").alias("_new"))
        if old_p.is_empty() or new_p.is_empty():
            fallback = mapped.filter((pl.col(_D) == roll_date) & (pl.col(_SYN) == syn))
            if not fallback.is_empty():
                blended.append(fallback.select(cols_out))
            continue
        if blend_join == "exact":
            both = old_p.join(new_p, on=_T, how="inner").sort(_T)
        else:
            both = old_p.sort(_T).join_asof(new_p.sort(_T), on=_T, strategy="nearest").drop_nulls(["_old", "_new"])
        if both.is_empty():
            continue
        n = both.height
        w = pl.lit(0.5) if n == 1 else pl.int_range(0, n, dtype=pl.Int64).cast(pl.Float64) / (n - 1)
        blended.append(both.with_columns(
            ((1.0 - w) * pl.col("_old") + w * pl.col("_new")).alias("price"),
            pl.lit(syn).alias(_SYN),
            pl.lit(roll_date).alias(_D),
        ).select(cols_out))
    return pl.concat([non_roll, *blended]).sort([_T, _SYN])


# ── public: continuous series ───────────────────────────────────────────

def map_to_continuous(bars, mapping, columns=None, right_labelled: bool = False) -> pl.DataFrame:
    """Keep only the rows of the contract selected for each series on each date.

    Adds ``series`` and ``days_to_expiry``. A bar belongs to several series
    if more than one series selects its contract that day.
    """
    cols = Columns.make(columns)
    df = _prepare_bars(bars, cols, right_labelled)
    m = _to_polars(mapping).rename({cols.contract: _C, "date": _D} if cols.contract != _C else {"date": _D})
    out = (df.lazy().join(m.lazy().select(_D, _C, _SYN, _DTE), on=[_D, _C], how="inner")
           .drop(_D).sort(_T).collect())
    return _restore(out, cols)


def contract_mapping(bars, expiries, target_days: Optional[list[int]] = None,
                     expiry_rank: Optional[list[int]] = None, min_dte: int = 7, start=None, end=None,
                     name: Optional[str] = None, columns=None, right_labelled: bool = False,
                     min_history: int = 30) -> pl.DataFrame:
    """Schedule + selection in one call: ``select_contracts(build_schedule(bars, expiries), ...)``.

    Returns ``date, contract, series, days_to_expiry`` plus ``weight=1.0``, so it
    can be passed straight to ``decompose_positions`` for series made by
    ``continuous_prices`` with the same arguments.
    """
    sched = build_schedule(bars, expiries, columns=columns, right_labelled=right_labelled)
    out = select_contracts(sched, target_days, expiry_rank, min_dte, start, end, name, columns=columns,
                           min_history=min_history)
    return out.with_columns(pl.lit(1.0).alias("weight"))


def continuous_prices(bars, expiries, target_days: Optional[list[int]] = None,
                      expiry_rank: Optional[list[int]] = None, min_dte: int = 7,
                      roll: Optional[Literal["blend", "instant"]] = "blend", panama: bool = True,
                      blend_join: Literal["exact", "nearest"] = "exact", start=None, end=None,
                      name: Optional[str] = None, columns=None, right_labelled: bool = False,
                      min_history: int = 30) -> pl.DataFrame:
    """Continuous price series built from individual contracts.

    Args:
        bars: long frame, one row per (time, contract), with a price column. Include
            every contract that may be selected (and the day before each roll, for
            Panama gaps).
        expiries: frame with contract and expiry columns, or ``{contract: expiry}``.
        target_days / expiry_rank / min_dte / start / end / name / min_history: see ``select_contracts``.
        roll: ``'blend'`` linearly moves from the old to the new contract across the
            roll day; ``'instant'`` switches at the roll; ``None`` returns the raw
            selected contract's prices with no roll handling.
        panama: back-adjust history by the roll gaps (ignored when ``roll=None``).
        blend_join: how old and new contract prices are aligned on a roll day —
            ``'exact'`` (common timestamps, for bars) or ``'nearest'`` (as-of, for ticks).
        right_labelled: bars are stamped at the end of their interval; the trading
            date is taken from the interval start.

    Returns:
        columns ``time, series, price`` (caller's time column name).
    """
    if roll not in (None, "blend", "instant"):
        raise ValueError(f"roll must be None, 'blend' or 'instant', got {roll!r}")
    cols = Columns.make(columns)
    data = _prepare_bars(bars, cols, right_labelled, keep=("price",))
    if "price" not in data.columns:
        raise KeyError(f"bars has no price column {cols.price!r}; pass columns={{'price': ...}}")
    exp = _prepare_expiries(expiries, cols)

    sched = build_schedule(data.drop(_D), exp, columns=Columns(time=_T, contract=_C, expiry=_E),
                           right_labelled=right_labelled)
    mapping = _select(sched, target_days, expiry_rank, min_dte, start, end, name, min_history)
    empty = pl.DataFrame(schema={cols.time: data.schema[_T], _SYN: pl.Utf8, "price": pl.Float64})
    if mapping.is_empty():
        return empty

    mapped = data.join(mapping.select(_D, _C, _SYN), on=[_D, _C], how="inner").sort(_T)
    if roll is None:
        return _restore(mapped.select(_T, _SYN, "price"), cols)

    rolls = _detect_rolls(mapping)
    spreads = None
    if panama and not rolls.is_empty():
        spreads = _roll_spreads(rolls, _daily_last(data, "price").rename({"price": "close"}))
    if roll == "instant":
        out = mapped.select(_T, _SYN, "price", _D)
    else:
        out = _blend_roll_days(data, mapped, rolls, blend_join, spreads)
    if spreads is not None:
        out = _apply_panama(out, spreads)
    return _restore(out.select(_T, _SYN, "price"), cols)


# ── public: weights ─────────────────────────────────────────────────────

def interpolation_weights(daily, target_days: Optional[list[int]] = None, name: Optional[str] = None,
                          columns=None) -> pl.DataFrame:
    """Constant-maturity weights: per date, the two contracts bracketing each target.

    For a target T between contracts with lo ≤ T ≤ hi days to expiry,
    weight_lo = (hi − T) / (hi − lo) and weight_hi = 1 − weight_lo. If only one side
    exists (or both are the same contract) it gets weight 1. ``0``/None = front month.

    Args:
        daily: one row per (date, contract) with ``date`` and ``days_to_expiry``.

    Returns:
        columns ``date, series, contract, weight, days_to_expiry``.
    """
    cols = Columns.make(columns)
    d = _to_polars(daily).rename({cols.contract: _C}) if cols.contract != _C else _to_polars(daily)
    return _restore(_interpolation_weights(d, target_days, name), cols)


def _interpolation_weights(daily: pl.DataFrame, target_days, name) -> pl.DataFrame:
    schema = {"date": pl.Date, _SYN: pl.Utf8, _C: daily.schema.get(_C, pl.Utf8), "weight": pl.Float64, _DTE: pl.Int32}
    if target_days is None:
        target_days = [0]
    results = []
    for target in target_days:
        if target == 0:
            results.append(
                daily.lazy().sort(["date", _DTE]).group_by("date").first()
                .with_columns(pl.lit(_series_name(name, "front")).alias(_SYN), pl.lit(1.0).alias("weight"))
                .select(list(schema)).sort("date").collect().cast(schema)
            )
            continue
        syn = _series_name(name, f"{target}d")
        lf = daily.lazy()
        lo = (lf.filter(pl.col(_DTE) <= target).sort(["date", _DTE]).group_by("date").last()
              .select("date", pl.col(_DTE).alias("dte_lo"), pl.col(_C).alias("c_lo")))
        hi = (lf.filter(pl.col(_DTE) >= target).sort(["date", _DTE]).group_by("date").first()
              .select("date", pl.col(_DTE).alias("dte_hi"), pl.col(_C).alias("c_hi")))
        joined = lo.join(hi, on="date", how="full", coalesce=True).sort("date").collect()
        rows = []
        for d, dte_lo, c_lo, dte_hi, c_hi in joined.select("date", "dte_lo", "c_lo", "dte_hi", "c_hi").iter_rows():
            if c_lo is not None and c_hi is not None and c_lo != c_hi:
                span = dte_hi - dte_lo
                w_lo = (dte_hi - target) / span if span > 0 else 1.0
                rows.append((d, syn, c_lo, w_lo, dte_lo))
                rows.append((d, syn, c_hi, 1.0 - w_lo, dte_hi))
            elif c_lo is not None:
                rows.append((d, syn, c_lo, 1.0, dte_lo))
            elif c_hi is not None:
                rows.append((d, syn, c_hi, 1.0, dte_hi))
        if rows:
            results.append(pl.DataFrame(rows, schema=schema, orient="row"))
    if not results:
        return pl.DataFrame(schema=schema)
    return pl.concat(results).sort(["date", _SYN])


def top_n_weights(values, schedule, top_n: int, min_dte: int = 7, start=None, end=None,
                  name: Optional[str] = None, label: str = "top", value_agg: Literal["last", "sum"] = "last",
                  columns=None, right_labelled: bool = False, value_ffill_limit: int = 5) -> pl.DataFrame:
    """Blend weights from the N contracts with the largest value (open interest, volume, ...).

    Per date the ``weight`` column is aggregated per contract (``value_agg``), the
    top N live contracts with at least ``min_dte`` days left are kept and normalised
    to sum to 1. Weights computed from date D apply on the next trading date, so
    they only use information available before that day. The nearest-expiry
    contract's weight can only fall while it stays the nearest; any excess is
    redistributed pro rata to the others (no cap when it is the only contract).

    Args:
        values: long frame with ``time``, ``contract`` and ``weight`` columns.
        schedule: output of ``build_schedule`` (gives trading dates and days to expiry).
        label: synthetic id tag, e.g. ``'top'`` → ``'FUT(top3)'``.
        value_ffill_limit: a contract's last value is carried forward over up to this
            many consecutive trading dates on which it is live but has no row, so a
            missing row does not drop it from the top N for a day.

    Returns:
        columns ``date, contract, series, weight, days_to_expiry``.
    """
    cols = Columns.make(columns)
    v = _prepare_bars(values, cols, right_labelled, keep=("weight",))
    if "weight" not in v.columns:
        raise KeyError(f"values has no weight column {cols.weight!r}; pass columns={{'weight': ...}}")
    agg = pl.col("weight").last() if value_agg == "last" else pl.col("weight").sum()
    v = v.sort(_T).group_by([_D, _C]).agg(agg.alias("_value"))
    sched = _to_polars(schedule).rename({cols.contract: _C}) if cols.contract != _C else _to_polars(schedule)
    out = _top_n_weights(v, sched, top_n, min_dte, start, end, _series_name(name, f"{label}{top_n}"), value_ffill_limit)
    return _restore(out, cols)


def _top_n_weights(v: pl.DataFrame, sched: pl.DataFrame, top_n, min_dte, start, end, syn,
                   value_ffill_limit: int = 5) -> pl.DataFrame:
    schema = {"date": pl.Date, _C: sched.schema.get(_C, pl.Utf8), _SYN: pl.Utf8, "weight": pl.Float64, _DTE: pl.Int32}
    s = sched.filter(pl.col(_DTE) >= min_dte).select("date", _C, _DTE)
    if end is not None:
        s = s.filter(pl.col("date") < _to_date(end))
    # Grid of live (date, contract); carry each contract's last value over short gaps.
    joined = (
        s.join(v, left_on=["date", _C], right_on=[_D, _C], how="left")
        .sort([_C, "date"])
        .with_columns(pl.col("_value").forward_fill(limit=max(value_ffill_limit, 0)).over(_C))
        .drop_nulls("_value")
        .rename({"date": _D})
    )
    if joined.is_empty():
        return pl.DataFrame(schema=schema)

    ranked = (
        joined.lazy()
        .sort([_D, "_value"], descending=[False, True])
        .with_columns(pl.col(_C).cum_count().over(_D).alias("_rank"))
        .filter(pl.col("_rank") <= top_n)
        .with_columns((pl.col("_value").cast(pl.Float64)
                       / pl.col("_value").cast(pl.Float64).sum().over(_D)).alias("weight"))
        .select(_D, _C, "weight", _DTE)
        .collect()
    )
    trading = s["date"].unique().sort().to_list()
    if len(trading) < 2:
        return pl.DataFrame(schema=schema)
    next_day = pl.DataFrame({_D: trading[:-1], "date": trading[1:]}, schema={_D: pl.Date, "date": pl.Date})
    lagged = ranked.join(next_day, on=_D, how="inner")

    # Ratchet: while the same contract stays nearest-expiry, its weight can't rise.
    rows = []
    prev_front = prev_w = None
    for d in lagged["date"].unique().sort().to_list():
        day = lagged.filter(pl.col("date") == d).sort(_DTE).to_dicts()
        if not day:
            continue
        front, w = day[0][_C], day[0]["weight"]
        others = sum(r["weight"] for r in day[1:])
        if front == prev_front and prev_w is not None and prev_w < w and others > 0:
            excess = w - prev_w
            day[0]["weight"] = prev_w
            scale = (others + excess) / others
            for r in day[1:]:
                r["weight"] *= scale
        prev_front, prev_w = front, day[0]["weight"]
        rows.extend(day)
    if not rows:
        return pl.DataFrame(schema=schema)
    out = pl.DataFrame(rows).with_columns(pl.lit(syn).alias(_SYN)).select(list(schema)).cast(schema)
    if start is not None:
        out = out.filter(pl.col("date") >= _to_date(start))
    return out.sort(["date", _C])


def continuous_weights(bars, expiries, target_days: Optional[list[int]] = None,
                       expiry_rank: Optional[list[int]] = None, top_n: Optional[int] = None,
                       values=None, min_dte: int = 7, start=None, end=None, name: Optional[str] = None,
                       label: str = "top", value_agg: Literal["last", "sum"] = "last",
                       columns=None, values_columns=None, right_labelled: bool = False,
                       min_history: int = 30, value_ffill_limit: int = 5) -> pl.DataFrame:
    """How each continuous series decomposes into actual contracts, per date.

    Exactly one of:
        target_days: constant-maturity interpolation between the two contracts
            bracketing each target (see ``interpolation_weights``). Contracts are
            those live per ``build_schedule`` (first observation through expiry) with
            at least ``min_dte`` days left, so a day without data for a contract does
            not change the weights. Target ``0`` is exactly ``contract_mapping``'s front.
        expiry_rank: one contract per rank, weight 1 (see ``select_contracts``).
        top_n: the N contracts with the largest ``weight`` column value — from
            ``values`` if given (e.g. an open-interest frame), else from ``bars``
            (e.g. ``columns={'weight': 'volume'}, value_agg='sum'``). See ``top_n_weights``.
            ``values_columns`` maps the ``values`` frame's columns when they differ from
            ``bars`` (defaults to ``columns``). ``value_ffill_limit``: see ``top_n_weights``.
    Defaults to front-month weights when none is given. ``min_history``: see ``select_contracts``.

    Returns:
        columns ``date, series, contract, weight, days_to_expiry``.
    """
    if sum(x is not None for x in (target_days, expiry_rank, top_n)) > 1:
        raise ValueError("target_days, expiry_rank and top_n are mutually exclusive")
    cols = Columns.make(columns)
    order = ["date", _SYN, cols.contract, "weight", _DTE]

    if top_n is not None:
        vcols = cols if values is None or values_columns is None else Columns.make(values_columns)
        sched = build_schedule(bars, expiries, columns=cols, right_labelled=right_labelled)
        sched = sched.rename({cols.contract: vcols.contract}) if cols.contract != vcols.contract else sched
        out = top_n_weights(bars if values is None else values, sched, top_n, min_dte, start, end, name,
                            label, value_agg, columns=vcols, right_labelled=right_labelled,
                            value_ffill_limit=value_ffill_limit)
        out = out.rename({vcols.contract: cols.contract}) if cols.contract != vcols.contract else out
        return out.select(order).sort(["date", _SYN])
    if expiry_rank is not None:
        out = contract_mapping(bars, expiries, None, expiry_rank, min_dte, start, end, name, cols, right_labelled,
                               min_history=min_history)
        return out.select(order).sort(["date", _SYN])

    sched = build_schedule(bars, expiries, columns=cols, right_labelled=right_labelled)
    sched = sched.rename({cols.contract: _C}) if cols.contract != _C else sched
    targets = [0] if target_days is None else list(target_days)
    parts = []
    if 0 in targets:
        front = _select(sched, [0], None, min_dte, start, end, name, min_history)
        parts.append(front.rename({_D: "date"}).with_columns(pl.lit(1.0).alias("weight")))
    others = [t for t in targets if t != 0]
    if others:
        daily = sched.filter(pl.col(_DTE) >= min_dte)
        if start is not None:
            daily = daily.filter(pl.col("date") >= _to_date(start))
        if end is not None:
            daily = daily.filter(pl.col("date") < _to_date(end))
        parts.append(_interpolation_weights(daily, others, name))
    out = pl.concat([p.select("date", _SYN, _C, "weight", _DTE) for p in parts]).sort(["date", _SYN])
    return _restore(out, cols).select(order)


def interpolate_daily(daily, target_days: Optional[list[int]] = None, name: Optional[str] = None,
                      fields: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
                      columns=None) -> pl.DataFrame:
    """Constant-maturity daily bars: interpolate ``fields`` between bracketing contracts.

    Uses the same bracketing weights as ``interpolation_weights``; where only one
    side exists its values are used as-is.

    Args:
        daily: one row per (date, contract) with ``date``, ``days_to_expiry`` and ``fields``
            (e.g. from ``daily_ohlcv`` joined with days to expiry).

    Returns:
        columns ``date, series, *fields``.
    """
    cols = Columns.make(columns)
    d = _to_polars(daily).rename({cols.contract: _C}) if cols.contract != _C else _to_polars(daily)
    fields = [f for f in fields if f in d.columns]
    if target_days is None:
        target_days = [0]
    results = []
    for target in target_days:
        if target == 0:
            results.append(
                d.lazy().sort(["date", _DTE]).group_by("date").first()
                .with_columns(pl.lit(_series_name(name, "front")).alias(_SYN))
                .select(["date", _SYN, *fields]).sort("date").collect()
            )
            continue
        lf = d.lazy()
        lo = (lf.filter(pl.col(_DTE) <= target).sort(["date", _DTE]).group_by("date").last()
              .select("date", pl.col(_DTE).alias("dte_lo"), *[pl.col(f).alias(f"{f}_lo") for f in fields]))
        hi = (lf.filter(pl.col(_DTE) >= target).sort(["date", _DTE]).group_by("date").first()
              .select("date", pl.col(_DTE).alias("dte_hi"), *[pl.col(f).alias(f"{f}_hi") for f in fields]))
        span = (pl.col("dte_hi") - pl.col("dte_lo")).cast(pl.Float64)
        w_lo = pl.coalesce(pl.when(span > 0).then((pl.col("dte_hi").cast(pl.Float64) - float(target)) / span)
                           .otherwise(1.0), pl.lit(0.0))
        results.append(
            lo.join(hi, on="date", how="full", coalesce=True)
            .with_columns(w_lo.alias("_w_lo"))
            .with_columns([
                pl.coalesce(pl.col(f"{f}_lo") * pl.col("_w_lo") + pl.col(f"{f}_hi") * (1.0 - pl.col("_w_lo")),
                            pl.col(f"{f}_lo"), pl.col(f"{f}_hi")).alias(f)
                for f in fields
            ])
            .with_columns(pl.lit(_series_name(name, f"{target}d")).alias(_SYN))
            .select(["date", _SYN, *fields]).sort("date").collect()
        )
    return pl.concat(results).sort(["date", _SYN])


def daily_ohlcv(bars, columns=None, price: Optional[str] = None, volume: Optional[str] = "volume",
                right_labelled: bool = False) -> pl.DataFrame:
    """Aggregate bars or ticks to daily OHLCV per (date, contract).

    Args:
        price: if given, OHLC are taken from this single column (ticks, or a
            ``microprice()`` column); otherwise from ``open/high/low/close``.
        volume: column summed into ``volume`` (skipped if absent or None).
    """
    cols = Columns.make(columns)
    df = _prepare_bars(bars, cols, right_labelled).sort(_T)
    if price is not None:
        o = h = lo = c = pl.col(price)
    else:
        o, h, lo, c = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
    aggs = [o.first().alias("open"), h.max().alias("high"), lo.min().alias("low"), c.last().alias("close")]
    if volume is not None and volume in df.columns:
        aggs.append(pl.col(volume).sum().cast(pl.Float64).alias("volume"))
    out = df.group_by([_D, _C]).agg(aggs).rename({_D: "date"}).sort(["date", _C])
    return _restore(out, cols)


def weighted_returns(bars, weights, normalize: bool = False, return_vol: bool = False, vol_estimator=None,
                     columns=None, right_labelled: bool = False) -> pl.DataFrame:
    """Blend per-contract price changes with daily contract weights.

    Price changes are taken within each (contract, date), so the first bar of a day
    contributes nothing and different days may use different weights. Returns are
    summed per timestamp across contracts.

    Args:
        weights: e.g. ``continuous_weights(..., top_n=3)`` (date, series, contract, weight).
        normalize / return_vol: divide by (and/or return) a daily volatility of the
            blended series, estimated on its cumulative index (``yang_zhang`` by
            default) and lagged one trading day. Needs some warm-up history.

    Returns:
        columns ``time, series, return`` (+ ``vol``).
    """
    cols = Columns.make(columns)
    data = _prepare_bars(bars, cols, right_labelled, keep=("price",))
    w = _to_polars(weights).rename({cols.contract: _C}) if cols.contract != _C else _to_polars(weights)
    data = data.sort([_C, _T]).with_columns(pl.col("price").diff().over(_C, _D).alias("_ret"))
    joined = data.join(w.select("date", _C, _SYN, "weight"), left_on=[_D, _C], right_on=["date", _C], how="inner")
    blended = (
        joined.lazy()
        .with_columns((pl.col("_ret") * pl.col("weight")).alias("_wret"))
        .group_by([_T, _SYN]).agg(pl.col("_wret").sum().alias("return"))
        .sort([_SYN, _T])
        .with_columns(_date_expr(_T, right_labelled))
        .collect()
    )
    if not (normalize or return_vol) or blended.is_empty():
        return _restore(blended.select(_T, _SYN, "return"), cols)

    import pandas as pd
    if vol_estimator is None:
        from ebt.simulate.stats import yang_zhang as vol_estimator
    # Offset keeps the index positive for estimators that take logs.
    idx = blended.with_columns((pl.col("return").cum_sum().over(_SYN) + 1e6).alias("_px"))
    ohlc = (
        idx.group_by([_D, _SYN]).agg(pl.col("_px").first().alias("open"), pl.col("_px").max().alias("high"),
                                     pl.col("_px").min().alias("low"), pl.col("_px").last().alias("close"))
        .sort(_D).to_pandas()
    )
    ohlc[_D] = pd.to_datetime(ohlc[_D])
    vol = vol_estimator(ohlc.pivot(index=_D, columns=_SYN, values=["open", "high", "low", "close"]).sort_index())
    vol_long = vol.stack().rename("vol").reset_index()
    vol_long.columns = [_D, _SYN, "vol"]
    vol_pl = (
        pl.from_pandas(vol_long).with_columns(pl.col(_D).cast(pl.Date))
        .sort([_SYN, _D])
        .with_columns(pl.col(_D).shift(-1).over(_SYN).alias("_applies"))
        .filter(pl.col("_applies").is_not_null())
        .select(pl.col("_applies").alias(_D), _SYN, "vol")
    )
    blended = blended.join(vol_pl, on=[_D, _SYN], how="left")
    if normalize:
        blended = blended.with_columns((pl.col("return") / pl.col("vol")).alias("return"))
    out_cols = [_T, _SYN, "return"] + (["vol"] if return_vol else [])
    return _restore(blended.select(out_cols), cols)


# ── public: positions ───────────────────────────────────────────────────

def decompose_positions(positions, mapping, columns=None, right_labelled: bool = False):
    """Turn positions in continuous series into positions in actual contracts.

    Args:
        positions: pandas DataFrame, DatetimeIndex, one column per series. NaN
            is carried forward within each series, as ``backtest`` does; NaN before the
            first value counts as flat.
        mapping: date, series, contract and (optional) weight per row — e.g.
            ``contract_mapping(...)`` for series built by ``continuous_prices``, or
            ``continuous_weights(...)`` for interpolated / top-N series.
        right_labelled: same value the mapping was built with; a stamp exactly at
            midnight then belongs to the previous trading date, as it did for the bars.

    Returns:
        pandas DataFrame with the same index and one column per contract. Positions
        from several series landing on the same contract are summed. Inactive
        periods are NaN, with an explicit 0 on the first bar after each active
        period so a forward-filling backtest closes the position.
    """
    import pandas as pd

    cols = Columns.make(columns)
    if positions.empty:
        return pd.DataFrame(index=positions.index)
    m = _to_polars(mapping)
    if "weight" not in m.columns:
        m = m.with_columns(pl.lit(1.0).alias("weight"))
    rows = m.select("date", _SYN, pl.col(cols.contract).alias(_C), "weight").iter_rows()
    rows = list(rows)

    mapped = {r[1] for r in rows}
    for col in positions.columns:
        if col not in mapped:
            _LOGGER.warning(f"decompose_positions: no mapping for {col!r}, skipping")

    contracts = sorted({r[2] for r in rows})
    c_idx = {c: i for i, c in enumerate(contracts)}
    s_idx = {s: i for i, s in enumerate(positions.columns)}

    original_index = positions.index
    needs_sort = not positions.index.is_monotonic_increasing
    if needs_sort:
        order = np.argsort(positions.index)
        positions = positions.iloc[order]

    out = np.zeros((len(positions), len(contracts)), dtype=np.float64, order="F")
    vals = positions.ffill().values
    # Trading date of every stamp, by the same rule the mapping's dates were made with.
    stamp_dates = (pl.from_pandas(pd.Series(positions.index, name=_T)).to_frame()
                   .select(_date_expr(_T, right_labelled))[_D].to_numpy().astype("datetime64[D]"))
    for d, syn, contract, weight in rows:
        if syn not in s_idx:
            continue
        day = np.datetime64(d, "D")
        lo, hi = stamp_dates.searchsorted(day, "left"), stamp_dates.searchsorted(day, "right")
        if lo >= hi:
            continue
        v = vals[lo:hi, s_idx[syn]]
        out[lo:hi, c_idx[contract]] += (1.0 if weight is None else weight) * np.where(np.isnan(v), 0.0, v)

    for j in range(out.shape[1]):
        col = out[:, j]
        nonzero = col != 0.0
        prev_nonzero = np.empty_like(nonzero)
        prev_nonzero[0] = False
        prev_nonzero[1:] = nonzero[:-1]
        close_marker = (~nonzero) & prev_nonzero
        col[(~nonzero) & (~close_marker)] = np.nan

    result = pd.DataFrame({c: out[:, i] for i, c in enumerate(contracts)}, index=positions.index)
    if needs_sort:
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        result = result.iloc[inv]
        result.index = original_index
    return result
