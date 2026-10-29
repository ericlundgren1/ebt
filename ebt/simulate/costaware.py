"""Cost-aware position sizing — spend the risk budget where trading is cheap.

Inverse-vol sizing (``signal / vol``) ignores what a unit of signal COSTS to
express. When trading costs are liquidity-tiered, two names that inverse-vol
sizing treats as interchangeable can differ in cost by an order of magnitude
or more.

This module provides cost-aware transforms. Four mechanisms, each usable alone
and each with an exact identity element so the untreated book is recoverable
as a CONTROL ARM rather than approximated:

``cost_vol`` / ``cost_weight``
    ``w ∝ signal / (vol * spread**gamma)``. ``gamma=0`` returns the input
    sizing BITWISE — ``spread**0`` is exactly ``1.0`` for every float
    (including NaN and inf), and ``x * 1.0 == x``. The control arm is the
    same code path, not a bypass.

``spread_floor``
    Refuse the expensive tail outright: zero every name wider than
    ``max_spread`` and re-express the freed risk in the names that remain.
    ``max_spread=None`` is the identity.

``no_trade_band``
    Per-name, cost-proportional rebalancing band: a name only trades when its
    deviation from target exceeds ``k * spread``. Cheap names rebalance daily,
    expensive names sit. ``k=0`` is the identity. This generalises banded
    rebalancing to a cross-sectional book: the band is calibrated in units of
    each name's OWN spread rather than an absolute position threshold.

``turnover_budget``
    Cap the gross traded per day and spend the cap greedily on the largest
    deviations. ``budget=None``/``inf`` is the identity.

Dollar neutrality
-----------------
A naive band breaks neutrality: suppressing one leg's trades while the other
leg rebalances leaves a net position that is a market bet, not a cost saving.
Both stateful mechanisms therefore work in **trade space** — they decide a
trade vector ``d`` and rescale its positive and negative halves to equal gross
before applying it, so ``sum(new) == sum(held)`` identically. Neutrality is
PRESERVED (whatever the book had, it keeps), never imposed. That also means
untraded names are never touched, so the turnover saving the band buys is not
handed back by a whole-book rescale.

All functions are causal: row ``t`` sees only rows ``<= t`` and the state
carried from ``t-1``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = ["cost_vol", "cost_weight", "spread_floor", "no_trade_band",
           "turnover_budget", "balance_trades", "greedy_trades"]

#: Spreads are return-unit fractions; anything at or below this is a data
#: error, not a free trade. Only used to keep ``s ** -gamma`` finite.
_EPS = 1e-9


# ══════════════════════════════════════════════════════════════════════════
# 1. Sizing transform
# ══════════════════════════════════════════════════════════════════════════

def cost_vol(vol: Optional[pd.DataFrame],
             spreads: Optional[pd.DataFrame],
             gamma: float = 0.0,
             eps: float = _EPS) -> Optional[pd.DataFrame]:
    """Effective denominator for cost-aware sizing: ``vol * spread**gamma``.

    Pass the result as the ``vol`` argument of an inverse-vol book
    constructor. Keeping the transform on the DENOMINATOR (rather than
    scaling weights after the fact) means any downstream vol floor, leg
    normalisation or minimum-names guard sees the cost-aware sizing,
    exactly as it sees inverse-vol sizing.

    Args:
        vol: per-name risk panel (e.g. rolling 63d return std), or ``None``
            for a book that does not inverse-vol size (e.g. a signal that
            is itself a volatility measure).
        spreads: per-name full spread in return units, same shape as ``vol``.
        gamma: cost aversion. 0 = plain inverse-vol sizing, 1 = weight inversely
            proportional to spread, >1 = concentrate hard in the liquid names.
        eps: spread floor, guarding ``0 ** -gamma``.

    Returns:
        ``vol * spread**gamma``; ``spread**gamma`` when ``vol`` is None and
        ``gamma`` is non-zero; ``None`` when there is nothing to divide by
        (``vol is None`` and ``gamma == 0``) so the caller's un-sized path is
        reached unchanged.

    Exactness: at ``gamma == 0`` the factor is ``spread**0``, which IEEE-754
    defines as exactly ``1.0`` for every input (NaN and inf included), and
    ``vol * 1.0`` is bitwise ``vol``. The control arm is therefore the same
    code path as the treatment, not a special case.

    NaN spreads propagate (the name drops out of the book) for ``gamma != 0``.
    That is deliberate: an unknown spread is an unknown cost. Impute the
    widest tier upstream if a name should still be tradeable.
    """
    if spreads is None:
        return vol
    if vol is None and gamma == 0.0:
        return None
    factor = spreads.clip(lower=eps) ** float(gamma)
    if vol is None:
        return factor
    # reindex_like BEFORE multiplying: a bare pandas multiply takes the UNION
    # of both axes, so a spread panel missing one name would NaN that column
    # out of the CONTROL arm — the identity claim above is only true once the
    # frames are aligned.
    return vol * factor.reindex_like(vol)


def cost_weight(signal: pd.DataFrame,
                vol: Optional[pd.DataFrame] = None,
                spreads: Optional[pd.DataFrame] = None,
                gamma: float = 0.0,
                eps: float = _EPS) -> pd.DataFrame:
    """``w = signal / (vol * spread**gamma)`` — the sizing transform itself.

    Use this when you own the weights directly; use ``cost_vol`` when you are
    feeding an inverse-vol book constructor. ``gamma=0`` reproduces ``signal / vol`` bitwise
    (and ``signal`` itself when ``vol`` is None).
    """
    denom = cost_vol(vol, spreads, gamma, eps=eps)
    return signal if denom is None else signal / denom


# ══════════════════════════════════════════════════════════════════════════
# 2. Refuse the expensive tail
# ══════════════════════════════════════════════════════════════════════════

def spread_floor(weights: pd.DataFrame,
                 spreads: pd.DataFrame,
                 max_spread: Optional[float],
                 renormalize: bool = True,
                 min_names: int = 0) -> pd.DataFrame:
    """Zero every name whose spread exceeds ``max_spread``.

    Args:
        weights: position book (dates x names).
        spreads: per-name full spread, aligned to ``weights``.
        max_spread: keep names with ``spread <= max_spread``. ``None`` is the
            identity (returns ``weights`` unchanged, same object semantics as
            a no-op arm).
        renormalize: re-express the freed risk in the surviving names by
            rescaling each leg back to the row's ORIGINAL gross, keeping the
            book dollar-neutral. Rows that lose a whole leg go flat — a
            one-sided book is not a cost saving, it is a market bet.
            **Set False for a book that is not dollar-neutral by
            construction**: rescaling the legs of a hedge-ratio-weighted book
            rewrites its hedge ratios and turns hedged positions into
            directional bets.
        min_names: rows left with fewer than this many names on either leg go
            flat. Without it a hard floor can concentrate the whole gross into
            a handful of survivors, which is a concentration experiment rather
            than a cost experiment. Only applied when ``renormalize``.

    NaN spreads are treated as ABOVE any threshold (unknown cost = dropped).
    """
    if max_spread is None:
        return weights
    keep = spreads.reindex_like(weights) <= float(max_spread)
    gross0 = weights.abs().sum(axis=1)
    out = weights.where(keep, 0.0)
    if renormalize:
        out = _neutralize_panel(out, gross0, min_names=min_names)
    return out


def _neutralize_panel(book: pd.DataFrame, gross: pd.Series,
                      min_names: int = 0) -> pd.DataFrame:
    """Rescale each leg of every row to ``gross/2`` (net 0, gross preserved).

    Rows with an empty (or too-thin) leg, or zero target gross, come back flat.
    """
    pos = book.clip(lower=0.0)
    neg = book.clip(upper=0.0)
    lg = pos.sum(axis=1)
    sg = -neg.sum(axis=1)
    ok = (lg > 0) & (sg > 0) & (gross > 0)
    if min_names:
        ok &= ((pos > 0).sum(axis=1) >= min_names) & \
              ((neg < 0).sum(axis=1) >= min_names)
    half = gross / 2.0
    return (pos.mul((half / lg.where(ok)).fillna(0.0), axis=0)
            + neg.mul((half / sg.where(ok)).fillna(0.0), axis=0))


# ══════════════════════════════════════════════════════════════════════════
# 3. Trade-space primitives (shared by the band and the budget)
# ══════════════════════════════════════════════════════════════════════════

def _split_targets(tot_p: float, tot_n: float, cap: float,
                   offset: float) -> tuple:
    """Gross to spend on each half so the trade nets exactly ``offset``.

    Returns ``(a, b)`` — the gross allocated to the positive and negative
    halves — satisfying ``a - b == offset``, ``a <= tot_p``, ``b <= tot_n``,
    ``a + b <= cap``, both non-negative, and ``a + b`` as large as possible
    (trade as much of the intended vector as the constraints allow).

    ``offset == 0`` is the plain balanced rule ``a = b = min(tot_p, tot_n, cap/2)``.

    When ``offset`` is unreachable — it exceeds the gross available on the
    side that would have to carry it — the result is the closest feasible
    point and the caller's trade nets something other than ``offset``. That
    is a deliberate corner: it means a MANDATED exit could not be fully
    offset by discretionary trades, and freezing the mandated exit to protect
    the arithmetic would be the wrong trade-off.
    """
    b = min(tot_n, tot_p - offset)
    if np.isfinite(cap):
        b = min(b, (cap - offset) / 2.0)
    b = max(b, -offset, 0.0)              # keeps a = b + offset >= 0
    a = b + offset
    a = min(max(a, 0.0), tot_p)
    b = min(max(b, 0.0), tot_n)
    if np.isfinite(cap) and a + b > cap and a + b > 0:
        s = cap / (a + b)                 # only reachable in the clamped corner
        a, b = a * s, b * s
    return a, b


def balance_trades(d: np.ndarray, cap: float = np.inf,
                   offset: float = 0.0) -> np.ndarray:
    """Scale a trade vector's two halves so it nets ``offset``, gross <= ``cap``.

    With the default ``offset=0`` the result satisfies ``sum(out) == 0`` (so
    applying it preserves the net exposure of whatever it is applied to) and
    ``sum(|out|) <= cap``. Each half is scaled PROPORTIONALLY, so the relative
    shape of the intended trade survives.

    ``offset`` exists for one caller: :func:`no_trade_band` applies MANDATORY
    exits in full and then needs the discretionary remainder to net the
    opposite of what those exits moved, so that the two together still leave
    net exposure untouched. Passing ``offset=0`` gives the plain balanced rule.
    """
    out = np.zeros_like(d)
    p = d > 0
    n = d < 0
    tot_p = float(d[p].sum())
    tot_n = float(-d[n].sum())
    a, b = _split_targets(tot_p, tot_n, cap, float(offset))
    if a > 0:
        out[p] = d[p] * (a / tot_p)
    if b > 0:
        out[n] = d[n] * (b / tot_n)
    return out


def greedy_trades(d: np.ndarray, cap: float,
                  balanced: bool = True, offset: float = 0.0) -> np.ndarray:
    """Spend a traded-gross budget ``cap`` on the LARGEST deviations first.

    Names are filled to their full deviation in descending ``|d|`` order; the
    marginal name gets a partial trade. With ``balanced=True`` the budget is
    split evenly between the buy and sell halves (``cap/2`` each) so the
    result sums to zero and applying it cannot move the book's net exposure —
    the cost of neutrality is that the smaller half throttles the larger one.

    ``cap = inf`` returns ``d`` unchanged when ``balanced=False``. With
    ``balanced=True`` it returns ``d`` unchanged only if ``d`` already sums to
    zero — which it does whenever the target and the held book are both
    neutral, the case the panel form is built for. An unbalanced ``d`` is
    throttled to ``min(buys, sells)`` even uncapped: that is the neutrality
    constraint doing its job, not the budget.

    ``offset`` (balanced form only) makes the result net ``offset`` instead of
    zero, so :func:`turnover_budget` can execute mandatory exits in full and
    still leave net exposure where it found it. ``offset=0`` is the plain balanced rule.
    """
    if not balanced:
        return _greedy_side(np.abs(d), cap) * np.sign(d)
    out = np.zeros_like(d)
    p = d > 0
    n = d < 0
    tot_p = float(d[p].sum())
    tot_n = float(-d[n].sum())
    a, b = _split_targets(tot_p, tot_n, cap, float(offset))
    if a > 0:
        out[p] = _greedy_side(d[p], a)
    if b > 0:
        out[n] = -_greedy_side(-d[n], b)
    return out


def _greedy_side(v: np.ndarray, m: float) -> np.ndarray:
    """Allocate ``m`` across non-negative ``v``, largest first (partial last)."""
    if not np.isfinite(m):
        return v.copy()
    order = np.argsort(-v, kind="stable")
    sv = v[order]
    before = np.cumsum(sv) - sv           # budget consumed before each element
    alloc = np.clip(m - before, 0.0, sv)
    out = np.empty_like(v)
    out[order] = alloc
    return out


# ══════════════════════════════════════════════════════════════════════════
# 4. Per-name no-trade band
# ══════════════════════════════════════════════════════════════════════════

def no_trade_band(target: pd.DataFrame,
                  spreads: pd.DataFrame,
                  k: float,
                  scale: Optional[pd.Series] = None,
                  neutralize: bool = True,
                  eligible: Optional[pd.DataFrame] = None,
                  mandatory_exit: bool = True) -> pd.DataFrame:
    """Rebalance a name only when its deviation exceeds ``k * spread``.

    The band is per-name and cost-proportional: with a wide spread range
    across the universe, a single ``k`` makes the cheap names track their
    target daily and the expensive ones sit for weeks. An absolute position
    band cannot express that.

    Args:
        target: desired book (dates x names). NaN means "no target update" —
            the held position is carried, exactly as ``backtest`` reads NaN.
        spreads: per-name full spread, aligned to ``target``.
        k: band width in spreads. ``0`` is the identity (returns ``target``
            row-for-row, bitwise).
        scale: optional per-row multiplier for the band, so ``k`` stays
            comparable across books of different gross. Pass
            ``target.abs().sum(axis=1)`` to express the band in units of
            "spread x book gross"; ``None`` = 1.0 (correct when the book is
            constructed at gross 1).
        neutralize: rescale the surviving trade vector's two halves to equal
            gross before applying it, so the band cannot move net exposure.
            Set False for a book that is deliberately not dollar-neutral
            (e.g. a hedge-ratio-weighted book).
        eligible: optional boolean panel of names the book is ALLOWED to hold.
            Anything false is forced to zero, overriding both the target and
            the band. Supply it when eligibility is a MANDATE rather than a
            preference — e.g. a short-eligibility list — so a violation is
            impossible by construction rather than merely unlikely. Missing
            entries count as ineligible.
        mandatory_exit: never block a trade that takes a held name to zero.
            **This is the correct default, not a conservatism knob.** See
            below. ``False`` lets the band block exits too, for comparison.

    Returns:
        The book actually HELD — the traded book, not the target.

    Mandatory exits — why the band must not block them
    --------------------------------------------------
    A band that treats an EXIT like a rebalance parks positions forever. When
    a name leaves the eligible pool its target becomes exactly ``0.0``; the
    resulting deviation is one position's worth, which for the wider spreads
    can be smaller than the band — so the name is never sold, sits in the
    book at zero cost, and the "turnover saving" the band appears to buy is
    partly just never exiting. Under an eligibility mandate that is not merely a cost artifact — a frozen position
    outside the permitted pool is a constraint violation.

    So a trade to zero is always executed, in full, and the discretionary
    remainder is then balanced to net the OPPOSITE of what those exits moved
    (``balance_trades(..., offset=...)``), which keeps ``sum(new) ==
    sum(held)`` exactly. Only if the discretionary side cannot supply the
    offset does net exposure move at all — and letting it move is the right
    call there, because the alternative is holding a position the mandate
    forbids.

    Causality: row ``t`` uses ``target[t]``, ``spreads[t]``, ``eligible[t]``
    and the state from ``t-1`` only.
    """
    # C-order: pandas hands back an F-contiguous view for a single-dtype
    # frame, which makes the row-at-a-time loop below a strided gather.
    tv = np.ascontiguousarray(target.to_numpy(dtype="float64"))
    sv = np.ascontiguousarray(
        spreads.reindex_like(target).to_numpy(dtype="float64"))
    if np.isnan(sv).any():
        # An unknown spread must not become a ZERO band (which would make the
        # name the most eagerly traded in the book — exactly backwards).
        # Widest observed is the conservative reading, matching spread_floor.
        sv[np.isnan(sv)] = np.nanmax(sv) if np.isfinite(sv).any() else 0.0
    n_rows, n_cols = tv.shape
    if scale is None:
        sc = np.ones(n_rows)
    else:
        sc = pd.Series(scale).reindex(target.index).to_numpy(dtype="float64")
        sc = np.nan_to_num(sc, nan=0.0)
    ev = None
    if eligible is not None:
        ev = np.ascontiguousarray(
            eligible.reindex(index=target.index, columns=target.columns)
            .fillna(False).to_numpy(dtype=bool))

    out = np.empty_like(tv)
    held = np.zeros(n_cols)
    kf = float(k)
    for t in range(n_rows):
        tgt = tv[t]
        if ev is not None:
            # A forbidden name has no target but zero — including when the
            # signal said NaN ("hold"), because the mandate outranks the hold.
            tgt = np.where(ev[t], tgt, 0.0)
        d = tgt - held
        # NaN target = no update. `d` is NaN there, so mask before comparing.
        miss = np.isnan(d)
        if miss.any():
            d = np.where(miss, 0.0, d)
        band = kf * sv[t] * sc[t]
        # An exit is not a rebalance: leaving is mandatory, so the band never
        # gets a vote on it (see the docstring).
        must = ((tgt == 0.0) & (held != 0.0)) if mandatory_exit \
            else np.zeros(n_cols, dtype=bool)
        blocked = (np.abs(d) <= band) & (d != 0.0) & ~must
        if not blocked.any():
            # Nothing was suppressed, so nothing needs rebalancing: go to
            # target exactly (k=0 is then the identity, bitwise) and hold the
            # names whose target is missing. Routing this row through
            # balance_trades instead would throttle every name whenever ONE
            # name had a NaN target — the identity would be a lie.
            held = tgt if not miss.any() else np.where(miss, held, tgt)
        else:
            forced = np.where(must, d, 0.0)
            disc = np.where(blocked | miss | must, 0.0, d)
            if neutralize:
                # The exits go through untouched; the discretionary trades are
                # scaled to net the opposite, so the pair leaves net exposure
                # exactly where it was.
                offset = -float(forced.sum())
                disc = balance_trades(disc, offset=offset)
                resid = offset - float(disc.sum())
                if resid != 0.0:
                    # The trades that would have absorbed the exit are
                    # themselves inside the band — the common case, since an
                    # exit and the renormalisation it triggers are the same
                    # size. Neutrality outranks the band, so unblock the
                    # MINIMUM needed, largest deviation first. Without this
                    # the band silently wins and expelling one long leaves the
                    # book short by that name's whole position.
                    pool = np.where(blocked & (np.sign(d) == np.sign(resid)),
                                    d, 0.0)
                    disc = disc + greedy_trades(pool, abs(resid),
                                                balanced=False)
            held = held + forced + disc
        out[t] = held
    return pd.DataFrame(out, index=target.index, columns=target.columns)


# ══════════════════════════════════════════════════════════════════════════
# 5. Daily turnover budget
# ══════════════════════════════════════════════════════════════════════════

def turnover_budget(target: Union[pd.DataFrame, pd.Series, np.ndarray],
                    held=None,
                    budget: Union[None, float, pd.Series] = None,
                    neutralize: bool = True,
                    eligible=None,
                    mandatory_exit: bool = True):
    """Cap the gross traded per day, spending the cap on the largest deviations.

    Two forms, matching the two ways this gets used:

    * **single day** — ``turnover_budget(target_row, held_row, budget)`` with
      1-D inputs, returning the new position vector: one rebalance step.
    * **panel** — ``turnover_budget(target_df, None, budget)``, which walks
      the panel statefully from flat and returns the held book.

    Args:
        target: desired positions. NaN = no target update (position held).
        held: previous positions (single-day form) or ``None`` (panel form).
        budget: max ``sum(|new - held|)`` per day. Scalar, or a per-row Series
            for the panel form — pass ``b * target.abs().sum(axis=1)`` to cap
            turnover as a FRACTION of book gross, which is both causal and
            comparable across books. ``None``/``inf`` is the identity.
        neutralize: split the budget evenly between the buy and sell halves so
            the trade sums to zero and net exposure is preserved.
        eligible: optional boolean mask of names the book may hold (panel or
            single row, matching ``target``). Anything false is forced to
            zero. Use it when eligibility is a mandate.
        mandatory_exit: execute trades that take a held name to zero in full,
            OUTSIDE the cap, before the cap is spent on anything else.
            **Correct default, not a knob.** A greedy cap fills the LARGEST
            deviations first, so a small exit is starved indefinitely and the
            name is parked — the same defect as in ``no_trade_band``. Exits
            are not discretionary rebalancing, so they
            do not compete for the cap; the day's traded gross can therefore
            exceed ``budget`` when names are being expelled, which is visible
            in the measured turnover and is the intended trade-off.
            ``False`` lets exits compete for the cap, for comparison.

    The cap is never exceeded by DISCRETIONARY trading: with ``neutralize``
    the two halves get ``budget/2`` each, without it the greedy fill stops at
    ``budget``. Mandatory exits are additional to it.
    """
    if isinstance(target, pd.DataFrame):
        if held is not None:
            raise ValueError("panel form takes held=None (state is internal)")
        return _turnover_budget_panel(target, budget, neutralize, eligible,
                                      mandatory_exit)

    tgt = np.asarray(target, dtype="float64")
    prev = (np.zeros_like(tgt) if held is None
            else np.asarray(held, dtype="float64"))
    ev = None if eligible is None else np.asarray(eligible, dtype=bool)
    new = _budget_step(tgt, prev, _cap(budget), neutralize, ev, mandatory_exit)
    if isinstance(target, pd.Series):
        return pd.Series(new, index=target.index)
    return new


def _cap(budget) -> float:
    return np.inf if budget is None else float(budget)


def _budget_step(tgt: np.ndarray, held: np.ndarray, cap: float,
                 neutralize: bool, elig: Optional[np.ndarray] = None,
                 mandatory_exit: bool = True) -> np.ndarray:
    if elig is not None:
        tgt = np.where(elig, tgt, 0.0)
    d = tgt - held
    miss = np.isnan(d)
    if miss.any():
        d = np.where(miss, 0.0, d)
    if np.isinf(cap):
        # No budget = the identity, and a NaN target must not drag the whole
        # row through the balancing step (that would move names AWAY from a
        # target they could have reached for free).
        return tgt.copy() if not miss.any() else np.where(miss, held, tgt)
    if not mandatory_exit:
        return held + greedy_trades(d, cap, balanced=neutralize)
    # Exits first and in full: a greedy cap fills the biggest deviations, so
    # a small exit would otherwise never be reached (see turnover_budget).
    must = (tgt == 0.0) & (held != 0.0)
    forced = np.where(must, d, 0.0)
    disc = np.where(must | miss, 0.0, d)
    if neutralize:
        offset = -float(forced.sum())
        # The companion trade that keeps the book neutral is the OTHER HALF of
        # the same mandate: expelling a long and not reducing a short leaves a
        # directional book. So it is not charged to the cap either — without
        # this the cap silently wins and the exit re-opens net exposure it was
        # never allowed to open.
        disc = greedy_trades(disc, max(cap, abs(offset)), balanced=True,
                             offset=offset)
    else:
        disc = greedy_trades(disc, cap, balanced=False)
    return held + forced + disc


def _turnover_budget_panel(target: pd.DataFrame, budget, neutralize: bool,
                           eligible=None, mandatory_exit: bool = True):
    tv = target.to_numpy(dtype="float64", copy=False)
    n_rows, n_cols = tv.shape
    if budget is None:
        caps = np.full(n_rows, np.inf)
    elif np.isscalar(budget):
        caps = np.full(n_rows, float(budget))
    else:
        caps = pd.Series(budget).reindex(target.index).to_numpy(dtype="float64")
        caps = np.nan_to_num(caps, nan=0.0)
    ev = None
    if eligible is not None:
        ev = np.ascontiguousarray(
            eligible.reindex(index=target.index, columns=target.columns)
            .fillna(False).to_numpy(dtype=bool))

    out = np.empty_like(tv)
    held = np.zeros(n_cols)
    for t in range(n_rows):
        held = _budget_step(tv[t], held, caps[t], neutralize,
                            None if ev is None else ev[t], mandatory_exit)
        out[t] = held
    return pd.DataFrame(out, index=target.index, columns=target.columns)
