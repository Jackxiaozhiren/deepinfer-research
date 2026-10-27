"""Selective-verification machinery (Milestone 3, Lane M3-L1).

Closes the project's first code-level gap (HANDOFF_PROMPT.md §2 gap 1): the
engine computes a calibration monitor (`batch_confidence_error`,
asc/sampler.py:556-558 -> sequence.py:248 -> stats_logger.py:236) that never
feeds the allocation policy. This module builds the missing loop in simulation:
rules that READ the predicted confidence values, and the waste they realize.

Waste semantics (M3 Ruling A, user ruling 2026-10-27): per (sequence, step)
the allocator returns `seqlens_q = 1 + wins`, of which only the `wins` draft
positions are verifications — the fresh base row is not one and enters
neither numerator nor denominator. With k = accepted prefix inside the window,
rejected = wins - k, and the aggregate is the POOLED ratio

    W = sum(wins - k) / sum(wins)

not the mean of per-row ratios (the two differ: they give 1/6 vs 1/2 on
fixture F-A2). At the budget floor sum(wins) = 0 and W is 0/0; that is
reported as nan, never coerced to 0 — zero waste would read as "no waste"
where nothing was verified at all (N4).

Calibration axis (M3 Ruling D): the head's reading is perturbed per request
TYPE by an exponent, r_pred = r_true ** gamma[type]. The exponent form is the
only one for which M3 §4.1's cumprod commutation is exact; an additive bias
breaks within-sequence monotonicity at the clip boundary and would destroy the
invariant the §4 design space rests on. gamma perturbs the HEAD, not the
generator, so acceptance is always sampled from the true rates (M2 Ruling-B
semantics carried over) and the fixed-5 calibration band is gamma-invariant by
construction — no level re-matching is performed (user ruling 5). The budget
knob that gamma DOES move is calibrated instead: theta is solved by bisection
on a named reference seed and then frozen (user ruling 5).

The three rule families and their selector classes (M3 §4.2, corrected per
user ruling 2 — the discriminating property is THRESHOLD EQUIVARIANCE, not
"reading values"):

  (a) rank top-k    — global top-k on cumprod; rank-relative.
  (b1) fixed-c      — absolute threshold; value-reading, gamma-vulnerable.
  (b2) quantile     — relative threshold; equivariant, so global-gamma
                      invariant like (a) but per-type-gamma sensitive.
  (b3) BH step-up   — absolute, data-adaptive; see the operability note.
  (c) hybrid        — eligibility clamp composed with top-k.

BH operability (user ruling 3, corrected by this module's own check): with
p = 1 - cumprod, the step-up rule rejects anything iff
alpha >= min_k n * p_(k) / k, which measures 0.4505 on V1 and 0.0645 on V2E at
conc 16 — so BH is NOT structurally inert on V1 (a first draft here claimed
alpha >= n * p_(1) = 2.51 was required, which is the condition for rejecting
only the single most significant hypothesis). What the measurement does show is
that on V1 the selection is a step: 0 slots for alpha <= 0.4 and 52 of 80 at
alpha = 0.5, saturating to all 80 by alpha = 0.9. A family that cannot be aimed
at a waste level is a poor competitor for the L2 sweep, so BH is retained as a
declared diagnostic with the operability criterion asserted in both directions.

Hybrid form (user ruling 7): eligibility is applied as a POST-RANK CLAMP,
wins = min(top-k wins, eligible wins). The alternative — zeroing ineligible
rates and re-running the primitive — is wrong: zeroed cells tie at 0.0 and win
slack slots by flat index, awarding verification to positions the rule
declared ineligible (measured: 1080 and 1200 of 1200 sequences at c = 0.8/0.9).
That is the same defect class as the upstream zero-rate slack finding, reached
independently from the other side of the codebase. A clamp under-spends the
nominal budget instead; the under-spend is a reported output, not a bug.

`python -m sim.selective` re-runs the two anchor families (Rulings B and C)
plus the BH operability criterion at the reference configuration and exits
nonzero with a `<FAILED ...>` line on any violation (N4).
"""

from __future__ import annotations

import sys

import numpy as np

from sim.allocator import accepted_prefix_in_window, dynamic_seqlens
from sim.rank_noise import V2E_LEVELS, noise_z
from sim.rates import (
    DEPTH_DECAY,
    EMIT_BONUS,
    EPS,
    NEXTN,
    PROGRESS_DECAY,
    TYPE_LEVELS,
    TYPE_WEIGHTS,
    rates_matrix,
    stratified_types,
)

# Named seed family, separate from 20261022 (L1), 20261101 (L2 bootstrap),
# 20261201 (M2 noise). The sigma axis reuses rank_noise.noise_z verbatim so an
# M3 point at sigma = 1.6 is the SAME noise realization as the M2 one.
GAMMA_SEED = 20261202
REFERENCE_SEED = GAMMA_SEED

# M3 §4.1/§4.3 axes, kept separate per Ruling D.
GAMMA_IDENTITY = (1.0, 1.0, 1.0)
# Lane-triage option B, user ruling 4: identity + three inflation singletons
# + two mixed vectors. Inflation (gamma < 1) is represented because under-
# confidence applied to the LOW type is provably inert under top-k.
GAMMA_TYPE_CONDITIONS = (
    (1.0, 1.0, 1.0),
    (0.5, 1.0, 1.0),
    (1.0, 0.5, 1.0),
    (1.0, 1.0, 0.5),
    (2.0, 1.0, 0.5),
    (1.5, 1.5, 0.5),
)
# Negative-control row (user ruling 4): run once, excluded from the L2 grid.
GAMMA_INERT_CONTROL = (1.0, 1.0, 2.0)
# The M2-calibrated sigma pair (pure ranking vs plateau), grid option B.
SIGMA_CONDITIONS = (0.0, 1.6)

# Anchor dynamic verified column (HANDOFF_PROMPT.md §2; published claims used
# as reference points, never tuning targets — N8). Units: draft positions.
ANCHOR_DYNAMIC_VERIFIED = {16: 4.44, 32: 3.65, 48: 3.94, 60: 3.58, 72: 3.24, 96: 3.24}

THETA_BISECTION_ITERS = 40
THETA_BISECTION_TOL = 1e-4
# B1 tolerance, frozen by user ruling 1's rule (2.7x the measured gap). The
# measurement denominator is 144 runs = 2 concs x 12 GAMMA_SEED seeds x the
# ANCHOR_C_GRID at 6000 steps: max |realized - predicted| = 0.002685, at the
# SMALLEST concurrency (conc 16; conc 96's max is 0.001075, because the pooled
# estimator's noise falls with cell count). 0.0075 = 2.79x that maximum.
# A first freeze of 0.002 was taken from a single-seed grid (max 0.00075) and
# would have failed inside the L2 seed family at conc 16; corrected before any
# sweep runs, recorded in ledger entry 29.
ANCHOR_IDENTITY_TOL = 0.0075

GLOBAL_GAMMA = "global"
PER_TYPE_GAMMA = "per-type"


def gamma_kind(gamma: tuple[float, float, float]) -> str:
    """Classify a gamma vector by its own shape (Ruling D axis hygiene).

    A declared-per-type run whose entries are all equal is really a global run,
    and conflating the two is exactly what Ruling D forbids, so the kind is
    derived at emit time rather than passed in as a label.
    """
    vec = _gamma_vector(gamma)
    return GLOBAL_GAMMA if np.all(vec == vec[0]) else PER_TYPE_GAMMA


def _gamma_vector(gamma: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(gamma, dtype=np.float64)
    assert vec.shape == (3,), f"gamma must have 3 type entries, got shape {vec.shape}"
    assert np.all(vec > 0), f"gamma exponents must be positive, got {vec}"
    return vec


def predicted_rates(
    r_true: np.ndarray,
    gamma: tuple[float, float, float],
    types: np.ndarray,
    sigma: float = 0.0,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """The head's reading of the true rates: per-type exponent, then value noise.

    Declared composition order (triage §3): the systematic calibration bias is
    applied first, the step-to-step value error second, both in RATE space
    because the allocator primitive consumes rates and re-cumprods internally
    (the same realization choice as sim/rank_noise.py:6-9). gamma = (1,1,1) and
    sigma = 0 return r_true bit-identically: np.power(r, 1.0) == r and
    np.exp(0.0) == 1.0 exactly.
    """
    vec = _gamma_vector(gamma)[types][:, None]
    out = np.clip(r_true**vec, EPS, 1.0 - EPS)
    if sigma:
        assert z is not None, "sigma > 0 requires the shared noise block z"
        out = np.clip(out * np.exp(sigma * z), EPS, 1.0 - EPS)
    return out


def predicted_cumprod(r_pred: np.ndarray) -> np.ndarray:
    """The quantity the rules actually read: cumprod along draft positions."""
    return np.cumprod(r_pred, axis=1)


# --- waste accounting (Ruling A) ------------------------------------------------


def waste_stats(wins: np.ndarray, k: np.ndarray) -> dict[str, float]:
    """Pooled waste over (sequence, step) cells, draft positions only.

    wins and k must already be window-truncated (k <= wins); pass seqlens_q
    instead of wins and the fresh row enters the denominator, which this
    function deliberately cannot do for you. sum(wins) == 0 gives nan, not 0.
    """
    wins = np.asarray(wins, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    assert wins.shape == k.shape, f"shape mismatch {wins.shape} vs {k.shape}"
    assert np.all(k <= wins), f"accepted {k.min()} may not exceed verified {wins.min()}"
    assert np.all(k >= 0) and np.all(wins >= 0), "counts must be non-negative"
    verified = float(wins.sum())
    rejected = float((wins - k).sum())
    return {
        "verified_total": verified,
        "rejected_total": rejected,
        "mean_verified": verified / wins.size if wins.size else float("nan"),
        "waste": rejected / verified if verified > 0 else float("nan"),
    }


def predicted_waste_from_pool(sel_sum: float, sel_count: int) -> float:
    """1 - E[predicted cumprod over the verified window] from pooled cells.

    Under exact calibration this is the quantity the policy could have bought
    from the head alone; Ruling B's literal `1 - c` is its marginal value at the
    threshold and bounds it from ABOVE, because every verified cell has
    cumprod >= c. Empty pool gives nan, matching waste_stats at the budget floor.
    """
    if sel_count == 0:
        return float("nan")
    return 1.0 - float(sel_sum) / int(sel_count)


def predicted_waste(cum_pred: np.ndarray, wins: np.ndarray) -> float:
    """Single-batch form of predicted_waste_from_pool (hand fixtures use it)."""
    cum_pred = np.asarray(cum_pred, dtype=np.float64)
    wins = np.asarray(wins)
    assert cum_pred.shape[0] == wins.shape[0], f"row mismatch {cum_pred.shape} vs {wins.shape}"
    assert np.all(wins <= cum_pred.shape[1]), f"wins exceeds nextn: {wins.max()} > {cum_pred.shape[1]}"
    mask = np.arange(cum_pred.shape[1], dtype=np.int64)[None, :] < wins[:, None]
    return predicted_waste_from_pool(float(np.where(mask, cum_pred, 0.0).sum()), int(mask.sum()))


# --- rule families (exact shapes per triage §3) ----------------------------------


def rank_topk_wins(total_q: int, r_pred: np.ndarray) -> np.ndarray:
    """(a) The unchanged M1/M2 allocator: global top-k over all cells."""
    return dynamic_seqlens(total_q, r_pred).astype(np.int64) - 1


def eligible_wins(c: float, cum_pred: np.ndarray) -> np.ndarray:
    """Prefix length whose predicted cumprod clears c (b1, and (c)'s mask)."""
    return (cum_pred >= c).sum(axis=1).astype(np.int64)


def fixed_c_wins(c: float, cum_pred: np.ndarray) -> np.ndarray:
    """(b1) Absolute threshold: verify the prefix above a fixed c."""
    return eligible_wins(c, cum_pred)


def quantile_wins(q: float, cum_pred: np.ndarray) -> np.ndarray:
    """(b2) Relative threshold: the batch's (1-q) quantile of predicted cumprods.

    A quantile commutes with a monotone transform, so this selector is invariant
    to a global gamma and sensitive only to per-type gamma — measured, and
    asserted in tests/test_selective_invariants.py (C3).
    """
    assert 0.0 <= q <= 1.0, f"q must be in [0, 1], got {q}"
    c = float(np.quantile(cum_pred, 1.0 - q))
    return eligible_wins(c, cum_pred)


def bh_threshold(alpha: float, cum_pred: np.ndarray) -> float:
    """BH step-up threshold on p = 1 - cumprod over the batch's cells.

    Returns -1.0 when nothing is rejected (the V1 case at every admissible
    alpha, see bh_operability_boundary).
    """
    assert 0.0 < alpha < 1.0, f"alpha must be in (0, 1), got {alpha}"
    n = cum_pred.size
    p = np.sort((1.0 - cum_pred).ravel())
    levels = alpha * np.arange(1, n + 1) / n
    rejected = p <= levels
    if not rejected.any():
        return -1.0
    return float(p[np.arange(1, n + 1)[rejected][-1] - 1])


def bh_wins(alpha: float, cum_pred: np.ndarray) -> np.ndarray:
    """(b3) BH-style step-up selection; prefix-closed because p is monotone."""
    thr = bh_threshold(alpha, cum_pred)
    if thr < 0.0:
        return np.zeros(cum_pred.shape[0], dtype=np.int64)
    return ((1.0 - cum_pred) <= thr).sum(axis=1).astype(np.int64)


def bh_operability_boundary(cum_pred_batch: np.ndarray) -> float:
    """The smallest alpha at which BH rejects ANYTHING: min_k n * p_(k) / k.

    BH's step-up rule rejects up to the largest k with p_(k) <= alpha * k / n,
    so the family is inert exactly when alpha is below that minimum. This is NOT
    n * p_(1) — the condition for rejecting the single most significant
    hypothesis — which is the error this module corrected in its own first
    draft (n * p_(1) = 2.51 on V1 would have said "never operative", while the
    measured first-rejection alpha is 0.4505, see the entry-29 correction).
    """
    cum = np.asarray(cum_pred_batch, dtype=np.float64)
    n = cum.size
    p = np.sort((1.0 - cum).ravel())
    return float((n * p / np.arange(1, n + 1)).min())


def hybrid_wins(total_q: int, c: float, r_pred: np.ndarray, cum_pred: np.ndarray) -> np.ndarray:
    """(c) Top-k restricted to the threshold-eligible prefix, as a post-rank clamp.

    min(top-k, eligible) IS the top-k of the restricted pool while the pool
    absorbs the budget (both value-ordered, so the intersection is the
    restriction), and is the only form that stays truthful under slack.
    """
    return np.minimum(rank_topk_wins(total_q, r_pred), eligible_wins(c, cum_pred))


# --- budget calibration for value-reading rules ----------------------------------


def calibrate_c_for_budget(
    target_verified: float,
    num_seqs: int,
    steps: int,
    seed: int,
    *,
    family: str = "fixed_c",
    q: float = 0.5,
    total_q: int | None = None,
    gamma: tuple[float, float, float] = GAMMA_IDENTITY,
    sigma: float = 0.0,
    iters: int = THETA_BISECTION_ITERS,
    levels: tuple[float, float, float] | None = None,
    weights: tuple[float, float, float] | None = None,
) -> tuple[float, float]:
    """Bisection on the fixed-c threshold c to hit a target mean verified length.

    Solved ONCE on a named reference seed and then frozen (user ruling 5), so
    the drift reading isolates gamma's effect on a fixed rule parameter. Returns
    (c, realized mean verified). Deterministic: the interval is halved, no RNG.

    family is restricted to the two rules that read an absolute c ("fixed_c" and
    "hybrid"); the quantile family needs no search because its own knob is a
    cell fraction — see q_for_target.
    """
    assert family in ("fixed_c", "hybrid"), f"only absolute-c families are searched, got {family!r}"
    lo, hi = EPS, 1.0 - EPS
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        realized = _mean_verified_for_c(mid, num_seqs, steps, seed, family=family, q=q,
                                        total_q=total_q, gamma=gamma, sigma=sigma,
                                        levels=levels, weights=weights)
        if realized > target_verified:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    return c, _mean_verified_for_c(c, num_seqs, steps, seed, family=family, q=q,
                                   total_q=total_q, gamma=gamma, sigma=sigma,
                                   levels=levels, weights=weights)


def q_for_target(target_verified: float) -> float:
    """The quantile family's aim at a target mean verified length: q = target / NEXTN.

    A q-quantile threshold admits roughly q * NEXTN draft positions per sequence,
    so this family needs no search where the absolute-c families do. The
    realized value is still measured, because prefix rounding makes admission
    approximate (measured: q = 0.65 realizes 3.250 verified at conc 16).
    """
    q = float(target_verified) / NEXTN
    assert 0.0 <= q <= 1.0, f"target {target_verified} is outside [0, NEXTN={NEXTN}]"
    return q


def _mean_verified_for_c(c: float, num_seqs: int, steps: int, seed: int, **kw) -> float:
    rule = kw.pop("family", "fixed_c")
    total_q = kw.pop("total_q", None)
    out = simulate_rule(rule, num_seqs, steps, seed, c=c, total_q=total_q, **kw)
    return out["rule"]["mean_verified"]


# --- the paired harness ---------------------------------------------------------


def simulate_rule(
    rule: str,
    num_seqs: int,
    steps: int,
    seed: int,
    *,
    c: float | None = None,
    q: float | None = None,
    alpha: float | None = None,
    total_q: int | None = None,
    gamma: tuple[float, float, float] = GAMMA_IDENTITY,
    sigma: float = 0.0,
    levels: tuple[float, float, float] | None = None,
    weights: tuple[float, float, float] | None = None,
    depth_decay: float | None = None,
    progress_decay: float | None = None,
    emit_bonus: int | None = None,
) -> dict[str, object]:
    """Run one rule arm against the fixed-5 reference over the same ambient state.

    rule ∈ {"rank_topk", "fixed_c", "quantile", "bh", "hybrid"}. Semantics carried
    from M2: the selection sees the head's reading, acceptance is sampled from
    the TRUE rates, and each arm advances its own progress trajectory. The u
    block is drawn from (seed, num_seqs) and the z block from
    (NOISE_SEED, num_seqs) exactly as sim/rank_noise.py does, so sigma = 0 here
    is bit-comparable with the M2 pipeline and sigma = 1.6 reuses the M2
    realization. Returns per-arm arrays plus the waste readings and, for budget-
    bound rules, the under-spend against the nominal extra budget.
    """
    if emit_bonus is None:
        emit_bonus = EMIT_BONUS
    levels = TYPE_LEVELS if levels is None else levels
    weights = TYPE_WEIGHTS if weights is None else weights
    depth_decay = DEPTH_DECAY if depth_decay is None else depth_decay
    progress_decay = PROGRESS_DECAY if progress_decay is None else progress_decay

    params = {"fixed_c": c, "quantile": q, "bh": alpha, "hybrid": c}
    if rule in params and params[rule] is None:
        raise ValueError(f"rule {rule!r} requires its threshold parameter")
    if rule == "rank_topk" and total_q is None:
        raise ValueError("rank_topk requires total_q")
    if rule == "hybrid" and total_q is None:
        raise ValueError("hybrid requires total_q (it clamps the top-k)")

    types = stratified_types(num_seqs, weights)
    u = np.random.default_rng((seed, num_seqs)).random((num_seqs, steps, NEXTN))
    z = noise_z(num_seqs, steps) if sigma else None

    k_fixed = np.zeros((num_seqs, steps), dtype=np.int64)
    k_rule = np.zeros((num_seqs, steps), dtype=np.int64)
    wins_rule = np.zeros((num_seqs, steps), dtype=np.int64)
    pred_waste_sum = 0.0
    pred_cell_count = 0
    generated_fixed = np.zeros(num_seqs, dtype=np.float64)
    generated_rule = np.zeros(num_seqs, dtype=np.float64)
    pos = np.arange(NEXTN, dtype=np.int64)[None, :]

    for t in range(steps):
        r_true_f = rates_matrix(levels, depth_decay, progress_decay, types, generated_fixed)
        accept_f = u[:, t, :] < r_true_f
        k_fixed[:, t] = np.cumprod(accept_f, axis=1, dtype=np.int64).sum(axis=1)
        generated_fixed += k_fixed[:, t] + emit_bonus

        r_true_r = rates_matrix(levels, depth_decay, progress_decay, types, generated_rule)
        accept_r = u[:, t, :] < r_true_r
        r_pred = predicted_rates(r_true_r, gamma, types, sigma=sigma, z=None if z is None else z[:, t, :])
        cum_pred = predicted_cumprod(r_pred)
        if rule == "rank_topk":
            w = rank_topk_wins(total_q, r_pred)
        elif rule == "fixed_c":
            w = fixed_c_wins(c, cum_pred)
        elif rule == "quantile":
            w = quantile_wins(q, cum_pred)
        elif rule == "bh":
            w = bh_wins(alpha, cum_pred)
        elif rule == "hybrid":
            w = hybrid_wins(total_q, c, r_pred, cum_pred)
        else:
            raise ValueError(f"unknown rule: {rule}")
        k_rule[:, t] = accepted_prefix_in_window(accept_r, w)
        wins_rule[:, t] = w
        sel = pos < w[:, None]
        pred_waste_sum += float(np.where(sel, cum_pred, 0.0).sum())
        pred_cell_count += int(sel.sum())
        generated_rule += k_rule[:, t] + emit_bonus

    waste = waste_stats(wins_rule, k_rule)
    verified_total = waste["verified_total"]
    pred_waste = predicted_waste_from_pool(pred_waste_sum, pred_cell_count)
    nominal_extra = None if total_q is None else (total_q - num_seqs) * steps
    return {
        "rule_name": rule,
        "gamma_kind": gamma_kind(gamma),
        "fixed": {"k": k_fixed, "emitted_mean": float((k_fixed + emit_bonus).mean())},
        "rule": {
            "k": k_rule,
            "wins": wins_rule,
            "emitted_mean": float((k_rule + emit_bonus).mean()),
            **waste,
            "pred_waste": pred_waste,
            "waste_gap": waste["waste"] - pred_waste,
            "under_spend": (
                float("nan") if nominal_extra is None or nominal_extra <= 0
                else 1.0 - verified_total / nominal_extra
            ),
        },
    }


# --- loud self-check (N4): `python -m sim.selective` ----------------------------

# Hand-derived waste fixtures (Ruling A). These are arithmetic, not
# measurements; re-deriving them here makes an accounting regression visible
# before any simulation runs.
WASTE_FIXTURES = (
    ("F-A1 single row v=5 k=3", [(5, 3)], 0.4),
    ("F-A2 pooled (5,5)+(1,0)", [(5, 5), (1, 0)], 1.0 / 6.0),
    ("F-A3 draft-only (4,3)+(0,0)", [(4, 3), (0, 0)], 0.25),
    ("F-A5 truncated k=v (2,2)", [(2, 2)], 0.0),
)

BH_INERT_ALPHAS_ON_V1 = (0.01, 0.05, 0.10, 0.25, 0.35, 0.40, 0.44)
BH_MONOTONE_ALPHAS = (0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.99)
BH_ALPHA_GRID = np.linspace(0.001, 0.999, 999)
BH_ALPHA_GRID_STEP = float(BH_ALPHA_GRID[1] - BH_ALPHA_GRID[0])
ANCHOR_C_GRID = (0.30, 0.45, 0.60, 0.75, 0.85, 0.92)
SELF_CHECK_C = 0.60
DRIFT_GAMMAS = (0.5, 1.0, 1.5, 2.0)
SELF_CHECK_CONC = 16
SELF_CHECK_STEPS = 6000


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else '<FAILED>'} {name}: {detail}")
    return ok


def _rate_block(conc: int, seed: int, levels: tuple[float, float, float] | None = None) -> np.ndarray:
    types = stratified_types(conc, TYPE_WEIGHTS)
    gen = np.random.default_rng((seed, conc)).integers(0, 20000, size=conc).astype(float)
    return rates_matrix(TYPE_LEVELS if levels is None else levels, DEPTH_DECAY, PROGRESS_DECAY, types, gen)


def _anchor_run(gamma: tuple[float, float, float] = GAMMA_IDENTITY, c: float = SELF_CHECK_C) -> dict:
    return simulate_rule(
        "fixed_c", SELF_CHECK_CONC, SELF_CHECK_STEPS, REFERENCE_SEED, c=c, gamma=gamma
    )["rule"]


def _per_type_displacement(gamma: tuple[float, float, float]) -> int:
    """Sequences whose top-k window changes under this gamma vector.

    Six anchor concs x 3 rate blocks at total_q = conc * 3, so 972 sequences.
    """
    changed = 0
    for conc in (16, 32, 48, 60, 72, 96):
        types = stratified_types(conc, TYPE_WEIGHTS)
        for s in range(3):
            r = _rate_block(conc, 100 + s)
            changed += int((rank_topk_wins(conc * 3, r) != rank_topk_wins(conc * 3, predicted_rates(r, gamma, types))).sum())
    return changed


def _bh_slots(alpha: float, cum: np.ndarray) -> int:
    """Draft slots a single BH batch selects on this predicted-cumprod block."""
    return int(bh_wins(alpha, cum).sum())


def _bh_first_rejection_alpha(cum: np.ndarray) -> float:
    """Empirical smallest alpha on the declared grid that rejects anything."""
    for a in BH_ALPHA_GRID:
        if _bh_slots(float(a), cum) > 0:
            return float(a)
    return float("nan")


def main() -> int:
    ok = True

    for name, rows, expected in WASTE_FIXTURES:
        wins = np.array([v for v, _ in rows], dtype=np.int64)
        k = np.array([a for _, a in rows], dtype=np.int64)
        got = waste_stats(wins, k)["waste"]
        ok &= _report(name, abs(got - expected) < 1e-12, f"W={got:.6f} expected={expected:.6f}")
    floor = waste_stats(np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64))["waste"]
    ok &= _report("F-A4 budget floor is nan, not 0", np.isnan(floor), f"W={floor}")

    ident = _rate_block(16, 7)
    ok &= _report(
        "gamma=identity with sigma=0 is an exact identity",
        np.array_equal(predicted_rates(ident, GAMMA_IDENTITY, stratified_types(16, TYPE_WEIGHTS)), ident),
        "predicted_rates returns its input bit-identically",
    )

    ref = _anchor_run()
    ok &= _report(
        "B1 calibrated-identity anchor",
        abs(ref["waste_gap"]) <= ANCHOR_IDENTITY_TOL,
        f"realized={ref['waste']:.5f} predicted={ref['pred_waste']:.5f} gap={ref['waste_gap']:+.5f} "
        f"tol={ANCHOR_IDENTITY_TOL} (c={SELF_CHECK_C}, conc {SELF_CHECK_CONC}, {SELF_CHECK_STEPS} steps, seed {REFERENCE_SEED})",
    )
    dividend = (1.0 - SELF_CHECK_C) - ref["waste"]
    ok &= _report(
        "B2 threshold selectivity dividend",
        ref["waste"] <= (1.0 - SELF_CHECK_C) + ANCHOR_IDENTITY_TOL and dividend > 0.05,
        f"realized={ref['waste']:.5f} <= 1-c={1.0 - SELF_CHECK_C:.5f}; dividend={dividend:.5f} "
        "(Ruling B's literal 1-c is the marginal bound, not the expectation)",
    )

    gaps = {g: _anchor_run(gamma=(g, g, g))["waste_gap"] for g in DRIFT_GAMMAS}
    ok &= _report(
        "B3 global-gamma drift: sign and ordering",
        abs(gaps[1.0]) <= ANCHOR_IDENTITY_TOL and gaps[0.5] > 0 > gaps[1.5] and gaps[2.0] < gaps[1.5],
        " ".join(f"gamma={g}:{gaps[g]:+.5f}" for g in DRIFT_GAMMAS),
    )

    comm = []
    for conc in (16, 32, 48, 60, 72, 96):
        total_q = int(round(conc * ANCHOR_DYNAMIC_VERIFIED[conc])) if conc == 16 else conc * 3
        for s in range(4):
            r = _rate_block(conc, s)
            base = rank_topk_wins(total_q, r)
            comm += [np.array_equal(base, rank_topk_wins(total_q, np.clip(r**g, EPS, 1.0 - EPS)))
                     for g in (0.5, 1.5, 2.0, 3.0)]
    ok &= _report("C1 global-gamma commutation", all(comm), f"{sum(comm)}/{len(comm)} allocations bit-identical")

    live = _per_type_displacement((2.0, 1.0, 1.0))
    inert = _per_type_displacement(GAMMA_INERT_CONTROL)
    ok &= _report(
        "C2 per-type gamma is live; low-type under-confidence is inert",
        live > 0 and inert == 0,
        f"gamma=(2,1,1) changes {live} sequences; gamma={GAMMA_INERT_CONTROL} changes {inert} "
        "(declared negative control, 972 sequences surveyed)",
    )

    v1_cum = predicted_cumprod(_rate_block(16, 7))
    v2e_cum = predicted_cumprod(_rate_block(16, 7, V2E_LEVELS))
    boundary = bh_operability_boundary(v1_cum)
    first_emp = _bh_first_rejection_alpha(v1_cum)
    inert_below = all(_bh_slots(a, v1_cum) == 0 for a in BH_INERT_ALPHAS_ON_V1)
    live_above = _bh_slots(boundary + 0.05, v1_cum) > 0
    slots_above = [_bh_slots(a, v1_cum) for a in BH_MONOTONE_ALPHAS]
    monotone = all(lo <= hi for lo, hi in zip(slots_above, slots_above[1:]))
    v2e_live = _bh_slots(0.20, v2e_cum) > 0
    ok &= _report(
        "BH operability criterion (two-sided)",
        inert_below and live_above and monotone and v2e_live
        and 0.0 <= first_emp - boundary < BH_ALPHA_GRID_STEP,
        f"V1 boundary min_k n*p_(k)/k = {boundary:.4f}; empirical first rejecting alpha = "
        f"{first_emp:.4f} (grid step {BH_ALPHA_GRID_STEP:.4f}); strictly inert below "
        f"(alpha <= {BH_INERT_ALPHAS_ON_V1[-1]} -> 0 slots), operative above, monotone in alpha; "
        f"at alpha={BH_MONOTONE_ALPHAS[0]} it already takes {slots_above[0]} of {v1_cum.size} cells "
        f"(a step, not a ramp); V2E operative at alpha=0.20: {v2e_live}",
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
