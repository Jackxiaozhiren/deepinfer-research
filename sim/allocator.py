"""Allocation policies and the paired fixed-5 vs dynamic step simulator (Lane L2).

Wraps ``dflash_alloc.allocate_seqlens_q`` (the approved primitive, HANDOFF
PROMPT §3 — reused, not reimplemented) as the dynamic policy, and a constant
full-depth policy as fixed-5.

Row convention (sim/convention.md; schedule_mtp_varlen.py:98): seqlens_q
includes each sequence's final fresh row, so fixed-5 packs NEXTN + 1 = 6 rows
per sequence, and the dynamic policy degenerates to fixed-5 exactly at
``total_q = num_seqs * (NEXTN + 1)`` (HANDOFF §7.2's "num_seqs × 5" treated
draft positions as rows; corrected per user ruling 2026-10-22).

Modeling choices frozen for L2 (user-approved triage, 2026-10-22):

- Idealized ConfidenceHead: the allocator reads the current step's true
  generator rates. ZOS one-step staleness and ranking-error injection are out
  of scope (Milestone 2 directions). The reported dynamic gains are therefore
  an IDEAL-INFORMATION UPPER BOUND; a negative gain at conc 16 that still
  reproduces under this bound is a stronger conclusion (user ruling 4).
- remap = identity: a steady batch, no arrivals or terminations.
- Both arms share the stratified type vector and (when seeds match) the same
  pre-drawn u block, so fixed-vs-dynamic comparisons are paired per draw; at
  the degenerate budget the two arms are bit-identical.
- Horizon = CALIB_STEPS (frozen in L1; the accepted-length statistic is
  horizon-declared).
"""

from __future__ import annotations

import numpy as np

from dflash_alloc import allocate_seqlens_q
from sim.rates import (
    CALIB_STEPS,
    DEPTH_DECAY,
    EMIT_BONUS,
    NEXTN,
    PROGRESS_DECAY,
    TYPE_LEVELS,
    TYPE_WEIGHTS,
    rates_matrix,
    stratified_types,
)


def fixed5_seqlens(num_seqs: int) -> np.ndarray:
    """Fixed-5 policy: every sequence verifies all NEXTN drafts plus its fresh row."""
    return np.full(num_seqs, NEXTN + 1, dtype=np.int32)


def dynamic_seqlens(total_q: int, rates: np.ndarray) -> np.ndarray:
    """Dynamic policy: delegate to the approved allocator primitive (identity remap)."""
    remap = np.arange(rates.shape[0], dtype=np.int32)
    return allocate_seqlens_q(total_q, remap, rates)


def accepted_prefix_in_window(accept: np.ndarray, wins: np.ndarray) -> np.ndarray:
    """Accepted-prefix length counted within each sequence's verified window.

    accept: [num_seqs, NEXTN] bool per-position conditional accept draws.
    wins: [num_seqs] verified draft positions (a prefix per the allocator's
    cumprod ranking). The accepted prefix inside the first wins positions is
    min(full prefix, wins) by prefix monotonicity.
    """
    full = np.cumprod(accept, axis=1, dtype=np.int64).sum(axis=1)
    return np.minimum(full, wins.astype(np.int64))


def _run_arm(
    u: np.ndarray,
    types: np.ndarray,
    levels: tuple[float, float, float],
    *,
    depth_decay: float,
    progress_decay: float,
    mode: str,
    total_q: int,
    emit_bonus: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_seqs, steps, nextn = u.shape
    k = np.zeros((num_seqs, steps), dtype=np.int64)
    wins_out = np.zeros((num_seqs, steps), dtype=np.int64)
    generated = np.zeros(num_seqs, dtype=np.float64)
    for t in range(steps):
        r = rates_matrix(levels, depth_decay, progress_decay, types, generated)
        accept = u[:, t, :] < r
        if mode == "fixed":
            w = np.full(num_seqs, nextn, dtype=np.int64)
            k[:, t] = np.cumprod(accept, axis=1, dtype=np.int64).sum(axis=1)
        elif mode == "dynamic":
            w = dynamic_seqlens(total_q, r).astype(np.int64) - 1
            k[:, t] = accepted_prefix_in_window(accept, w)
        else:
            raise ValueError(f"unknown mode: {mode}")
        wins_out[:, t] = w
        generated += k[:, t] + emit_bonus
    return k, wins_out


def simulate_pair(
    num_seqs: int,
    steps: int,
    seed: int,
    *,
    total_q: int,
    levels: tuple[float, float, float] | None = None,
    weights: tuple[float, float, float] | None = None,
    depth_decay: float | None = None,
    progress_decay: float | None = None,
    dynamic_seed: int | None = None,
    emit_bonus: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Run the paired fixed-5 and dynamic arms over the same ambient state.

    Both arms draw their own u block from (seed, num_seqs) — with matching
    seeds the blocks are bit-identical, which is the pairing. Passing
    ``dynamic_seed`` different from ``seed`` deliberately breaks the pairing
    (the N5 mutant hook). Returns {"fixed": {"k", "wins"}, "dynamic": {"k",
    "wins"}}, arrays int64 [num_seqs, steps].
    """
    if emit_bonus is None:
        emit_bonus = EMIT_BONUS
    levels = TYPE_LEVELS if levels is None else levels
    weights = TYPE_WEIGHTS if weights is None else weights
    depth_decay = DEPTH_DECAY if depth_decay is None else depth_decay
    progress_decay = PROGRESS_DECAY if progress_decay is None else progress_decay

    types = stratified_types(num_seqs, weights)
    u_fixed = np.random.default_rng((seed, num_seqs)).random((num_seqs, steps, NEXTN))
    u_dyn = np.random.default_rng((dynamic_seed if dynamic_seed is not None else seed, num_seqs)).random(
        (num_seqs, steps, NEXTN)
    )
    k_fixed, wins_fixed = _run_arm(
        u_fixed, types, levels,
        depth_decay=depth_decay, progress_decay=progress_decay,
        mode="fixed", total_q=total_q, emit_bonus=emit_bonus,
    )
    k_dyn, wins_dyn = _run_arm(
        u_dyn, types, levels,
        depth_decay=depth_decay, progress_decay=progress_decay,
        mode="dynamic", total_q=total_q, emit_bonus=emit_bonus,
    )
    return {
        "fixed": {"k": k_fixed, "wins": wins_fixed},
        "dynamic": {"k": k_dyn, "wins": wins_dyn},
    }


def arm_stats(k: np.ndarray, wins: np.ndarray, emit_bonus: int | None = None) -> dict[str, float]:
    """Mean verified draft positions, accepted k, and emitted tokens per (sequence, step)."""
    if emit_bonus is None:
        emit_bonus = EMIT_BONUS
    return {
        "verified": float(wins.mean()),
        "accepted": float(k.mean()),
        "emitted": float((k + emit_bonus).mean()),
    }


def pairing_check(
    num_seqs: int,
    steps: int,
    seed: int,
    *,
    dynamic_seed: int | None = None,
) -> tuple[bool, str]:
    """Criterion: at the degenerate budget total_q = num_seqs * (NEXTN + 1) the
    two arms are bit-identical (every position wins; wins ≡ NEXTN).

    Why: extra_budget = NEXTN * num_seqs covers all num_seqs * NEXTN positions,
    so the allocator returns seqlens_q = NEXTN + 1 for every sequence
    (schedule_mtp_varlen.py:98; sim/convention.md) and the windows match the
    fixed arm. Flip: any allocator, draw-order, or pairing change makes the
    arrays diverge.
    """
    result = simulate_pair(
        num_seqs, steps, seed, total_q=num_seqs * (NEXTN + 1), dynamic_seed=dynamic_seed
    )
    identical = np.array_equal(result["fixed"]["k"], result["dynamic"]["k"])
    full_windows = bool((result["dynamic"]["wins"] == NEXTN).all())
    ok = identical and full_windows
    detail = (
        f"k arrays identical={identical}; dynamic wins all == {NEXTN}={full_windows}; "
        f"dynamic_seed={dynamic_seed if dynamic_seed is not None else seed}"
    )
    return ok, detail
