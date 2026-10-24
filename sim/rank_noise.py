"""Ranking-noise machinery (Milestone 2, Lane M2-L1).

Mechanism (user ruling 2026-10-24, option (i)): per-cell multiplicative
lognormal value noise. The allocator primitive consumes per-position rates
and re-cumprods internally (dflash_alloc.allocate_seqlens_q unchanged), so
the noise is applied in RATE space — r' = clip(r * exp(sigma * z), EPS,
1-EPS) with z ~ N(0, 1) i.i.d. per (sequence, position, step) — and the
cumprod ordering that the global top-k actually ranks carries the noise.
The triage's "C' = C * exp(sigma * z)" formula was shorthand for this
per-cell value noise; the rate-space realization preserves every approved
property (per-cell i.i.d., sigma -> infinity scrambling, monotone
degradation, Ruling B semantics, and the sigma = 0 bit-identity, since
exp(0) == 1.0 exactly). Kendall tau_K is measured on the cumprod ordering
(the ranking-relevant object) and thresholds are stated in tau_K.

Failure modes (declared, user ruling 1):
1. Heavy tails can let a single cell dominate the global top-k — part of the
   miscalibration model; per-run tau_K and per-row gains expose it.
2. The sigma -> tau_K map depends on batch geometry — thresholds are stated
   in tau_K (reported per run), never in sigma.
3. Per-cell i.i.d. noise ignores autocorrelation of miscalibration across
   steps — out of scope.

Rejected mechanisms (recorded per user ruling 1):
- (ii) pairwise-swap on the global ordering: indirect parameter-to-tau_K
  map; the engine's errors are value errors, not order surgeries.
- (iii) per-sequence level confusion: preserves within-sequence ordering
  perfectly, so tau_K cannot approach 0 — the near-total-scrambling endpoint
  is structurally unreachable.

Window-length semantics (Ruling B, load-bearing): the perturbation feeds
ONLY the allocator's ranking values, hence only each sequence's chosen
verification-window LENGTH (wins'). Acceptance is always sampled from the
TRUE rates over the first wins' draft positions; k_dyn = min(k_full,
wins'). Upstream justification: the allocator emits a length per sequence
(seqlens_q = 1 + wins, schedule_mtp_varlen.py:98) and the verifier accepts
a prefix (effective_accept, tilelang_ops/sampler.py:246, :778).

Seeds: NOISE_SEED = 20261201 (named, separate from the 20261022/20261101
families). The noise block z is drawn from (NOISE_SEED, num_seqs) AFTER the
u block, whose consumption order is therefore unchanged by sigma; the same
z is reused across all sigma (common random numbers, paired across the
grid) and by both arms (the fixed arm does not consume it). At sigma = 0
the pipeline is bit-identical to the M1 path.

V2E generator variant (Ruling D adoption, entry 10): high level pinned at
1-EPS — a DECLARED boundary, not a silent clip — mid level solved by
bisection on [0.40, 0.98] with 40 iterations to hold the fixed-5 accepted
length at the frozen L1 mean 3.3042, low level fixed at 0.50 BY CONVENTION:
the Ruling-D criteria do not uniquely determine (m, l), and the (c)
diagnostic's sensitivity to l is unmapped (declared). Candidate numbers
including the four rejected variants live in ledger entry 10.
"""

from __future__ import annotations

import numpy as np

from sim.allocator import accepted_prefix_in_window, dynamic_seqlens
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

NOISE_SEED = 20261201
V2E_LEVELS = (1.0 - EPS, 0.8682562200903792, 0.5)
V2E_TARGET_MEAN = 3.3042
TAU_STEPS = 64  # evenly spaced steps per run on which tau_K is evaluated


def noise_z(num_seqs: int, steps: int) -> np.ndarray:
    """The shared per-cell noise block, drawn after the u block."""
    return np.random.default_rng((NOISE_SEED, num_seqs)).standard_normal((num_seqs, steps, NEXTN))


def perturbed_rates(r_true: np.ndarray, z: np.ndarray, sigma: float) -> np.ndarray:
    """Per-cell multiplicative lognormal noise in rate space (sigma = 0 is the
    identity by construction: exp(0) == 1.0 exactly)."""
    return np.clip(r_true * np.exp(sigma * z), EPS, 1.0 - EPS)


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau_b between two aligned 1-D arrays (vectorized pairwise
    signs; ties in both variables enter the denominator — Kendall-b)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, 1)
    sx = np.sign(x[iu[0]] - x[iu[1]])
    sy = np.sign(y[iu[0]] - y[iu[1]])
    prod = sx * sy
    conc = int(np.sum(prod > 0))
    disc = int(np.sum(prod < 0))
    _, tx = np.unique(x, return_counts=True)
    _, ty = np.unique(y, return_counts=True)
    n0 = n * (n - 1) / 2.0
    n1 = float(np.sum(tx * (tx - 1) / 2))
    n2 = float(np.sum(ty * (ty - 1) / 2))
    denom = float(np.sqrt((n0 - n1) * (n0 - n2)))
    if denom <= 0:
        return float("nan")
    return (conc - disc) / denom


def kendall_tau_bruteforce(x: np.ndarray, y: np.ndarray) -> float:
    """Independent O(n^2) double-loop reference for the cross-check tests."""
    x = list(np.asarray(x, dtype=np.float64).ravel())
    y = list(np.asarray(y, dtype=np.float64).ravel())
    n = len(x)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = int(x[i] > x[j]) - int(x[i] < x[j])
            sy = int(y[i] > y[j]) - int(y[i] < y[j])
            p = sx * sy
            if p > 0:
                conc += 1
            elif p < 0:
                disc += 1
    _, tx = np.unique(np.asarray(x), return_counts=True)
    _, ty = np.unique(np.asarray(y), return_counts=True)
    n0 = n * (n - 1) / 2.0
    n1 = float(np.sum(tx * (tx - 1) / 2))
    n2 = float(np.sum(ty * (ty - 1) / 2))
    denom = float(np.sqrt((n0 - n1) * (n0 - n2)))
    if denom <= 0:
        return float("nan")
    return (conc - disc) / denom


def degradation_check(
    mean_accepted_by_sigma: dict[float, float], *, tol: float, endpoint_margin: float
) -> tuple[bool, str]:
    """Monotone-degradation criterion (user ruling 3): adjacent grid points
    non-increasing within the declared tolerance `tol` (the seed-noise
    equivalent criterion), plus strict endpoint degradation —
    mean(sigma_max) < mean(sigma_min) - endpoint_margin — which carries the
    falsification weight (the identity-noise mutant fails exactly here).
    Returns (ok, detail)."""
    sigmas = sorted(mean_accepted_by_sigma)
    values = [mean_accepted_by_sigma[s] for s in sigmas]
    violations = []
    for s_a, s_b, v_a, v_b in zip(sigmas, sigmas[1:], values, values[1:]):
        if v_b > v_a + tol:
            violations.append(f"sigma {s_a}->{s_b}: +{v_b - v_a:.4f} > tol {tol}")
    endpoint_drop = values[0] - values[-1]
    strict_ok = endpoint_drop > endpoint_margin
    ok = (not violations) and strict_ok
    detail = (
        f"means={ {s: round(v, 4) for s, v in zip(sigmas, values)} }; "
        f"adjacent violations={violations}; endpoint drop={endpoint_drop:.4f} (margin {endpoint_margin})"
    )
    return ok, detail


def simulate_pair_noisy(
    num_seqs: int,
    steps: int,
    seed: int,
    *,
    total_q: int,
    sigma: float,
    levels: tuple[float, float, float] | None = None,
    weights: tuple[float, float, float] | None = None,
    depth_decay: float | None = None,
    progress_decay: float | None = None,
    emit_bonus: int | None = None,
) -> dict[str, object]:
    """The M1 paired pipeline with ranking noise on the dynamic arm's
    allocation input (Ruling B semantics). At sigma = 0 it is bit-identical
    to allocator.simulate_pair. Returns per-arm k/wins arrays, the fixed
    arm's sigma-independent arrays, and the run's mean Kendall tau_K between
    true and perturbed cumprod orderings on TAU_STEPS evenly spaced steps.
    """
    if emit_bonus is None:
        emit_bonus = EMIT_BONUS
    levels = TYPE_LEVELS if levels is None else levels
    weights = TYPE_WEIGHTS if weights is None else weights
    depth_decay = DEPTH_DECAY if depth_decay is None else depth_decay
    progress_decay = PROGRESS_DECAY if progress_decay is None else progress_decay

    types = stratified_types(num_seqs, weights)
    u = np.random.default_rng((seed, num_seqs)).random((num_seqs, steps, NEXTN))
    z = noise_z(num_seqs, steps)

    k_fixed = np.zeros((num_seqs, steps), dtype=np.int64)
    k_dyn = np.zeros((num_seqs, steps), dtype=np.int64)
    wins_dyn = np.zeros((num_seqs, steps), dtype=np.int64)
    k_true_prefix = np.zeros((num_seqs, steps), dtype=np.int64)
    generated_fixed = np.zeros(num_seqs, dtype=np.float64)
    generated_dyn = np.zeros(num_seqs, dtype=np.float64)
    tau_samples: list[float] = []
    tau_idx = set(np.linspace(0, steps - 1, min(TAU_STEPS, steps)).astype(np.int64).tolist())

    for t in range(steps):
        # Fixed arm: full-depth window, own progress trajectory.
        r_true_f = rates_matrix(levels, depth_decay, progress_decay, types, generated_fixed)
        accept_f = u[:, t, :] < r_true_f
        k_fixed[:, t] = np.cumprod(accept_f, axis=1, dtype=np.int64).sum(axis=1)
        generated_fixed += k_fixed[:, t] + emit_bonus

        # Dynamic arm: own progress trajectory; the RANKING sees perturbed
        # rates while acceptance samples the TRUE rates (Ruling B).
        r_true_d = rates_matrix(levels, depth_decay, progress_decay, types, generated_dyn)
        accept_d = u[:, t, :] < r_true_d
        k_true_prefix[:, t] = np.cumprod(accept_d, axis=1, dtype=np.int64).sum(axis=1)
        r_rank = perturbed_rates(r_true_d, z[:, t, :], sigma)
        w = dynamic_seqlens(total_q, r_rank).astype(np.int64) - 1
        k_dyn[:, t] = accepted_prefix_in_window(accept_d, w)
        wins_dyn[:, t] = w
        generated_dyn += k_dyn[:, t] + emit_bonus

        if t in tau_idx:
            c_true = np.cumprod(r_true_d, axis=1).ravel()
            c_pert = np.cumprod(r_rank, axis=1).ravel()
            tau_samples.append(kendall_tau(c_true, c_pert))

    return {
        "fixed": {"k": k_fixed},
        "dynamic": {"k": k_dyn, "wins": wins_dyn, "k_true_prefix": k_true_prefix},
        "tau_k": float(np.mean(tau_samples)),
    }
