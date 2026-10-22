"""Seeded acceptance-rate generator and fixed-5 calibration (Lane L1).

Implements the rate generator family frozen in sim/convention.md:

    r_ij = clip( level_type(i) * depth_decay^(j-1) * exp(-progress_decay * G_i), eps, 1-eps )

- level_type(i): per-sequence acceptance level, drawn from a fixed 3-type
  mixture. Cross-sequence heterogeneity is the load-bearing modeling
  assumption of the whole study (HANDOFF_PROMPT.md §7.1): a homogeneous
  population makes dynamic ~ fixed by construction, because a global top-k
  allocator has no room to gain.
- depth_decay^(j-1): within-step draft-depth decay (deeper drafts are
  conditioned on more unverified predictions).
- exp(-progress_decay * G_i): request-progress decay, where G_i counts tokens
  already emitted by sequence i. THE PROGRESS TERM IS DESIGN-DRIVEN, NOT
  EMPIRICAL: it exists so the emission convention (EMIT_BONUS) is observable
  in the accepted-length statistic, which lets the N5 mutant test go red
  (EMIT_BONUS=0 slows progress accumulation, raises acceptance, and pushes
  the statistic out of the calibration band). No traffic evidence supports
  it; no calibration conclusion may rely on its empirical meaning
  (sim/convention.md, user note 2026-10-22). The accepted-length statistic is
  DEFINED AT THE FROZEN HORIZON CALIB_STEPS: with the frozen progress_decay
  the per-sequence accepted length drifts by a few tenths of a token between
  a fresh request and a tens-of-thousands-token-old request, and the
  statistic would shift if the horizon changed.

Draw order (frozen; any change moves every seeded result):
  1. types = stratified_types(num_seqs, weights)  (deterministic, no RNG)
  2. u[num_seqs, steps, NEXTN] = rng.random(...)

The same u block is consumed with and without the EMIT_BONUS flip, so
base-vs-mutant comparisons are paired per draw: with identical types and
draws, smaller accumulated progress means weakly higher rates at every
(sequence, step), so the mutant arm dominates pointwise. Acceptance is a
sequential per-position Bernoulli process; k is the accepted prefix length; a
step emits k + EMIT_BONUS tokens and advances G by the same amount.

`python -m sim.rates` re-runs the deterministic calibration search, verifies
the frozen constants against its output plus the band and mutant constraints,
and exits nonzero with a `<FAILED ...>` line on any violation.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

EMIT_BONUS = 1
NEXTN = 5
EPS = 1e-6

CONCURRENCY_ROWS = (16, 32, 48, 60, 72, 96)

# Calibration band from the anchor table's fixed-5 accepted column
# (doc/inference_deployment_qa.md:249-256, transcribed to HANDOFF_PROMPT.md §2).
BAND_MEAN = (3.29, 3.32)
BAND_ROW = (3.25, 3.32)

CALIB_SEED = 20261022
CALIB_STEPS = 6000

# Frozen by the deterministic search in this module (run `python -m sim.rates`
# to re-verify): first stage-2-feasible candidate in (|statistic - 3.305|,
# progress_decay) order; seeds CALIB_SEED, horizon CALIB_STEPS.
TYPE_LEVELS = (0.97, 0.85, 0.7)
TYPE_WEIGHTS = (0.5, 0.25, 0.25)
DEPTH_DECAY = 0.99
PROGRESS_DECAY = 1e-06


def _levels_array(levels: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(levels, dtype=np.float64)
    assert arr.shape == (3,), f"levels must have 3 entries, got shape {arr.shape}"
    return arr


def _weights_array(weights: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    assert arr.shape == (3,), f"weights must have 3 entries, got shape {arr.shape}"
    assert np.all(arr > 0), f"weights must be positive, got {arr}"
    return arr / arr.sum()


def stratified_types(num_seqs: int, weights: tuple[float, float, float]) -> np.ndarray:
    """Type assignment with exactly the mixture proportions (largest remainder).

    The engine's benchmark statistics average over a long window of request
    arrivals, so the composition noise of i.i.d. type draws (±1.4/sqrt(conc)
    on the accepted length) is not part of the modeled quantity; stratifying
    removes it while keeping the population heterogeneous. Type t occupies a
    contiguous block (sorted arrangement).
    """
    w = _weights_array(weights)
    counts = np.floor(w * num_seqs).astype(np.int64)
    remainder = num_seqs - int(counts.sum())
    if remainder > 0:
        frac = w * num_seqs - counts
        for idx in np.argsort(-frac, kind="stable")[:remainder]:
            counts[idx] += 1
    assert int(counts.sum()) == num_seqs, f"stratified counts {counts} != num_seqs {num_seqs}"
    return np.repeat(np.arange(3, dtype=np.int64), counts)


def rates_matrix(
    levels: tuple[float, float, float],
    depth_decay: float,
    progress_decay: float,
    types: np.ndarray,
    generated: np.ndarray,
) -> np.ndarray:
    """Per-(sequence, position) conditional acceptance rates, shape [num_seqs, NEXTN]."""
    positions = np.arange(NEXTN, dtype=np.float64)
    r = (
        _levels_array(levels)[types][:, None]
        * (depth_decay**positions)[None, :]
        * np.exp(-progress_decay * generated)[:, None]
    )
    return np.clip(r, EPS, 1.0 - EPS)


def simulate_fixed5(
    num_seqs: int,
    steps: int,
    seed: int,
    *,
    levels: tuple[float, float, float],
    weights: tuple[float, float, float],
    depth_decay: float,
    progress_decay: float,
    emit_bonus: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the fixed-5 arm: every sequence verifies all NEXTN drafts per step.

    Returns (k, emitted), both int64 [num_seqs, steps]: accepted-prefix
    lengths and emitted token counts. emit_bonus defaults to the module-level
    EMIT_BONUS read at call time, so the N5 mutant can flip it.
    """
    if emit_bonus is None:
        emit_bonus = EMIT_BONUS
    rng = np.random.default_rng((seed, num_seqs))
    types = stratified_types(num_seqs, weights)
    u = rng.random((num_seqs, steps, NEXTN))

    k = np.zeros((num_seqs, steps), dtype=np.int64)
    generated = np.zeros(num_seqs, dtype=np.float64)
    for t in range(steps):
        r = rates_matrix(levels, depth_decay, progress_decay, types, generated)
        accepted = u[:, t, :] < r
        k[:, t] = np.cumprod(accepted, axis=1, dtype=np.int64).sum(axis=1)
        generated += k[:, t] + emit_bonus

    emitted = k + emit_bonus
    return k, emitted


def fixed5_statistic(
    num_seqs: int,
    steps: int,
    seed: int,
    *,
    levels: tuple[float, float, float],
    weights: tuple[float, float, float],
    depth_decay: float,
    progress_decay: float,
    emit_bonus: int | None = None,
) -> float:
    """Mean accepted draft count k per (sequence, step), fixed-5 arm."""
    k, _ = simulate_fixed5(
        num_seqs,
        steps,
        seed,
        levels=levels,
        weights=weights,
        depth_decay=depth_decay,
        progress_decay=progress_decay,
        emit_bonus=emit_bonus,
    )
    return float(k.mean())


def fixed5_report(
    *,
    seed: int = CALIB_SEED,
    steps: int = CALIB_STEPS,
    concs: tuple[int, ...] = CONCURRENCY_ROWS,
    emit_bonus: int | None = None,
    levels: tuple[float, float, float] | None = None,
    weights: tuple[float, float, float] | None = None,
    depth_decay: float | None = None,
    progress_decay: float | None = None,
) -> dict[int, float]:
    """Per-concurrency-row accepted-length statistic.

    Defaults to the frozen constants. The same seed family ((seed, num_seqs))
    is used with and without the EMIT_BONUS flip, so base-vs-mutant row
    differences are paired.
    """
    kwargs = dict(
        levels=TYPE_LEVELS if levels is None else levels,
        weights=TYPE_WEIGHTS if weights is None else weights,
        depth_decay=DEPTH_DECAY if depth_decay is None else depth_decay,
        progress_decay=PROGRESS_DECAY if progress_decay is None else progress_decay,
        emit_bonus=emit_bonus,
    )
    return {
        c: fixed5_statistic(c, steps, seed, **kwargs)  # type: ignore[arg-type]
        for c in concs
    }


def band_check(vals: dict[int, float]) -> tuple[bool, str]:
    """Calibration-band criterion (N6). Returns (ok, detail)."""
    mean = sum(vals.values()) / len(vals)
    out_rows = {c: v for c, v in vals.items() if not (BAND_ROW[0] <= v <= BAND_ROW[1])}
    ok = BAND_MEAN[0] <= mean <= BAND_MEAN[1] and not out_rows
    rows = ", ".join(f"{c}: {v:.4f}" for c, v in sorted(vals.items()))
    detail = (
        f"mean={mean:.4f} band={BAND_MEAN}; rows=({rows}); "
        f"row band={BAND_ROW}; out-of-band rows={out_rows}"
    )
    return ok, detail


def _search_grids() -> tuple:
    # Weights are pinned to (0.5, 0.25, 0.25): divisible by 1/4 = gcd of the
    # anchor concurrencies, so stratified_types is exact at every row. The
    # progress_decay grid brackets the smallest value whose mutant signal
    # (~9 * xi * CALIB_STEPS / 2) clears the band edge with margin.
    hs = (0.96, 0.97, 0.98)
    ms = (0.85, 0.90)
    ls = (0.60, 0.65, 0.70)
    weight_pairs = ((0.5, 0.25),)
    rhos = (0.985, 0.99, 0.995)
    xis = (5e-7, 1e-6, 2e-6, 4e-6)
    return hs, ms, ls, weight_pairs, rhos, xis


_SEARCH_CHEAP_CONC = 16
_SEARCH_TARGET = 3.305
_SEARCH_TOLERANCE = 0.008
_SEARCH_FULL_MAX = 8


def calibrate() -> dict:
    """Deterministic grid search for parameters satisfying every N6 constraint.

    Stage 1 filters on the smallest concurrency row (cheap eval), stage 2 runs
    the full six-row report plus the mutant constraints in deterministic
    |value - target| order and returns the first fully feasible candidate.
    Exits nonzero with a `<FAILED ...>` line if nothing is feasible.
    """
    hs, ms, ls, weight_pairs, rhos, xis = _search_grids()
    survivors: list[tuple[tuple, float]] = []
    for h, m, low, (wh, wm), rho, xi in itertools.product(hs, ms, ls, weight_pairs, rhos, xis):
        weights = (wh, wm, round(1.0 - wh - wm, 4))
        if weights[2] <= 0:
            continue
        levels = (h, m, low)
        val = fixed5_statistic(
            _SEARCH_CHEAP_CONC,
            CALIB_STEPS,
            CALIB_SEED,
            levels=levels,
            weights=weights,
            depth_decay=rho,
            progress_decay=xi,
        )
        if abs(val - _SEARCH_TARGET) <= _SEARCH_TOLERANCE:
            survivors.append(((levels, weights, rho, xi), val))
    survivors.sort(key=lambda item: (round(abs(item[1] - _SEARCH_TARGET), 3), item[0][3]))

    for params, _ in survivors[:_SEARCH_FULL_MAX]:
        levels, weights, rho, xi = params
        vals = fixed5_report(levels=levels, weights=weights, depth_decay=rho, progress_decay=xi)
        ok, detail = band_check(vals)
        if not ok:
            continue
        mut_vals = fixed5_report(
            levels=levels, weights=weights, depth_decay=rho, progress_decay=xi, emit_bonus=0
        )
        mut_ok, mut_detail = band_check(mut_vals)
        mut_mean = sum(mut_vals.values()) / len(mut_vals)
        direction = min(mut_vals.values()) > max(vals.values())
        if not mut_ok and direction and mut_mean > BAND_MEAN[1]:
            return {"params": params, "report": vals, "mutant_report": mut_vals}

    print(
        f"<FAILED> no feasible calibration point: {len(survivors)} stage-1 survivors, "
        f"none passed the full band + mutant constraints"
    )
    sys.exit(1)


def _main() -> int:
    found = calibrate()
    (levels, weights, rho, xi) = found["params"]
    frozen = (tuple(TYPE_LEVELS), tuple(TYPE_WEIGHTS), float(DEPTH_DECAY), float(PROGRESS_DECAY))
    print("search output (paste-ready):")
    print(f"  TYPE_LEVELS     = {levels}")
    print(f"  TYPE_WEIGHTS    = {weights}")
    print(f"  DEPTH_DECAY     = {rho!r}")
    print(f"  PROGRESS_DECAY  = {xi!r}")
    if frozen != (tuple(levels), tuple(weights), float(rho), float(xi)):
        print(f"<FAILED> frozen constants {frozen} != search output; paste the values above")
        return 1
    ok, detail = band_check(found["report"])
    print(f"frozen constants verified; band check {'OK' if ok else 'RED'}: {detail}")
    if not ok:
        return 1
    mut_ok, mut_detail = band_check(found["mutant_report"])
    mut_mean = sum(found["mutant_report"].values()) / len(found["mutant_report"])
    direction = min(found["mutant_report"].values()) > max(found["report"].values())
    print(
        f"mutant EMIT_BONUS=0: band_check={'OK (BAD: must be red)' if mut_ok else 'RED (expected)'}; "
        f"mutant mean={mut_mean:.4f} (must be > {BAND_MEAN[1]}); "
        f"paired direction min(mut)>max(base)={direction}"
    )
    print(f"  mutant detail: {mut_detail}")
    if mut_ok or not direction or mut_mean <= BAND_MEAN[1]:
        print("<FAILED> mutant constraint violated (N5 control cannot fire)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
