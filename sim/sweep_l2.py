"""Lane L2 sensitivity sweep (HANDOFF §7.2 + user rulings 2026-10-22).

Paired fixed-5 vs dynamic simulation across concurrency, budget, cost-model
parameters, and the two design-driven generator axes (PROGRESS_DECAY and
heterogeneity level spread). Writes results/sweep_l2.md.

Design points frozen for this sweep:

- Both axes are MEAN-MATCHED by a uniform shift of the whole level vector:
  the shift is solved by deterministic bisection so the fixed-5 accepted
  length stays at the frozen L1 mean (3.3042) while the axis parameter
  varies. A uniform shift preserves every type gap, so the heterogeneity
  structure is held exactly (an h-only re-solve would compress the gaps and
  corrupt the spread axis; a larger-PROGRESS_DECAY point can also be
  unreachable that way). The pre-match vector, the applied shift, and any
  clipping are recorded — no silent deformation (user ruling 5). Without
  matching, a larger PROGRESS_DECAY drags the base statistic far below the
  anchor band and the mutant-red criterion would fail through base drift,
  not through signal loss.
- The sign conclusion per axis point is PAIRED (both arms share the variant's
  generator), so it is meaningful even where the absolute calibration is
  re-matched rather than frozen.
- Everything is seeded and deterministic (N3); violations print `<FAILED ...>`
  and exit nonzero (N4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sim import allocator
from sim import cost_model as cm
from sim import rates
from sim.rates import BAND_MEAN, CALIB_STEPS, CONCURRENCY_ROWS, EPS, NEXTN, TYPE_LEVELS, TYPE_WEIGHTS

L2_SWEEP_SEED = 20261101
N_SWEEP_SEEDS = 12
BOOTSTRAP_SEED = 20261101
BOOTSTRAP_RESAMPLES = 10000

# rows/seq including the fresh row; covers every anchor-matched budget
# (verified_dyn + 1) plus the 4.0 and 6.0 endpoints. Corrected from the
# triage message's {4.25, 4.5} to {4.24, 4.58} — the triage grid missed the
# conc-60/72/96 matched budgets despite stating full coverage.
BUDGET_GRID = (4.0, 4.24, 4.58, 4.65, 4.94, 5.44, 6.0)
ANCHOR_MATCHED_QBAR = {conc: round(verified + 1, 2) for conc, (verified, _, _, _, _) in cm.ANCHOR.items()}

COST_AB = tuple(float(x) for x in np.logspace(1, 4, 10))
# The anchor-implied a/b values from the linear-exclusion derivation join the
# t_pow grid so the reference tables read the exact pinned points. The three
# band-center values sit between the conc-16 and conc-32 flip curves (derived
# analytically from this sweep's own emitted ratios; see the flip-curve
# section) — they verify the predicted anchor-pattern band, they are not a fit.
COST_AB_EXTENDED = tuple(sorted(set(COST_AB) | {910.9, 2676.7, 465.0, 1550.0, 4900.0}))
COST_C = (0.9, 1.0, 1.1, 1.3, 1.5)
COST_BETA = (0.0, 0.5, 2.0)
SAT_REF_V = 500.0
def _ref_cost_points() -> tuple:
    """Reference (a/b, c) points; the fourth uses an exact t_pow grid ab."""
    return ((2676.7, 1.0), (2676.7, 1.1), (910.9, 1.3), (COST_AB[7], 1.3))

XI_GRID = (1e-06, 2e-06, 3e-06)
# 4e-06 was probed and excluded: holding the fixed-5 mean at 3.3042 there
# requires a level shift that pushes the high type past 1 - EPS (measured:
# h at the bisection ceiling 0.999 still yields only mean 3.1281), so the
# operating point is not reachable; the axis covers the mean-matchable,
# mutant-red range instead.
HETERO_GRID = (0.5, 1.0, 1.5)
MEAN_MATCH_TARGET = 3.3042
MEAN_MATCH_TOL = 0.005
SHIFT_LO, SHIFT_HI, BISECT_ITERS = -0.15, 0.15, 16

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "sweep_l2.md"


def bootstrap_ci(
    deltas: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Mean and 2.5/97.5 percentile CI of per-seed paired log-ratio deltas (numpy only)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    assert deltas.ndim == 1 and deltas.size >= 2, f"deltas shape {deltas.shape}"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_resamples, deltas.size))
    means = deltas[idx].mean(axis=1)
    return float(deltas.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _seeds() -> list[int]:
    return [L2_SWEEP_SEED + i for i in range(N_SWEEP_SEEDS)]


def _run_pair_stats(conc: int, seed: int, total_q: int, **gen_kwargs) -> dict[str, dict[str, float]]:
    res = allocator.simulate_pair(conc, CALIB_STEPS, seed, total_q=total_q, **gen_kwargs)
    return {
        "seed": seed,
        "fixed": allocator.arm_stats(res["fixed"]["k"], res["fixed"]["wins"]),
        "dynamic": allocator.arm_stats(res["dynamic"]["k"], res["dynamic"]["wins"]),
    }


def run_main_sweep() -> dict[tuple[int, float], list[dict]]:
    """Paired sims over (conc, budget) x seeds, frozen generator parameters."""
    data: dict[tuple[int, float], list[dict]] = {}
    for conc in CONCURRENCY_ROWS:
        for qbar in BUDGET_GRID:
            total_q = int(round(conc * qbar))
            data[(conc, qbar)] = [_run_pair_stats(conc, seed, total_q) for seed in _seeds()]
        print(f"  main sweep conc={conc} done", flush=True)
    return data


def run_anchor_matched_variant(**gen_kwargs) -> dict[int, list[dict]]:
    """Paired sims at each anchor-matched budget under variant generator parameters."""
    out: dict[int, list[dict]] = {}
    for conc in CONCURRENCY_ROWS:
        total_q = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
        out[conc] = [_run_pair_stats(conc, seed, total_q, **gen_kwargs) for seed in _seeds()]
    return out


def match_levels_shift(
    base_levels: tuple[float, float, float], progress_decay: float
) -> tuple[tuple[float, float, float], float, float, bool]:
    """Bisect a uniform level shift so the six-row fixed-5 mean stays at
    MEAN_MATCH_TARGET at the given progress_decay.

    A uniform shift preserves every type gap (the heterogeneity structure).
    Returns (levels, achieved_mean, delta, post_shift_clipped). The base
    vector is clipped into (EPS, 1-EPS) first; post_shift_clipped must be
    False for the operating point to count as held.
    """
    base = np.clip(np.asarray(base_levels, dtype=np.float64), EPS, 1.0 - EPS)

    def mean_at(delta: float) -> float:
        rep = rates.fixed5_report(levels=tuple(base + delta), progress_decay=progress_decay)
        return sum(rep.values()) / len(rep)

    lo, hi = SHIFT_LO, SHIFT_HI
    for _ in range(BISECT_ITERS):
        mid = (lo + hi) / 2
        if mean_at(mid) < MEAN_MATCH_TARGET:
            lo = mid
        else:
            hi = mid
    delta = (lo + hi) / 2
    levels = tuple(base + delta)
    clipped = bool(any(x <= EPS or x >= 1.0 - EPS for x in levels))
    achieved = mean_at(delta)
    return levels, achieved, delta, clipped


def mutant_red_check(levels: tuple[float, float, float], progress_decay: float) -> dict:
    """Mutant-red criterion at the given generator point: with EMIT_BONUS=0 the
    fixed-5 statistic must leave the band with its mean above the band top."""
    base = rates.fixed5_report(levels=levels, progress_decay=progress_decay)
    mut = rates.fixed5_report(levels=levels, progress_decay=progress_decay, emit_bonus=0)
    mut_ok, mut_detail = rates.band_check(mut)
    mut_mean = sum(mut.values()) / len(mut)
    base_mean = sum(base.values()) / len(base)
    return {
        "levels": levels,
        "progress_decay": progress_decay,
        "base_mean": base_mean,
        "base_in_band": BAND_MEAN[0] <= base_mean <= BAND_MEAN[1],
        "mut_mean": mut_mean,
        "mut_red": (not mut_ok) and mut_mean > BAND_MEAN[1],
        "mut_detail": mut_detail,
    }


def t_ratio_pow(v_fixed: float, v_dynamic: float, ab: float, c: float) -> float:
    return (ab + v_fixed**c) / (ab + v_dynamic**c)


def t_ratio_sat(v_fixed: float, v_dynamic: float, ab: float, beta: float) -> float:
    def t(v: float) -> float:
        return ab + v + beta * v * v / SAT_REF_V

    return t(v_fixed) / t(v_dynamic)


def _per_seed_deltas(rows: list[dict], v_fixed: float, v_dynamic: float, t_ratio: float) -> np.ndarray:
    """Per-seed paired log-throughput-ratio deltas: log(emitted_dyn/emitted_fix) + log(T(v_f)/T(v_d))."""
    out = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i] = np.log(row["dynamic"]["emitted"] / row["fixed"]["emitted"]) + np.log(t_ratio)
    return out


def _classify(mean: float, lo: float, hi: float) -> str:
    if hi < 0.0:
        return "-"
    if lo > 0.0:
        return "+"
    return "."


def sign_map(rows_by_setting: dict, matched_only: bool, t_ratio_fn) -> dict[tuple, dict[int, tuple[str, float, float, float]]]:
    """Sign of the paired gain per cost point and conc, with bootstrap CI.

    rows_by_setting: {(conc, qbar): rows}; matched_only restricts to the
    anchor-matched budgets. Returns {(cost point): {conc: (sign, mean, lo, hi)}}.
    """
    settings = {conc: qbar for (conc, qbar) in rows_by_setting if not matched_only or np.isclose(qbar, ANCHOR_MATCHED_QBAR[conc])}
    out: dict[tuple, dict[int, tuple[str, float, float, float]]] = {}
    for conc, qbar in settings.items():
        rows = rows_by_setting[(conc, qbar)]
        v_fixed = conc * (NEXTN + 1)
        v_dynamic = int(round(conc * qbar))
        for cost_point in _cost_point_list(t_ratio_fn):
            t_ratio = t_ratio_fn(v_fixed, v_dynamic, *cost_point)
            deltas = _per_seed_deltas(rows, v_fixed, v_dynamic, t_ratio)
            mean, lo, hi = bootstrap_ci(deltas)
            gains = np.exp(deltas) - 1.0
            gmean = float(np.mean(gains))
            out.setdefault(cost_point, {})[conc] = (_classify(mean, lo, hi), gmean, mean, hi - lo)
    return out


def _cost_point_list(t_ratio_fn) -> list[tuple]:
    if t_ratio_fn is t_ratio_pow:
        return [(ab, c) for ab in COST_AB_EXTENDED for c in COST_C]
    return [(ab, beta) for ab in COST_AB for beta in COST_BETA]


def _anchor_pattern(signs: dict[int, tuple[str, float, float, float]]) -> bool:
    return signs[16][0] == "-" and all(signs[c][0] == "+" for c in (32, 48, 60, 72, 96))


def _fmt_gain(entry: tuple[str, float, float, float]) -> str:
    sign, gmean, _, _ = entry
    return f"{sign}{gmean * 100:+.2f}%"


def _flip_curve(data: dict, c: float) -> dict[int, float]:
    """a/b where the paired gain crosses zero at each conc (t_pow family, beta=0).

    ab* = (v_f**c - v_d**c / E) / (1/E - 1) with E the mean emitted ratio at
    the anchor-matched budget in `data` ({(conc, qbar): rows}).
    """
    out = {}
    for conc in CONCURRENCY_ROWS:
        qbar = ANCHOR_MATCHED_QBAR[conc]
        setting_rows = None
        for (conc_key, qbar_key), candidate in data.items():
            if conc_key == conc and np.isclose(qbar_key, qbar):
                setting_rows = candidate
                break
        assert setting_rows is not None, f"no rows for conc {conc} qbar {qbar}"
        e_ratio = float(np.mean([row["dynamic"]["emitted"] / row["fixed"]["emitted"] for row in setting_rows]))
        v_fixed = conc * (NEXTN + 1)
        v_dynamic = int(round(conc * qbar))
        denom = 1.0 / e_ratio - 1.0
        num = v_fixed**c - v_dynamic**c / e_ratio
        out[conc] = num / denom if denom > 0 and num > 0 else float("nan")
    return out


def _flip_band(data: dict, c: float = 1.1) -> tuple[float, float, bool]:
    """The anchor-pattern band at exponent c: between the conc-16 and conc-32
    zero-gain curves. Non-empty iff ab*_16 < ab*_32."""
    curve = _flip_curve(data, c)
    a16, a32 = curve[16], curve[32]
    return a16, a32, bool(np.isfinite(a16) and np.isfinite(a32) and a16 < a32)


def _band_summary(data: dict) -> str:
    """First exponent c (ascending) whose anchor-pattern band is non-empty."""
    for c in COST_C:
        a16, a32, ok = _flip_band(data, c)
        if ok:
            return f"c={c} [{a16:.0f}, {a32:.0f}]"
    return "none in swept c"


def _matched_dict(variant: dict[int, list[dict]]) -> dict[tuple[int, float], list[dict]]:
    return {(conc, ANCHOR_MATCHED_QBAR[conc]): rows for conc, rows in variant.items()}


_MAIN_DATA: dict = {}


def _build_markdown(sections: list[str]) -> str:
    head = [
        "# Lane L2 sweep — allocator integration, cost model, sensitivity (2026-10-22)",
        "",
        f"- Seeds: fixed arm and dynamic arm share (L2_SWEEP_SEED + i, num_seqs), i in [0, {N_SWEEP_SEEDS}); "
        f"bootstrap seed {BOOTSTRAP_SEED}, {BOOTSTRAP_RESAMPLES} resamples.",
        f"- Horizon: CALIB_STEPS = {CALIB_STEPS} steps per run (frozen L1 horizon).",
        "- Generator: frozen L1 constants (ledger entry 3); axis points mean-matched as described per section.",
        "- v = conc x rows/seq, rows including each sequence's fresh row (fixed-5: 6 rows/seq).",
        "- Gains are paired per-seed throughput ratios minus 1; signs classified by the bootstrap CI "
        "('-' = CI wholly negative, '+' = CI wholly positive, '.' = not significant).",
        "",
    ]
    return "\n".join(head + sections)


def _main() -> int:
    assert set(ANCHOR_MATCHED_QBAR.values()) <= set(BUDGET_GRID), (
        f"anchor-matched budgets {sorted(set(ANCHOR_MATCHED_QBAR.values()))} not covered by {BUDGET_GRID}"
    )
    sections: list[str] = []

    print("running main sweep...", flush=True)
    global _MAIN_DATA
    _MAIN_DATA = run_main_sweep()

    # Degenerate-budget pairing must be bit-identical for every (conc, seed).
    pairing_failures = []
    for conc in CONCURRENCY_ROWS:
        for row in _MAIN_DATA[(conc, 6.0)]:
            ef, ed = row["fixed"]["emitted"], row["dynamic"]["emitted"]
            vf, vd = row["fixed"]["verified"], row["dynamic"]["verified"]
            if ef != ed or vf != vd:
                pairing_failures.append((conc, row["seed"], ef, ed))
    if pairing_failures:
        print(f"<FAILED> degenerate-budget pairing broken: {pairing_failures[:5]}")
        return 1

    # 1. achieved-length sweep table
    lines = ["## Achieved lengths per (conc, budget)", "",
             "| conc | rows/seq budget | total_q | dynamic verified | dynamic accepted | fixed accepted |", 
             "|---:|---:|---:|---:|---:|---:|"]
    for conc in CONCURRENCY_ROWS:
        for qbar in BUDGET_GRID:
            rows = _MAIN_DATA[(conc, qbar)]
            ver = np.array([r["dynamic"]["verified"] for r in rows])
            acc_d = np.array([r["dynamic"]["accepted"] for r in rows])
            acc_f = np.array([r["fixed"]["accepted"] for r in rows])
            lines.append(
                f"| {conc} | {qbar:.2f} | {int(round(conc * qbar))} | {ver.mean():.3f}±{ver.std():.3f} "
                f"| {acc_d.mean():.3f}±{acc_d.std():.3f} | {acc_f.mean():.3f}±{acc_f.std():.3f} |"
            )
    sections.append("\n".join(lines))

    # 2. sign maps
    for title, fn in (("T(v) = a + b*v^c (beta = 0)", t_ratio_pow), ("T(v) = a + b*v + d*v^2 (saturation)", t_ratio_sat)):
        smap = sign_map(_MAIN_DATA, matched_only=True, t_ratio_fn=fn)
        cps = _cost_point_list(fn)
        lines = [f"## Sign map at anchor-matched budgets — {title}", "",
                 "cost point = " + ("(a/b, c)" if fn is t_ratio_pow else "(a/b, beta)") +
                 "; cell = sign and mean paired gain at that conc", "",
                 "| cost point | " + " | ".join(str(c) for c in CONCURRENCY_ROWS) + " | matches anchor sign pattern |",
                 "|---|" + "---:|" * 6 + ":---:|"]
        matching = []
        for cp in cps:
            signs = smap[cp]
            cells = " | ".join(_fmt_gain(signs[c]) for c in CONCURRENCY_ROWS)
            hit = _anchor_pattern(signs)
            if hit:
                matching.append(cp)
            lines.append(f"| {cp} | {cells} | {'YES' if hit else 'no'} |")
        lines.append("")
        if matching:
            lines.append(f"Cost points reproducing the anchor sign pattern (16 negative, 32-96 positive): {matching}.")
        else:
            lines.append(
                "FINDING (N8): no cost point in the swept grid reproduces the anchor sign pattern; "
                "recorded as a finding, not a failure — the near-miss structure is visible in the table."
            )
        sections.append("\n".join(lines))

    # 3. flip curves (t_pow, beta = 0): a/b where the gain crosses zero per conc
    lines = ["## Flip curves — a/b where the paired gain crosses zero (T = a + b*v^c, beta = 0)", "",
             "Conc where dynamic loses: a/b above the flip value at that conc and c.", "",
             "| c | " + " | ".join(f"conc {c}" for c in CONCURRENCY_ROWS) + " |",
             "|---:|" + "---:|" * 6]
    for c in COST_C:
        curve = _flip_curve(_MAIN_DATA, c)
        lines.append(f"| {c} | " + " | ".join(f"{curve[conc]:.0f}" for conc in CONCURRENCY_ROWS) + " |")
    sections.append("\n".join(lines))

    # 4. reference gains at the main sweep (frozen generator)
    lines = ["## Reference paired gains at anchor-matched budgets (frozen generator)", "",
             "| (a/b, c) | " + " | ".join(str(c) for c in CONCURRENCY_ROWS) + " |",
             "|---|" + "---:|" * 6]
    smap = sign_map(_MAIN_DATA, matched_only=True, t_ratio_fn=t_ratio_pow)
    for cp in _ref_cost_points():
        lines.append(f"| {cp} | " + " | ".join(_fmt_gain(smap[cp][c]) for c in CONCURRENCY_ROWS) + " |")
    sections.append("\n".join(lines))

    # 5. PROGRESS_DECAY axis
    lines = ["## PROGRESS_DECAY axis (mutant-red range, mean-matched)", "",
             "| xi | matched levels (uniform shift) | fixed-5 mean | mutant mean | mutant red | conc-16 gain at (2676.7, 1.1) | first non-empty anchor band |",
             "|---:|---:|---:|---:|:---:|---:|---:|"]
    xi_summary = {}
    for xi in XI_GRID:
        if np.isclose(xi, 1e-06):
            levels = tuple(TYPE_LEVELS)
            achieved = MEAN_MATCH_TARGET
            delta = 0.0
        else:
            levels, achieved, delta, clipped = match_levels_shift(tuple(TYPE_LEVELS), xi)
            if clipped:
                print(f"<FAILED> mean-matching at xi={xi} clips the level vector: {levels}")
                return 1
        check = mutant_red_check(levels, xi)
        if not check["mut_red"]:
            print(f"<FAILED> mutant not red at xi={xi}: {check}")
            return 1
        variant = run_anchor_matched_variant(levels=levels, progress_decay=xi)
        vmap = sign_map(_matched_dict(variant), matched_only=True, t_ratio_fn=t_ratio_pow)
        c16 = _fmt_gain(vmap[(2676.7, 1.1)][16])
        band = _band_summary(_matched_dict(variant))
        xi_summary[xi] = band
        lines.append(
            f"| {xi:g} | shift {delta:+.4f} -> {tuple(round(float(x), 4) for x in levels)} | {achieved:.4f} | {check['mut_mean']:.4f} "
            f"| {'RED' if check['mut_red'] else 'NOT RED'} | {c16} | {band} |"
        )
    sections.append("\n".join(lines))

    # 6. heterogeneity axis
    lines = ["## Heterogeneity axis (level spread s, mean-matched)", "",
             "| s | scaled levels (pre-match) | matched levels (uniform shift) | fixed-5 mean | conc-16 gain at (2676.7, 1.1) | first non-empty anchor band |",
             "|---:|---|---|---:|---:|---:|"]
    mu = sum(TYPE_LEVELS) / 3.0
    for s in HETERO_GRID:
        if np.isclose(s, 1.0):
            scaled = tuple(TYPE_LEVELS)
            levels = tuple(TYPE_LEVELS)
            achieved = MEAN_MATCH_TARGET
            delta = 0.0
        else:
            scaled = tuple(mu + s * (x - mu) for x in TYPE_LEVELS)
            levels, achieved, delta, clipped = match_levels_shift(scaled, 1e-06)
            if clipped:
                print(f"<FAILED> mean-matching at s={s} clips the level vector: {levels}")
                return 1
        variant = run_anchor_matched_variant(levels=levels)
        vmap = sign_map(_matched_dict(variant), matched_only=True, t_ratio_fn=t_ratio_pow)
        c16 = _fmt_gain(vmap[(2676.7, 1.1)][16])
        band = _band_summary(_matched_dict(variant))
        pre_clip = " (pre-match vector clipped into (eps, 1-eps): recorded per ruling 5)" if not np.isclose(s, 1.0) and any(
            x <= EPS or x >= 1.0 - EPS for x in scaled
        ) else ""
        lines.append(
            f"| {s:g} | {tuple(round(float(x), 4) for x in scaled)}{pre_clip} | shift {delta:+.4f} -> {tuple(round(float(x), 4) for x in levels)} "
            f"| {achieved:.4f} | {c16} | {band} |"
        )
    sections.append("\n".join(lines))

    # 7. evidence boundary
    sections.append("\n".join([
        "## What this does NOT prove",
        "",
        "- IDEAL-INFORMATION UPPER BOUND: the allocator reads the current step's true generator rates",
        "  (idealized ConfidenceHead) and the batch is steady (identity remap, no arrivals or",
        "  terminations). Every dynamic gain above is an upper bound under ideal confidence; ZOS",
        "  one-step staleness and ranking errors can only reduce it. A negative gain at conc 16 that",
        "  still reproduces under this bound is a STRONGER conclusion (user ruling 4).",
        "- The cost families T(v) and the throughput identity num_seqs * E[emitted] / T(v) are",
        "  inference assumptions (cost_model.py docstring), not code facts; the sweep maps which",
        "  parameter regions reproduce the anchor's signs, it does not fit production hardware.",
        "- The anchor table is a set of published claims (N8). Reproducing a sign pattern is not",
        "  verification of production truth; failing to reproduce it is a finding, not a bug.",
        "- The horizon (6000 steps), the stratified type composition, and the progress term are",
        "  declared modeling choices (sim/convention.md); PROGRESS_DECAY and the heterogeneity",
        "  spread are design-driven and carry no empirical meaning (user note, 2026-10-22).",
        "- No fresh-sequence dynamics, no request termination, no zero-rate pathways (including the",
        "  post-EOS masking pathway), and no per-step budget adaptation are modeled in this lane.",
    ]))

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(_build_markdown(sections), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
