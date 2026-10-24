"""Lane M2-L2: gain-versus-ranking sweep (HANDOFF_PROMPT_M2.md §5.2).

Declarations mirrored from ledger entry 11 (recorded before this sweep ran):

- Cost slices: primary (465.0, 1.1); sensitivity {(476.45, 1.1), (1550.0,
  1.3)}. Gains per slice are analytic transforms of one shared simulation
  set (the cost model enters only through the step-time ratio).
- sigma grid: the full 10-point grid {0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2,
  6.4, 12.8} for BOTH generators (V1 frozen levels = continuity anchor;
  V2E = adopted primary, sim/rank_noise.py).
- Threshold semantics (Ruling 5): rows whose sigma=0 gain is <= 0 report
  tau* and tau-half as "n/a (negative at sigma=0)"; no interpolation, no
  blanks. Otherwise tau* = the first grid sigma whose mean paired gain
  crosses zero, tau-half = the first whose mean gain drops below half its
  sigma=0 value; each reported as (sigma, tau_K at that sigma).
- Non-monotonicity: EVERY adjacent grid increase in mean gain is recorded
  with magnitude and the paired-difference bootstrap CI; increases whose CI
  is wholly positive are non-monotonicity findings (N8).
- Seeds: L2_SWEEP_SEED + i shared by both arms; the noise realization is
  shared across seeds at a given (conc, sigma) — seed-level CIs capture
  acceptance-stream variance only (declared limitation, carried to the M2-L3
  report).
- Loud continuity check: the V1 sigma=0 gains at the primary slice must
  reproduce the committed (465.0, 1.1) sign-map row (the sigma=0 pipeline is
  the M1 pipeline).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sim import rank_noise as rn
from sim.rates import CALIB_STEPS
from sim.sweep_l2 import (
    ANCHOR_MATCHED_QBAR,
    CONCURRENCY_ROWS,
    bootstrap_ci,
    t_ratio_pow,
)

SIGMA_GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)
COST_SLICES = {
    "primary_465_11": (465.0, 1.1),
    "band_edge_476_11": (476.45, 1.1),
    "second_1550_13": (1550.0, 1.3),
}
GENERATORS = {"V1": None, "V2E": rn.V2E_LEVELS}
N_SEEDS = 12
SEED_BASE = 20261101
COMMITTED_ROW_465_11 = {16: -0.05, 32: 0.16, 48: 3.63, 60: 4.99, 72: 6.09, 96: 9.17}
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "sweep_m2.md"


def _seeds() -> list[int]:
    return [SEED_BASE + i for i in range(N_SEEDS)]


def run_sweep() -> dict:
    """2 generators x 6 concs x 10 sigmas x 12 seeds paired sims (1440 runs)."""
    data: dict = {}
    for gen_name, levels in GENERATORS.items():
        data[gen_name] = {}
        for conc in CONCURRENCY_ROWS:
            total_q = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
            data[gen_name][conc] = {}
            for sigma in SIGMA_GRID:
                points = []
                for seed in _seeds():
                    res = rn.simulate_pair_noisy(
                        conc, CALIB_STEPS, seed, total_q=total_q, sigma=sigma, levels=levels
                    )
                    points.append(
                        {
                            "ef": float((res["fixed"]["k"] + 1).mean()),
                            "ed": float((res["dynamic"]["k"] + 1).mean()),
                            "acc": float(res["dynamic"]["k"].mean()),
                            "ver": float(res["dynamic"]["wins"].mean()),
                            "tau": res["tau_k"],
                        }
                    )
                data[gen_name][conc][sigma] = points
            print(f"  {gen_name} conc={conc} done", flush=True)
    return data


def _slice_gain(points: list[dict], conc: int, ab: float, c: float) -> tuple[float, float, float]:
    v_fixed = conc * 6
    v_dynamic = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
    t_ratio = t_ratio_pow(v_fixed, v_dynamic, ab, c)
    ratios = np.array([p["ed"] / p["ef"] for p in points])
    deltas = np.log(ratios) + np.log(t_ratio)
    mean, lo, hi = bootstrap_ci(deltas)
    return float(np.mean(np.exp(deltas) - 1.0)), lo, hi


def _thresholds(gains: dict[float, float], taus: dict[float, float]) -> dict[str, str]:
    """Ruling-5 semantics: n/a rows stay n/a; crossings reported as (sigma, tau_K)."""
    g0 = gains[0.0]
    if g0 <= 0.0:
        return {
            "tau_star": "n/a (negative at sigma=0)",
            "tau_half": "n/a (negative at sigma=0)",
        }
    out = {}
    for name, bound in (("tau_star", 0.0), ("tau_half", g0 / 2.0)):
        hit = next((s for s in sorted(gains) if gains[s] <= bound), None)
        out[name] = (
            f"sigma={hit:g}, tau_K={taus[hit]:.3f}" if hit is not None else f"no crossing in grid (gain stays > {bound:+.3f})"
        )
    return out


def _paired_diff_ci(points_a: list[dict], points_b: list[dict], conc: int, ab: float, c: float) -> tuple[float, float, float]:
    v_fixed = conc * 6
    v_dynamic = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
    t_ratio = t_ratio_pow(v_fixed, v_dynamic, ab, c)
    ra = np.array([p["ed"] / p["ef"] for p in points_a])
    rb = np.array([p["ed"] / p["ef"] for p in points_b])
    deltas = np.log(rb) - np.log(ra)
    return bootstrap_ci(deltas)


def _build_report(data: dict) -> str:
    lines = []
    lines.append("# Lane M2-L2 sweep — gain versus ranking quality (2026-10-24)")
    lines.append("")
    lines.append("- Generators: V1 (frozen L1 levels, continuity anchor) and V2E (adopted primary, entry 10).")
    lines.append(f"- sigma grid: {SIGMA_GRID} (extended per the entry-10 plateau rule).")
    lines.append(f"- Cost slices: {COST_SLICES} — one shared simulation set, gains are analytic transforms.")
    lines.append("- Seeds: L2_SWEEP_SEED + i shared by both arms; the noise realization is shared across seeds")
    lines.append("  at a given (conc, sigma) — seed-level CIs capture acceptance-stream variance only.")
    lines.append("- Signs classified by the bootstrap CI ('-' wholly negative, '+' wholly positive, '.' not significant).")
    lines.append("- Anchor-table values are published claims (N8).")
    lines.append("")

    # sigma = 0 continuity check (loud)
    cont = {}
    for conc in CONCURRENCY_ROWS:
        g, _, _ = _slice_gain(data["V1"][conc][0.0], conc, *COST_SLICES["primary_465_11"])
        cont[conc] = round(g * 100, 2)
    lines.append("## Continuity check (V1 at sigma = 0, primary slice)")
    lines.append("")
    lines.append(f"Reproduces the committed (465.0, 1.1) sign-map row: {cont};")
    lines.append(f"committed: {COMMITTED_ROW_465_11}. Flip: any cell mismatch.")
    lines.append("")

    # primary slice: full per-sigma table
    lines.append("## Primary slice (465.0, 1.1) — mean paired gain [CI], accepted/verified, tau_K")
    lines.append("")
    for gen in GENERATORS:
        lines.append(f"### {gen}")
        lines.append("")
        lines.append("| conc | sigma | gain [CI] | sign | acc/ver | tau_K |")
        lines.append("|---:|---:|---|:---:|---:|---:|")
        for conc in CONCURRENCY_ROWS:
            for sigma in SIGMA_GRID:
                pts = data[gen][conc][sigma]
                g, lo, hi = _slice_gain(pts, conc, *COST_SLICES["primary_465_11"])
                sign = "-" if hi < 0 else ("+" if lo > 0 else ".")
                acc = float(np.mean([p["acc"] for p in pts]))
                ver = float(np.mean([p["ver"] for p in pts]))
                tau = float(np.mean([p["tau"] for p in pts]))
                lines.append(f"| {conc} | {sigma:g} | {g * 100:+.2f}% [{lo * 100:+.2f}, {hi * 100:+.2f}] | {sign} | {acc / ver:.4f} | {tau:.3f} |")
        lines.append("")

    # sensitivity slices: compact gain tables
    for slice_name, (ab, c) in COST_SLICES.items():
        if slice_name == "primary_465_11":
            continue
        lines.append(f"## Sensitivity slice ({ab}, {c}) — mean paired gain [CI]")
        lines.append("")
        for gen in GENERATORS:
            lines.append(f"### {gen}")
            lines.append("")
            lines.append("| conc | " + " | ".join(f"{s:g}" for s in SIGMA_GRID) + " |")
            lines.append("|---:|" + "---:|" * len(SIGMA_GRID))
            for conc in CONCURRENCY_ROWS:
                cells = []
                for sigma in SIGMA_GRID:
                    g, lo, hi = _slice_gain(data[gen][conc][sigma], conc, ab, c)
                    cells.append(f"{g * 100:+.2f}")
                lines.append(f"| {conc} | " + " | ".join(cells) + " |")
            lines.append("")

    # thresholds
    lines.append("## Thresholds (ruling-5 semantics; each stated as (sigma, tau_K at that sigma))")
    lines.append("")
    lines.append("- tau*: the first grid sigma whose mean paired gain crosses zero. Criterion: the point where")
    lines.append("  ranking noise consumes the allocator's entire paired advantage. Flip: the CI at that sigma")
    lines.append("  crossing zero from above.")
    lines.append("- tau-half: the first grid sigma whose mean gain drops below half its sigma=0 value. Criterion:")
    lines.append("  half-retention of the ideal-band gain. Flip: the mean gain recovering above half.")
    lines.append("- Rows whose sigma=0 gain is <= 0 are n/a (negative at sigma=0) — never interpolated.")
    lines.append("")
    for slice_name, (ab, c) in COST_SLICES.items():
        lines.append(f"### Slice ({ab}, {c})")
        lines.append("")
        lines.append("| generator | conc | tau* | tau-half |")
        lines.append("|---|---:|---|---|")
        for gen in GENERATORS:
            for conc in CONCURRENCY_ROWS:
                gains = {s: _slice_gain(data[gen][conc][s], conc, ab, c)[0] for s in SIGMA_GRID}
                taus = {s: float(np.mean([p["tau"] for p in data[gen][conc][s]])) for s in SIGMA_GRID}
                th = _thresholds(gains, taus)
                lines.append(f"| {gen} | {conc} | {th['tau_star']} | {th['tau_half']} |")
        lines.append("")

    # non-monotonicity findings
    lines.append("## Non-monotonicity record (every adjacent increase; CI-wholly-positive = FINDING)")
    lines.append("")
    findings = 0
    for gen in GENERATORS:
        for slice_name, (ab, c) in COST_SLICES.items():
            for conc in CONCURRENCY_ROWS:
                gains = {s: _slice_gain(data[gen][conc][s], conc, ab, c)[0] for s in SIGMA_GRID}
                for s_a, s_b in zip(SIGMA_GRID, SIGMA_GRID[1:]):
                    if gains[s_b] > gains[s_a]:
                        _, dlo, dhi = _paired_diff_ci(data[gen][conc][s_a], data[gen][conc][s_b], conc, ab, c)
                        flag = "FINDING (N8)" if dlo > 0 else "within CI"
                        if dlo > 0:
                            findings += 1
                        lines.append(
                            f"- {gen} slice ({ab}, {c}) conc {conc}: sigma {s_a:g} -> {s_b:g}: "
                            f"{gains[s_a] * 100:+.3f}% -> {gains[s_b] * 100:+.3f}% (diff CI [{dlo * 100:+.3f}, {dhi * 100:+.3f}]) — {flag}"
                        )
    if findings == 0:
        lines.append("- No adjacent increase had a CI-wholly-positive paired difference.")
    lines.append("")

    # plateau per conc + V1-vs-V2 ratio diagnostic
    lines.append("## tau_K plateau per conc (sigma = 12.8) and the V1-vs-V2 ratio diagnostic")
    lines.append("")
    lines.append("| generator | conc | tau_K at sigma=12.8 |")
    lines.append("|---|---:|---:|")
    for gen in GENERATORS:
        for conc in CONCURRENCY_ROWS:
            tau = float(np.mean([p["tau"] for p in data[gen][conc][12.8]]))
            lines.append(f"| {gen} | {conc} | {tau:.3f} |")
    lines.append("")
    lines.append("| conc | sigma | acc/ver V1 | acc/ver V2E | V2E - V1 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for conc in CONCURRENCY_ROWS:
        for sigma in SIGMA_GRID:
            r1 = float(np.mean([p["acc"] / p["ver"] for p in data["V1"][conc][sigma]]))
            r2 = float(np.mean([p["acc"] / p["ver"] for p in data["V2E"][conc][sigma]]))
            lines.append(f"| {conc} | {sigma:g} | {r1:.4f} | {r2:.4f} | {r2 - r1:+.4f} |")
    lines.append("")
    lines.append("## What this does NOT prove")
    lines.append("")
    lines.append("- IDEAL-INFORMATION UPPER BOUND carries over (L2 triage ruling 4): ideal confidence, steady")
    lines.append("  batch, no arrivals/terminations.")
    lines.append("- The noise realization is shared across seeds at a given (conc, sigma): seed-level CIs capture")
    lines.append("  acceptance-stream variance only, not noise-realization variance; thresholds condition on one")
    lines.append("  realized noise pattern per batch geometry.")
    lines.append("- Thresholds are stated within the swept grid and the t_pow cost family; the sigma -> tau_K map")
    lines.append("  is batch-geometry-dependent, so the auditable bound must be read per concurrency.")
    lines.append("- The anchor table remains published claims (N8); M2-L3 attributes the M1 tension residual")
    lines.append("  with V1 and V2E side by side.")
    return "\n".join(lines) + "\n"


def _main() -> int:
    data = run_sweep()

    cont = {}
    for conc in CONCURRENCY_ROWS:
        g, _, _ = _slice_gain(data["V1"][conc][0.0], conc, *COST_SLICES["primary_465_11"])
        cont[conc] = round(g * 100, 2)
    if cont != COMMITTED_ROW_465_11:
        print(f"<FAILED> V1 sigma=0 continuity broken: {cont} != {COMMITTED_ROW_465_11}")
        return 1
    for gen in GENERATORS:
        for conc in CONCURRENCY_ROWS:
            for sigma in SIGMA_GRID:
                taus = [p["tau"] for p in data[gen][conc][sigma]]
                if not all(-1.0 <= t <= 1.0 and np.isfinite(t) for t in taus):
                    print(f"<FAILED> tau_K out of range at {gen} conc {conc} sigma {sigma}: {taus}")
                    return 1

    RESULTS_PATH.write_text(_build_report(data), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
