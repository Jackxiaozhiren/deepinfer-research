"""Lane L3: anchor-table replay under the freeze-then-score protocol.

Freeze rule (declared 2026-10-23, user rulings on the L3 triage):

- Family: T(v) = a + b * v**c only. The saturation family is excluded by the
  entry-5 FINDING (no swept saturation point reproduces the anchor sign
  pattern).
- Selection set: {conc 16}, the negative anchor. Scoring set: {32, 48, 60,
  72, 96}.
- Criterion: among the swept t_pow cost points (COST_AB_EXTENDED x COST_C,
  the same grid as the committed results/sweep_l2.md sign map — 15 a/b values
  x 5 exponents = 75 rows), keep those whose conc-16 paired gain is negative
  with the seed-level bootstrap CI wholly below zero (the anchor's conc-16
  sign). Among the survivors select
      argmin | mean_gain_16 - target |,
  where target is the selection row's anchor gain computed from the committed
  constants as 4816/4891 - 1 = -0.0153363... (displayed as -1.5% in the Q&A
  table) — a zero-width magnitude window; no window width is invented.
- NON-BLIND PRE-REGISTRATION: the committed L2 sign map was in view at freeze
  time; the rule is declared in executable form requiring only selection-set
  information (the calibration criteria and the conc-16 row).

Structural argument (pre-declared in the freeze entry; numbers land in the
outcome entry and results/m1_report.md sections 3-4):

  (a) c = 0.9 and 1.0: if flip_32(c) < flip_16(c) (machine-precision values
      from fresh paired sims), the anchor pattern (conc-16 negative AND
      conc-32 positive) is structurally impossible at that exponent — no a/b
      satisfies both.
  (b) c = 1.1, 1.3, 1.5: at a/b = flip_32(c) exactly (the conc-32 zero-gain
      point, machine precision — not the rounded table integers), compute the
      conc-16 gain — three new cells giving the in-band conc-16 magnitude
      upper bound. Interpolation expectation: about -0.06 / -0.5 / -0.55%,
      all far above (smaller in magnitude than) the anchor's -1.5%. If the
      computed values contradict the expectation, that is a finding, not a
      bug.

`select` prints the rule, the survivor set, and the slice. `score` (added
after the freeze entry lands) re-derives the slice, asserts it equals the
ledger-frozen slice, scores all six rows, and writes results/m1_report.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sim import rates
from sim.cost_model import ANCHOR
from sim.sweep_l2 import (
    ANCHOR_MATCHED_QBAR,
    N_SWEEP_SEEDS,
    _classify,
    _cost_point_list,
    _per_seed_deltas,
    _run_pair_stats,
    bootstrap_ci,
    t_ratio_pow,
)

FROZEN_TARGET = ANCHOR[16][4] / ANCHOR[16][3] - 1.0
REPO_ROOT = Path(__file__).resolve().parents[1]


def count_sign_map_rows(text: str) -> dict[str, int]:
    """Count data rows in the committed L2 sign-map tables.

    The single source for every sign-map count quoted in results/m1_report.md
    (ruling: report counts must be derivable from the committed artifact).
    """
    counts: dict[str, int] = {}
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            if "beta = 0)" in line:
                section = "t_pow"
            elif "saturation)" in line:
                section = "saturation"
            else:
                section = None
            continue
        if section is not None and line.startswith("| ("):
            counts[section] = counts.get(section, 0) + 1
    return counts


def select_slice(target: float | None = None, *, maximize: bool = False) -> dict:
    """Execute the declared freeze rule on fresh conc-16 paired sims.

    target defaults to FROZEN_TARGET; the knobs exist so the N5 mutants can
    run a deliberately wrong rule and be shown to pick a different slice.
    """
    target = FROZEN_TARGET if target is None else target
    calib = rates.fixed5_report()
    ok, detail = rates.band_check(calib)
    if not ok:
        print(f"<FAILED> calibration left the band before selection: {detail}")
        sys.exit(1)

    conc = 16
    total_q = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
    rows = [_run_pair_stats(conc, seed, total_q) for seed in _l3_seeds()]
    v_fixed = conc * (5 + 1)
    v_dynamic = total_q

    survivors = []
    for cost_point in _cost_point_list(t_ratio_pow):
        ab, c = cost_point
        t_ratio = t_ratio_pow(v_fixed, v_dynamic, ab, c)
        deltas = _per_seed_deltas(rows, v_fixed, v_dynamic, t_ratio)
        mean, lo, hi = bootstrap_ci(deltas)
        if hi < 0.0:
            survivors.append({"ab": ab, "c": c, "mean_gain16": float(np.mean(np.exp(deltas) - 1.0)), "mean_log": mean})
    if not survivors:
        print("<FAILED> no cost point has a conc-16 CI-wholly-negative gain")
        sys.exit(1)

    key = (lambda s: abs(s["mean_gain16"] - target)) if not maximize else (lambda s: -abs(s["mean_gain16"] - target))
    chosen = min(survivors, key=key)
    return {
        "slice": (chosen["ab"], chosen["c"]),
        "mean_gain16": chosen["mean_gain16"],
        "target": target,
        "survivors": survivors,
        "calib": calib,
        "calib_detail": detail,
    }


def _l3_seeds() -> list[int]:
    from sim.sweep_l2 import L2_SWEEP_SEED

    return [L2_SWEEP_SEED + i for i in range(N_SWEEP_SEEDS)]


SCORE_CONCS = (16, 32, 48, 60, 72, 96)
ANCHOR_SIGNS = {16: "-", 32: "+", 48: "+", 60: "+", 72: "+", 96: "+"}
SCORE_C_LOW = (0.9, 1.0)
SCORE_C_HIGH = (1.1, 1.3, 1.5)
INTERPOLATION_EXPECTATION = {1.1: -0.06, 1.3: -0.5, 1.5: -0.55}


def _flip_ab(e_ratio: float, v_fixed: float, v_dynamic: float, c: float) -> float:
    """a/b where the paired gain crosses zero: ab* = (v_f**c - v_d**c/E)/(1/E - 1)."""
    denom = 1.0 / e_ratio - 1.0
    num = v_fixed**c - v_dynamic**c / e_ratio
    return num / denom if denom > 0 and num > 0 else float("nan")


def _row_summary(rows: list[dict], conc: int, ab: float, c: float) -> dict:
    v_fixed = conc * 6
    v_dynamic = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
    t_ratio = t_ratio_pow(v_fixed, v_dynamic, ab, c)
    deltas = _per_seed_deltas(rows, v_fixed, v_dynamic, t_ratio)
    mean_log, lo_log, hi_log = bootstrap_ci(deltas)
    gains = np.exp(deltas) - 1.0
    return {
        "mean": float(gains.mean()),
        "lo": float(np.exp(lo_log) - 1.0),
        "hi": float(np.exp(hi_log) - 1.0),
        "sign": _classify(mean_log, lo_log, hi_log),
        "verified_dyn": float(np.mean([r["dynamic"]["verified"] for r in rows])),
        "accepted_dyn": float(np.mean([r["dynamic"]["accepted"] for r in rows])),
        "accepted_fix": float(np.mean([r["fixed"]["accepted"] for r in rows])),
        "e_ratio": float(np.mean([r["dynamic"]["emitted"] / r["fixed"]["emitted"] for r in rows])),
    }


def check_slice(derived: tuple, provided: tuple) -> None:
    """Loud guard: the ledger-frozen slice must equal the rule's re-derivation."""
    if provided != derived:
        print(f"<FAILED> provided frozen slice {provided} != rule re-derivation {derived}")
        sys.exit(1)


def run_score(frozen_slice: tuple[float, float]) -> dict:
    """Score all six rows under the frozen slice and assemble the report data."""
    ab, c = frozen_slice
    calib = rates.fixed5_report()
    ok, detail = rates.band_check(calib)
    if not ok:
        print(f"<FAILED> calibration left the band before scoring: {detail}")
        sys.exit(1)
    rows_by_conc = {}
    summaries = {}
    for conc in SCORE_CONCS:
        total_q = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
        rows_by_conc[conc] = [_run_pair_stats(conc, seed, total_q) for seed in _l3_seeds()]
        summaries[conc] = _row_summary(rows_by_conc[conc], conc, ab, c)
        print(f"  scored conc={conc}: {summaries[conc]['mean'] * 100:+.2f}% [{summaries[conc]['lo'] * 100:+.2f}, {summaries[conc]['hi'] * 100:+.2f}]", flush=True)

    e16 = summaries[16]["e_ratio"]
    e32 = summaries[32]["e_ratio"]
    v_f16, v_d16 = 16 * 6, int(round(16 * ANCHOR_MATCHED_QBAR[16]))
    v_f32, v_d32 = 32 * 6, int(round(32 * ANCHOR_MATCHED_QBAR[32]))
    structural_a = {}
    for cc in SCORE_C_LOW:
        f16 = _flip_ab(e16, v_f16, v_d16, cc)
        f32 = _flip_ab(e32, v_f32, v_d32, cc)
        structural_a[cc] = {"flip16": f16, "flip32": f32, "impossible": bool(f32 < f16)}
    structural_b = {}
    for cc in SCORE_C_HIGH:
        f32 = _flip_ab(e32, v_f32, v_d32, cc)
        t16_at_f32 = t_ratio_pow(v_f16, v_d16, f32, cc)
        gain16 = e16 * t16_at_f32 - 1.0
        structural_b[cc] = {"flip32": f32, "gain16_at_flip32": gain16, "expectation": INTERPOLATION_EXPECTATION[cc]}
    return {
        "slice": frozen_slice,
        "calib": calib,
        "calib_detail": detail,
        "summaries": summaries,
        "structural_a": structural_a,
        "structural_b": structural_b,
    }


def _build_report(data: dict) -> str:
    ab, c = data["slice"]
    s = data["summaries"]
    sa = data["structural_a"]
    sb = data["structural_b"]
    counts = count_sign_map_rows((REPO_ROOT / "results" / "sweep_l2.md").read_text(encoding="utf-8"))
    lines = []
    lines.append("# Milestone 1 report — anchor-table replay (Lane L3, 2026-10-23)")
    lines.append("")
    lines.append(f"- Frozen slice (ledger entry 7): (a/b = {ab!r}, c = {c}); commit at freeze `c842e85`.")
    lines.append("- Seeds: L2_SWEEP_SEED + i, i in [0, 12), shared by both arms; bootstrap seed 20261101, 10000 resamples; horizon 6000 steps.")
    lines.append("- Anchor-table values are published claims (N8), not ground truth.")
    lines.append("")
    lines.append("## 1. Convention derivation summary")
    lines.append("")
    lines.append("The engine emits **k+1** tokens per verification step (accepted prefix plus the final")
    lines.append("fresh row): decisive sites are the emitted-stop-token docstring \"(accepted prefix plus")
    lines.append("final row)\" (tilelang_ops/sampler.py:466-467), the finalize-path comment +")
    lines.append("`emitted_rows = (effective_accept | is_final) & stop_hits` (tilelang_ops/sampler.py:376-379),")
    lines.append("and the KV advance `seqlens_k += seqlens_q - (nextn - accepted)` (compute_seqlens_k_zos.py:22-23)")
    lines.append("which returns k+1 under full-depth verification; fresh sequences pack exactly 1 row")
    lines.append("(schedule_mtp_varlen.py:98). `EMIT_BONUS = 1` in sim/rates.py encodes this. The Q&A")
    lines.append("\"accepted length\" statistic is frozen as the mean accepted draft count k per step")
    lines.append("(excluding the bonus; num_generated_tokens / num_accepted_tokens_gpu semantics); the")
    lines.append("k+1 reading is the declared first re-freeze candidate (sim/convention.md).")
    lines.append("")
    lines.append("## 2. Calibration table")
    lines.append("")
    lines.append("Fixed-5 accepted length k per concurrency row (frozen generator constants, seed 20261022,")
    lines.append("horizon 6000):")
    lines.append("")
    lines.append("| conc | fixed-5 accepted k |")
    lines.append("|---:|---:|")
    for conc, val in sorted(data["calib"].items()):
        lines.append(f"| {conc} | {val:.4f} |")
    lines.append("")
    calib_mean = sum(data["calib"].values()) / len(data["calib"])
    calib_ok, _ = rates.band_check(data["calib"])
    lines.append(
        f"Criterion: mean across rows in [3.29, 3.32] and every row in [3.25, 3.32] — "
        f"{'SATISFIED' if calib_ok else 'VIOLATED'}; mean = {calib_mean:.4f}. "
        f"Flip: any row or the mean leaving the band."
    )
    lines.append("")
    lines.append("## 3. Replay table (frozen slice)")
    lines.append("")
    lines.append("| conc | anchor gain | anchor sign | model gain (mean [CI]) | model sign | sign match | model verified_dyn | model accepted dyn/fix |")
    lines.append("|---:|---:|:---:|---:|:---:|:---:|---:|---|")
    for conc in SCORE_CONCS:
        r = s[conc]
        match = "YES" if r["sign"] == ANCHOR_SIGNS[conc] else "NO"
        anchor_sign_cell = ANCHOR_SIGNS[conc]
        anchor_gain_text = f"{(ANCHOR[conc][4] / ANCHOR[conc][3] - 1) * 100:+.1f}%"
        lines.append(
            f"| {conc} | {anchor_gain_text} | {anchor_sign_cell} | {r['mean'] * 100:+.2f}% [{r['lo'] * 100:+.2f}, {r['hi'] * 100:+.2f}] "
            f"| {r['sign']} | {match} | {r['verified_dyn']:.3f} | {r['accepted_dyn']:.3f} / {r['accepted_fix']:.3f} |"
        )
    lines.append("")
    lines.append("Per-row criteria and flip conditions (N6):")
    lines.append("")
    lines.append("- conc 16 (selection row): criterion — the model's paired gain is negative with the CI wholly")
    lines.append("  below zero, matching the anchor's -1.5% sign; magnitude closeness was the declared argmin")
    lines.append("  constraint at freeze time. Flip: CI crossing zero or a positive mean.")
    for conc in (32, 48, 60, 72, 96):
        lines.append(
            f"- conc {conc} (scoring row): criterion — the model's paired-gain sign matches the anchor's "
            f"{ANCHOR_SIGNS[conc]} sign; flip — the CI classifies the opposite sign or is not significant."
        )
    lines.append("")
    lines.append("Accepted/verified tension, exact construction (ruling: never as bare 0.97-vs-0.82): the anchor's")
    lines.append(f"conc-96 dynamic accepted/verified = 3.15/3.24 = {3.15 / 3.24:.4f}, while the generator at the matched")
    lines.append(f"budget yields {s[96]['accepted_dyn']:.3f}/{s[96]['verified_dyn']:.3f} = {s[96]['accepted_dyn'] / s[96]['verified_dyn']:.4f}.")
    lines.append("The anchor implies near-perfect ranking over sharply separated rates; this generator's")
    lines.append("realized dynamic acceptance is the binding discrepancy behind every scoring-row mismatch above.")
    lines.append("")
    lines.append("## 4. Sensitivity summary — where the family reproduces the anchor's sign flip")
    lines.append("")
    lines.append("Structural argument (a) — exponents c = 0.9 and 1.0, machine-precision zero-gain curves from")
    lines.append("the fresh paired sims:")
    lines.append("")
    lines.append("| c | flip_16 (a/b) | flip_32 (a/b) | flip_32 < flip_16 | anchor pattern possible? |")
    lines.append("|---:|---:|---:|:---:|:---:|")
    for cc in SCORE_C_LOW:
        row = sa[cc]
        lines.append(
            f"| {cc} | {row['flip16']:.1f} | {row['flip32']:.1f} | {'yes' if row['impossible'] else 'no'} "
            f"| {'no — structurally impossible' if row['impossible'] else 'not excluded by this argument'} |"
        )
    lines.append("")
    lines.append("Structural argument (b) — conc-16 gain evaluated AT a/b = flip_32(c) exactly (three new cells;")
    lines.append("the conc-32 boundary is the best case for the anchor pattern at that exponent):")
    lines.append("")
    lines.append("| c | flip_32 (a/b, exact) | conc-16 gain at that a/b | interpolation expectation | consistent? |")
    lines.append("|---:|---:|---:|---:|:---:|")
    for cc in SCORE_C_HIGH:
        row = sb[cc]
        consistent = (row["gain16_at_flip32"] < 0.0) and (abs(row["gain16_at_flip32"] * 100) < abs(row["expectation"]) * 2)
        lines.append(
            f"| {cc} | {row['flip32']:.2f} | {row['gain16_at_flip32'] * 100:+.3f}% | {row['expectation']:+.2f}% "
            f"| {'yes' if consistent else 'NO — finding (N8)'} |"
        )
    lines.append("")
    lines.append(f"Reading: at every exponent where the anchor-pattern band is non-empty, the conc-16 magnitude at the")
    lines.append(f"band's conc-32 edge stays far above (smaller than) the anchor's -1.5% — so the anchor's conc-16")
    lines.append(f"magnitude and its scoring-row signs are jointly irreproducible within the t_pow family under this")
    lines.append(f"generator, independent of the argmin slice. Where the pattern IS reproducible ((465.0, 1.1),")
    lines.append(f"(1550.0, 1.3), (4900.0, 1.5) in the committed map), conc-16 lands at -0.05% to -0.37%.")
    lines.append("")
    lines.append(f"Committed-artifact counts (ruling 1): the L2 sign map evaluates {counts['t_pow']} t_pow cost points")
    lines.append(f"(15 a/b values x 5 exponents) and {counts['saturation']} saturation points (10 x 3); the saturation")
    lines.append("family reproduces no anchor-pattern point (entry-5 FINDING) and is excluded from slice selection.")
    lines.append("")
    lines.append("## 5. What this does NOT prove")
    lines.append("")
    lines.append("- IDEAL-INFORMATION UPPER BOUND: the allocator reads current-step true rates (idealized")
    lines.append("  ConfidenceHead); the batch is steady (identity remap, no arrivals/terminations). All dynamic")
    lines.append("  gains above are upper bounds under ideal confidence; staleness and ranking errors can only")
    lines.append("  reduce them (user ruling 4 of the L2 triage).")
    lines.append("- The cost families and the throughput identity num_seqs * E[emitted] / T(v) are inference")
    lines.append("  assumptions (cost_model.py docstring), not code facts; this report maps which parameter")
    lines.append("  regions reproduce the anchor's signs — it does not fit production hardware.")
    lines.append("- The anchor table is a set of published claims (N8). A scoring-row mismatch is a finding about")
    lines.append("  this model family under these assumptions, not a bug to tune away; reproducing a sign is not")
    lines.append("  verification of production truth.")
    lines.append("- NON-BLIND: the committed L2 sign map was in view at freeze time (entry 7); the freeze rule is")
    lines.append("  executable from selection-set information alone, but the scoring-row outcomes were already")
    lines.append("  public in the repo when the slice was frozen.")
    lines.append("- Declared modeling choices: horizon 6000 steps; stratified type composition; the progress term")
    lines.append("  (design-driven, no empirical meaning); no fresh-sequence dynamics, no request termination, no")
    lines.append("  zero-rate pathways, no per-step budget adaptation (sim/convention.md).")
    return "\n".join(lines) + "\n"


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "select":
        out = select_slice()
        print("freeze rule: t_pow only; conc-16 bootstrap CI wholly < 0; argmin |mean_gain16 - target|,")
        print(f"  target = 4816/4891 - 1 = {out['target']:.6f} (the Q&A table's -1.5% row)")
        print(f"calibration band check: OK — {out['calib_detail']}")
        print(f"survivors (CI-wholly-negative at conc 16): {len(out['survivors'])} cost points")
        for s in sorted(out["survivors"], key=lambda s: abs(s["mean_gain16"] - out["target"]))[:6]:
            print(f"  (a/b={s['ab']}, c={s['c']}) mean_gain16={s['mean_gain16'] * 100:+.3f}% |delta|={abs(s['mean_gain16'] - out['target']) * 100:.3f}pp")
        print(
            f"FROZEN SLICE: (a/b={out['slice'][0]!r}, c={out['slice'][1]!r}) "
            f"mean_gain16={out['mean_gain16'] * 100:+.3f}%"
        )
        return 0
    if len(argv) == 4 and argv[1] == "score" and argv[2] == "--slice":
        provided = tuple(float(x) for x in argv[3].split(","))
        derived = select_slice()["slice"]
        check_slice(derived, provided)
        data = run_score(provided)
        report = _build_report(data)
        out_path = REPO_ROOT / "results" / "m1_report.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"wrote {out_path}")
        return 0
    print("usage: .venv/bin/python -m sim.replay_qa select | score --slice <ab,c>")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
