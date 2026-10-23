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

import numpy as np

from sim import rates
from sim.cost_model import ANCHOR
from sim.sweep_l2 import (
    ANCHOR_MATCHED_QBAR,
    N_SWEEP_SEEDS,
    _cost_point_list,
    _per_seed_deltas,
    _run_pair_stats,
    bootstrap_ci,
    t_ratio_pow,
)

FROZEN_TARGET = ANCHOR[16][4] / ANCHOR[16][3] - 1.0


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


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] != "select":
        print("usage: .venv/bin/python -m sim.replay_qa select")
        return 2
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


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
