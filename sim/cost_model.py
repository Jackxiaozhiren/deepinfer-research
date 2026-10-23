"""Step-time cost models and the anchor-ratio exclusion analysis (Lane L2).

Families (v = per-step BATCH row count, v = num_seqs * rows_per_seq, rows
including each sequence's fresh row — sim/convention.md):

    T_lin(v) = a + b * v              (pure linear + per-step overhead)
    T_pow(v) = a + b * v**c           (MAIN family; c = 1 is excluded by the
                                       anchor ratios — user ruling 2026-10-22,
                                       mechanically re-proven below)
    T_sat(v) = a + b * v + d * v**2   (saturation second family)

INFERENCE ASSUMPTION (declared per user ruling 2026-10-22): throughput per
rank is modeled as ``num_seqs * E[emitted] / T(v)``. This treats a step as
producing ``num_seqs * E[emitted]`` tokens at cost ``T(v)`` and ignores
pipeline, memory-bandwidth, and scheduling coupling between v and step time
beyond T's parametric form. It is an inference, not a code fact.

Plausible regions (module contract, HANDOFF §7.2):

- a/b: per-step overhead vs per-row cost. The anchor's own conc-16 operating
  point implies a/b of about 9.1e2 (statistic=k reading) to 2.7e3 (statistic
  =k+1 reading) under batch totals; smaller a/b lets dynamic's row savings
  pay proportionally more, larger a/b approaches the overhead-bound regime
  where verification is nearly free and dynamic can only lose.
- c: convexity of per-row cost in batch size. c = 1 is excluded (see
  ``exclude_linear_family``); c > 1 (superlinear batch cost, e.g. cache/
  bandwidth pressure at large packed batches) reproduces the anchor's growing
  gain with concurrency; c < 1 (amortization) makes dynamic's savings shrink
  with concurrency and cannot produce the anchor's sign pattern.
- d: quadratic saturation onset, swept as d = beta * b / 500 so the quadratic
  term equals beta * (b * 500) at v = 500 rows; beta in {0, 0.5, 2}.

The anchor constants below are PUBLISHED CLAIMS, not ground truth (rule N8).
"""

from __future__ import annotations

from typing import Any

# Anchor table transcribed from doc/inference_deployment_qa.md:249-256
# (also HANDOFF_PROMPT.md §2). conc: (verified_dyn, accepted_fixed,
# accepted_dyn, tok_s_fixed, tok_s_dyn).
ANCHOR: dict[int, tuple[float, float, float, float, float]] = {
    16: (4.44, 3.32, 3.24, 4891, 4816),
    32: (3.65, 3.25, 3.17, 8964, 9111),
    48: (3.94, 3.3, 3.26, 12032, 12467),
    60: (3.58, 3.3, 3.21, 13730, 14466),
    72: (3.24, 3.29, 3.12, 14962, 15961),
    96: (3.24, 3.3, 3.15, 16817, 18660),
}

NEXTN = 5


def t_linear(v: float, a: float, b: float) -> float:
    return a + b * v


def t_pow(v: float, a: float, b: float, c: float) -> float:
    return a + b * v**c


def t_sat(v: float, a: float, b: float, d: float) -> float:
    return a + b * v + d * v**2


def throughput(num_seqs: int, emitted_mean: float, t_value: float) -> float:
    """Inference assumption: tokens per second per rank (see module docstring)."""
    return num_seqs * emitted_mean / t_value


def implied_ab(v_fixed: float, v_dynamic: float, t_ratio: float) -> float:
    """a/b implied by a required step-time ratio under T = a + b*v.

    Solves (a + b*v_fixed) / (a + b*v_dynamic) = t_ratio for a/b. Requires
    t_ratio > 1 and v_fixed > t_ratio * v_dynamic (a positive overhead).
    """
    assert t_ratio > 1.0, f"t_ratio {t_ratio} must exceed 1"
    num = v_fixed - t_ratio * v_dynamic
    assert num > 0, f"no positive a/b fits: v_fixed {v_fixed} <= t_ratio * v_dynamic {t_ratio * v_dynamic}"
    return num / (t_ratio - 1.0)


def predicted_ratio_pow(v_fixed: float, v_dynamic: float, ab: float, c: float) -> float:
    """T(v_fixed)/T(v_dynamic) under T = b*(ab + v**c) with ab = a/b."""
    return (ab + v_fixed**c) / (ab + v_dynamic**c)


def exclude_linear_family(c: float = 1.0) -> dict[str, Any]:
    """Mechanical re-proof of the linear-family exclusion (ledger entry 3 Next).

    Derivation: the anchor's gain g = tok_dyn/tok_fix = emit_ratio * T(v_f)/
    T(v_d), with v = conc * rows/seq, rows including the fresh row (fixed-5:
    6 rows/seq). Row conc-16 pins the unique a/b under the linear family for
    each reading of the accepted-length statistic; the conc-96 row's required
    T ratio then exceeds the linear prediction — no linear (a, b) fits both
    rows. The c parameter exists for the test's discriminating-power arm: at
    c = 1.3 (the main family) the verdict flips, showing the checker can fail.

    Returns {"ok": all readings excluded, "readings": {...}}.
    """
    gain16 = ANCHOR[16][4] / ANCHOR[16][3]
    gain96 = ANCHOR[96][4] / ANCHOR[96][3]
    v_f16, v_f96 = 16 * (NEXTN + 1), 96 * (NEXTN + 1)
    v_d16 = 16 * (ANCHOR[16][0] + 1)
    v_d96 = 96 * (ANCHOR[96][0] + 1)
    readings = {
        "statistic=k (frozen): emitted=k+1": ((ANCHOR[16][2] + 1) / (ANCHOR[16][1] + 1), (ANCHOR[96][2] + 1) / (ANCHOR[96][1] + 1)),
        "statistic=k+1 (alternative): emitted=k": (ANCHOR[16][2] / ANCHOR[16][1], ANCHOR[96][2] / ANCHOR[96][1]),
    }
    out: dict[str, Any] = {"c": c, "readings": {}}
    all_excluded = True
    for name, (emit16, emit96) in readings.items():
        ab16 = implied_ab(v_f16, v_d16, gain16 / emit16)
        pred96 = predicted_ratio_pow(v_f96, v_d96, ab16, c)
        required96 = gain96 / emit96
        excluded = pred96 < required96
        all_excluded = all_excluded and excluded
        out["readings"][name] = {
            "ab16": ab16,
            "predicted96": pred96,
            "required96": required96,
            "excluded": excluded,
        }
    out["ok"] = all_excluded
    return out
