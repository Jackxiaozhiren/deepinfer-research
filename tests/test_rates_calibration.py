"""Calibration tests for sim/rates.py (Lane L1).

N6 criteria with flip conditions; N5 mutants demonstrate the controls can
fire. All randomness flows through the frozen seeds.
"""

import numpy as np

from sim import rates
from sim.rates import BAND_MEAN, CALIB_SEED, NEXTN


def test_calibration_in_band():
    """Criterion: with the frozen parameters and seeds the fixed-5 accepted
    length k averages into [3.29, 3.32] across the six anchor rows and every
    row lands in [3.25, 3.32].

    Why: the deterministic search in sim/rates.py imposed exactly these
    constraints (plus the mutant constraint below) and the frozen constants
    reproduce its output bit-for-bit (`python -m sim.rates` re-verifies).
    Flip: any change to the frozen constants, seeds, draw order, or numpy's
    random stream behavior moves the statistic out of the band.
    """
    vals = rates.fixed5_report()
    ok, detail = rates.band_check(vals)
    assert ok, f"calibration left the band: {detail}"


def test_mutant_emit_bonus_red(monkeypatch):
    """N5 mutant: flipping EMIT_BONUS to 0 must push the statistic out of the
    calibration band, mean above the band top (slower progress -> less decay
    -> higher acceptance), with the paired per-row direction preserved.

    Why: PROGRESS_DECAY is design-driven precisely to make the emission
    convention observable in the accepted-length statistic (user note 3).
    Flip: if the mutant stays in band, the N5 control cannot fire and the
    generator family is miscalibrated for its purpose.
    """
    base = rates.fixed5_report()
    base_ok, base_detail = rates.band_check(base)
    assert base_ok, f"frozen calibration broken: {base_detail}"
    base_max = max(base.values())

    monkeypatch.setattr(rates, "EMIT_BONUS", 0)
    mut = rates.fixed5_report()
    mut_ok, mut_detail = rates.band_check(mut)
    mut_mean = sum(mut.values()) / len(mut)
    assert not mut_ok, f"mutant EMIT_BONUS=0 stayed in band — N5 control cannot fire: {mut_detail}"
    assert mut_mean > BAND_MEAN[1], f"mutant mean {mut_mean:.4f} did not clear the band top {BAND_MEAN[1]}"
    assert min(mut.values()) > base_max, (
        f"paired direction violated: min(mut)={min(mut.values()):.4f} <= max(base)={base_max:.4f}"
    )


def test_heterogeneity_present():
    """Criterion: the frozen mixture produces distinct per-type populations —
    the level spread is at least 0.15 and the simulated per-sequence mean k
    has std > 0.05 across sequences.

    Why: cross-sequence heterogeneity is the load-bearing assumption
    (HANDOFF §7.1): a homogeneous population makes dynamic ~ fixed by
    construction, which would invalidate lane L2 before it starts.
    Flip: levels collapse together or the mixture degenerates to one type.
    """
    levels = rates.TYPE_LEVELS
    assert max(levels) - min(levels) >= 0.15
    k, _ = rates.simulate_fixed5(
        num_seqs=rates.CONCURRENCY_ROWS[-1],
        steps=2000,
        seed=CALIB_SEED,
        levels=levels,
        weights=rates.TYPE_WEIGHTS,
        depth_decay=rates.DEPTH_DECAY,
        progress_decay=rates.PROGRESS_DECAY,
    )
    per_seq = k.mean(axis=1)
    assert float(per_seq.std()) > 0.05, (
        f"per-sequence mean-k std {float(per_seq.std()):.4f} — population too homogeneous"
    )


def test_seeded_determinism():
    """Criterion: identical seeds reproduce identical draws bit-for-bit; a
    different seed moves them.

    Why: N3 — every stochastic step takes an explicit seed and every frozen
    number must be reproducible. Flip: an RNG call escapes the seeded stream
    or the draw order drifts.
    """
    kwargs = dict(
        levels=rates.TYPE_LEVELS,
        weights=rates.TYPE_WEIGHTS,
        depth_decay=rates.DEPTH_DECAY,
        progress_decay=rates.PROGRESS_DECAY,
    )
    k1, e1 = rates.simulate_fixed5(num_seqs=16, steps=200, seed=CALIB_SEED, **kwargs)
    k2, e2 = rates.simulate_fixed5(num_seqs=16, steps=200, seed=CALIB_SEED, **kwargs)
    assert np.array_equal(k1, k2) and np.array_equal(e1, e2)
    k3, _ = rates.simulate_fixed5(num_seqs=16, steps=200, seed=CALIB_SEED + 1, **kwargs)
    assert not np.array_equal(k1, k3)
    assert int(k1.max()) <= NEXTN
