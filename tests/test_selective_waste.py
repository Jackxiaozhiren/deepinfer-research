"""M3-L1 waste accounting (Ruling A). N6 criteria + N5 planted mutants.

Every fixture value here is hand-derived arithmetic, not a measurement, so a
green run certifies the accounting and not the generator. The mutants are
planted wrong implementations of the SAME quantity inside the tests: each one
must produce a different answer from the pinned fixture, which is what makes
the fixture a control rather than a restatement (N5).
"""

import numpy as np
import pytest

from sim import rates
from sim import selective as sel


def _w(rows):
    wins = np.array([v for v, _ in rows], dtype=np.int64)
    k = np.array([a for _, a in rows], dtype=np.int64)
    return sel.waste_stats(wins, k)


def test_waste_fixture_single_row():
    """Criterion: v=5 verified draft positions, k=3 accepted gives
    W = (5-3)/5 = 0.4 exactly.

    Why: this is Ruling A's definition in its smallest form — rejected draft
    positions over verified draft positions, one sequence, one step.
    Flip: any change to numerator (rejected) or denominator (verified).
    """
    assert _w([(5, 3)])["waste"] == pytest.approx(0.4, abs=1e-12)


def test_waste_fixture_pooling_convention():
    """Criterion: rows (5,5) and (1,0) give POOLED W = (0+1)/(5+1) = 1/6, while
    the mean of per-row ratios is (0/5 + 1/1)/2 = 0.5.

    Why: the two conventions differ by 3x on this input, so the choice is
    load-bearing and cannot be implied. Ruling A's "per step" wording is the
    pooled reading: the run-level statistic is total rejected over total
    verified, matching the anchor table's accepted/verified columns.
    Flip: production returning the mean-of-ratios value.
    """
    got = _w([(5, 5), (1, 0)])["waste"]
    assert got == pytest.approx(1.0 / 6.0, abs=1e-12)
    mutant = np.mean([0.0 / 5.0, 1.0 / 1.0])
    assert mutant == pytest.approx(0.5, abs=1e-12)
    assert abs(got - mutant) > 0.3, "pooling convention is not distinguishable"


def test_waste_fixture_fresh_row_excluded():
    """Criterion: rows (v=4,k=3) and (v=0,k=0) give W = 1/4 = 0.25 with draft
    positions only, whereas counting each sequence's fresh base row gives
    (1+1)/(5+1) = 0.50.

    Why: Ruling A pins that the fresh row is not a verification and enters
    neither numerator nor denominator; seqlens_q = 1 + wins (sim/convention.md)
    is the row count, not the verified count.
    Flip: any accounting that feeds seqlens_q where wins belongs.
    """
    got = _w([(4, 3), (0, 0)])["waste"]
    assert got == pytest.approx(0.25, abs=1e-12)
    mutant = _w([(5, 3), (1, 0)])["waste"]  # wins replaced by seqlens_q
    assert mutant == pytest.approx(0.5, abs=1e-12)
    assert abs(got - mutant) > 0.2


def test_waste_budget_floor_is_nan_not_zero():
    """Criterion: when no draft position is verified at all (total_q = num_seqs,
    wins all zero) the pooled ratio is 0/0 and must report nan.

    Why: a zero here would read as "this policy wastes nothing" for a policy
    that verified nothing, which is the N4 failure mode (a zero count is not a
    success). M3 §9 Q4 resolved this by declaring nan.
    Flip: any fallback that coerces the empty denominator to 0.
    """
    got = _w([(0, 0), (0, 0), (0, 0)])["waste"]
    assert np.isnan(got), f"budget floor must be nan, got {got}"
    assert got != 0.0


def test_waste_rejects_untruncated_accepted_length():
    """Criterion: waste_stats refuses k > wins with an AssertionError rather
    than returning a negative waste.

    Why: accepted-prefix length must be truncated to the verified window
    (k = min(k_full, wins)); without truncation a row with v=2, k_full=4 would
    report W = -1.0. The guard turns that arithmetic into a loud failure
    (N4), which is the stronger form of the triage's F-A5 expectation.
    Flip: the guard being widened to accept k > wins, or the truncation moving
    into the accounting function where it would silently mask the harness.
    """
    with pytest.raises(AssertionError, match="may not exceed"):
        sel.waste_stats(np.array([2], dtype=np.int64), np.array([4], dtype=np.int64))
    truncated = sel.waste_stats(np.array([2], dtype=np.int64), np.array([2], dtype=np.int64))
    assert truncated["waste"] == 0.0


def test_fixed5_waste_is_a_derivative_of_the_frozen_calibration_band():
    """Criterion: under the frozen L1 generator the fixed-5 arm's waste lands in
    [1 - 3.32/5, 1 - 3.29/5], the band DERIVED from the already-frozen accepted
    band BAND_MEAN, and equals 1 - mean(k)/NEXTN identically.

    Why: fixed-5 verifies all NEXTN draft positions every step, so waste is a
    monotone function of the accepted-length statistic Milestone 1 already
    calibrated. Asserting the derived band adds no new numeric claim (N8: the
    published 1 - 3.30/5.00 = 0.34 reading is a reference point, not a target
    this test can tune to) and pins Ruling A's denominator to the draft count.
    Flip: any generator drift, or counting the fresh row (which would move the
    value to 1 - mean(k)/6, outside this band).
    """
    k, _ = rates.simulate_fixed5(
        16,
        rates.CALIB_STEPS,
        rates.CALIB_SEED,
        levels=rates.TYPE_LEVELS,
        weights=rates.TYPE_WEIGHTS,
        depth_decay=rates.DEPTH_DECAY,
        progress_decay=rates.PROGRESS_DECAY,
    )
    w_draft = sel.waste_stats(np.full(k.shape, rates.NEXTN, dtype=np.int64), k)["waste"]
    assert w_draft == pytest.approx(1.0 - float(k.mean()) / rates.NEXTN, abs=1e-12)
    lo, hi = 1.0 - rates.BAND_MEAN[1] / rates.NEXTN, 1.0 - rates.BAND_MEAN[0] / rates.NEXTN
    assert lo <= w_draft <= hi, f"fixed-5 waste {w_draft:.6f} outside derived band [{lo}, {hi}]"
    row_incorrect = 1.0 - float(k.mean()) / (rates.NEXTN + 1)
    assert not (lo <= row_incorrect <= hi), "row-counting mutant must fall outside the band"


def test_predicted_waste_matches_pooled_definition():
    """Criterion: predicted_waste(cum, wins) equals
    1 - sum(cum over verified cells)/count(verified cells) on a hand-built block,
    and returns nan when the window is empty.

    Why: B1's anchor side is this pool; a off-by-one in the window mask
    (idx <= wins) is exactly "the fresh row counted", so the two conventions
    must give different numbers here too.
    Flip: the mask, or the empty-pool branch returning 0.
    """
    cum = np.array([[0.9, 0.8, 0.7, 0.6, 0.5], [0.4, 0.2, 0.1, 0.05, 0.02]])
    wins = np.array([3, 0], dtype=np.int64)
    got = sel.predicted_waste(cum, wins)
    assert got == pytest.approx(1.0 - (0.9 + 0.8 + 0.7) / 3.0, abs=1e-12)
    assert np.isnan(sel.predicted_waste(cum, np.array([0, 0], dtype=np.int64)))
    fresh_row_mutant = 1.0 - (0.9 + 0.8 + 0.7) / (3.0 + 2)
    assert abs(got - fresh_row_mutant) > 0.1
