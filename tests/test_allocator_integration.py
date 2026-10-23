"""Allocator-integration tests (Lane L2). N6 criteria + N5 mutants."""

import numpy as np

from sim import allocator
from sim.allocator import (
    accepted_prefix_in_window,
    dynamic_seqlens,
    fixed5_seqlens,
    pairing_check,
    simulate_pair,
)
from sim.rates import NEXTN
from sim.sweep_l2 import bootstrap_ci


def test_dynamic_degenerates_to_fixed5_at_full_budget():
    """Criterion: dynamic_seqlens at total_q = num_seqs*(NEXTN+1) returns
    NEXTN+1 for every sequence, for arbitrary rates including all-zero; and
    total_q = num_seqs*NEXTN does NOT degenerate.

    Why: extra_budget = NEXTN*num_seqs covers all num_seqs*NEXTN positions,
    so every position wins and seqlens_q = 1 + NEXTN (schedule_mtp_varlen.py:98;
    sim/convention.md). HANDOFF §7.2's "num_seqs × 5" treated draft positions
    as rows — spec error corrected per user ruling 2026-10-22. Flip: any
    change to the allocator primitive's ranking or the row convention.
    """
    rng = np.random.default_rng(11)
    for num_seqs in (1, 3, 16):
        rates = rng.random((num_seqs, NEXTN)) * 0.9 + 0.05
        got = dynamic_seqlens(num_seqs * (NEXTN + 1), rates)
        assert np.array_equal(got, fixed5_seqlens(num_seqs))
    zero = dynamic_seqlens(3 * (NEXTN + 1), np.zeros((3, NEXTN)))
    assert np.array_equal(zero, fixed5_seqlens(3))
    under = dynamic_seqlens(3 * NEXTN, np.full((3, NEXTN), 0.5))
    assert not np.array_equal(under, fixed5_seqlens(3))
    assert int(under.sum()) == 3 * NEXTN


def test_allocator_budget_invariant():
    """Criterion: dynamic_seqlens sums to total_q for any budget in
    [num_seqs, num_seqs*(NEXTN+1)] and never goes below 1 row per sequence.

    Why: the approved primitive's contract (dflash_alloc.allocate_seqlens_q:
    1 base row + won extra slots). Flip: a wrapper bug that drops the fresh
    row or leaks budget.
    """
    rng = np.random.default_rng(12)
    rates = rng.random((8, NEXTN)) * 0.9 + 0.05
    for total_q in (8, 11, 20, 33, 48):
        out = dynamic_seqlens(total_q, rates)
        assert int(out.sum()) == total_q
        assert int(out.min()) >= 1


def test_windowed_prefix_count():
    """Criterion: the accepted prefix inside a verified window equals
    min(full prefix, wins).

    Why: acceptance is a sequential prefix process, monotone in position.
    Flip: a windowed count that re-draws accepts or mis-truncates.
    """
    accept = np.array(
        [[True, True, False, False, False], [True, True, True, True, True]], dtype=bool
    )
    assert accepted_prefix_in_window(accept, np.array([3, 2])).tolist() == [2, 2]
    assert accepted_prefix_in_window(accept, np.array([5, 5])).tolist() == [2, 5]
    assert accepted_prefix_in_window(accept, np.array([0, 0])).tolist() == [0, 0]
    assert accepted_prefix_in_window(accept, np.array([1, 4])).tolist() == [1, 4]


def test_pairing_bit_identical_at_degenerate_budget():
    """Criterion: with matching seeds the fixed and dynamic arms are
    bit-identical at total_q = num_seqs*(NEXTN+1).

    Why: identical stratified type vectors and identical u blocks, and at the
    degenerate budget the dynamic windows equal the fixed windows, so the
    trajectories coincide step by step. Flip: any ambient-state divergence
    between arms (seed, draw order, window logic).
    """
    ok, detail = pairing_check(num_seqs=8, steps=200, seed=101)
    assert ok, detail


def test_pairing_mutant_red():
    """N5 mutant: reseeding only the dynamic arm breaks bit-identity — the
    pairing check must go red (HANDOFF §7.2's pairing mutant).

    Flip: if the broken pairing still passes, the pairing check cannot detect
    ambient-state divergence and every downstream A/B claim is unsound.
    """
    ok, detail = pairing_check(num_seqs=8, steps=200, seed=101, dynamic_seed=102)
    assert not ok, f"broken pairing still passes the check: {detail}"


def test_paired_ci_width_zero_vs_broken():
    """Criterion: at the degenerate budget the paired per-seed log-ratio
    deltas are exactly 0, so the bootstrap CI width is 0; under the broken
    pairing the deltas are nonzero and the width exceeds 0.

    Why: bit-identical arms give emitted_dyn/emitted_fixed = 1 exactly, so
    the paired deltas vanish; broken pairing leaves real dispersion. This is
    HANDOFF §7.2's CI-width mutant control. Flip: a pairing break that leaves
    the zero-width assertion green means the width test is miscalibrated.
    """
    seeds = [20261101 + i for i in range(4)]
    paired = []
    for s in seeds:
        res = simulate_pair(num_seqs=8, steps=200, seed=s, total_q=8 * (NEXTN + 1))
        ef = allocator.arm_stats(res["fixed"]["k"], res["fixed"]["wins"])["emitted"]
        ed = allocator.arm_stats(res["dynamic"]["k"], res["dynamic"]["wins"])["emitted"]
        paired.append(np.log(ed / ef))
    mean_p, lo_p, hi_p = bootstrap_ci(np.array(paired), n_resamples=2000, seed=7)
    assert mean_p == 0.0
    assert hi_p - lo_p == 0.0

    broken = []
    for s in seeds:
        res = simulate_pair(
            num_seqs=8, steps=200, seed=s, total_q=8 * (NEXTN + 1), dynamic_seed=s + 500
        )
        ef = allocator.arm_stats(res["fixed"]["k"], res["fixed"]["wins"])["emitted"]
        ed = allocator.arm_stats(res["dynamic"]["k"], res["dynamic"]["wins"])["emitted"]
        broken.append(np.log(ed / ef))
    mean_b, lo_b, hi_b = bootstrap_ci(np.array(broken), n_resamples=2000, seed=7)
    assert hi_b - lo_b > 0.0
