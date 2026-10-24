"""M2-L1 ranking-noise tests. N6 criteria + N5 mutants; Ruling B semantics."""

import numpy as np
import pytest

from sim import allocator
from sim import rank_noise as rn
from sim import rates
from sim.rates import CALIB_STEPS
from sim.sweep_l2 import ANCHOR_MATCHED_QBAR, _per_seed_deltas, t_ratio_pow

COMMITTED_ROW_465_11 = {16: -0.05, 32: 0.16, 48: 3.63, 60: 4.99, 72: 6.09, 96: 9.17}
SIGMA_GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)


def test_kendall_tau_bruteforce_cross_check():
    """Criterion: the vectorized production tau_b equals an independent
    pure-python O(n^2) reference on random tie-free inputs, and the extremes
    are exactly +-1.

    Why: tau_K is the reported metric every threshold is stated in; a wrong
    implementation would corrupt every bound. Flip: any mismatch on the
    random cases or the extremes.
    """
    rng = np.random.default_rng(20261201)
    for _ in range(20):
        x = rng.random(12)
        y = rng.random(12)
        assert abs(rn.kendall_tau(x, y) - rn.kendall_tau_bruteforce(x, y)) < 1e-12
    x = rng.random(30)
    assert rn.kendall_tau(x, x) == 1.0
    assert rn.kendall_tau(x, -x) == -1.0


def test_kendall_tau_tied_hand_case():
    """Criterion: x = (1, 1, 2), y = (1, 2, 2) gives tau_b = +0.5 by hand —
    re-derived: pair (0, 2) is concordant (both increase), pairs (0, 1) and
    (1, 2) carry an x-tie and a y-tie respectively, so C = 1, D = 0,
    n0 = 3, n1 = n2 = 1, denom = 2. Flip: any change to the tie handling.
    """
    x = np.array([1.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 2.0])
    assert abs(rn.kendall_tau(x, y) - 0.5) < 1e-12
    assert abs(rn.kendall_tau_bruteforce(x, y) - 0.5) < 1e-12


@pytest.mark.slow
def test_tau0_pin_bit_identity_and_committed_row():
    """Criterion: at sigma = 0 the noisy pipeline is bit-identical to the M1
    path (both arms' k arrays), and the (465.0, 1.1) paired gains reproduce
    the committed sign-map row to its printed precision.

    Why: the pin proves the machinery adds nothing at sigma = 0 (Ruling B
    semantics + draw-order discipline) and anchors every sigma > 0 result to
    the committed L2 artifact. Flip: any deviation in draw order, window
    logic, or aggregation.
    """
    for conc in (16, 32, 48, 60, 72, 96):
        total_q = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
        res_n = rn.simulate_pair_noisy(conc, CALIB_STEPS, 20261101, total_q=total_q, sigma=0.0)
        res_ref = allocator.simulate_pair(conc, CALIB_STEPS, 20261101, total_q=total_q)
        assert np.array_equal(res_n["fixed"]["k"], res_ref["fixed"]["k"]), f"fixed arm diverged at conc {conc}"
        assert np.array_equal(res_n["dynamic"]["k"], res_ref["dynamic"]["k"]), f"dynamic arm diverged at conc {conc}"
    committed = {}
    for conc in (16, 32, 48, 60, 72, 96):
        total_q = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
        gains = []
        for i in range(12):
            res = rn.simulate_pair_noisy(conc, CALIB_STEPS, 20261101 + i, total_q=total_q, sigma=0.0)
            ef = float((res["fixed"]["k"] + 1).mean())
            ed = float((res["dynamic"]["k"] + 1).mean())
            gains.append(ed / ef)
        deltas = np.log(np.array(gains)) + np.log(t_ratio_pow(conc * 6, total_q, 465.0, 1.1))
        mean_gain = float(np.mean(np.exp(deltas) - 1.0))
        committed[conc] = round(mean_gain * 100, 2)
    assert committed == COMMITTED_ROW_465_11, f"sigma=0 row deviates from the committed sign map: {committed}"


def test_monotone_degradation_and_identity_mutant(monkeypatch):
    """Criterion (two parts, per ruling 3): (i) the mean dynamic accepted
    length is non-increasing in sigma over the grid under a declared
    tolerance of 0.01 tokens between adjacent points; (ii) ENDPOINT strict
    degradation — mean(sigma=3.2) < mean(sigma=0) - 0.05 — carries the
    falsification weight.

    Why: window noise can only redistribute a fixed budget, so scrambling
    must cost accepted length; an implementation that preserves or improves
    ordering must fail (ii). Flip: the identity-noise mutant (perturbation
    ignores sigma) leaves the endpoint intact and this test goes red.
    """
    conc, steps = 16, 600
    total_q = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))

    def run_grid() -> dict[float, float]:
        means = {}
        for sigma in SIGMA_GRID:
            accs = []
            for i in range(4):
                res = rn.simulate_pair_noisy(conc, steps, 20261101 + i, total_q=total_q, sigma=sigma)
                accs.append(float(res["dynamic"]["k"].mean()))
            means[sigma] = float(np.mean(accs))
        return means

    means = run_grid()
    ok, detail = rn.degradation_check(means, tol=0.01, endpoint_margin=0.05)
    assert ok, f"degradation check failed on the real machinery: {detail}"

    monkeypatch.setattr(rn, "perturbed_rates", lambda r, z, sigma: r)
    means_mutant = run_grid()
    ok_m, detail_m = rn.degradation_check(means_mutant, tol=0.01, endpoint_margin=0.05)
    assert not ok_m, f"identity-noise mutant passed the degradation check: {detail_m}"


def test_ruling_b_window_semantics():
    """Criterion: the perturbation changes ONLY the window — elementwise
    k_dyn == min(k_true_prefix, wins') with acceptance drawn from the TRUE
    rates; the budget invariant holds every step; the fixed arm is
    sigma-independent and the u-block order is unchanged by sigma.

    Why: a perturbation model that perturbs acceptance OUTCOMES instead of
    window lengths is a category error (Ruling B). Flip: any path where the
    min-identity, the budget sum, or the fixed-arm invariance breaks.
    """
    conc, steps = 8, 120
    total_q = 34
    res0 = rn.simulate_pair_noisy(conc, steps, 20261101, total_q=total_q, sigma=0.0)
    res3 = rn.simulate_pair_noisy(conc, steps, 20261101, total_q=total_q, sigma=3.2)
    assert np.array_equal(res3["dynamic"]["k"], np.minimum(res3["dynamic"]["k_true_prefix"], res3["dynamic"]["wins"]))
    wins_sums = res3["dynamic"]["wins"].sum(axis=0)
    assert np.array_equal(wins_sums, np.full(steps, total_q - conc, dtype=np.int64))
    assert np.array_equal(res0["fixed"]["k"], res3["fixed"]["k"])


def test_v2e_frozen_constants():
    """Criterion: the adopted V2E levels (entry 10) hold the fixed-5 mean at
    3.3042 within bisection precision (|delta| <= 1e-5), every base row in
    [3.25, 3.32], and every EMIT_BONUS = 0 mutant row above the band top
    3.32 (L1's margin discipline, per-row).

    Why: Ruling D's adoption conditions (a) and (b), re-checked against the
    frozen constants rather than the probe that selected them. Flip: any
    constant transcription error or numpy stream change.
    """
    base = rates.fixed5_report(levels=rn.V2E_LEVELS)
    mean = sum(base.values()) / len(base)
    assert abs(mean - rn.V2E_TARGET_MEAN) <= 1e-5, f"V2E mean {mean} off target"
    assert all(3.25 <= v <= 3.32 for v in base.values()), f"V2E base rows out of band: {base}"
    mut = rates.fixed5_report(levels=rn.V2E_LEVELS, emit_bonus=0)
    assert all(v > 3.32 for v in mut.values()), f"V2E mutant rows not all above band top: {mut}"


def test_v2e_tau0_pin_on_adopted_levels():
    """Criterion: at sigma = 0 under the ADOPTED V2E levels, the paired
    pipeline reduces to the M1 path bit-for-bit (both arms, one seed per
    anchor conc) — the machinery is generator-agnostic.

    Flip: any sigma-path leakage into the sigma = 0 trajectory.
    """
    for conc in (16, 96):
        total_q = int(round(conc * ANCHOR_MATCHED_QBAR[conc]))
        res_n = rn.simulate_pair_noisy(conc, 400, 20261101, total_q=total_q, sigma=0.0, levels=rn.V2E_LEVELS)
        res_ref = allocator.simulate_pair(conc, 400, 20261101, total_q=total_q, levels=rn.V2E_LEVELS)
        assert np.array_equal(res_n["fixed"]["k"], res_ref["fixed"]["k"])
        assert np.array_equal(res_n["dynamic"]["k"], res_ref["dynamic"]["k"])
