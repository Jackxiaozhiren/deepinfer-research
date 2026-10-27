"""M3-L1 invariant and anchor tests (Rulings B and C) plus the invariance
signature table. N6 criteria + N5 planted mutants.

Every structural claim in HANDOFF_PROMPT_M3.md §4 is asserted here in BOTH
directions: the claim that holds (global gamma cannot corrupt a rank-relative
selector) and the claim that must fail for the axis to be live (per-type gamma
does corrupt it, absolute thresholds do). A one-sided test would stay green on
a vacuous universe, which is what Ruling C's "if it fails, that is a finding"
requires us to be able to notice.
"""

import numpy as np
import pytest

from sim import allocator
from sim import rates
from sim import rank_noise as rn
from sim import selective as sel

CONCS = rates.CONCURRENCY_ROWS
SURVEY_BLOCKS = 4
COMMUTATION_GAMMAS = (0.5, 1.5, 2.0, 3.0)
ANCHOR_SEEDS = (sel.GAMMA_SEED, sel.GAMMA_SEED + 1)
SURVEY_SEEDS = 3
SEQS_SURVEYED = sum(CONCS) * SURVEY_SEEDS


def _rate_block(conc: int, seed: int, levels=None) -> np.ndarray:
    types = rates.stratified_types(conc, rates.TYPE_WEIGHTS)
    gen = np.random.default_rng((seed, conc)).integers(0, 20000, size=conc).astype(float)
    return rates.rates_matrix(
        rates.TYPE_LEVELS if levels is None else levels,
        rates.DEPTH_DECAY,
        rates.PROGRESS_DECAY,
        types,
        gen,
    )


def _budget(conc: int) -> int:
    return int(round(conc * sel.ANCHOR_DYNAMIC_VERIFIED[conc])) if conc == 16 else conc * 3


def _displacement(gamma, *, concs=CONCS, blocks=SURVEY_SEEDS, mult: float | None = None) -> int:
    """Sequences whose top-k window changes under this gamma vector.

    mult sets total_q = round(conc * mult) rows/sequence; with mult=None the
    anchor-matched budget is used (conc * 4.44 at conc 16, conc * 3 elsewhere).
    """
    changed = 0
    for conc in concs:
        types = rates.stratified_types(conc, rates.TYPE_WEIGHTS)
        for s in range(blocks):
            r = _rate_block(conc, 100 + s)
            tq = _budget(conc) if mult is None else int(round(conc * mult))
            changed += int((sel.rank_topk_wins(tq, r) != sel.rank_topk_wins(tq, sel.predicted_rates(r, gamma, types))).sum())
    return changed


def test_identity_gamma_and_zero_sigma_reproduce_the_m2_pipeline():
    """Criterion: with gamma = (1,1,1) and sigma = 0, the rank_topk arm of
    simulate_rule is bit-identical (k and wins arrays) to the committed
    allocator.simulate_pair dynamic arm at the same seed and budget, at conc 16
    and conc 96.

    Why: M3 adds an axis on top of machinery Milestones 1-2 already published
    numbers with. If the new harness moved anything at its own identity point,
    every M3 reading would be suspect. The draw order is inherited verbatim
    from sim/rank_noise.py, so a change in u- or z-consumption shows up here.
    Flip: any change to draw order, window truncation, or per-arm progress.
    """
    for conc in (16, 96):
        tq = _budget(conc)
        ref = allocator.simulate_pair(conc, 6000, 20261101, total_q=tq)
        got = sel.simulate_rule("rank_topk", conc, 6000, 20261101, total_q=tq)
        assert np.array_equal(got["fixed"]["k"], ref["fixed"]["k"]), f"fixed arm moved at conc {conc}"
        assert np.array_equal(got["rule"]["k"], ref["dynamic"]["k"]), f"rule arm k moved at conc {conc}"
        assert np.array_equal(got["rule"]["wins"], ref["dynamic"]["wins"]), f"rule arm wins moved at conc {conc}"


def test_c1_global_gamma_leaves_topk_allocation_bit_identical():
    """Criterion: for a GLOBAL exponent gamma in {0.5, 1.5, 2.0, 3.0}, the top-k
    windows over 6 anchor concs x 4 rate blocks equal the gamma = 1 windows,
    array for array (96 comparisons, zero differences).

    Why: r' = r^gamma commutes with cumprod and with the ranking, because
    x^gamma is monotone on (0,1) and the primitive's tie-break is by flat index
    (schedule_mtp_varlen.py:85), which no value transform reorders. This is
    M3 §4.1's claim, and Ruling C makes its failure a finding about §4.
    Flip: a per-type gamma, a non-monotone transform, or applying gamma after
    cumprod instead of to the rates.
    """
    for conc in CONCS:
        tq = _budget(conc)
        for s in range(SURVEY_BLOCKS):
            r = _rate_block(conc, s)
            types = rates.stratified_types(conc, rates.TYPE_WEIGHTS)
            base = sel.rank_topk_wins(tq, r)
            for g in COMMUTATION_GAMMAS:
                perturbed = sel.predicted_rates(r, (g, g, g), types)
                assert np.array_equal(base, sel.rank_topk_wins(tq, perturbed)), f"gamma={g} moved top-k at conc {conc}"


def test_c1_mutant_additive_bias_does_move_the_allocation():
    """Criterion: the additive form r + delta (clipped) DOES change the top-k
    windows, so the C1 assertion above is discriminating rather than vacuous.

    Why: an exponent is chosen precisely because it commutes; a bias does not,
    since clipping flattens a run of cells and the prefix ordering is lost.
    Without this test, "gamma cannot corrupt the allocation" would also be
    satisfied by a perturbation function that silently did nothing.
    Flip: the additive mutant becoming a no-op (clip boundaries, or delta 0).
    """
    moved = 0
    comparisons = 0
    for conc in CONCS:
        tq = _budget(conc)
        for s in range(SURVEY_BLOCKS):
            r = _rate_block(conc, s)
            base = sel.rank_topk_wins(tq, r)
            for delta in (0.05, -0.05):
                alt = np.clip(r + delta, rates.EPS, 1.0 - rates.EPS)
                comparisons += 1
                if not np.array_equal(base, sel.rank_topk_wins(tq, alt)):
                    moved += 1
    assert moved > 0, "additive mutant left the allocation untouched: survey is blind"


def test_c2_per_type_gamma_is_a_live_axis():
    """Criterion: gamma = (2,1,1) changes the top-k window of at least 10% of the
    surveyed sequences (measured 488 of 972).

    Why: per-type gamma is THE miscalibration axis for M3 (Ruling D). If it did
    not reshuffle cross-sequence rankings, the whole milestone would be measuring
    an inert knob.
    Flip: the type mixture becoming homogeneous, or gamma applied globally.
    """
    changed = _displacement((2.0, 1.0, 1.0))
    assert changed > 0.10 * SEQS_SURVEYED, f"per-type gamma changed only {changed} of {SEQS_SURVEYED}"


def test_c2_negative_control_gamma_bite_is_budget_conditional():
    """Criterion: the gamma axis's bite depends on where the budget cut falls.
    Measured over the 972-sequence survey:

      gamma=(1,1,2) at total_q = conc*3 -> 0 windows changed (exactly inert)
      gamma=(1,1,2) at total_q = conc*4 -> 146 changed (revived)
      gamma=(2,1,1) at total_q = conc*4 -> 744 changed
      gamma=(2,1,1) at total_q = conc*5.44 -> 0 changed (saturated, inert)

    Why: under-confidence can only move a cell that sits near the cut. At a
    tight budget the low type is already below the cut and shrinking it changes
    nothing; at a near-saturating budget almost every cell wins anyway, so even
    the high type's ordering has no room to matter. Asserting all four rows is
    what keeps the axis honest in both directions (user ruling 4), and the last
    row is a warning for L2: sweeping only the anchor-matched full budget would
    understate gamma's effect.
    Flip: a budget grid that is too narrow to show the revival, or a homogeneous
    type mixture. The live rows are asserted as fractional floors of the 972
    surveyed sequences rather than exact integers, because one libm rounding
    difference at a marginal rank would otherwise flip the suite without meaning
    anything; the inert rows stay exact, since those are structural claims.
    """
    assert _displacement(sel.GAMMA_INERT_CONTROL, mult=3.0) == 0
    assert _displacement(sel.GAMMA_INERT_CONTROL, mult=4.0) > 0.10 * SEQS_SURVEYED
    assert _displacement((2.0, 1.0, 1.0), mult=4.0) > 0.50 * SEQS_SURVEYED
    assert _displacement((2.0, 1.0, 1.0), mult=5.44) == 0


def test_c3_quantile_threshold_is_equivariant_to_global_gamma():
    """Criterion: for q in {0.35, 0.5, 0.65} at conc 16 over 400 steps, the
    quantile family's windows, mean verified length and pooled waste are
    identical between gamma = 1 and gamma = 2.

    Why: a quantile commutes with a monotone transform, so this threshold rule —
    unlike the fixed-c one — is rank-relative and immune to global gamma. That is
    the correction to M3 §4.2 (user ruling 2): the discriminating property is
    threshold equivariance, not "reading values".
    Flip: the threshold being taken on absolute values, or per-type gamma.
    """
    steps = 400
    for q in (0.35, 0.5, 0.65):
        base = sel.simulate_rule("quantile", 16, steps, sel.GAMMA_SEED, q=q)
        alt = sel.simulate_rule("quantile", 16, steps, sel.GAMMA_SEED, q=q, gamma=(2.0, 2.0, 2.0))
        assert np.array_equal(base["rule"]["wins"], alt["rule"]["wins"]), f"q={q} windows moved"
        assert base["rule"]["mean_verified"] == alt["rule"]["mean_verified"]
        assert base["rule"]["waste"] == alt["rule"]["waste"]


def test_fixed_c_absolute_threshold_does_move_under_global_gamma():
    """Criterion: the absolute-c family's verified length falls materially under
    gamma = 2 (measured 3.250 -> 2.093 at c = 0.60, conc 16, 6000 steps).

    Why: cumprod^gamma >= c iff cumprod >= c^(1/gamma) — the same gamma that
    cannot touch a rank-relative selector reinterprets an absolute threshold.
    This is the mechanism error control is built on, so it must be asserted
    alongside C3 rather than assumed from prose.
    Flip: c at a boundary where the whole batch is above or below it.
    """
    base = sel.simulate_rule("fixed_c", 16, 6000, sel.GAMMA_SEED, c=0.60)
    alt = sel.simulate_rule("fixed_c", 16, 6000, sel.GAMMA_SEED, c=0.60, gamma=(2.0, 2.0, 2.0))
    assert abs(base["rule"]["mean_verified"] - 3.250) < 0.01
    assert alt["rule"]["mean_verified"] < base["rule"]["mean_verified"] - 0.5


def test_b1_calibrated_identity_anchor_and_its_two_mutants():
    """Criterion: under exact calibration (gamma = identity, sigma = 0), the
    fixed-c rule's realized pooled waste equals
    1 - E[predicted cumprod over the verified window] within
    ANCHOR_IDENTITY_TOL for every c in ANCHOR_C_GRID.

    Why: this is Ruling B's anchor in the only form the prefix structure
    supports (user ruling 1) — the window mean, not the marginal 1 - c. A
    failure here is an accounting bug, never a finding.
    Flip: counting the fresh row on the prediction side, or any gamma != 1.
    """
    for seed in ANCHOR_SEEDS:
        for c in sel.ANCHOR_C_GRID:
            out = sel.simulate_rule("fixed_c", 16, 6000, seed, c=c)["rule"]
            assert abs(out["waste_gap"]) <= sel.ANCHOR_IDENTITY_TOL, (
                f"B1 anchor broken at c={c} seed={seed}: realized={out['waste']:.5f} "
                f"predicted={out['pred_waste']:.5f}"
            )

    ref = sel.simulate_rule("fixed_c", 16, 6000, ANCHOR_SEEDS[0], c=0.60)["rule"]
    v = ref["verified_total"]
    sel_sum = (1.0 - ref["pred_waste"]) * v
    rows = 16 * 6000
    mutant_rejected_row = 1.0 - sel_sum / (v + rows)
    mutant_accepted_row = 1.0 - (sel_sum + rows) / (v + rows)
    assert abs(ref["waste"] - mutant_rejected_row) > sel.ANCHOR_IDENTITY_TOL
    assert abs(ref["waste"] - mutant_accepted_row) > sel.ANCHOR_IDENTITY_TOL
    biased = sel.simulate_rule("fixed_c", 16, 6000, ANCHOR_SEEDS[0], c=0.60, gamma=(1.5, 1.5, 1.5))["rule"]
    assert abs(biased["waste_gap"]) > sel.ANCHOR_IDENTITY_TOL, "anchor must break under miscalibration"


def test_b2_upper_bound_and_threshold_selectivity_dividend():
    """Criterion: for every c, realized waste never exceeds 1 - c (plus tol), and
    at c = 0.60 it is BELOW that bound by more than 0.05 (measured dividend
    0.221). The gap (1 - c) - W is named the threshold selectivity dividend.

    Why: every verified cell has cumprod >= c, so the window mean sits above the
    threshold and 1 - c is an upper bound rather than the expectation. The
    dividend is the mechanism by which a threshold buys waste control — it is
    promoted to a named L2 record quantity (user ruling 1).
    Flip: a bound violation means the window mask is wrong; a zero dividend means
    the selector stopped discriminating.
    """
    for c in sel.ANCHOR_C_GRID:
        out = sel.simulate_rule("fixed_c", 16, 6000, sel.GAMMA_SEED, c=c)["rule"]
        assert out["waste"] <= (1.0 - c) + sel.ANCHOR_IDENTITY_TOL, f"bound broken at c={c}"
    ref = sel.simulate_rule("fixed_c", 16, 6000, sel.GAMMA_SEED, c=0.60)["rule"]
    dividend = (1.0 - 0.60) - ref["waste"]
    assert dividend > 0.05, f"selectivity dividend collapsed: {dividend:.5f}"


def test_b3_global_gamma_drift_is_signed_and_ordered():
    """Criterion: the realized-minus-predicted waste gap is within tol at
    gamma = 1, positive at gamma = 0.5, negative at gamma = 1.5, and more
    negative at gamma = 2.0 than at 1.5.

    Why: under-confidence (gamma > 1) makes the head overstate its own waste and
    cut verification; over-confidence (gamma < 1) makes it understate waste and
    over-verify. The gap is the quantity upstream's batch_confidence_error
    monitors, so its sign structure is the milestone's reason to exist.
    Flip: either sign flipping, or the ordering inverting.
    """
    gaps = {
        g: sel.simulate_rule("fixed_c", 16, 6000, sel.GAMMA_SEED, c=0.60, gamma=(g, g, g))["rule"]["waste_gap"]
        for g in (0.5, 1.0, 1.5, 2.0)
    }
    assert abs(gaps[1.0]) <= sel.ANCHOR_IDENTITY_TOL
    assert gaps[0.5] > 0 > gaps[1.5]
    assert gaps[2.0] < gaps[1.5]


def test_bh_operability_criterion_holds_in_both_directions():
    """Criterion: min_k n*p_(k)/k predicts the smallest alpha at which BH rejects
    anything, to within the declared scan grid step; selection is empty strictly
    below it, non-empty above it, monotone in alpha, and non-empty on V2E at
    alpha = 0.20.

    Why: Ruling E forbids attaching a guarantee to BH here, so the honest
    statement is an operability criterion. It also corrects this lane's own first
    draft, which used n*p_(1) = 2.51 (the condition for rejecting only the most
    significant hypothesis) and concluded BH could never fire on V1; the measured
    first rejecting alpha is 0.4505.
    Flip: the closed form drifting from the empirical scan, or monotonicity
    breaking.
    """
    cum = sel.predicted_cumprod(_rate_block(16, 7))
    boundary = sel.bh_operability_boundary(cum)
    alphas = sel.BH_ALPHA_GRID
    first = next((float(a) for a in alphas if int(sel.bh_wins(float(a), cum).sum()) > 0), float("nan"))
    assert 0.0 <= first - boundary < sel.BH_ALPHA_GRID_STEP
    assert all(int(sel.bh_wins(a, cum).sum()) == 0 for a in sel.BH_INERT_ALPHAS_ON_V1)
    assert int(sel.bh_wins(boundary + 0.05, cum).sum()) > 0
    slots = [int(sel.bh_wins(a, cum).sum()) for a in sel.BH_MONOTONE_ALPHAS]
    assert all(lo <= hi for lo, hi in zip(slots, slots[1:]))
    v2e = sel.predicted_cumprod(_rate_block(16, 7, rn.V2E_LEVELS))
    assert int(sel.bh_wins(0.20, v2e).sum()) > 0
    first_hypothesis_boundary = cum.size * float((1.0 - cum).min())
    assert abs(first_hypothesis_boundary - first) > 0.05, "the wrong closed form would have passed"


def test_bh_selection_is_prefix_closed():
    """Criterion: wherever BH rejects, the rejected cells in each sequence are
    exactly its first wins positions, over 60 blocks at conc 16.

    Why: p = 1 - cumprod is monotone increasing in position, so any threshold on
    p selects a prefix — the allocator's output shape (one length per sequence,
    schedule_mtp_varlen.py:98) is preserved without extra machinery. If this
    failed, the rule would not be expressible in the engine's contract at all.
    Flip: a non-monotone p-value construction.
    """
    for s in range(60):
        cum = sel.predicted_cumprod(_rate_block(16, 300 + s))
        thr = sel.bh_threshold(0.60, cum)
        wins = sel.bh_wins(0.60, cum)
        if thr < 0.0:
            assert int(wins.sum()) == 0
            continue
        mask = (1.0 - cum) <= thr
        pos = np.arange(rates.NEXTN, dtype=np.int64)[None, :]
        assert np.array_equal(mask, pos < mask.sum(axis=1)[:, None]), "BH selection is not a prefix"


def test_hybrid_clamp_never_awards_ineligible_positions():
    """Criterion: across 120 rate blocks at total_q = 4 * num_seqs, the clamp
    form awards zero ineligible positions at EVERY c, and the two forms split
    exactly where the eligibility mask starts binding:

      c = 0.4 / 0.6 -> zeroing leaks in 0 sequences, clamp == zeroing in
                      120/120 blocks, mean slots 48.0 (under-spend 0.0%)
      c = 0.7        -> leak appears (113 sequences), 7/120 agree, 3.9% unspent
      c = 0.8 / 0.9  -> leak in 1080 / 1200 sequences, 0/120 agree,
                      33.0% / 68.9% of the nominal budget unspent

    Why: zeroed cells tie at cumprod 0.0 and take slack slots by flat index —
    the same defect class as the upstream zero-rate slack finding, reached from
    the other side of the codebase (user ruling 7). The clamp spends less than
    the nominal budget instead and reports the under-spend, so budget parity in
    L2 must be stated on realized length for the value-reading families.
    Flip: the clamp losing its min() (violations appear), or the survey losing
    the slack that makes the zeroing form diverge (then the comparison is blind).
    """
    tq = 16 * 4
    blocks = 120
    for c in (0.4, 0.6, 0.7, 0.8, 0.9):
        leak = agree = under_any = 0
        slot_totals = []
        for s in range(blocks):
            r = _rate_block(16, 1300 + s)
            types = rates.stratified_types(16, rates.TYPE_WEIGHTS)
            cum = sel.predicted_cumprod(sel.predicted_rates(r, sel.GAMMA_IDENTITY, types))
            elig = sel.eligible_wins(c, cum)
            clamp = sel.hybrid_wins(tq, c, r, cum)
            assert int(np.maximum(clamp - elig, 0).sum()) == 0, f"clamp awarded an ineligible position at c={c}"
            slot_totals.append(int(clamp.sum()))
            mask = np.arange(rates.NEXTN, dtype=np.int64)[None, :] < elig[:, None]
            zeroed = sel.rank_topk_wins(tq, np.where(mask, r, 0.0))
            leak += int((np.maximum(zeroed - elig, 0) > 0).sum())
            agree += int(np.array_equal(clamp, zeroed))
        under = 1.0 - float(np.mean(slot_totals)) / (tq - 16)
        if c <= 0.6:
            assert leak == 0 and agree == blocks and under == 0.0
        else:
            assert leak > 0, f"the zeroing mutant stopped leaking at c={c}: survey is blind"
            assert under > 0.0, f"under-spend not reported at c={c}"


def test_gamma_condition_declarations_and_axis_classification():
    """Criterion: GAMMA_TYPE_CONDITIONS holds exactly one identity vector, every
    entry positive, at least one entry below 1 in every non-identity condition,
    and the inert control is not part of the grid; gamma_kind classifies by the
    vector's own shape (all-equal is global, otherwise per-type).

    Why: Ruling D forbids conflating the two axes, and user ruling 4 requires the
    inflation direction to be represented because under-confidence is provably
    inert on the low type. Deriving the kind from the vector rather than from a
    passed label makes a mislabelled run impossible at emit time.
    Flip: an all-equal vector being classed per-type, or a grid drifting to
    gamma > 1 only.
    """
    grid = list(sel.GAMMA_TYPE_CONDITIONS)
    assert grid.count(sel.GAMMA_IDENTITY) == 1
    assert all(all(g > 0 for g in vec) for vec in grid)
    assert all(min(vec) < 1.0 for vec in grid if vec != sel.GAMMA_IDENTITY)
    assert sel.GAMMA_INERT_CONTROL not in grid
    assert sel.gamma_kind(sel.GAMMA_IDENTITY) == sel.GLOBAL_GAMMA
    assert sel.gamma_kind((2.0, 1.0, 1.0)) == sel.PER_TYPE_GAMMA
    assert sel.gamma_kind((0.5, 0.5, 0.5)) == sel.GLOBAL_GAMMA
    with pytest.raises(AssertionError):
        sel.gamma_kind((1.0, 1.0, 0.0))
    with pytest.raises(AssertionError):
        sel.gamma_kind((1.0, 1.0))
    out = sel.simulate_rule("quantile", 16, 50, sel.GAMMA_SEED, q=0.5, gamma=(1.5, 1.5, 1.5))
    assert out["gamma_kind"] == sel.GLOBAL_GAMMA


def test_theta_calibration_hits_the_anchor_matched_budget():
    """Criterion: bisection on c reaches the target mean verified length to
    within 0.05 draft positions, and an arbitrary uncalibrated c misses the same
    target by more than that.

    Why: value-reading rules emit a budget rather than consuming one, so
    comparison against the fixed-5 and top-k arms is only meaningful at matched
    realized length (user ruling 5 freezes c at the calibrated point on the
    reference seed). The second half proves the calibration earns its keep.
    Flip: monotonicity of verified length in c breaking, or the search saturating.
    """
    target = sel.ANCHOR_DYNAMIC_VERIFIED[16]
    c, realized = sel.calibrate_c_for_budget(target, 16, 1200, sel.GAMMA_SEED, iters=24)
    assert abs(realized - target) <= 0.05, f"calibration missed: c={c} realized={realized}"
    naive = sel.simulate_rule("fixed_c", 16, 1200, sel.GAMMA_SEED, c=0.5)["rule"]["mean_verified"]
    assert abs(naive - target) > 0.05, "an uncalibrated c hit the target: the test is vacuous"


def test_quantile_aim_matches_its_realized_budget():
    """Criterion: q = target / NEXTN realizes the target mean verified length to
    within 0.05 draft positions (q = 0.65 measures 3.250 at conc 16).

    Why: the relative family needs no bisection, which is why it is the natural
    budget-preserving competitor under per-type gamma. The realized value is
    still measured because prefix rounding makes admission approximate.
    Flip: the quantile interpolation convention drifting, or NEXTN changing.
    """
    q = sel.q_for_target(3.25)
    out = sel.simulate_rule("quantile", 16, 600, sel.GAMMA_SEED, q=q)["rule"]
    assert abs(out["mean_verified"] - 3.25) <= 0.05, f"q-aim realized {out['mean_verified']}"


def test_declared_gamma_conditions_have_bite_at_declared_budgets():
    """Criterion: the survey table over the six declared conditions x three
    budgets (total_q = conc * mult, 972 sequences per cell) is

                                  mult=3     mult=4   mult=5.44
      identity (1,1,1)             0          0          0
      (0.5,1,1)                  486          0          0
      (1,0.5,1)                  448        559          0
      (1,1,0.5)                  430        340        486
      (2,1,0.5)                  614        785        486
      (1.5,1.5,0.5)              486        656        486
      (1,1,2) inert control        0        146          0

    asserted as exact zeros where the row is inert and floors of 10% where it is
    live; every non-identity condition is live at at least one budget.

    Why: this is L2's grid-design constraint, discovered here — three of the six
    conditions are exactly inert at some budget, and (0.5,1,1) is live ONLY at
    the tight one. A sweep at the anchor-matched full budget alone would see the
    gamma axis vanish for four of six conditions, which would be reported as
    "calibration does not matter" instead of "the budget saturated the ranking".
    Flip: a budget grid narrowed to one point (rows go missing), or the type
    mixture homogenized (all rows go to zero).
    """
    live = []
    for vec in sel.GAMMA_TYPE_CONDITIONS[1:]:
        row = [_displacement(vec, mult=m) for m in (3.0, 4.0, 5.44)]
        assert any(ch > 0.10 * SEQS_SURVEYED for ch in row), f"{vec} is inert at every budget"
        live.append(tuple(0 if ch == 0 else 1 for ch in row))
    assert live[0] == (1, 0, 0), f"(0.5,1,1) bite pattern drifted: {live[0]}"
    assert live[2] == (1, 1, 1), f"(1,1,0.5) bite pattern drifted: {live[2]}"
    assert _displacement(sel.GAMMA_IDENTITY, mult=4.0) == 0
    assert _displacement(sel.GAMMA_INERT_CONTROL, mult=5.44) == 0


@pytest.mark.slow
def test_anchor_grid_across_all_concs_and_gamma_conditions():
    """Criterion: the B1 identity holds within tol at every anchor conc for the
    whole c grid at the full CALIB_STEPS horizon, and a global gamma leaves the
    top-k windows untouched at every conc under both the anchor-matched and the
    mult=4 budgets.

    Why: the tolerance was frozen on conc 16 and conc 96 only; this is the run
    that shows it holds at the four intermediate concs too, so L2 inherits a
    tolerance rather than a surprise. The global-gamma half re-checks C1 at the
    budgets L2 will actually sweep.
    Flip: any conc whose pooled estimator noise exceeds ANCHOR_IDENTITY_TOL.
    """
    for conc in CONCS:
        for c in sel.ANCHOR_C_GRID:
            out = sel.simulate_rule("fixed_c", conc, rates.CALIB_STEPS, sel.GAMMA_SEED, c=c)["rule"]
            assert abs(out["waste_gap"]) <= sel.ANCHOR_IDENTITY_TOL, f"B1 broken at conc {conc}, c={c}"
    for g in (0.5, 1.5, 2.0):
        vec = (g, g, g)
        for mult in (None, 4.0):
            assert _displacement(vec, mult=mult, blocks=2) == 0, f"global gamma={g} moved top-k"
