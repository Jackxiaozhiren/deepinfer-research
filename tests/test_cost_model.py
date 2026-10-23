"""Cost-model tests (Lane L2). N6 criteria + the linear-exclusion negative control."""

import numpy as np

from sim import cost_model as cm


def test_cost_models_monotone_in_v():
    """Criterion: every family is strictly increasing in v >= 0 and T(0) = a.

    Why: more packed rows cannot take less step time; the families are
    defined with positive coefficients on non-decreasing terms.
    Flip: negative b/d, c <= 0, or a reordered family definition.
    """
    v = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])
    for a, b in ((1.0, 0.01), (100.0, 1.0), (2676.7, 1.0)):
        for c in (0.9, 1.0, 1.1, 1.3, 1.5):
            vals = cm.t_pow(v, a, b, c)
            assert bool(np.all(np.diff(vals) > 0))
            assert vals[0] == a
        lin = cm.t_linear(v, a, b)
        assert bool(np.all(np.diff(lin) > 0))
        assert lin[0] == a
        for beta in (0.0, 0.5, 2.0):
            sat = cm.t_sat(v, a, b, beta * b / 500.0)
            assert bool(np.all(np.diff(sat) > 0))
            assert sat[0] == a


def test_linear_family_excluded_by_anchor_ratios():
    """NEGATIVE CONTROL (ledger entry 3 Next; user ruling 2026-10-22): the
    pure-linear family cannot reproduce the anchor's conc-96 gain at the
    unique a/b implied by the conc-16 row, under EITHER reading of the
    accepted-length statistic.

    Why: with v = conc * rows/seq (fresh row included), row 16 pins a/b at
    about 2676.7 (emitted = k+1) or 910.9 (emitted = k); the linear
    prediction for row 96 (1.0548 / 1.1282) falls short of the required
    1.1497 / 1.1625, so no linear (a, b) fits both rows.
    Flip: any reading reporting excluded=False puts the linear family back on
    the table — stop and investigate before proceeding.

    Discriminating power (N5): the same checker at c = 1.3 — the convex main
    family the exclusion motivates — must NOT exclude (verdict flips),
    proving the control can fire in both directions.
    """
    res = cm.exclude_linear_family(c=1.0)
    assert res["ok"], f"linear family not excluded: {res}"
    for name, r in res["readings"].items():
        assert r["excluded"], f"{name}: linear not excluded: {r}"
        assert r["predicted96"] < r["required96"], f"{name}: {r}"
    res13 = cm.exclude_linear_family(c=1.3)
    assert not res13["ok"], f"c=1.3 wrongly excluded — checker cannot discriminate: {res13}"


def test_cost_model_declares_inference_assumption_and_plausible_regions():
    """Criterion: the module docstring carries the INFERENCE ASSUMPTION
    declaration (throughput = num_seqs * E[emitted] / T(v)) and the
    plausible-regions contract (user ruling 2; HANDOFF §7.2).

    Flip: either declaration removed from the docstring. The match runs on
    whitespace-normalized text so rewrapping cannot blind the guard.
    """
    doc = " ".join((cm.__doc__ or "").split())
    assert "INFERENCE ASSUMPTION" in doc
    assert "Plausible regions" in doc
    assert "num_seqs * E[emitted] / T(v)" in doc
