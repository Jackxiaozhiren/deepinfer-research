"""Lane L3 freeze-rule tests. N6 criteria + N5 mutants; freeze-phase scope."""

from pathlib import Path

from sim import replay_qa
from sim.sweep_l2 import _classify  # shared sign classifier with the committed sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SLICE = (464.15888336127773, 0.9)


def test_slice_rule_regression():
    """Criterion: the declared freeze rule on fresh conc-16 paired sims selects
    (a/b = 464.15888336127773, c = 0.9).

    Why: the rule (t_pow only; conc-16 bootstrap CI wholly < 0; argmin
    |mean_gain16 - (4816/4891 - 1)|) is executable from selection-set
    information alone; the committed L2 sign map places that point at
    mean_gain16 = -1.60%, |delta| = 0.10pp — the closest survivor to the
    anchor's -1.5%.
    Flip: any change to the seeds, the cost grid, the CI filter, or the
    target moves the selection and this test goes red.
    """
    out = replay_qa.select_slice()
    assert out["slice"] == FROZEN_SLICE, (
        f"rule selected {out['slice']}, expected the frozen slice {FROZEN_SLICE}"
    )


def test_slice_rule_mutant_maximize():
    """N5 mutant: maximizing the distance to the target (deliberately wrong
    rule) must select a DIFFERENT slice — the rule is load-bearing.

    Flip: if the wrong rule still selected the frozen slice, the regression
    test could not detect a corrupted rule.
    """
    out = replay_qa.select_slice(maximize=True)
    assert out["slice"] != FROZEN_SLICE, "argmax mutant selected the frozen slice"


def test_slice_rule_mutant_flipped_target():
    """N5 mutant: flipping the target sign (aiming at +1.5% instead of -1.5%)
    must select a DIFFERENT slice among the CI-negative survivors."""
    out = replay_qa.select_slice(target=-replay_qa.FROZEN_TARGET)
    assert out["slice"] != FROZEN_SLICE, "flipped-target mutant selected the frozen slice"


def test_sign_classifier_matches_committed_convention():
    """Criterion: the shared classifier marks a row '-' iff the CI is wholly
    below zero, '+' iff wholly above, '.' otherwise — the same convention as
    the committed results/sweep_l2.md sign maps.

    Flip: any change to the classification boundaries.
    """
    assert _classify(-0.02, -0.03, -0.01) == "-"
    assert _classify(0.02, 0.01, 0.03) == "+"
    assert _classify(0.0, -0.01, 0.01) == "."
    assert _classify(-0.005, -0.01, 0.0) == "."


def test_committed_sign_map_counts():
    """Criterion: the committed L2 sign map contains 75 t_pow rows (15 a/b
    values x 5 exponents) and 30 saturation rows (10 a/b x 3 beta) — the only
    counts the m1 report may cite (ruling 1: counts must be derivable from
    the committed artifact).

    Flip: a grid change in sim/sweep_l2.py or results/sweep_l2.md that is not
    reflected in the report's stated counts.
    """
    text = (REPO_ROOT / "results" / "sweep_l2.md").read_text(encoding="utf-8")
    counts = replay_qa.count_sign_map_rows(text)
    assert counts == {"t_pow": 75, "saturation": 30}, f"committed sign-map counts changed: {counts}"


def test_slice_guard_rejects_wrong_and_accepts_frozen():
    """Criterion: check_slice passes the frozen slice and exits loudly on any
    other — both directions asserted (an honest input must pass).

    Why: score re-derives the slice and refuses to run against a ledger value
    that disagrees with the rule. Flip: a guard that accepts wrong slices or
    rejects the frozen one.
    """
    replay_qa.check_slice(FROZEN_SLICE, FROZEN_SLICE)
    try:
        replay_qa.check_slice(FROZEN_SLICE, (465.0, 1.1))
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("check_slice accepted a slice that is not the frozen one")


def test_m1_report_structure_and_counts():
    """Criterion: results/m1_report.md carries the five mandated sections, the
    exact accepted/verified tension construction (ruling 2), and sign-map
    counts equal to the committed artifact's (ruling 1).

    Flip: a section removal, a bare-ratio restatement of the tension, or a
    count that no longer matches the parsed committed map.
    """
    text = (REPO_ROOT / "results" / "m1_report.md").read_text(encoding="utf-8")
    for marker in (
        "## 1. Convention derivation summary",
        "## 2. Calibration table",
        "## 3. Replay table (frozen slice)",
        "## 4. Sensitivity summary — where the family reproduces the anchor's sign flip",
        "## 5. What this does NOT prove",
        "3.15/3.24 = 0.9722",
        "2.656/3.240 = 0.8198",
        "75 t_pow cost points",
        "30 saturation points",
    ):
        assert marker in text, f"results/m1_report.md lost marker: {marker}"
