"""M2-L3 report guards: every stated count/range must equal the value parsed
from the committed results/sweep_m2.md (Ruling 1 discipline)."""

from pathlib import Path

from sim.build_m2_report import REPO_ROOT, parse_sweep

REPORT_PATH = REPO_ROOT / "results" / "m2_report.md"
SWEEP_PATH = REPO_ROOT / "results" / "sweep_m2.md"


def _report_text() -> str:
    return REPORT_PATH.read_text(encoding="utf-8")


def test_m2_report_sections():
    """Criterion: the report carries the five mandated sections (M2 §5.3).
    Flip: any section removal."""
    text = _report_text()
    for marker in (
        "## 1. Question and mechanism recap",
        "## 2. Machinery, window-length semantics, and the reachable ranking-quality band",
        "## 3. Curves and thresholds",
        "## 4. M1-tension diagnosis",
        "## 5. What this does NOT prove",
    ):
        assert marker in text, f"m2_report.md lost section: {marker}"


def test_m2_report_counts_match_parsed_artifact():
    """Criterion: the report's FINDING count (24 = 18 V1 + 6 V2E, all at
    sigma 6.4 -> 12.8) equals the count parsed from the committed sweep.

    Why: Ruling 1b corrected the count; chat-memory numbers are forbidden.
    Flip: the committed sweep regenerates with different findings, or the
    report's stated count drifts from the artifact.
    """
    parsed = parse_sweep(SWEEP_PATH.read_text(encoding="utf-8"))
    findings = parsed["findings"]
    n_v1 = sum(1 for f in findings if f.startswith("- V1 "))
    n_v2 = sum(1 for f in findings if f.startswith("- V2E "))
    all_6412 = all("sigma 6.4 -> 12.8" in f for f in findings)
    text = _report_text()
    assert f"**{len(findings)} FINDING cells ({n_v1} V1 + {n_v2} V2E)**" in text
    assert (len(findings), n_v1, n_v2) == (24, 18, 6)
    assert all_6412


def test_m2_report_tension_numbers_match_parsed_artifact():
    """Criterion: the tension-diagnosis numbers equal the parsed ratio table
    and the frozen anchor constant — exact construction, never bare ratios
    (M2-L1 acceptance Ruling 2).

    Flip: any drift between the report's stated values and
    results/sweep_m2.md / sim.cost_model.ANCHOR.
    """
    from sim.cost_model import ANCHOR

    parsed = parse_sweep(SWEEP_PATH.read_text(encoding="utf-8"))
    v1_ratio, v2_ratio, _ = parsed["ratio"][(96, 0.0)]
    anchor = ANCHOR[96][2] / ANCHOR[96][0]
    text = _report_text()
    assert f"3.15/3.24 = {anchor:.4f}" in text
    assert f"V1 = {v1_ratio:.4f}" in text
    assert f"V2E = {v2_ratio:.4f}" in text
    closed_pct = (v2_ratio - v1_ratio) / (anchor - v1_ratio) * 100.0
    assert f"**{closed_pct:.1f}%**" in text
    assert (round(anchor, 4), v1_ratio, v2_ratio) == (0.9722, 0.8198, 0.8747)


def test_m2_report_multiplier_rows_complete():
    """Criterion: the report's multiplier list contains exactly the
    both-positive rows with both parsed gains (Ruling 1c: row-by-row, both
    values) — 14 rows; the conc-32 pair reaches 27.4x / 6.7x.

    Flip: a row added/dropped, or a value drifting from the parsed artifact.
    """
    parsed = parse_sweep(SWEEP_PATH.read_text(encoding="utf-8"))
    expected = []
    for sk in ("primary", "band_edge", "second"):
        for conc in (16, 32, 48, 60, 72, 96):
            v1 = parsed["sigma0_gain"].get(("V1", sk, conc))
            v2 = parsed["sigma0_gain"].get(("V2E", sk, conc))
            if v1 is not None and v2 is not None and v1 > 0 and v2 > 0:
                expected.append((sk, conc, v1, v2))
    text = _report_text()
    for sk, conc, v1, v2 in expected:
        marker = f"- {sk} conc {conc}: V2E {v2:+.2f}% vs V1 {v1:+.2f}%"
        assert marker in text, f"multiplier row missing or drifted: {marker}"
    assert len(expected) == 14
    assert "only the two conc-32 rows reach 6.7-27.4x" in text
