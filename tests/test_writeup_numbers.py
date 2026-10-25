"""Lane MS-L1 guards: the public writeup may only carry artifact-derived numbers.

Contract (HANDOFF_PROMPT_MS.md §5.3, extended by this session's rulings 1-2):
  G1 freshness    - results/writeup_numbers.json equals what the committed
                    artifacts yield right now.
  G2 completeness - every claim appears verbatim in WRITEUP.md and in the
                    report's Outcomes section.
  G3 soundness    - no NOVEL numeric token appears in either prose surface.
                    G2 alone cannot catch an invented number: appending a
                    fabricated value leaves every real claim present, so G2
                    stays green. G3 is what makes the N5 mutant fire.
  G4 attribution  - the paired claims sit near the entity they belong to.
                    G2/G3 police token identity, not slot correctness; G4 is
                    the defense against the entry-13 failure class (a real
                    number transcribed onto the wrong row).

Prose is whitespace-folded before matching so a rewrap in either the artifact or
the writeup cannot silently break the guard; structural spans (git shas, ISO
dates, file:line citations, section refs, heading ordinals) are stripped before
the token scan because they are locators, not claims.
"""

from __future__ import annotations

import json
import re

import pytest

from sim.build_writeup_numbers import (
    EXEMPTIONS,
    OUT_PATH,
    REPO_ROOT,
    TOKEN_RE,
    allow_list,
    derive,
    fold,
    strip_structural,
)

WRITEUP_PATH = REPO_ROOT / "WRITEUP.md"
REPORT_PATH = REPO_ROOT / "research_DeepInfer_20261022.md"
OUTCOMES_HEADING = "## 10. Outcomes (2026-10-24)"
TIER2_WINDOW = 180

WRITEUP_SECTIONS = (
    "## 1. What was audited, and with what",
    "## 2. The mechanism, as read from the open-source engine",
    "## 3. Milestone 1 — replaying the anchor table",
    "## 4. Milestone 2 — how good must the ranking be?",
    "## 5. What this does NOT prove",
    "## 6. Reproduce it yourself",
    "## 7. Attribution",
)
OUTCOMES_BLOCKS = (
    "**M1 — anchor-table replay.**",
    "**M2 — rank sufficiency.**",
    "**Tension attribution.**",
    "**Reproducibility.**",
    "**Artifacts.**",
    "**Evidence boundary.**",
)


def _fixture() -> list[dict]:
    return json.loads(OUT_PATH.read_text(encoding="utf-8"))["entries"]


def _writeup() -> str:
    return WRITEUP_PATH.read_text(encoding="utf-8")


def _outcomes() -> str:
    text = REPORT_PATH.read_text(encoding="utf-8")
    idx = text.find(OUTCOMES_HEADING)
    assert idx >= 0, f"Outcomes section missing from {REPORT_PATH.name}"
    return text[idx:]


def _surfaces() -> dict[str, str]:
    return {"WRITEUP.md": _writeup(), "Outcomes section": _outcomes()}


def test_g0_structure_of_both_artifacts():
    """Criterion: WRITEUP.md carries the seven mandated sections in order and the
    Outcomes section carries the six mandated blocks (MS §5.1-5.2).

    Why: the lane's deliverable is a fixed structure; a dropped section would
    otherwise be reported as done.
    Flip: any section heading or block label removed or reordered.
    """
    w = _writeup()
    positions = [w.find(s) for s in WRITEUP_SECTIONS]
    assert all(p >= 0 for p in positions), [
        WRITEUP_SECTIONS[i] for i, p in enumerate(positions) if p < 0]
    assert positions == sorted(positions), "WRITEUP.md sections are out of order"
    o = _outcomes()
    missing = [b for b in OUTCOMES_BLOCKS if b not in o]
    assert not missing, f"Outcomes section lost blocks: {missing}"


def test_g1_fixture_is_fresh_against_the_artifacts():
    """Criterion: the committed fixture equals a live re-derivation from the
    artifacts, field by field, and every locator matched.

    Why: the fixture is a snapshot; without re-derivation an edit to
    results/m1_report.md or results/m2_report.md would leave the guard certifying
    a stale contract (a fallback snapshot needs a freshness guard).
    Flip: any artifact number or wrapping edited, any locator regex drifting, or
    the fixture being hand-edited.
    """
    entries, failures = derive()
    assert not failures, "\n".join(failures)
    committed = _fixture()
    fields = ("key", "kind", "printed", "tokens", "tier2")
    assert [ {k: e[k] for k in fields} for e in committed ] == [
        {k: e[k] for k in fields} for e in entries], (
        "results/writeup_numbers.json is stale — regenerate with "
        ".venv/bin/python -m sim.build_writeup_numbers")
    claims = [e for e in committed if e["kind"] == "claim"]
    assert len(claims) >= 30, f"claim set shrank to {len(claims)} — fixture is vacuous"
    assert len([e for e in claims if e["tier2"]]) >= 8, (
        "fewer than 8 paired claims carry attribution contexts")


def test_g2_artifact_claims_appear_verbatim():
    """Criterion: every claim string from the fixture appears verbatim in
    WRITEUP.md and in the Outcomes section.

    Why: MS §5.3's contract — headline numbers are transcribed by rule from the
    committed artifacts, never from memory.
    Flip: a claim dropped from either prose surface, or an artifact value edited
    without regenerating the fixture and the prose.
    """
    prose = {name: fold(text) for name, text in _surfaces().items()}
    missing = {
        name: [e["printed"] for e in _fixture()
               if e["kind"] == "claim" and e["printed"] not in body]
        for name, body in prose.items()
    }
    offenders = {k: v for k, v in missing.items() if v}
    assert not offenders, f"claims absent verbatim from the prose: {offenders}"


def test_g3_no_novel_numeric_token_in_prose():
    """Criterion: after stripping structural spans, every numeric token in
    WRITEUP.md and in the Outcomes section is either captured from a committed
    artifact or listed in EXEMPTIONS with a reason.

    Why: this is the only check that can fail for an INVENTED number — G2 proves
    presence of the real ones and is insensitive to additions (N5 mutant 1).
    Flip: any new figure typed into the prose, or an exemption whose token stops
    appearing (dead weight must be deleted, not carried).
    """
    allowed = allow_list(_fixture())
    offenders = {}
    for name, raw in _surfaces().items():
        novel = sorted(set(TOKEN_RE.findall(strip_structural(raw))) - allowed)
        if novel:
            offenders[name] = novel
    assert not offenders, (
        f"numbers not derived from a committed artifact: {offenders} — quote an "
        "artifact value (add a locator) or, for a structural identifier, add a "
        "reasoned exemption in sim/build_writeup_numbers.py")
    for ex in EXEMPTIONS:
        assert ex["reason"].strip(), f"exemption {ex['token']} has no reason"
        used = any(ex["token"] in TOKEN_RE.findall(strip_structural(raw))
                   for raw in _surfaces().values())
        assert used, (
            f"exemption {ex['token']!r} is dead weight: its token appears in "
            "neither prose surface — delete it")


def _paired_claim_keys() -> list[str]:
    return [e["key"] for e in _fixture() if e["kind"] == "claim" and e["tier2"]]


@pytest.mark.parametrize("key", _paired_claim_keys())
def test_g4_paired_claims_are_attributed(key: str):
    """Criterion: each paired claim has at least one occurrence whose
    +-180-character window matches its attribution regex, in BOTH surfaces.

    Why: token identity is not slot correctness. The entry-13 correction chain
    (26.4x vs 27.4x, computed from the neighbouring row's cell) is exactly this
    failure class: a real number attached to the wrong generator/concurrency.
    Flip: a claim moved next to a different entity, or its locator's tier2
    regex tightened/degraded.
    """
    entry = next(e for e in _fixture() if e["key"] == key)
    value, ctx = entry["printed"], re.compile(entry["tier2"])
    for name, raw in _surfaces().items():
        body = fold(raw)
        windows = []
        at = body.find(value)
        while at >= 0:
            windows.append(body[max(0, at - TIER2_WINDOW): at + len(value) + TIER2_WINDOW])
            at = body.find(value, at + 1)
        assert windows, f"{key} ({value!r}) never appears in {name}"
        assert any(ctx.search(w) for w in windows), (
            f"{key}: {value!r} appears in {name} but never near its attribution "
            f"{entry['tier2']!r} — real number, wrong slot")
