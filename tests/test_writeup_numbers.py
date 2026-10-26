"""Public prose may only carry artifact-derived numbers — on every surface.

Contract (HANDOFF_PROMPT_MS.md §5.3, extended by lane R ruling A):
  G0 structure    - WRITEUP.md keeps its seven sections, the Outcomes section its
                    six blocks, README.md its seven, and the release body its
                    generated-header provenance.
  G1 freshness    - results/writeup_numbers.json equals what the committed
                    artifacts yield right now, field by field, including which
                    surfaces each claim is published on.
  G2 completeness - every claim appears verbatim in each surface that declares it.
  G3 soundness    - no NOVEL numeric token appears in any policed surface.
                    G2 alone cannot catch an invented number: appending a
                    fabricated value leaves every real claim present, so G2
                    stays green. G3 is what makes the N5 mutant fire.
  G4 attribution  - the paired claims sit near the entity they belong to, on each
                    surface that publishes them.
                    G2/G3 police token identity, not slot correctness; G4 is
                    the defense against the entry-13 failure class (a real
                    number transcribed onto the wrong row).
  G5 sharing      - the reproduce block is byte-identical across the surfaces that
                    publish it, and the release body equals what the builder
                    renders from the current fixture.

Prose surfaces come from sim.build_writeup_numbers.prose_surfaces(): WRITEUP.md,
the report's Outcomes section, README.md, results/release_v1.0.0.md, CITATION.cff.
A surface that disappeared fails loudly instead of shrinking the census (N4).

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
    CITATION_PATH,
    EXEMPTIONS,
    METADATA_SURFACES,
    OUT_PATH,
    README_CLAIM_KEYS,
    README_PATH,
    RELEASE_CLAIM_KEYS,
    RELEASE_PATH,
    REPO_ROOT,
    TOKEN_RE,
    allow_list,
    derive,
    fence_after,
    fold,
    prose_surfaces,
    render_release_notes,
    section,
    strip_structural,
)

REPORT_PATH = REPO_ROOT / "research_DeepInfer_20261022.md"
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
README_SECTIONS = (
    "## What this is, and what it is not",
    "## Headline results",
    "## Quickstart",
    "## Repository map",
    "## Reproduce and cite",
)
README_ANCHOR = "# A published speculative-decoding speedup, audited on a laptop"
RELEASE_BLOCKS = (
    "## v1.0.0 — the audited front door",
    "### What the audit found",
    "### Reproduce",
    "### Guarded prose surfaces",
)


def _fixture() -> list[dict]:
    return json.loads(OUT_PATH.read_text(encoding="utf-8"))["entries"]


def _writeup() -> str:
    return (REPO_ROOT / "WRITEUP.md").read_text(encoding="utf-8")


def _surfaces() -> dict[str, str]:
    return prose_surfaces()


def _ordered(body: str, markers: tuple[str, ...], name: str) -> None:
    positions = [body.find(m) for m in markers]
    missing = [markers[i] for i, p in enumerate(positions) if p < 0]
    assert not missing, f"{name} lost sections: {missing}"
    assert positions == sorted(positions), f"{name} sections are out of order"


def test_g0_structure_of_every_prose_surface():
    """Criterion: WRITEUP.md carries the seven mandated sections in order, the
    Outcomes section its six blocks, README.md its title plus five mandated
    sections in order, and the release body its generated heading plus the three
    blocks the renderer emits.

    Why: these lanes' deliverables are fixed structures; a dropped section would
    otherwise be reported as done (MS §5.1-5.2, MR §5.4).
    Flip: any heading or block label removed or reordered, in any of the four.
    """
    _ordered(_writeup(), WRITEUP_SECTIONS, "WRITEUP.md")
    o = _surfaces()["Outcomes section"]
    missing = [b for b in OUTCOMES_BLOCKS if b not in o]
    assert not missing, f"Outcomes section lost blocks: {missing}"
    _ordered(README_PATH.read_text(encoding="utf-8"),
             (README_ANCHOR,) + README_SECTIONS, "README.md")
    _ordered(RELEASE_PATH.read_text(encoding="utf-8"), RELEASE_BLOCKS,
             "results/release_v1.0.0.md")


def test_g1_fixture_is_fresh_against_the_artifacts():
    """Criterion: the committed fixture equals a live re-derivation from the
    artifacts, field by field (including the per-surface publication set), every
    locator matched, and every declared surface name registered in the census.

    Why: the fixture is a snapshot; without re-derivation an edit to
    results/m1_report.md or results/m2_report.md would leave the guard certifying
    a stale contract (a fallback snapshot needs a freshness guard). The surfaces
    field is what decides where a number must appear, so a typo in a surface name
    would un-publish a number silently — containment makes that red.
    Flip: any artifact number or wrapping edited, any locator regex drifting, the
    fixture being hand-edited, or a claim declared on a surface that the census
    does not police.
    """
    entries, failures = derive()
    assert not failures, "\n".join(failures)
    committed = _fixture()
    fields = ("key", "kind", "printed", "tokens", "tier2", "surfaces")
    assert [ {k: e[k] for k in fields} for e in committed ] == [
        {k: e[k] for k in fields} for e in entries], (
        "results/writeup_numbers.json is stale — regenerate with "
        ".venv/bin/python -m sim.build_writeup_numbers")
    policed = set(prose_surfaces())
    undeclared = {e["key"]: sorted(set(e["surfaces"]) - policed) for e in committed
                  if set(e["surfaces"]) - policed}
    assert not undeclared, f"claims declared on surfaces nobody polices: {undeclared}"
    claims = [e for e in committed if e["kind"] == "claim"]
    assert len(claims) >= 30, f"claim set shrank to {len(claims)} — fixture is vacuous"
    assert len([e for e in claims if e["tier2"]]) >= 8, (
        "fewer than 8 paired claims carry attribution contexts")
    by_key = {e["key"]: e for e in committed}
    for name, keys in (("README_CLAIM_KEYS", README_CLAIM_KEYS),
                       ("RELEASE_CLAIM_KEYS", RELEASE_CLAIM_KEYS)):
        for key in sorted(keys):
            assert key in by_key, f"{name} names no locator: {key}"
            assert by_key[key]["kind"] == "claim", f"{name} {key} is not a claim"


def test_g2_artifact_claims_appear_verbatim():
    """Criterion: every claim string from the fixture appears verbatim in each
    prose surface that declares it.

    Why: MS §5.3's contract — headline numbers are transcribed by rule from the
    committed artifacts, never from memory. Ruling A extends it to the front door,
    per surface: README.md and the release body publish a declared subset, so a
    global "all claims everywhere" reading would be unsatisfiable for a one-screen
    landing page (this session's ruling 2).
    Flip: a claim dropped from a surface that declares it, or an artifact value
    edited without regenerating the fixture and the prose.
    """
    prose = {name: fold(text) for name, text in _surfaces().items()}
    offenders = {}
    for name, body in prose.items():
        missing = [e["printed"] for e in _fixture()
                   if e["kind"] == "claim" and name in e["surfaces"]
                   and e["printed"] not in body]
        if missing:
            offenders[name] = missing
    assert not offenders, f"claims absent verbatim from the prose: {offenders}"
    for name in prose:
        if name in METADATA_SURFACES:
            continue
        assert any(name in e["surfaces"] for e in _fixture() if e["kind"] == "claim"), (
            f"{name} is policed but declares no claim — the census would certify an "
            "empty surface")


def test_g3_no_novel_numeric_token_in_prose():
    """Criterion: after stripping structural spans, every numeric token in every
    policed prose surface is either captured from a committed artifact or listed
    in EXEMPTIONS with a reason.

    Why: this is the only check that can fail for an INVENTED number — G2 proves
    presence of the real ones and is insensitive to additions (N5 mutant 1). It
    now covers README.md, the release body and CITATION.cff as well, because a
    repository that audits its numbers cannot publish unguarded ones (Ruling A).
    Flip: any new figure typed into any published surface, or an exemption whose
    token stops appearing (dead weight must be deleted, not carried).
    """
    allowed = allow_list(_fixture())
    surfaces = _surfaces()
    offenders = {}
    for name, raw in surfaces.items():
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
                   for raw in surfaces.values())
        assert used, (
            f"exemption {ex['token']!r} is dead weight: its token appears in "
            "no policed surface — delete it")


def _paired_publications() -> list[tuple[str, str]]:
    """(claim key, surface) pairs: every paired claim on every surface publishing it."""
    return [(e["key"], name) for e in _fixture()
            if e["kind"] == "claim" and e["tier2"] for name in e["surfaces"]]


@pytest.mark.parametrize("key,surface", _paired_publications())
def test_g4_paired_claims_are_attributed(key: str, surface: str):
    """Criterion: each paired claim has at least one occurrence whose
    +-180-character window matches its attribution regex, on every surface that
    publishes it.

    Why: token identity is not slot correctness. The entry-13 correction chain
    (26.4x vs 27.4x, computed from the neighbouring row's cell) is exactly this
    failure class: a real number attached to the wrong generator/concurrency.
    Ruling A means the front door is not exempt: README.md gets its own window.
    Flip: a claim moved next to a different entity on any surface, or its
    locator's tier2 regex tightened/degraded.
    """
    entry = next(e for e in _fixture() if e["key"] == key)
    value, ctx = entry["printed"], re.compile(entry["tier2"])
    body = fold(_surfaces()[surface])
    windows = []
    at = body.find(value)
    while at >= 0:
        windows.append(body[max(0, at - TIER2_WINDOW): at + len(value) + TIER2_WINDOW])
        at = body.find(value, at + 1)
    assert windows, f"{key} ({value!r}) never appears in {surface}"
    assert any(ctx.search(w) for w in windows), (
        f"{key}: {value!r} appears in {surface} but never near its attribution "
        f"{entry['tier2']!r} — real number, wrong slot")


def _fences(body: str, heading: str, name: str) -> str:
    return fence_after(section(body, heading, name), name)


def test_g5_reproduce_fence_is_shared_not_retyped():
    """Criterion: README.md's Quickstart fence and WRITEUP.md §6's fence are the
    same ten lines, byte for byte, and both are the first fence in their section.

    Why: MR §5.4 publishes one reproduce path on two surfaces (and a third in the
    release body). A reader who copy-pastes from the landing page must get the
    commands the writeup verified; a hand-copied fence drifts silently — a
    `python3` becoming `python`, a pin losing its `==`. Comparing the two fences
    makes drift a test failure rather than a documentation bug.
    Flip: either fence edited alone, or a second fenced block appearing earlier
    in either section.
    """
    from_readme = _fences(README_PATH.read_text(encoding="utf-8"),
                          "## Quickstart", "README.md")
    from_writeup = _fences(_writeup(), "## 6. Reproduce it yourself", "WRITEUP.md")
    assert from_readme == from_writeup, "the reproduce fences drifted apart"
    lines = from_readme.splitlines()
    assert len(lines) == 10, f"expected the shared 10-line fence, saw {len(lines)}"
    assert lines[0] == "```bash" and lines[-1] == "```", "fence markup moved"

    release = _fences(RELEASE_PATH.read_text(encoding="utf-8"),
                      "### Reproduce", "results/release_v1.0.0.md")
    assert release == from_writeup, "the release body's fence drifted from §6"


def test_g5_release_body_matches_the_builder():
    """Criterion: the committed results/release_v1.0.0.md equals what
    render_release_notes() produces from a live re-derivation right now.

    Why: Ruling B makes the release notes a derived surface. A hand-edit to the
    published tag body would otherwise be invisible to every other guard — this
    pins it to the builder, the same way G1 pins the fixture to the artifacts.
    Flip: any edit to the release file that the builder did not make, or a fixture
    regeneration that changes a claim string.
    """
    entries, failures = derive()
    assert not failures, "\n".join(failures)
    assert RELEASE_PATH.read_text(encoding="utf-8") == render_release_notes(entries), (
        "results/release_v1.0.0.md is stale — regenerate with "
        ".venv/bin/python -m sim.build_writeup_numbers")


def test_g3_citation_surface_keeps_its_structural_identifiers_alive():
    """Criterion: CITATION.cff's three exempted tokens are exactly
    cff-version/version/year, and each still appears in that file.

    Why: CITATION.cff is machine-readable metadata, not a claim surface, yet it is
    policed (Ruling A). Without this pin a fourth exemption could be added for a
    real published number and dressed up as "metadata".
    Flip: an exemption added for anything other than these three identifiers, the
    identifiers renamed out of the file, or METADATA_SURFACES grown to smuggle a
    claim-bearing surface past the anti-vacuity rule in G2.
    """
    assert METADATA_SURFACES == frozenset({"CITATION.cff"}), (
        f"METADATA_SURFACES changed: {sorted(METADATA_SURFACES)} — a surface that "
        "publishes numbers belongs in the claim set, not here")
    cite_tokens = {x["token"] for x in EXEMPTIONS} & {"1.0.0", "1.2.0", "2026"}
    assert cite_tokens == {"1.0.0", "1.2.0", "2026"}, sorted(cite_tokens)
    raw = CITATION_PATH.read_text(encoding="utf-8")
    found = set(TOKEN_RE.findall(strip_structural(raw)))
    assert cite_tokens <= found, f"CITATION.cff no longer carries {cite_tokens - found}"
    assert not (found - cite_tokens), (
        f"CITATION.cff carries unexplained numbers: {sorted(found - cite_tokens)}")

