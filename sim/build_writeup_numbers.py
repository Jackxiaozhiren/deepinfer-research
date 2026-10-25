"""Lane MS-L1: derive the headline numeric strings for the public writeup.

Every entry in results/writeup_numbers.json is regex-captured from a COMMITTED
artifact (results/m1_report.md, results/m2_report.md, RESULTS_LEDGER.md) —
nothing is retyped from chat memory (HANDOFF_PROMPT_MS.md §5.3; N3/N8).

Curation lives in the locator list: each locator names an artifact, a heading
that scopes the search, and a regex whose capture group is the printed claim.
A locator that stops matching fails loudly (N4): `<FAILED locator ...>` lines
plus a nonzero exit, so a silently shrinking list is impossible.

Entry kinds:
  claim   - a headline number. G2 requires it verbatim in WRITEUP.md AND in the
            report's Outcomes section; optional tier2 context (G4).
  context - an artifact clause that is not quoted in prose; its numeric tokens
            feed G3's allow-list only.

Division of labor (approved rulings 1-2 of this session):
  G2 proves a derived value APPEARS (completeness).
  G3 proves no NOVEL numeric spelling appears (soundness) - it polices novel
    tokens, not a legitimate token reused in a wrong slot.
  G4 closes that slot-attribution gap for the paired claims by requiring a
    context regex within the same window as the value.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "m1": REPO_ROOT / "results" / "m1_report.md",
    "m2": REPO_ROOT / "results" / "m2_report.md",
    "sweep": REPO_ROOT / "results" / "sweep_m2.md",
    "sweep_code": REPO_ROOT / "sim" / "sweep_m2.py",
    "ledger": REPO_ROOT / "RESULTS_LEDGER.md",
}
OUT_PATH = REPO_ROOT / "results" / "writeup_numbers.json"

M1_S3 = "## 3. Replay table (frozen slice)"
M1_S4 = "## 4. Sensitivity summary — where the family reproduces the anchor's sign flip"
M1_S2 = "## 2. Calibration table"
M1_S1 = "## 1. Convention derivation summary"
M2_S2 = "## 2. Machinery, window-length semantics, and the reachable ranking-quality band"
M2_S3 = "## 3. Curves and thresholds (all counts parsed from results/sweep_m2.md)"
M2_S4 = "## 4. M1-tension diagnosis — does ranking quality close the gap?"
M2_S5 = "## 5. What this does NOT prove"
LEDGER_15 = ("## 2026-10-24 — Milestone-S lane MS-L1 session start / "
             "§4 baseline capture (entry 15)")

# Numeric cores, unsigned: 36.0% / 0.9722 / 75 / 3.12.14 each read as one token.
TOKEN_RE = re.compile(r"\d+(?:\.\d+)*%?")
# Artifact prose is hard-wrapped; claims are stored and compared whitespace-folded
# so a rewrap in either the artifact or the writeup cannot silently break the guard.
FOLD_RE = re.compile(r"\s+")
# Spans removed before the G3 token scan (structural, not claims).
STRIP_RES = (
    re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,8}\b"),  # git commit shas (letters only)
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),      # ISO dates
    re.compile(r"[\w./-]+:\d+(?:-\d+)?"),      # file:line citations (locators, not claims)
    re.compile(r"§\d+"),                        # section references
    re.compile(r"^#{2,4} \d+\.", re.MULTILINE),  # heading ordinals
)


@dataclass(frozen=True)
class Locator:
    key: str
    artifact: str
    heading: str | None
    pattern: str
    kind: str = "claim"          # claim | context
    tier2: str | None = None     # G4 context regex, required near the value
    multi: bool = False          # context: findall + join


LOCATORS: tuple[Locator, ...] = (
    # ---- M1 convention + calibration -------------------------------------
    Locator("emit_convention", "m1", M1_S1, r"emits \*\*(k\+1)\*\* tokens",
            tier2=r"EMIT_BONUS|accepted prefix|bonus"),
    Locator("calib_mean", "m1", M1_S2, r"mean = (\d+(?:\.\d+)?)"),
    Locator("anchor_gain16", "m1", M1_S3, r"^\| 16 \| ([-+][\d.]+%)",
            tier2=r"conc\S* ?16|\banchor\b"),
    Locator("anchor_gain96", "m1", M1_S3, r"^\| 96 \| ([-+][\d.]+%)",
            tier2=r"conc\S* ?96"),
    Locator("model_gain16", "m1", M1_S3, r"^\| 16 \| [-+][\d.]+% \| - \| ([-+][\d.]+%) \["),
    Locator("anchor_ratio96", "m1", M1_S3, r"= (0\.9722)", tier2=r"3\.15/3\.24"),
    Locator("model_ratio_v1", "m1", M1_S3, r"= (0\.8198)", tier2=r"V1"),
    Locator("calib_band_lo", "m1", M1_S2, r"mean across rows in \[([\d.]+),"),
    Locator("calib_band_hi", "m1", M1_S2, r"mean across rows in \[[\d.]+, ([\d.]+)\]"
            , tier2=r"calibrat|fixed-5"),
    Locator("gen_verified_dyn16", "m1", M1_S3, r"^\| 16 \|.*\| ([\d.]+) \| [\d.]+ / [\d.]+ \|",
            tier2=r"verif"),
    Locator("gen_acc_dyn16", "m1", M1_S3, r"^\| 16 \|.*\| [\d.]+ \| ([\d.]+) / "),
    Locator("gen_acc_fix16", "m1", M1_S3, r"^\| 16 \|.*\| [\d.]+ / ([\d.]+) \|"),
    Locator("model_ci16", "m1", M1_S3, r"-1\.60% (\[-1\.62, -1\.59\])", kind="context"),
    Locator("conc16_inband_lo", "m1", M1_S4, r"conc-16 lands at ([-\d.]+%) to", tier2=r"conc\S* ?16"),
    Locator("conc16_inband_hi", "m1", M1_S4, r"conc-16 lands at [-\d.]+% to ([-\d.]+%)"),
    Locator("l2_tpow_points", "m1", M1_S4, r"evaluates (\d+) t_pow cost points", tier2=r"t_pow"),
    Locator("l2_sat_points", "m1", M1_S4, r"and (\d+) saturation points", tier2=r"saturation"),
    Locator("seed_gen", "m1", M1_S2, r"(seed 20261022,\s*horizon 6000)"),
    Locator("seed_bootstrap", "m1", None, r"(bootstrap seed 20261101, 10000 resamples; horizon 6000 steps)",
            kind="context"),
    Locator("seed_i", "m1", None, r"(i in \[0, 12\))", kind="context"),
    Locator("m1_title", "m1", None, r"(Milestone 1 report)", kind="context"),
    Locator("fixed5", "m1", M1_S2, r"(Fixed-5 accepted length k per concurrency row)", kind="context"),
    Locator("l2_counts_ctx", "m1", M1_S4,
            r"\(15 a/b values x 5 exponents\) and 30 saturation points \(10 x 3\)", kind="context"),
    Locator("repro_cells", "m1", M1_S4,
            r"\(\(465\.0, 1\.1\),\s*\(1550\.0, 1\.3\), \(4900\.0, 1\.5\)", kind="context"),
    Locator("replay_concs", "m1", M1_S3, r"^\| (16|32|48|60|72|96) \|", kind="context", multi=True),
    Locator("replay_rows", "m1", M1_S3,
            r"^\| (?:16|32|48|60|72|96) \| .*\|$", kind="context", multi=True),
    Locator("calib_rows", "m1", M1_S2, r"^\| (16|32|48|60|72|96) \| ([\d.]+) \|",
            kind="context", multi=True),
    # ---- M2 machinery + reachable band -----------------------------------
    Locator("m2_sims", "m2", None, r"(1440 paired sims)", kind="context"),
    Locator("m2_slice_semantics", "sweep", None,
            r"(one shared simulation set, gains are analytic transforms)", kind="context"),
    Locator("m2_run_math", "sweep_code", None,
            r"(2 generators x 6 concs x 10 sigmas x 12 seeds paired sims "
            r"\(1440 runs\))", kind="context"),
    Locator("m2_title", "m2", None, r"(Milestone 2 report)", kind="context"),
    Locator("plateau_v1_lo", "m2", M2_S2, r"V1 plateaus at ([\d.]+)-", tier2=r"V1"),
    Locator("plateau_v1_hi", "m2", M2_S2, r"V1 plateaus at [\d.]+-([\d.]+)"),
    Locator("plateau_v2e_lo", "m2", M2_S2, r"V2E at ([\d.]+)-", tier2=r"V2E"),
    Locator("plateau_v2e_hi", "m2", M2_S2, r"V2E at [\d.]+-([\d.]+)"),
    Locator("floor_band", "m2", M2_S5, r"tau_K floor \((~[\d.]+-[\d.]+)\)"),
    # ---- M2 thresholds + findings ----------------------------------------
    Locator("finding_cells", "m2", M2_S3, r"\*\*(\d+) FINDING cells"),
    Locator("finding_split", "m2", M2_S3, r"\((\d+ V1 \+ \d+ V2E)\)"),
    Locator("finding_sigma", "m2", M2_S3, r"(sigma 6\.4 -> 12\.8)", kind="context"),
    Locator("tauhalf_v1_lo", "m2", M2_S3, r"\*\*V1 \[([\d.]+),", tier2=r"V1"),
    Locator("tauhalf_v1_hi", "m2", M2_S3, r"\*\*V1 \[[\d.]+, ([\d.]+)\]\*\*"),
    Locator("tauhalf_v2e_lo", "m2", M2_S3, r"\*\*V2E \[([\d.]+),", tier2=r"V2E"),
    Locator("tauhalf_v2e_hi", "m2", M2_S3, r"\*\*V2E \[[\d.]+, ([\d.]+)\]\*\*"),
    Locator("tauhalf_v2e_conc32", "m2", M2_S3, r"V2E conc-32 .*?tau_half = ([\d.]+)",
            tier2=r"conc\S* ?32"),
    Locator("tauhalf_v1_conc32", "m2", M2_S3, r"V1 conc-32 \(primary/band-edge\): tau_half = ([\d.]+)",
            tier2=r"DEGENERATE|degenerate"),
    Locator("ideal_gain_v2e_conc32", "m2", M2_S3,
            r"V2E conc-32 .*?larger ideal gain (\+[\d.]+%)", tier2=r"V2E"),
    Locator("ideal_gain_v1_conc32", "m2", M2_S3,
            r"V1 conc-32 .*?DEGENERATE: ideal gain (\+[\d.]+%)", tier2=r"V1"),
    Locator("tauhalf_v2e_conc16", "m2", M2_S3, r"V2E conc-16 \(all slices\): tau_half = ([\d.]+)",
            tier2=r"conc\S* ?16"),
    Locator("tauhalf_second_v1", "m2", M2_S3, r"V1 conc-96 \(second regime\): tau_half = ([\d.]+)",
            tier2=r"V1"),
    Locator("tauhalf_second_v2e", "m2", M2_S3, r"V2E conc-96 \(second regime\): tau_half = ([\d.]+)",
            tier2=r"V2E"),
    Locator("mult_lo", "m2", M2_S3, r"only the two conc-32 rows reach ([\d.-]+x)"),
    Locator("mult_hi_band", "m2", M2_S3, r"every other row is ([\d.-]+x)"),
    Locator("m2_concs", "m2", M2_S3, r"conc (\d+)", kind="context", multi=True),
    Locator("second_regime_gains", "m2", M2_S3,
            r"(second conc 96: V2E \+20\.25% vs V1 \+14\.64% = 1\.4x)", kind="context"),
    # ---- M2 tension attribution ------------------------------------------
    Locator("model_ratio_v2e", "m2", M2_S4, r"V2E = (0\.8747)", tier2=r"V2E"),
    Locator("closure_gap_part", "m2", M2_S4, r"closed (0\.\d+) of the"),
    Locator("closure_gap_total", "m2", M2_S4, r"of the (0\.\d+) gap"),
    Locator("closure_pct", "m2", M2_S4, r"gap = \*\*(36\.0%)\*\*", tier2=r"V2E"),
    Locator("residual_pct", "m2", M2_S4, r"the residual (64\.0%)", tier2=r"36\.0%|residual"),
    Locator("tension_anchor_pair", "m2", M2_S4, r"(3\.15/3\.24)", kind="context"),
    Locator("tension_ideal_accept", "m2", M2_S4, r"(accepting at ~0\.87 of the verified length)",
            kind="context"),
    Locator("conc16_v1", "m2", M2_S4, r"V1 ([-+][\d.]+%) and"),
    Locator("conc16_v2e", "m2", M2_S4, r"and V2E (\+[\d.]+%)", tier2=r"V2E"),
    # ---- M2 evidence boundary --------------------------------------------
    Locator("retention_def", "m2", M2_S5, r"retaining >= (\d+%)"),
    Locator("realization_claim", "m2", M2_S5, r"(ONE noise realization)", kind="context"),
    # ---- Environment (ledger entry 15, append-only committed) -------------
    Locator("env_python", "ledger", LEDGER_15, r"→ `(Python 3\.12\.14)`", kind="context"),
    Locator("env_versions", "ledger", LEDGER_15, r"→ `(2\.5\.4 9\.1\.1)`", kind="context"),
)

# Numbers that are structural, not claims. Each carries an explicit reason and
# is enumerated by tests/test_writeup_numbers.py (an unexplained novel token
# must go red).
EXEMPTIONS: tuple[dict, ...] = (
    {"token": "4.1", "reason": "product identifier inside the model name V4.1-Flash "
                               "(research_DeepInfer_20261022.md §7 sources)"},
    {"token": "2.0", "reason": "license identifier Apache-2.0 (upstream README, §7 sources)"},
)


class LocatorError(RuntimeError):
    """A locator's scoping heading is absent from its artifact."""


def _scope(text: str, heading: str | None) -> str:
    lines = text.splitlines()
    if heading is None:
        end = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines))
        return "\n".join(lines[:end])
    if heading not in lines:
        raise LocatorError(f"heading not present in artifact: {heading!r}")
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def fold(text: str) -> str:
    return FOLD_RE.sub(" ", text).strip()


def strip_structural(text: str) -> str:
    for rx in STRIP_RES:
        text = rx.sub(" ", text)
    return text


def derive() -> tuple[list[dict], list[str]]:
    entries: list[dict] = []
    failures: list[str] = []
    texts = {k: p.read_text(encoding="utf-8") for k, p in ARTIFACTS.items()}
    for loc in LOCATORS:
        try:
            region = _scope(texts[loc.artifact], loc.heading)
        except LocatorError as exc:
            failures.append(f"<FAILED locator {loc.key}: {exc}>")
            continue
        if loc.multi:
            found = re.findall(loc.pattern, region, re.MULTILINE)
            printed = " ".join(x if isinstance(x, str) else " ".join(x) for x in found)
        else:
            m = re.search(loc.pattern, region, re.MULTILINE)
            if not m:
                failures.append(
                    f"<FAILED locator {loc.key}: pattern did not match {loc.artifact} "
                    f"under {loc.heading or 'preamble'}: {loc.pattern!r}>")
                continue
            printed = m.group(1) if m.lastindex else m.group(0)
        if not printed:
            failures.append(f"<FAILED locator {loc.key}: captured an empty claim>")
            continue
        printed = fold(printed)
        entries.append({
            "key": loc.key,
            "kind": loc.kind,
            "printed": printed,
            "tokens": sorted(set(TOKEN_RE.findall(strip_structural(printed)))),
            "artifact": loc.artifact,
            "artifact_path": str(ARTIFACTS[loc.artifact].relative_to(REPO_ROOT)),
            "heading": loc.heading,
            "pattern": loc.pattern,
            "tier2": loc.tier2,
        })
    return entries, failures


def allow_list(entries: list[dict]) -> set[str]:
    return {t for e in entries for t in e["tokens"]} | {x["token"] for x in EXEMPTIONS}




def build(write: bool = True) -> list[dict]:
    entries, failures = derive()
    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        raise SystemExit(f"{len(failures)} locator(s) failed — refusing to emit a "
                         "shrunken fixture (N4).")
    if write:
        payload = {
            "_generated_by": "sim/build_writeup_numbers.py",
            "_note": "Every 'printed' string is regex-captured from a committed artifact. "
                     "Regenerate: .venv/bin/python -m sim.build_writeup_numbers",
            "entries": entries,
            "exemptions": list(EXEMPTIONS),
        }
        OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        n_claim = sum(1 for e in entries if e["kind"] == "claim")
        print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}: {len(entries)} entries "
              f"({n_claim} claims, {len(entries) - n_claim} context), "
              f"{len(allow_list(entries))} allowed tokens")
    return entries


if __name__ == "__main__":
    build()
