# MASTER PROMPT — Milestone S: Synthesis and public writeup

> **How to use this document.** Open it from inside a coding agent working in
> the repository at `/Users/jackson/deepinfer-research`. Protocol is
> HANDOFF_PROMPT.md Sections 4-6 and 8-10 (absolute, referenced not copied);
> HANDOFF_PROMPT_M2.md §5 documents the Milestone-2 lanes (historical). One
> lane, this file. Do not paste any handoff document into a chat window —
> read from disk.

## 1. How to use this document

Single-lane milestone. The triage stop (§6 of HANDOFF_PROMPT.md) applies: present the writeup outline and the guard design before writing any file. The non-negotiables N1-N10 and the §10 command surface are absolute. Milestones 1 and 2 are CLOSED (research-repo `be55c16`, ledger entries 1-14); this lane adds no new simulation.

## 2. Mission

Milestones 1-2 produced, from a Mac-only CPU simulator, a complete auditable arc against DeepInfer's published serving numbers: mechanism adjudication (k+1 emission, statistic = k) → anchor-table sign-pattern audit (reproduction region + joint irreproducibility, structural) → ranking-quality machinery (reachable band [per-conc plateau, 1]) → the auditable bound (half-retention tau_K thresholds) → the M1-tension attribution (ranking quality necessary but not sufficient: 36.0% closure, 64.0% residual). This lane packages that arc:

1. **Update** `research_DeepInfer_20261022.md`: append an "Outcomes (2026-10-24)" section summarizing M1/M2 with pointers to `results/m1_report.md`, `results/m2_report.md`, `results/sweep_m2.md`; mark its §6 milestone list as executed (pointer to the ledger). Do not rewrite the original sections — the report is a living document, the ledger is the history.
2. **Create** `WRITEUP.md`: the public-facing artifact (blog/HN/Zhihu-ready, English body per the deliverable convention).

## 3. Verified state (reference)

Research repo HEAD `be55c16`, clean; clone `/Users/jackson/DeepInfer` @ `9bd9cf2` untouched. Committed sources of truth: `results/m1_report.md`, `results/m2_report.md`, `results/sweep_l2.md`, `results/sweep_m2.md`, `sim/convention.md`, `RESULTS_LEDGER.md` (entries 1-14), `dflash_alloc.py`, the sim package (38 fast tests + 1 slow pin, all green). Key verified numbers live in those artifacts; this lane transcribes none of them by hand (§5.3).

## 4. Lane MS-L1 specification

### 5.1 Outcomes section (research report)

Structure fixed: M1 headline (joint irreproducibility, structural, binding = dynamic accepted length); M2 deliverables (reachable band, tau-half thresholds, 36.0%/64.0% attribution); the reproducibility statement (seeds + commit pins + dual suites); pointers to the four result artifacts. Every number quoted from the committed artifacts, never from memory.

### 5.2 WRITEUP.md

Audience: technical public (HN / r/LocalLLaMA / Zhihu readers; benchmark-literate, not spec-decode specialists). Structure:

1. **Hook** — what was audited (DeepSeek's published serving numbers for V4.1-Flash inference), with what (a laptop, numpy, no GPU, no model weights), and the headline results.
2. **The mechanism** — DSpark's confidence-scheduled verification as read from the open-source engine: ConfidenceHead → global top-k on cumulative products under a batch budget (a greedy knapsack); the k+1 emission convention with its citation chain; window-length semantics.
3. **M1 result** — the anchor's sign pattern is reproducible only in convex-cost bands; the conc-16 magnitude and scoring-row signs are JOINTLY irreproducible (structural, flip-curve geometry); stated as a finding about internal consistency of published claims, not an accusation (N8 tone).
4. **M2 result** — the reachable ranking-quality band (a probability-constrained head can never go below the plateau); the auditable bound (retaining >= 50% of the ideal-band gain requires tau_K >= ~0.39-0.55 per regime); the tension attribution (36%/64%).
5. **What this does NOT prove** — carry the evidence boundary verbatim-in-spirit: published claims vs ground truth; ideal-information upper bound; single-noise-realization conditioning; the anchor as self-consistency audit only.
6. **Reproduce it yourself** — the exact commands (`git clone` of the research repo state, `.venv/bin/python -m sim.sweep_l2`, `-m sim.sweep_m2`, `-m pytest`), seeds, and commit pins.
7. **Attribution** — DeepInfer/DSpark facts from the public repo @ `9bd9cf2` (Apache-2.0); this analysis is independent and unaffiliated; anchor numbers quoted as published claims.

Style: dense, no padding, every number from an artifact; full-width punctuation not required (English body).

### 5.3 The guard (emit-time contract, extended to prose)

`tests/test_writeup_numbers.py`: a generator script extracts the headline numeric strings FROM the committed artifacts (parsed, never hand-typed) into a frozen list; the test asserts each list entry appears verbatim in `WRITEUP.md` and in the Outcomes section. N5 mutant: inject an invented number into WRITEUP.md and show the guard goes red. The list generation must be deterministic and committed (the list itself lives in the repo as a generated fixture with its generator referenced).

## 5. Non-goals

No Milestone-3 implementation (FDR wrapper / ZOS staleness — proposal only, already recorded in m2_report.md §5); no new simulation runs; no upstream-clone edits; no new dependencies.

## 6. Definition of Done

`WRITEUP.md` + Outcomes section + guard tests committed by explicit path; dual suites green (fast + `-m slow`, durations recorded); ledger entry (baseline + outcome, with the derived-number fixture generator noted); tree clean. The session report follows §9 (seven subsections).
