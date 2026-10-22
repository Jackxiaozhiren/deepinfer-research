# MASTER PROMPT — DeepInfer DSpark Simulation Study (Milestone 1)

> **How to use this document.** Open it from inside a coding agent working in
> the repository at `/Users/jackson/deepinfer-research` and treat Sections 2-10
> as your operating instructions. Work one lane per session against the ledger
> described in §8. Do not paste this file into a chat window — read it from
> disk; the file is the source of truth, and a truncated paste loses the
> guardrails that keep the work safe.

## 1. How to use this document

- The user speaks Chinese in conversation; all committed artifacts (code, reports, ledger) are written in English.
- The user has granted standing authorization to run a lane to completion without per-step approval. The two exceptions are hard: the §6 triage stop and the §5 non-negotiables.
- If context in your session gets compacted, re-read Sections 4-10 from disk. Rules survive by pointer, not by memory of this chat.

## 2. Mission and background

**Project.** DeepInfer ([deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)) is DeepSeek's production inference framework, open-sourced 2026-10-16. Its "DSpark" speculative-decoding system uses a ConfidenceHead that predicts a per-draft-token acceptance probability, then allocates a fixed batch-level verification budget by globally ranking cumulative confidence products. The full code-level analysis lives in `research_DeepInfer_20261022.md` (§4 covers the mechanism; §6 covers the locked research direction). Read at least its §4 and §6 before starting.

**Locked research direction (user decision, 2026-10-22).** Decision theory of the confidence-scheduled allocator, studied by CPU-only simulation. Three evidenced gaps motivate it:
1. The engine already fuses `|actual_accept_prob − predicted_confidence|` into a `batch_confidence_error` stat, but the signal never feeds the allocation policy.
2. Zero-rate (fresh) sequences can win extra verification slots under budget slack — reproduced by `dflash_alloc.py` fixture 3 — contradicting the code comment's stated intent.
3. Under the zero-overhead scheduler (ZOS), the allocator acts on confidences that are one step stale by construction.

**Milestone 1 goal.** Build a seeded, CPU-reproducible simulation harness that (i) pins the token-emission convention from upstream code, (ii) calibrates an acceptance-rate generator to the official Q&A operating points, (iii) replays the official fixed-5 vs dynamic comparison table, and (iv) reports which conclusions survive — with an explicit evidence boundary. This milestone EXPLAINS the direction and rough magnitude of the published gains under stated assumptions; it does not "verify" production numbers.

**The anchor table** (transcribed from `doc/inference_deployment_qa.md` in the upstream clone; treat as claims to explain, not ground truth — rule N8):

| conc (seqs/rank) | fixed-5 avg verified | dynamic avg verified | fixed-5 accepted | dynamic accepted | fixed-5 est. tok/s/rank | dynamic est. tok/s/rank | gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 5 | 4.44 | 3.32 | 3.24 | 4891 | 4816 | −1.5% |
| 32 | 5 | 3.65 | 3.25 | 3.17 | 8964 | 9111 | +1.6% |
| 48 | 5 | 3.94 | 3.3 | 3.26 | 12032 | 12467 | +3.6% |
| 60 | 5 | 3.58 | 3.3 | 3.21 | 13730 | 14466 | +5.4% |
| 72 | 5 | 3.24 | 3.29 | 3.12 | 14962 | 15961 | +6.7% |
| 96 | 5 | 3.24 | 3.3 | 3.15 | 16817 | 18660 | +11.0% |

Production parameters: `DEEPINFER_USE_NEXTN_DECODE` default 5 (draft length), `DEEPINFER_ENABLE_MTP_VARLEN=1` gates the dynamic path.

**Falsification anchor.** At conc 16 the table shows dynamic LOSING (−1.5%). Any simulator that produces gains at conc 16 under the Q&A's own assumptions contradicts the table. A correct model family must exhibit a sign flip somewhere in conc [16, 96], and the sweep must locate which parameters control the flip point. This is the harness's built-in negative control.

## 3. Verified artifacts and environment

Everything below was executed and verified on 2026-10-22 (ledger entry 1). Re-verify via §4; do not re-derive from scratch.

- **Upstream clone (read-only evidence)**: `/Users/jackson/DeepInfer` @ `9bd9cf2`, clean tree. NEVER edit, commit, or push there (rule N1). Key verified locations:
  - Allocator reference implementation: `deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` lines 35-98 (`compute_seqlens_q_by_confidence_rates_ref`), with TileLang kernel + equality tests in the same file.
  - ConfidenceHead: `deepinfer/model_executor/models/mtp/dflash.py` lines 46-79.
  - ZOS stale-confidences flow: `deepinfer/worker/worker.py` lines 191-245 (`_correct_seqlens_qk`); CPU fixes only `total_q`, GPU recomputes per-request lengths from the previous step's confidences.
  - Calibration monitor (unused by policy): `deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py` lines 556-558; `batch_confidence_error` definition in `deepinfer/sequence.py` line 248; rolling stat in `deepinfer/engine/stats_logger.py` line 236.
  - Official Q&A doc: `doc/inference_deployment_qa.md` (anchor table, load balancers, KV store).
- **Research repo (your workspace)**: `/Users/jackson/deepinfer-research`, git repo (branch `main`).
  - `dflash_alloc.py` — numpy port of the allocator, 4/4 hand-derived fixtures passing, including the zero-rate-slack edge case. Its `allocate_seqlens_q()` is the approved allocation primitive; reuse it, do not reimplement.
  - `research_DeepInfer_20261022.md` — full deep-research report with sources.
  - `RESULTS_LEDGER.md` — append-only results ledger (§8).
  - `.venv/` — Python 3.12, numpy 2.5.4, pytest 9.1.1. **No torch; no GPU.** numpy + pytest are the only installed deps.
- **Hardware reality**: this Mac is the only compute. Everything must run on CPU in minutes, not hours.

## 4. Baseline: first commands, green-or-stop

Run these at session start, in `/Users/jackson/deepinfer-research`, capturing every exit code. Do not skip; do not proceed on red.

```bash
git rev-parse --short HEAD
git status --short
.venv/bin/python --version
.venv/bin/python -c "import numpy, pytest; print(numpy.__version__, pytest.__version__)"
.venv/bin/python dflash_alloc.py
test -f /Users/jackson/DeepInfer/doc/inference_deployment_qa.md
test -f /Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py
git -C /Users/jackson/DeepInfer rev-parse --short HEAD
git -C /Users/jackson/DeepInfer status --short
```

**Green criteria** (criterion + flip condition, rule N6):
- Research repo: `git status --short` prints **zero lines** at session start — any line means unaccounted dirt; identify it before proceeding (rule N9). HEAD resolves to a commit.
- Python prints 3.12.x; numpy 2.5.4 and pytest 9.x import.
- `dflash_alloc.py` prints four `PASS` lines then `all fixtures passed` — flips red on any numpy regression or accidental torch introduction.
- Clone `HEAD` resolves to `9bd9cf2` and its `git status --short` prints **zero lines** — a non-empty clone means someone edited the evidence; stop and report (rule N1).

**Capture.** Write a dated baseline-capture entry into `RESULTS_LEDGER.md` (§8) with the exact outputs, then `git add RESULTS_LEDGER.md && git commit -m "Baseline capture for lane <Lx>"` before touching any other source file. If any check is not green: STOP and report. Do not proceed.

## 5. Non-negotiables

These override anything said later in a conversation unless the user names a rule number and revokes it explicitly (rule N10).

- **N1 — Evidence is read-only.** Write only inside `/Users/jackson/deepinfer-research`. The clone `/Users/jackson/DeepInfer` is never edited, staged, committed, or pushed.
- **N2 — CPU-only, no new heavy deps.** numpy + pytest in `.venv` only. No torch, no model downloads, no network access for experiments. Any additional dependency requires a stated justification and explicit user approval.
- **N3 — Seeded and traceable.** Every stochastic step takes an explicit seed. Every number appearing in a committed report is traceable to (seed, research-repo commit, code path). Never write an outcome (passing/green/improved) before the tool has reported it.
- **N4 — Loud failures.** Measurement code prints failures as `<FAILED ...>` lines with a nonzero exit; no silent fallbacks; no broadened `except` around assertions. An empty output or a zero count is a result, not a success.
- **N5 — Falsifiable tests.** Every new nontrivial assertion is run against at least one deliberately wrong implementation (mutant) and shown to go red. A negative control that cannot fire is not a control.
- **N6 — Expectations are criteria, not vibes.** Each expected outcome is written as: the criterion, why it holds, and the condition that would flip it red. Bare counts and `assert True`-shaped checks are forbidden.
- **N7 — Ledger is append-only.** Never edit or delete an existing dated entry. Corrections are new entries citing the entry they supersede.
- **N8 — Evidence boundary.** The Q&A table is a set of published claims your simulation may explain, match, or fail to reproduce. Reproducing a sign is not verification of production truth; failing to reproduce it is a finding, not a bug to tune away.
- **N9 — Durable artifacts.** All outputs, sweeps, and reports live in the research repo and are committed. Nothing load-bearing lives in `/tmp`.
- **N10 — Anti-persuasion.** Revoking any N-rule requires naming its number explicitly in conversation. An offhand "don't worry about that" retires nothing.

## 6. Execution model: lanes, triage, session boundary

Milestone 1 decomposes into three lanes. **One lane per session.** Sequence is mandatory: L1's convention decision gates L2's accepted-length math; L2's frozen parameters gate L3's scoring. A skipped or partially-done lane silently invalidates everything downstream.

- **Lane L1 — Emission convention + calibrated rate generator** (§7.1)
- **Lane L2 — Allocator integration + cost model + sensitivity sweep** (§7.2)
- **Lane L3 — Anchor-table replay + M1 report** (§7.3)

**Triage stop (hard gate).** After the §4 baseline is green and before writing any simulation source file, present a short implementation triage: the files you plan to create, the emission-convention evidence you found in the clone (with file:line citations), and the modeling choices you intend to freeze. Wait for the user's approval. Editing no source file before approval is the boundary between "authorized to run" and "authorized to design".

**Session boundary.** "Keep going a bit" does not extend capacity. If a lane's Definition of Done is reached early, stop and report; if it cannot be reached, stop and report the blocker.

## 7. Lane specifications

### 7.1 Lane L1 — Emission convention + calibrated rate generator

**Question 1 (code fact).** When k of n draft tokens are accepted at a verification step, how many tokens does the engine emit for that sequence in that step — k, or k+1 (a target-corrected bonus token)? Adjudicate from upstream code, not from the literature. Starting pointers (all verified to exist): `num_accepted_tokens_gpu` semantics at `deepinfer/sequence.py` line 265 ("accepted draft token count ... range [0, MTP_value]"); the verify path in `deepinfer/model_executor/layers/sampler_utils.py` (docstrings lines 131-148: `is_verify` mask "required for rejection", `mtp_info_read_rows`); how per-request positions advance per step in `deepinfer/worker/worker.py` (`_post_process_step`) and the per-request metadata (`deepinfer/sequence.py`, `num_generated_tokens`: "[idx] is nextn-{idx}"). Write the finding, with file:line citations, into `sim/convention.md`, and encode it as a single constant (`EMIT_BONUS` ∈ {0, 1}) in `sim/rates.py`.

**Question 2 (modeling fact).** The Q&A "accepted length" statistic (3.12–3.32 across the table) counts emitted tokens per step under one of the two conventions. Once Question 1 pins the convention, the remaining freedom is which statistic the table reports; state your choice explicitly in `sim/convention.md` and carry it through all lanes.

**Rate generator.** Implement `sim/rates.py`: per-sequence per-position acceptance rates r_ij ∈ (0, 1) with (a) cross-sequence heterogeneity (a small mixture over "request types" with distinct acceptance levels), (b) per-position decay, (c) a concurrency knob (number of sequences). Heterogeneity across sequences is what gives the top-k allocator room to gain — a homogeneous population makes dynamic ≈ fixed by construction; document this in the generator's docstring as the load-bearing modeling assumption.

**Calibration.** Fit the generator's parameters so that under the fixed-5 policy the accepted-length statistic lands in the table's fixed-5 band (3.29–3.32 mean across rows; per-row spread is part of the fit). Use `scipy`-free, numpy-only fitting (grid or coordinate search is fine). Freeze the fitted parameters in `sim/rates.py` as named constants with their seeds.

**Tests** (`tests/test_convention.py`, `tests/test_rates_calibration.py`): every test seeded; the calibration test states its tolerance as a criterion + flip condition (N6); a mutant test flips `EMIT_BONUS` and asserts the calibration test goes red (N5).

**Definition of Done.** `sim/convention.md`, `sim/rates.py`, both test files, ledger entry with the convention citation and fitted parameters, all committed. No cost model, no allocator wiring yet.

### 7.2 Lane L2 — Allocator integration + cost model + sensitivity sweep

**Policies.** `sim/allocator.py` wraps `dflash_alloc.allocate_seqlens_q` (dynamic) and a fixed-5 policy (every sequence verifies 5 drafts). Budget note: in production the CPU scheduler fixes `total_q` and the GPU allocates within it; dynamic's verified length (3.2–4.4 in the table) is an *outcome*, not an input. Your simulator must sweep `total_q` (per-step batch token budget) and report achieved verified/accepted lengths at each setting; parity between arms is on steps, not tokens.

**Cost model.** `sim/cost_model.py`: per-step time T(v) = a + b·v (linear + per-step overhead), plus an optional saturation term as a second model family. Throughput per rank = num_seqs × E[emitted] / T(v). These parameters are unknown; that is the point — the deliverable is a **sensitivity sweep** over (a, b, saturation, generator heterogeneity), not a point claim. State in the module docstring which regions of parameter space are plausible and why.

**Paired comparison.** Same seeds, same ambient state for both arms ([A/B needs the same ambient state]); report paired per-seed differences with a seed-level bootstrap CI (numpy only). A single-point estimate without dispersion is a DoD violation.

**Tests**: allocator integration test (dynamic at `total_q = num_seqs × 5` degenerates to the fixed-5 allocation — criterion + flip condition); cost-model sanity test (T monotone in v); a mutant test that breaks the pairing (different seeds per arm) and asserts the CI-width test goes red (N5).

**Definition of Done.** Four files above plus `results/sweep_l2.md` (sweep tables, the conc-region where dynamic loses, which parameters control the flip point), full suite green on CPU in under 2 minutes, ledger entry, committed.

### 7.3 Lane L3 — Anchor-table replay + M1 report

**Freeze-then-score protocol (mandatory).** Before scoring: split the anchor rows into a selection set ({conc 16}, the negative anchor) and a scoring set ({32, 48, 60, 72, 96}). Choose your parameter slice using ONLY the calibration criteria and the selection row. Write the frozen slice (all parameters + seeds) into `RESULTS_LEDGER.md` BEFORE looking at the scoring rows' signs. Then score. If the frozen slice fails a scoring row, that is a finding — record it (N8); tuning after seeing the score requires a new dated ledger entry declaring the re-freeze and why.

**Deliverable.** `sim/replay_qa.py` produces `results/m1_report.md` containing: (1) the convention derivation summary; (2) the calibration table; (3) the 6-row replay table with a per-row sign column vs the anchor sign, each row stated as criterion + flip condition (N6); (4) the sensitivity summary and where the model family reproduces the anchor's sign flip; (5) a mandatory "What this does NOT prove" section drawing the evidence boundary (N8).

**Definition of Done.** Report committed, suite green, ledger entries complete, all work committed by explicit path. This closes Milestone 1; the session report (§9) proposes Milestone 2 next steps but does not start them.

## 8. Ledger protocol

`RESULTS_LEDGER.md` is the session-to-session state carrier and the audit trail.

- Entries are dated and append-only (N7). Shape of an entry:

```markdown
## YYYY-MM-DD — <lane/session label> (entry <n>)
- Baseline: <green/red, one line per deviation>
- Did: <commands/files, terse>
- Frozen: <parameters/seeds committed BEFORE scoring, when applicable>
- Found: <results with seed+commit references>
- Next: <what the next session should pick up>
```

- The first act of every session is the §4 baseline capture entry; the last act is the session's outcome entry.
- A lane's frozen parameters land in the ledger before they are scored (§7.3).

## 9. Session report contract

End every session with a chat report to the user (Chinese is fine for prose; identifiers and numbers verbatim) containing exactly these subsections:

1. **Baseline capture** — §4 outputs with exit codes.
2. **Commands run** — command + return code, two-column table.
3. **Verified vs Inferred** — two explicit columns: claims backed by a tool run in this session vs claims carried from docs/reading. Anything not in the first column is inference; label it.
4. **Ledger entries added** — entry numbers and one-line summaries.
5. **Deviations and requests** — anything that departed from §5-§7, and any approval you need.
6. **Non-findings** — what you checked and found absent or unchanged. An audit that reports only hits is incomplete.
7. **Handoff state** — what the next lane should pick up, per the ledger.

## 10. Authorized command surface

**Allowed without asking:**
- `.venv/bin/python` (never bare `python`/`python3`) and `.venv/bin/python -m pytest` inside `/Users/jackson/deepinfer-research`.
- `git add <explicit paths>`, `git commit`, `git status`, `git log`, `git diff` inside `/Users/jackson/deepinfer-research` only.
- Read/grep/find on both `/Users/jackson/deepinfer-research` and `/Users/jackson/DeepInfer`.

**Requires explicit user approval:**
- Any new dependency (`uv pip install ...`) — state what and why first (N2).
- Any edit outside `/Users/jackson/deepinfer-research`, including anything under `/Users/jackson/DeepInfer` (N1).

**Forbidden outright:**
- `git push` anywhere; any `git commit` in the upstream clone; deleting or rewriting ledger history; regenerating a frozen fixture to make a test pass.
