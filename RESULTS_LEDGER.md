# Results ledger (append-only)

Entries are dated. Never edit or delete an existing entry; corrections are new entries that cite the entry they supersede (HANDOFF_PROMPT.md §5 rule N7, §8).

## 2026-10-22 — project launch / direction-locking session (entry 1)

- Environment: macOS arm64. Python 3.12 in `.venv`, numpy 2.5.4, pytest 9.1.1. No torch installed; no GPU available (Mac-only, per user decision).
- Upstream clone: `/Users/jackson/DeepInfer` @ `9bd9cf2`, clean tree, read-only evidence (N1).
- `dflash_alloc.py`: 4/4 fixtures PASS — fixture1 prefix-strict `[3, 1]`; fixture2 fresh-gets-one `[1, 3]`; fixture3 zero-rate-wins-under-slack `[2, 3]`; fixture4 no-slack `[1, 1, 1]`.
- Research repo initial commit: `9fadd38` (deep-research report + allocator port).
- Locked direction: Path A — decision theory of DSpark's confidence-scheduled verification (report §6). Milestone 1 lanes L1 → L2 → L3 defined in HANDOFF_PROMPT.md §7.
- Anchor table (fixed-5 vs dynamic, conc 16-96) transcribed to HANDOFF_PROMPT.md §2 from `doc/inference_deployment_qa.md`; conc-16 row (−1.5%) designated the falsification anchor.

## 2026-10-22 — Lane L1 session start / §4 baseline capture (entry 2)

- Baseline: GREEN — all nine §4 commands ran with exit code 0; zero deviations from the §4 green criteria.
- Did: ran the §4 first-commands in order. Exact outputs (each rc 0):
  - `git rev-parse --short HEAD` → `7bc986c`
  - `git status --short` → 0 lines
  - `.venv/bin/python --version` → `Python 3.12.14`
  - `.venv/bin/python -c "import numpy, pytest; print(numpy.__version__, pytest.__version__)"` → `2.5.4 9.1.1`
  - `.venv/bin/python dflash_alloc.py` → `PASS fixture1 prefix-strict: seqlens_q=[3, 1] (sum=4)` / `PASS fixture2 fresh-gets-one: seqlens_q=[1, 3] (sum=4)` / `PASS fixture3 zero-rate-wins-under-slack: seqlens_q=[2, 3] (sum=5)` / `PASS fixture4 no-slack: seqlens_q=[1, 1, 1] (sum=3)` / `all fixtures passed`
  - `test -f /Users/jackson/DeepInfer/doc/inference_deployment_qa.md` → rc 0
  - `test -f /Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` → rc 0
  - `git -C /Users/jackson/DeepInfer rev-parse --short HEAD` → `9bd9cf2`
  - `git -C /Users/jackson/DeepInfer status --short` → 0 lines
- Found: green criteria satisfied — research repo clean at `7bc986c` (commit log: `7bc986c` handoff docs on top of `9fadd38` initial artifacts); Python 3.12.14 with numpy 2.5.4 + pytest 9.1.1 importing; `dflash_alloc.py` 4/4 fixtures PASS; clone HEAD `9bd9cf2` clean (N1 intact). No torch present; no new dependencies introduced.
- Next: Lane L1 (§7.1) only. Per §6 hard gate: implementation triage presented to the user for approval BEFORE any simulation source file is written.

## 2026-10-22 — Lane L1 complete: emission convention + calibrated rate generator (entry 3)

- Baseline: green (entry 2). Triage approved by the user with four notes, all applied: (1) `effective_accept` cited at `tilelang_ops/sampler.py:246` (triage chat had misattributed it to `layers/sampler.py:246`) plus the added independent k+1 evidence `emitted_rows = (effective_accept | is_final) & stop_hits` at `tilelang_ops/sampler.py:376-379`; (2) the post-EOS confidence-zeroing pathway `masked_fill_(emitted_stop_token..., 0.0)` at `layers/sampler.py:252-254` recorded as a second production zero-rate pathway; (3) PROGRESS_DECAY marked design-driven, not empirical, in `sim/convention.md` and the `sim/rates.py` docstring, with a doc guard test enforcing it; (4) the L2 cost-model note recorded below under Next.
- Did: wrote `sim/convention.md` (Question 1/2 adjudication, 8-point citation chain, declared alternative reading, zero-rate pathways, generator family, calibration criteria), `sim/rates.py` (generator + fixed-5 simulator + deterministic search + frozen constants), `conftest.py`, `sim/__init__.py`, `tests/test_convention.py` (8 tests), `tests/test_rates_calibration.py` (4 tests).
- Frozen: `EMIT_BONUS = 1`; `NEXTN = 5`; `TYPE_LEVELS = (0.97, 0.85, 0.7)`; `TYPE_WEIGHTS = (0.5, 0.25, 0.25)`; `DEPTH_DECAY = 0.99`; `PROGRESS_DECAY = 1e-06`; `CALIB_SEED = 20261022`; `CALIB_STEPS = 6000`; band `BAND_MEAN = (3.29, 3.32)`, `BAND_ROW = (3.25, 3.32)`. Research-repo commit at freeze: `468bfe6` (ledger) → L1 commit pending this entry.
- Found:
  - Question 1: the engine emits **k+1** tokens per verification step (accepted prefix plus final fresh row). Decisive sites: `tilelang_ops/sampler.py:376-379` (comment + `emitted_rows`), `:466-467` (emitted_stop_token docstring), `:442-443`, `:458-459`; KV advance `seqlens_k += seqlens_q - (nextn - accepted)` at `compute_seqlens_k_zos.py:22-23` gives k+1 under full-depth verification; fresh sequences pack exactly 1 row (`schedule_mtp_varlen.py:98`).
  - Question 2 (frozen choice): the Q&A "accepted length" statistic is the mean accepted draft count k per step, excluding the bonus (`num_generated_tokens`/`num_accepted_tokens_gpu` semantics, `tilelang_ops/sampler.py:460-461`, `layers/sampler.py:289`). The k+1 reading is the declared first re-freeze candidate if L3 scoring fails (N7/N8).
  - Calibration (seed `20261022`, horizon 6000 steps, stratified types): base rows 16: 3.3058, 32: 3.3007, 48: 3.3050, 60: 3.3036, 72: 3.3062, 96: 3.3040; mean 3.3042 — in band, rows flat. Mutant (`EMIT_BONUS=0`, same seeds): rows 16: 3.3325, 32: 3.3268, 48: 3.3314, 60: 3.3303, 72: 3.3328, 96: 3.3297; mean 3.3306 — all six rows above the band top, paired direction holds. Signal = +0.0264.
  - Two rejected calibration designs, kept as process findings: (a) i.i.d. type draws make the row statistic wobble ±0.15 via type-composition noise (observed: 3.42 next to 3.13) — fixed by stratified composition with weights divisible by 1/4 = gcd of the anchor concurrencies; (b) a capped progress term (`PROGRESS_CAP=2000`) truncates the mutant signal to +0.0007..0.0018 because the base arm saturates at t~465 — removed; the signal/drift ratio of the uncapped term is 1/(2·advance) ≈ 0.116 and the horizon dependence is declared instead.
  - Flip-condition demonstrations (N5): h 0.97→0.973 puts mean at 3.3222 with five rows out of band; h→0.96 puts mean at 3.2451 (both red). Mutant tests assert the EMIT_BONUS=0 control fires; the emission-accounting mutant shows 6→5 tokens/step.
  - Suite: 12/12 passed in 1.04 s (`.venv/bin/python -m pytest tests/ -v`).
- Next: Lane L2 (§7.2) picks up from the frozen L1 constants. User note 4 (2026-10-22, pre-planted for L2, hand-computed — L2 must re-verify mechanically): the anchor table's throughput ratios exclude the pure-linear cost family T(v)=a+b·v; conc-16 implies a≈58b, substituted at conc-96 the linear family predicts T(5)/T(3.24)≈1.029 while the anchor requires 1.162 — contradictory under both the k and k+1 readings of the accepted-length statistic. L2's main cost family must be T(v)=a+b·v^c, and L2 adds `test_linear_family_excluded_by_anchor_ratios` as a negative control: the linear family must FAIL that test; if it passes, the checker is miscalibrated.

## 2026-10-22 — Lane L2 session start / §4 baseline capture (entry 4)

- Baseline: GREEN — all nine §4 commands ran with exit code 0; zero deviations from the §4 green criteria. Tree had moved since entry 3 exactly by the L1 commit (below), nothing unaccounted.
- Did: ran the §4 first-commands in order. Exact outputs (each rc 0):
  - `git rev-parse --short HEAD` → `c6d5a21`
  - `git status --short` → 0 lines
  - `.venv/bin/python --version` → `Python 3.12.14`
  - `.venv/bin/python -c "import numpy, pytest; print(numpy.__version__, pytest.__version__)"` → `2.5.4 9.1.1`
  - `.venv/bin/python dflash_alloc.py` → `PASS fixture1 prefix-strict: seqlens_q=[3, 1] (sum=4)` / `PASS fixture2 fresh-gets-one: seqlens_q=[1, 3] (sum=4)` / `PASS fixture3 zero-rate-wins-under-slack: seqlens_q=[2, 3] (sum=5)` / `PASS fixture4 no-slack: seqlens_q=[1, 1, 1] (sum=3)` / `all fixtures passed`
  - `test -f /Users/jackson/DeepInfer/doc/inference_deployment_qa.md` → rc 0
  - `test -f /Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` → rc 0
  - `git -C /Users/jackson/DeepInfer rev-parse --short HEAD` → `9bd9cf2`
  - `git -C /Users/jackson/DeepInfer status --short` → 0 lines
- L1 review ratification (recorded verbatim from the user's 2026-10-22 message):
  - "L1 independently re-verified: suite 12/12; frozen constants reproduce bit-for-bit (base mean 3.3042 in band; mutant mean 3.3306, all six rows above band top); tree clean at c6d5a21; clone untouched at 9bd9cf2. All three declared deviations (stratified type composition, uncapped progress term with declared horizon, tightened mutant acceptance) are ratified."
  - "Report hygiene: the L1 chat report's command table contained a bare `python -c` row; bare python does not exist on this machine, so that row was shorthand, not the command run. Your §9 command table must name every command exactly as executed, full path included."
  - "L2 sensitivity addition (within §7.2's existing sweep mandate, not new scope): PROGRESS_DECAY and the heterogeneity level are design-driven, not empirical — include both as sweep axes, so the dynamic-vs-fixed sign conclusion is shown robust across the PROGRESS_DECAY range where the mutant stays red and across heterogeneity levels, not only at the frozen point."
- Found: green criteria satisfied — research repo clean at `c6d5a21` (L1 deliverables commit); Python 3.12.14 with numpy 2.5.4 + pytest 9.1.1; `dflash_alloc.py` 4/4 fixtures PASS; clone `9bd9cf2` clean (N1 intact); no new dependencies (N2 intact).
- Next: Lane L2 (§7.2) only. Per §6 hard gate: implementation triage presented to the user for approval BEFORE any L2 source file is written. No approval inherited from the L1 session.

## 2026-10-22 — Lane L2 complete: allocator integration + cost model + sensitivity sweep (entry 5)

- Baseline: green (entry 4). Triage approved with five rulings, all applied: (1) the §7.2 "total_q = num_seqs × 5" degeneracy spec is an error (draft positions are not rows) — tests use num_seqs × (NEXTN+1) = 6N with a ×5 counterexample, citing schedule_mtp_varlen.py:98 and sim/convention.md; (2) batch-form cost model (v = conc × rows/seq) adopted; the implied a/b values (2676.7 under the frozen statistic=k reading, 910.9 under the alternative) were independently re-verified by the user, and the throughput identity num_seqs · E[emitted] / T(v) is declared an inference assumption in the cost_model.py docstring; (3) fifth file sim/sweep_l2.py approved as the paired-comparison home; (4) results/sweep_l2.md carries the ideal-information-upper-bound evidence boundary ("What this does NOT prove"); (5) per-axis pre-match and post-match level vectors are recorded, clipping never silent.
- Did: wrote `sim/allocator.py` (policies wrapping `dflash_alloc.allocate_seqlens_q`, windowed prefix counting, paired step simulator sharing the stratified type vector and u block, `pairing_check`), `sim/cost_model.py` (T = a+b·v, a+b·v^c, a+b·v+d·v²; anchor constants; `exclude_linear_family`), `sim/sweep_l2.py` (paired sweep, seed-level bootstrap CI, sign maps, flip curves, two design-driven axes), `tests/test_allocator_integration.py` (6 tests), `tests/test_cost_model.py` (3 tests); ran the sweep to `results/sweep_l2.md`.
- Frozen (gates L3): `L2_SWEEP_SEED = 20261101`, `N_SWEEP_SEEDS = 12` (seeds 20261101..20261112), `BOOTSTRAP_SEED = 20261101` with 10000 resamples, `BUDGET_GRID = (4.0, 4.24, 4.58, 4.65, 4.94, 5.44, 6.0)` rows/seq, `COST_AB` logspace(1,4,10) + {465.0, 910.9, 1550.0, 2676.7, 4900.0}, `COST_C = (0.9, 1.0, 1.1, 1.3, 1.5)`, `COST_BETA = (0.0, 0.5, 2.0)`, `XI_GRID = (1e-06, 2e-06, 3e-06)`, `HETERO_GRID = (0.5, 1.0, 1.5)`, horizon `CALIB_STEPS = 6000`. L1 constants untouched (ledger entry 3).
- Found:
  - Suite 21/21 passed in 2.91 s. Degeneracy: dynamic at total_q = 6·num_seqs equals fixed-5 for arbitrary rates including all-zero; at 5·num_seqs it does not (budget sum invariant holds).
  - Linear-family exclusion mechanically re-proven under batch totals, both statistic readings: a/b pinned at 2676.7 (emitted=k+1) / 910.9 (emitted=k) by the conc-16 row; linear predicts 1.0548 / 1.1282 at conc-96 vs required 1.1497 / 1.1625. Discriminating power verified: at c=1.3 the same checker does NOT exclude (verdict flips).
  - The anchor sign pattern (conc-16 negative, 32-96 positive) is reproducible in the T = a+b·v^c family: verified points (465.0, 1.1), (1550.0, 1.3), (4900.0, 1.5) plus grid neighbors (464.16, 1.1), (4641.59, 1.5). The flip point is controlled jointly by a/b and c: the pattern lives in the band between the conc-16 and conc-32 zero-gain curves, [ab*_16(c), ab*_32(c)] = [453, 476] at c=1.1, [1386, 1716] at c=1.3, [4081, 5870] at c=1.5 (flip-curve table in the report).
  - FINDING (N8): the pure-quadratic saturation family (T = a+b·v+d·v²) reproduces NO anchor-pattern point in the swept grid — wherever conc-16 is negative, conc-32 stays negative too (its flip sits between conc 32 and 48, not 16 and 32).
  - PROGRESS_DECAY axis (user robustness requirement): mutant RED at every swept point (mutant means 3.3306 / 3.3548 / 3.3777 vs band top 3.32); conc-16 gain stable at (2676.7, 1.1): −2.03% / −2.11% / −2.18%; the anchor band stays non-empty at c=1.1 ([453,476] / [437,465] / [420,455]). xi = 4e-06 excluded: the mean-matched operating point is unreachable (level shift would push the high type past 1−ε; recorded in the XI_GRID comment).
  - Heterogeneity axis: the anchor band is non-empty at every spread level but the required convexity moves with the spread — s=0.5: c=0.9 band [60, 63]; s=1.0: c=1.1 [453, 476]; s=1.5: c=1.5 band [7265, 8159]; conc-16 gain stays negative (−3.76% / −2.03% / −1.03%). At s=1.5 the c=1.1 band INVERTS ([837, 691]) — recorded, not hidden.
  - Observation carried into L3: the anchor's dynamic accepted/verified ratio ≈ 0.97 (3.15/3.24 at conc 96) while this generator yields ≈ 0.82 (2.656/3.240 at the matched budget) — the anchor implies near-perfect ranking with sharply separated rates; expect tension on the accepted columns in L3 scoring (a finding per N8, potential declared re-freeze).
  - Process: three loud assertion catches during development (float 4.9399… ≠ 4.94 grid membership; two missing cost-point lookups; one unbound variable) fixed in source; the report is the artifact of the final deterministic run.
- Next: Lane L3 (§7.3). Freeze-then-score: selection set {conc 16}, scoring set {32, 48, 60, 72, 96}. The parameter slice must be chosen using only the calibration criteria and the conc-16 row, frozen in the ledger BEFORE scoring. Candidate slices live inside the anchor band mapped here (e.g., around (465.0, 1.1) / (1550.0, 1.3)); the accepted/verified ratio tension and the saturation-family exclusion are pre-declared findings for the report's evidence boundary.

## 2026-10-23 — Lane L3 session start / §4 baseline capture (entry 6)

- Baseline: GREEN — all nine §4 commands ran with exit code 0; zero deviations from the §4 green criteria. Tree had moved since entry 5 exactly by the L2 commit (below), nothing unaccounted.
- Did: ran the §4 first-commands in order. Exact outputs (each rc 0):
  - `git rev-parse --short HEAD` → `5f34d07`
  - `git status --short` → 0 lines
  - `.venv/bin/python --version` → `Python 3.12.14`
  - `.venv/bin/python -c "import numpy, pytest; print(numpy.__version__, pytest.__version__)"` → `2.5.4 9.1.1`
  - `.venv/bin/python dflash_alloc.py` → `PASS fixture1 prefix-strict: seqlens_q=[3, 1] (sum=4)` / `PASS fixture2 fresh-gets-one: seqlens_q=[1, 3] (sum=4)` / `PASS fixture3 zero-rate-wins-under-slack: seqlens_q=[2, 3] (sum=5)` / `PASS fixture4 no-slack: seqlens_q=[1, 1, 1] (sum=3)` / `all fixtures passed`
  - `test -f /Users/jackson/DeepInfer/doc/inference_deployment_qa.md` → rc 0
  - `test -f /Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` → rc 0
  - `git -C /Users/jackson/DeepInfer rev-parse --short HEAD` → `9bd9cf2`
  - `git -C /Users/jackson/DeepInfer status --short` → 0 lines
- L2 review ratification (recorded verbatim from the user's 2026-10-23 message):
  - "Independently re-verified: suite 21/21; commits ec6a216/5f34d07, tree clean, 7 files; clone untouched at 9bd9cf2. Two sign cells of results/sweep_l2.md hand-recomputed from the committed achieved-lengths table: (465.0, 1.1) at conc 16 = -0.05% and at conc 96 = +4.99%, matched to within hand-rounding; the c=1.1 conc-16 flip value 453 reproduces zero gain by hand. All five rulings verified as landed."
  - "Report-precision items carried into L3: 1. The L2 chat report's count \"62+30 sign points\" does not match the committed sign map (75 t_pow rows + 30 saturation rows). Chat-report counts must be derivable from the committed artifact; results/m1_report.md must not repeat untraceable counts. 2. The accepted/verified ratio tension enters m1_report.md with its exact construction as recorded in ledger entry 5 (anchor 3.15/3.24 at conc 96 vs generator 2.656/3.240 at the matched budget), never as bare 0.97-vs-0.82."
  - "L3-specific rulings: 3. Slice selection: t_pow family bands only (the saturation family is excluded by the entry-5 FINDING). The selection-set criterion must be DECLARED in the freeze entry before scoring: minimum = anchor sign at conc 16; if you also constrain magnitude closeness at conc 16, state that window explicitly in the freeze entry. 4. Recommended: during the L3 baseline, re-run `.venv/bin/python -m sim.sweep_l2` in the background as a bit-reproducibility check. Afterwards `git diff results/sweep_l2.md` must show at most the header date line; ANY numeric cell difference is a determinism finding, not a nuisance. 5. Freeze-then-score discipline unchanged: selection set {conc 16}; scoring set {32, 48, 60, 72, 96}; any re-freeze requires a new dated ledger entry declaring what failed and the new slice BEFORE looking at scoring rows again."
- Found: green criteria satisfied — research repo clean at `5f34d07`; Python 3.12.14 with numpy 2.5.4 + pytest 9.1.1; `dflash_alloc.py` 4/4 fixtures PASS; clone `9bd9cf2` clean (N1 intact); no new dependencies (N2 intact).
- Next: Lane L3 (§7.3) only. Per §6 hard gate: implementation triage presented to the user for approval BEFORE any L3 source file is written. No approval inherited from previous sessions. Ruling-4 bit-reproducibility check (background sweep + `git diff results/sweep_l2.md`) runs during this baseline phase; its result lands in the freeze entry.

## 2026-10-23 — Lane L3 FREEZE: parameter slice frozen before scoring (entry 7)

- Baseline: green (entry 6).
- Ruling-4 bit-reproducibility check (recorded per ruling 1): re-ran `.venv/bin/python -m sim.sweep_l2` in the background during the baseline phase; exit code 0; then `git diff results/sweep_l2.md` printed ZERO BYTES — the regenerated report is byte-identical to the committed one, including the header date line, which is a hardcoded string (not a clock read), so not even the date line differs. Bit-reproducibility confirmed; no determinism finding.
- Frozen slice (all parameters + seeds, committed BEFORE any scoring):
  - Family: t_pow only, T(v) = a + b·v^c (saturation family excluded by the entry-5 FINDING; ruling 3).
  - Slice: **(a/b = 464.15888336127773, c = 0.9)** — selected by the declared rule, fresh conc-16 paired sims, mean_gain16 = −1.605%.
  - Declared selection criterion: t_pow cost points on the committed grid (COST_AB_EXTENDED × COST_C, 15 × 5 = 75) whose conc-16 paired gain is negative with the seed-level bootstrap CI wholly below zero (anchor sign at the selection row); among survivors, argmin |mean_gain16 − target| with target = 4816/4891 − 1 = −0.0153363 (the selection row's anchor gain, committed-constant-derived; displayed as −1.5% in the Q&A table). Zero-width magnitude window; no window width invented. Under the rounded −1.50% reading the same point wins (|Δ| 0.10pp vs 0.11pp runner-up), consistent with the user's independent check (ruling 2); the exact-target |Δ| is 0.071pp vs 0.073pp.
  - Seeds: L2_SWEEP_SEED + i for i in [0, 12) shared by both arms; bootstrap seed 20261101, 10000 resamples; horizon CALIB_STEPS = 6000; all L1 generator constants and L2 sweep constants by reference (entries 3 and 5).
  - select output: 40 CI-wholly-negative survivors; runner-ups (465.0, 0.9) −1.606%, (10000.0, 1.5) −1.408%, (1550.0, 1.1) −1.673%.
- Non-blind pre-registration: the committed L2 sign map was in view at freeze time; the rule is declared in executable form requiring only selection-set information.
- Structural argument pre-declaration (ruling 4; numbers land in entry 8 and results/m1_report.md sections 3-4, computed only during scoring): joint-irreproducibility must not rest on the argmin single point. (a) For c = 0.9 and 1.0: compute flip_16(c) and flip_32(c) at machine precision from fresh paired sims; if flip_32(c) < flip_16(c), the anchor pattern (conc-16 negative AND conc-32 positive) is structurally impossible at that exponent. (b) For c = 1.1, 1.3, 1.5: at a/b = flip_32(c) exactly (machine precision, not the rounded table integers), compute the conc-16 gain — three new cells giving the in-band conc-16 magnitude upper bound; interpolation expectation ≈ −0.06 / −0.5 / −0.55%, all far smaller in magnitude than the anchor's −1.5%. If a computed value contradicts the expectation, that is a finding, not a bug.
- Scoring prediction (openly recorded, from the committed sign map): the frozen slice's scoring rows are expected to fail — the committed map shows (464.16, 0.9) at 32: −6.10%, 48: −2.73%, 60: −4.50%, 72: −6.74%, 96: −5.00%, all against positive anchor signs. A scoring failure is a finding (N8), not a bug to tune away; any re-freeze requires a new dated entry per ruling 5.
- Did: wrote `sim/replay_qa.py` (select subcommand + shared count helper) and `tests/test_replay_qa.py` (regression pin, two N5 mutants — argmax and flipped-target both select different slices — sign-classifier check, committed-artifact count check 75/30). Suite 26/26 passed in 10.56 s.
- Next: score subcommand + scoring runs, then results/m1_report.md and the outcome entry. The scoring set {32, 48, 60, 72, 96} has not been evaluated under the frozen slice in this lane yet.

## 2026-10-23 — Lane L3 scored: Milestone 1 closed (entry 8)

- Baseline: green (entry 6); freeze at entry 7, commit `c842e85`. Scoring ran only after the freeze commit landed.
- Did: appended the `score` subcommand to `sim/replay_qa.py` (re-derives the slice, asserts equality with the ledger-frozen value, runs fresh paired sims at all six anchor-matched budgets, computes the pre-declared structural arguments, writes `results/m1_report.md`); appended two score-phase tests (slice guard asserted in BOTH directions; report structure + exact tension construction + artifact-derived counts). Suite 28/28 passed in 8.76 s.
- Found:
  - Scoring verdict under the frozen slice (a/b = 464.15888336127773, c = 0.9): conc-16 (selection row) −1.60% [−1.62, −1.59] — sign YES; scoring rows 32: −6.10% [−6.12, −6.08], 48: −2.73% [−2.74, −2.72], 60: −4.50% [−4.51, −4.49], 72: −6.74% [−6.76, −6.71], 96: −5.00% [−5.01, −4.99] — ALL CI-wholly-negative against positive anchor signs: **5/5 scoring rows sign-FAIL** (finding per N8, pre-declared in entry 7).
  - Structural argument (a): machine-precision zero-gain curves — c = 0.9: flip_16 = 139.7 > flip_32 = 121.2; c = 1.0: flip_16 = 254.0 > flip_32 = 243.8 → the anchor pattern is structurally impossible at c ≤ 1.0 (no a/b gives conc-16 negative AND conc-32 positive).
  - Structural argument (b): conc-16 gain evaluated AT a/b = flip_32(c) exactly — c = 1.1: flip_32 = 476.45 → −0.097%; c = 1.3: 1716.04 → −0.414%; c = 1.5: 5870.10 → −0.688%. All negative and far smaller in magnitude than the anchor's −1.5%; all consistent with the pre-declared interpolation expectations (−0.06 / −0.5 / −0.55%) within tolerance — no contradiction, no new finding from (b).
  - HEADLINE (N8): the anchor's conc-16 magnitude and its scoring-row signs are JOINTLY IRREPRODUCIBLE in the t_pow family under this generator — established structurally, independent of the argmin slice. Where the family DOES reproduce the sign flip (the committed band: (465.0, 1.1), (1550.0, 1.3), (4900.0, 1.5)), conc-16 lands at −0.05% to −0.37%.
  - Binding discrepancy: the dynamic accepted length. Exact construction (ruling 2): anchor conc-96 accepted/verified = 3.15/3.24 = 0.9722 vs generator 2.656/3.240 = 0.8198 at the matched budget — fresh values reproducing entry 5 bit-wise.
  - Bit-reproducibility: the fresh score sims reproduce the committed L2 numbers exactly (achieved lengths, per-row gains); the ruling-4 sweep re-run produced a zero-byte `git diff results/sweep_l2.md`.
  - Calibration: mean 3.3042, all rows in band — SATISFIED (report section 2), bit-identical to entries 3/5.
- Milestone 1 closes per §7.3 DoD: `results/m1_report.md` committed with the five mandated sections; suite green; ledger complete; all work committed by explicit path. No re-freeze was needed (no scoring outcome influenced the frozen slice; the prediction in entry 7 was confirmed, not tuned toward).
- Next: Milestone 2 (not started, per ruling 6). Proposals with priorities in the session report: ranking-error injection (rank-sufficiency quantification), ZOS staleness regret, FDR-style selective-verification wrapper — each builds directly on this harness.
