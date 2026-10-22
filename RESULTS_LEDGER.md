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
