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
