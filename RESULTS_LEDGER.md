# Results ledger (append-only)

Entries are dated. Never edit or delete an existing entry; corrections are new entries that cite the entry they supersede (HANDOFF_PROMPT.md §5 rule N7, §8).

## 2026-10-22 — project launch / direction-locking session (entry 1)

- Environment: macOS arm64. Python 3.12 in `.venv`, numpy 2.5.4, pytest 9.1.1. No torch installed; no GPU available (Mac-only, per user decision).
- Upstream clone: `/Users/jackson/DeepInfer` @ `9bd9cf2`, clean tree, read-only evidence (N1).
- `dflash_alloc.py`: 4/4 fixtures PASS — fixture1 prefix-strict `[3, 1]`; fixture2 fresh-gets-one `[1, 3]`; fixture3 zero-rate-wins-under-slack `[2, 3]`; fixture4 no-slack `[1, 1, 1]`.
- Research repo initial commit: `9fadd38` (deep-research report + allocator port).
- Locked direction: Path A — decision theory of DSpark's confidence-scheduled verification (report §6). Milestone 1 lanes L1 → L2 → L3 defined in HANDOFF_PROMPT.md §7.
- Anchor table (fixed-5 vs dynamic, conc 16-96) transcribed to HANDOFF_PROMPT.md §2 from `doc/inference_deployment_qa.md`; conc-16 row (−1.5%) designated the falsification anchor.
