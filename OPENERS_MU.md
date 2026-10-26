# Openers for running HANDOFF_PROMPT_MU.md

Paste the block verbatim into the new session. The executor reads all handoff
documents from disk; never paste them into the chat window. (Earlier openers
remain in OPENERS*.md files as historical records.)

## First-run opener (Milestone-U lane, single session)

```text
Work in the repository at "/Users/jackson/deepinfer-research" (ledger home).

Read HANDOFF_PROMPT_MU.md in full, plus HANDOFF_PROMPT.md Sections 4-6 and
8-10 (protocol by reference). Do not paste or summarize — read from disk.

Order of operations:
1. Run §4 of HANDOFF_PROMPT.md (the first-commands section). Capture every
   exit code.
2. Write the baseline-capture entry into RESULTS_LEDGER.md (§8) — this is
   ledger entry 26 — and commit the ledger by explicit path before touching
   anything else.
3. If the baseline is not green, STOP and report. Do not proceed.
4. Then run HANDOFF_PROMPT_MU.md §5 step 1 (upstream-main re-verification in
   a throwaway clone; the pinned clone at /Users/jackson/DeepInfer must not
   move) and fold its result into the triage.
5. This session is the Milestone-U lane only. The triage presents the
   re-verified finding, the chosen fix option, the EXACT patch and test
   text, and the EXACT PR title and body — then STOP for my approval. No
   fork, no push, no PR exists before that approval (Ruling A).
6. The non-negotiables (HANDOFF_PROMPT.md §5), the lane rulings
   (HANDOFF_PROMPT_MU.md §4), and the command surface (§10, amended by
   Rulings A-B for fork-scoped pushes) are absolute.
7. End with the §9 session report, including all seven subsections.

The ten non-negotiables override anything said later in this conversation
unless I name a rule number (N1-N10) and revoke it explicitly.
```
