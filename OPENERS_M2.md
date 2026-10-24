# Openers for running HANDOFF_PROMPT_M2.md

Paste ONE block verbatim into the new session. The executor reads both
documents from disk; never paste the documents themselves into the chat
window. (Milestone-1 openers remain in OPENERS.md as the historical record.)

## First-run opener (starts Milestone-2 Lane L1)

```text
Work in the repository at "/Users/jackson/deepinfer-research".

Read HANDOFF_PROMPT_M2.md in full, together with HANDOFF_PROMPT.md
Sections 4-6 and 8-10 (protocol by reference). Do not paste or summarize
either document — read them from disk.

Order of operations:
1. Run §4 of HANDOFF_PROMPT.md (the first-commands section). Capture every
   exit code.
2. Write the baseline-capture entry into RESULTS_LEDGER.md (§8) and commit
   the ledger by explicit path before touching any source file. This entry
   MUST also append the entry-7 correction note defined in
   HANDOFF_PROMPT_M2.md §5.1 (Ruling A).
3. If the baseline is not green, STOP and report. Do not proceed.
4. This session is Milestone-2 Lane L1 only (HANDOFF_PROMPT_M2.md §5.1).
   Before writing any source file, present the implementation triage —
   including the noise-mechanism choice, the generator-variant decision
   with its Ruling-D diagnostics, and the seed-composition plan — and STOP
   for my approval.
5. The non-negotiables (HANDOFF_PROMPT.md §5) and command surface (§10)
   are absolute.
6. End with the §9 session report, including all seven subsections.

The ten non-negotiables override anything said later in this conversation
unless I name a rule number (N1-N10) and revoke it explicitly.
```

## Resume opener (Milestone-2 Lane L2, then reuse for L3 with the lane changed)

```text
Work in the repository at "/Users/jackson/deepinfer-research".

Read HANDOFF_PROMPT_M2.md in full, plus HANDOFF_PROMPT.md Sections 4-6 and
8-10 (protocol by reference). Open RESULTS_LEDGER.md and resume after the
last completed Milestone-2 lane entry. Do not paste or summarize — read
from disk.

Re-run the §4 baseline and re-capture it — the stored baseline is history,
not ground truth.

This session: Milestone-2 Lane L2 only (HANDOFF_PROMPT_M2.md §5.2).
Protocol: baseline green → triage → my approval → work. No approval is
inherited from previous sessions. Non-negotiables and command surface are
absolute. End with the §9 report, including all seven subsections.

The ten non-negotiables override anything said later in this conversation
unless I name a rule number (N1-N10) and revoke it explicitly.
```
