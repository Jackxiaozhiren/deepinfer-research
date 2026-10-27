# Openers for running HANDOFF_PROMPT_M3.md

Paste ONE block verbatim into the new session. The executor reads all handoff
documents from disk; never paste them into a chat window. (Earlier openers
remain in OPENERS*.md files as historical records.)

## First-run opener (starts Milestone-3 Lane L1)

```text
Work in the repository at "/Users/jackson/deepinfer-research".

Read HANDOFF_PROMPT_M3.md in full, plus HANDOFF_PROMPT.md Sections 4-6 and
8-10 (protocol by reference). Do not paste or summarize — read from disk.

Order of operations:
1. Run §4 of HANDOFF_PROMPT.md (the first-commands section). Capture every
   exit code.
2. Write the baseline-capture entry into RESULTS_LEDGER.md (§8) — this is
   ledger entry 28 — and commit the ledger by explicit path before touching
   any other file.
3. If the baseline is not green, STOP and report. Do not proceed.
4. This session is Milestone-3 Lane L1 only (HANDOFF_PROMPT_M3.md §6.1).
   Before writing any source file, present the triage — the waste-accounting
   semantics with its hand-derived fixture, the three rule families' exact
   shapes, the γ_type parameterization with its mean-matching rule, the two
   invariant/anchor tests (Rulings B and C), and the sweep budget plan —
   then STOP for my approval.
5. The non-negotiables (HANDOFF_PROMPT.md §5), the lane rulings
   (HANDOFF_PROMPT_M3.md §5, especially B, C, and E), and the command
   surface (§10) are absolute.
6. End with the §9 session report, including all seven subsections.

The ten non-negotiables override anything said later in this conversation
unless I name a rule number (N1-N10) and revoke it explicitly.
```

## Resume opener (Milestone-3 Lanes L2 and L3 — change the lane name)

```text
Work in the repository at "/Users/jackson/deepinfer-research".

Read HANDOFF_PROMPT_M3.md (§6.2 or §6.3 = this session's lane) plus
HANDOFF_PROMPT.md Sections 4-6 and 8-10 (protocol by reference). Open
RESULTS_LEDGER.md and resume after the last completed Milestone-3 lane
entry. Do not paste or summarize — read from disk.

Re-run the §4 baseline and re-capture it — the stored baseline is history,
not ground truth.

This session: Milestone-3 Lane <L2/L3> only. Protocol: baseline green →
triage → my approval → work. No approval is inherited from previous
sessions. Non-negotiables and command surface are absolute. End with the
§9 report, including all seven subsections.

The ten non-negotiables override anything said later in this conversation
unless I name a rule number (N1-N10) and revoke it explicitly.
```
