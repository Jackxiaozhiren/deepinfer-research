# Openers for running HANDOFF_PROMPT_MR.md

Paste the block verbatim into the new session. The executor reads all handoff
documents from disk; never paste them into the chat window. (Earlier openers
remain in OPENERS.md / OPENERS_M2.md / OPENERS_MS.md as historical records.)

## First-run opener (Milestone-R lane, single session)

```text
Work in the repository at "/Users/jackson/deepinfer-research".

Read HANDOFF_PROMPT_MR.md in full, plus HANDOFF_PROMPT.md Sections 4-6 and
8-10 (protocol by reference). Do not paste or summarize — read from disk.

Order of operations:
1. Run §4 of HANDOFF_PROMPT.md (the first-commands section). Capture every
   exit code.
2. Write the baseline-capture entry into RESULTS_LEDGER.md (§8) — this entry
   MUST be ledger entry 20 recording the gold-check datum from
   HANDOFF_PROMPT_MR.md §3 — and commit the ledger by explicit path before
   touching any other file.
3. If the baseline is not green, STOP and report. Do not proceed.
4. This session is the Milestone-R lane only (HANDOFF_PROMPT_MR.md §5).
   Before changing any file, present the triage — exemplar findings with
   URLs, the README outline, the exact About/topics text, the release-notes
   draft, the guard-extension design, and the CI decision — then STOP for
   my approval.
5. The non-negotiables (HANDOFF_PROMPT.md §5), the lane rulings
   (HANDOFF_PROMPT_MR.md §4, especially A and G), and the command surface
   (§10) are absolute.
6. End with the §9 session report, including all seven subsections.

The ten non-negotiables override anything said later in this conversation
unless I name a rule number (N1-N10) and revoke it explicitly.
```
