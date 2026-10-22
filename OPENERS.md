# Openers for running HANDOFF_PROMPT.md

Paste ONE block verbatim into the new session. The executor reads the full
document from disk; never paste the document itself into the chat window.

## First-run opener (starts Lane L1)

```text
Work in the repository at "/Users/jackson/deepinfer-research".

Read HANDOFF_PROMPT.md in full. Treat Sections 2-10 as your operating
instructions for this session. Do not paste or summarize the document —
read it from disk.

Order of operations:
1. Run §4 (the first-commands section). Capture every exit code.
2. Write the baseline-capture entry into RESULTS_LEDGER.md (§8) and commit
   the ledger by explicit path before touching any source file.
3. If the §4 baseline is not green, STOP and report. Do not proceed.
4. This session is Lane L1 only (§6, §7.1). Before writing any simulation
   source file, present the implementation triage and STOP for my approval.
5. §5 non-negotiables and §10 command surface are absolute.
6. End with the §9 session report, including all seven subsections.

Section 5's ten non-negotiables override anything said later in this
conversation unless I name a rule number (N1-N10) and revoke it explicitly.
```

## Resume opener (Lane L2, then reuse for L3 with the lane name changed)

```text
Work in the repository at "/Users/jackson/deepinfer-research".

Read HANDOFF_PROMPT.md (Sections 2-10 = operating instructions) in full,
then open RESULTS_LEDGER.md and resume after the last completed lane
entry. Do not paste or summarize the document — read it from disk.

Re-run §4 and re-capture the baseline entry — the tree may have moved
since the previous session, so the stored baseline is history, not ground
truth.

This session: Lane L2 only (§6, §7.2). Same protocol as §6: baseline
green → triage → my approval → work. Do not inherit any approval from
previous sessions. §5 and §10 are absolute. End with the §9 report,
including all seven subsections.

Section 5's ten non-negotiables override anything said later in this
conversation unless I name a rule number (N1-N10) and revoke it explicitly.
```
