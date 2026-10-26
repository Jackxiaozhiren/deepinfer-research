# MASTER PROMPT — Milestone U: upstream contribution (zero-rate-slack finding)

> **How to use this document.** Open it from inside a coding agent working in
> the repository at `/Users/jackson/deepinfer-research` (ledger home), with the
> work happening in a SEPARATE clone of the fork (see §5). Protocol is
> HANDOFF_PROMPT.md Sections 4-6 and 8-10 (absolute); Milestones 1/2/S/R are
> closed (ledger entries 1-25). Do not paste any handoff document into a chat
> window — read from disk.

## 1. How to use this document

Single lane, but with an external-visibility gate that overrides the usual
flow: **nothing external happens until the user has seen and approved the
exact PR title, body, and patch at the triage** (Ruling A). The §6 triage stop
therefore presents more than a plan — it presents the artifacts of the
contribution itself.

## 2. Mission

Deliver Milestone 1's edge-case finding to upstream deepseek-ai/DeepInfer as a
contribution:

**The finding.** In the confidence-scheduled allocator
(`deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py`, ref
implementation lines 35-98 @ `9bd9cf2`), positions with zero cumulative
marginal participate in the global top-k with a lowest-flat-index stable
tie-break. Under budget slack (`total_q − num_seqs` exceeding the count of
positive marginals), zero-confidence sequences win extra verification slots —
contradicting the code comment's stated intent ("fresh sequences get rates 0
so cumprod stays 0" — implying they stay at one token). Verified by our
fixture 3 (`dflash_alloc.py`): rates `[[0.5, 0.2]]`, remap `[−1, 0]`,
`total_q = 5` ⇒ seqlens `[2, 3]`, not `[1, 4]`.

**Why it is more than a nitpick — two production pathways deliver zero-rate
rows:** (1) fresh sequences (`remap == −1`); (2) post-stop requests, whose
next-round draft confidences are masked to 0.0
(`deepinfer/model_executor/layers/sampler.py:252-254`).

## 3. Verified state (reference)

- Our upstream clone `/Users/jackson/DeepInfer` is pinned @ `9bd9cf2` (2026-10-21) and MUST NOT move (N1) — upstream-main inspection happens in a separate clone (§5 step 1).
- The finding is fixture-reproduced and documented in `dflash_alloc.py` (fixture 3) and `sim/convention.md` ("Zero-rate production pathways").
- Research-repo ledger: entries 1-25; this lane's baseline is **entry 26**.
- Upstream repo metadata at 2026-10-22: `has_issues: false`, `has_pull_requests: true` (PR-only contribution path) — RE-VERIFY, it may have changed.
- Identity: future commits use the noreply email
  `104724357+Jackxiaozhiren@users.noreply.github.com` (the research-repo local
  config is a precedent; the fork clone needs its own local config).

## 4. Lane rulings

- **Ruling A — external-visibility gate.** The fork creation, branch push, and
  PR opening are individually gated: the triage presents the exact patch,
  test, PR title, and PR body, and the user approves the wording before any
  fork/push/PR exists. A generic "push authorization" does not cover this —
  the text itself is the approval object.
- **Ruling B — flow.** `gh repo fork deepseek-ai/DeepInfer --clone=false` (the
  fork appears on the user's profile — gated at triage), branch, patch + test,
  push to the FORK only, `gh pr create` against upstream. There is no write
  access to upstream and none is sought.
- **Ruling C — the research repo is read-mostly here.** Its ledger takes this
  lane's entries; its code/artifacts change not at all. The working clone for
  the patch is the fork clone in its own directory.
- **Ruling D — fix options adjudicated at triage**, with the upstream test
  suite as the gate and maintainer preference as the criterion:
  (a) behavioral fix aligning the allocator with the documented intent
      (e.g., exclude non-positive marginals from ranking eligibility) + test;
  (b) documentation-only (align the comment with the actual behavior) + an
      edge-case test that pins current behavior;
  (c) finding-report PR without a behavior change (description carries the
      full writeup; patch minimal or empty).
  Note the severity context for that decision: production budgets are usually
  tight (the slack case is rare), which argues for the lowest-friction option
  unless upstream signals otherwise.
- **Ruling E — the PR description must carry the two production pathways with
  `file:line` citations.** That is what elevates this from nitpick to finding.
- **Ruling F — identity.** Fork commits carry the noreply email (local config
  in the fork clone); PR text in English; no school email anywhere.
- **Ruling G — honesty boundaries stated, not hidden.** The upstream test file
  hardcodes `device="cuda"`: their GPU suite cannot run on this Mac. Local
  verification of the patch runs the reference implementation on CPU tensors
  via a small driver (torch CPU + numpy in the fork clone's own venv — a new
  dependency, approved at triage per §10), and the PR states exactly what was
  and was not verified locally. Their CI (if any) runs their own suite.

## 5. Lane U-L1 specification

Ordered steps:

1. **Baseline** (research-repo ledger entry 26: §4 of HANDOFF_PROMPT.md, green
   criteria) **+ upstream-main re-verification**: in a THROWAWAY clone of
   upstream main, check whether `schedule_mtp_varlen.py` (and the reference
   implementation, the sampler masking site, the docstring) changed since
   `9bd9cf2`; re-run the finding against current main if changed; report
   either way. Also re-check `has_issues`/discussions state via `gh api`.
   The throwaway clone is scratch (N9-hygiene: delete after).
2. **Triage** (hard stop): present the re-verified finding, the chosen fix
   option (D) with the maintainer-preference argument, the exact patch and
   test text, the exact PR title and body, and the fork/identity plan. STOP
   for approval.
3. **Execute the approved plan**: fork → branch → patch + test → local
   CPU-driver verification (Ruling G) → push to fork → open the PR with the
   approved text.
4. **Close-out**: record the PR URL in the ledger (entry 27); both repos clean
   (research repo: ledger-only; fork: committed work); upstream response is
   outside this lane's control and is NOT monitored automatically.

## 6. Non-goals

No Milestone-3 work; no changes to the research repo's code/artifacts (ledger
only); no contact with DeepSeek outside the PR (no emails, DMs, or mentions);
no social-preview or WRITEUP edits; no surveillance of the PR after opening.

## 7. Definition of Done

Either (a) the PR is opened with the user-approved text and its URL is
recorded in the ledger; or (b) step 1 shows upstream changed enough that the
finding needs re-adjudication, and the user re-decides with the fresh
evidence. In both cases: ledger complete, both repos clean, the fork state
matches exactly what the user approved.

## 8. Open questions the triage must answer

1. Upstream-main re-verification result (finding still present? code drift?).
2. Issue vs PR-only path (current `has_issues` state).
3. Fix option (a)/(b)/(c) and the maintainer-preference argument.
4. The exact PR title/body (English, carrying the two pathways with
   citations) and the exact patch + test.
5. Fork/identity plan (noreply email, branch name, commit messages).
6. The local-verification boundary: what the CPU driver does and does not
   prove, stated as criterion + flip condition.
