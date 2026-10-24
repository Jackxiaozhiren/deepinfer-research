# MASTER PROMPT — DeepInfer DSpark Rank-Sufficiency Study (Milestone 2)

> **How to use this document.** Open it from inside a coding agent working in
> the repository at `/Users/jackson/deepinfer-research`. This document defines
> Milestone 2's lanes and rulings; the operating protocol is HANDOFF_PROMPT.md
> Sections 4-6 and 8-10, which remain absolute (Section 4 of this file). Do not
> paste either file into a chat window — read them from disk; a truncated paste
> loses the guardrails that keep the work safe.

## 1. How to use this document

- Protocol (baseline, non-negotiables N1-N10, execution model, ledger, report contract, command surface) lives in HANDOFF_PROMPT.md and is referenced, not copied. One scope unit (lane) per session; the §6 triage stop and the §5 non-negotiables are hard.
- HANDOFF_PROMPT.md §7 (lanes L1-L3) is now the historical record of Milestone 1 — do not execute it.
- The user speaks Chinese in conversation; committed artifacts are English. Standing authorization covers running an approved lane to completion.

## 2. Mission and background

Milestone 1 (closed at research-repo commit `ac08b91`, ledger entries 1-8) established, under a bit-reproducible CPU simulator:

- The engine emits k+1 tokens per verification step; the Q&A "accepted length" statistic is the accepted draft count k (sim/convention.md).
- The anchor table's sign pattern (conc-16 negative, conc-32-96 positive) is reproducible in the T(v) = a + b·v^c family only inside convexity bands (c ≈ 1.1-1.5, moving with heterogeneity); linear and quadratic-saturation families are excluded.
- **Headline (N8 finding):** the anchor's conc-16 magnitude (−1.5%) and its scoring-row signs are JOINTLY IRREPRODUCIBLE in the t_pow family — structurally (flip-curve geometry), not just at the argmin slice. Binding discrepancy: the dynamic accepted length — anchor conc-96 accepted/verified = 3.15/3.24 = 0.9722 vs generator 2.656/3.240 = 0.8198 at the matched budget. The anchor implies near-perfect ranking over sharply separated rates.

**Milestone 2 question.** The M1 binding discrepancy makes ranking quality the load-bearing variable. M2 quantifies it: **how good must the ConfidenceHead's ranking be for DSpark's allocation to retain its benefit — and is the M1 accepted/verified tension a ranking-quality gap, a rate-separation gap, or both?** This is the positive-formulation of the user's research line (model scores → auditable decisions): the deliverable is an auditable bound of the form "to retain ≥ X% of the ideal-band gain at concurrency c, the cumprod ordering must achieve Kendall τ_K ≥ t(c)".

## 3. Verified state (all frozen; reference, do not re-derive)

- Research repo: git, branch main, HEAD `ac08b91`, clean. Upstream clone `/Users/jackson/DeepInfer` @ `9bd9cf2`, clean, read-only (N1).
- Frozen constants: L1 generator (ledger entry 3: TYPE_LEVELS, TYPE_WEIGHTS, DEPTH_DECAY, PROGRESS_DECAY, CALIB_SEED=20261022, CALIB_STEPS=6000, EMIT_BONUS=1, NEXTN=5); L2 sweep (entry 5: L2_SWEEP_SEED=20261101 ×12, BOOTSTRAP_SEED=20261101, BUDGET_GRID=(4.0, 4.24, 4.58, 4.65, 4.94, 5.44, 6.0) rows/seq, COST grids, XI_GRID, HETERO_GRID); L3 slice (entry 7: (464.15888336127773, 0.9), frozen under non-blind pre-registration).
- Committed artifacts: `sim/rates.py`, `sim/allocator.py`, `sim/cost_model.py`, `sim/sweep_l2.py`, `sim/replay_qa.py`, `results/sweep_l2.md`, `results/m1_report.md`, `sim/convention.md`, `dflash_alloc.py` (allocator primitive — reuse, never reimplement). Suite: 28 tests, all green.
- Committed reference values for pins: the sign-map row (465.0, 1.1) = conc-16 −0.05%, conc-32 +0.16%, conc-48 +3.63%, conc-60 +4.99%, conc-72 +6.09%, conc-96 +9.17%; achieved lengths per (conc, budget) in results/sweep_l2.md; the machine-precision flip curves in m1_report.md §4 (flip_32 exact: 476.45 @c=1.1, 1716.04 @c=1.3, 5870.10 @c=1.5).

## 4. Protocol by reference

HANDOFF_PROMPT.md remains operative: §4 (baseline first-commands, green-or-stop, ledger capture), §5 (N1-N10), §6 (triage stop, one lane per session, no inherited approvals), §8 (append-only ledger protocol), §9 (seven-subsection session report; commands verbatim with full paths), §10 (command surface; `uv pip install` still requires approval). Milestone-2 lane sequence below replaces §7.

## 5. Milestone-2 lanes

### 5.1 Lane M2-L1 — Ranking-noise machinery + generator decision

**Ruling A (entry-7 correction, record in the M2-L1 baseline ledger entry):** ledger entry 7 shows the selection target as "4816/4891 − 1 = −0.0153363"; the exact value is 75/4891 = −0.0153343 (verified by computation; the code computes the ratio, so no selection was affected — argmin margins 0.0716pp vs 0.0726pp as stated). Append as a correction note citing entry 7; do not edit entry 7 (N7).

**Window-length semantics (Ruling B, load-bearing):** ranking noise perturbs ONLY the allocator's input values, hence only each sequence's chosen verification-window LENGTH. Acceptance is always sampled from the TRUE rates over the first `wins'` draft positions; `k_dyn = min(k_full, wins')`. Justification from upstream code: the allocator emits a length per sequence (`seqlens_q = 1 + wins`, schedule_mtp_varlen.py:98) and the verifier accepts a prefix (`effective_accept` prefix acceptance, tilelang_ops/sampler.py:246, :778). A perturbation model that instead perturbs acceptance outcomes is a category error — the τ=0 identity pin (below) and the semantics test must distinguish the two.

**Machinery** (`sim/rank_noise.py`, new file; M1 sources are import-only): given the true per-(seq, position) cumprods C and a mechanism parameter τ, produce perturbed values C′ that feed the SAME global top-k allocation (`dflash_alloc.allocate_seqlens_q` unchanged). Candidate mechanisms — triage picks one and declares why: (i) multiplicative lognormal noise on log C′, (ii) a pairwise-swap process on the global ordering, (iii) per-sequence level confusion. Requirements, each as criterion + flip condition:
1. **τ = 0 identity pin:** at τ = 0 the full paired pipeline reproduces the committed (465.0, 1.1) sign-map row bit-for-bit (§3 reference values) — flips red on any deviation.
2. **Monotone degradation in expectation:** E[accepted extras] is non-increasing in τ over the grid (seeded, paired); a "noise" implementation that accidentally preserves or improves ordering (e.g., sorting by true values) must go red.
3. **Metric:** Kendall τ_K between the perturbed and true cumprod orderings, reported per run; implementation checked against a brute-force O(n²) version on small n.
4. **Pairing:** the noise stream has its own named seed and is consumed identically by both arms (same ambient state); the u-block consumption order is unchanged by τ (semantics test).

**Generator decision (Ruling D, declared re-freeze protocol):** the M1 tension (0.9722 vs 0.8198) motivates a sharper-separation variant of the rate family. The triage proposes exact V2 levels under mean-matching to the calibration band; V2 is ADOPTED only if all of: (a) calibration band holds (mean ∈ [3.29, 3.32], every row ∈ [3.25, 3.32]); (b) the EMIT_BONUS = 0 mutant stays red with all rows above the band top (L1's margin discipline); (c) at the primary cost slice the matched-budget accepted/verified ratio moves ≥ +0.05 toward 0.9722 (from 0.8198); (d) every constant and seed frozen in a dated ledger entry BEFORE any sweep consumes V2. If adopted, L2 runs V1 (continuity anchor) and V2 (primary); if rejected, the rejection and its numbers are recorded and M2 proceeds on V1 with the tension carried as a declared limitation.

**Definition of Done:** machinery + tests green (including both mutants and the semantics test); generator decision frozen in a dated entry (adoption or rejection, with numbers); baseline + correction note recorded; all committed by explicit path.

### 5.2 Lane M2-L2 — Gain-versus-ranking sweep

**Cost slice (Ruling C):** primary (a/b, c) = (465.0, 1.1) — mid-band, anchor-sign-reproducible. Sensitivity set: (476.45, 1.1) (the committed exact conc-32 flip — the band's edge) and (1550.0, 1.3) (a second regime). Triage may extend with justification; the set is declared in the ledger before sweeping.

**Grid:** mechanism parameter τ over a geometric grid (triage-proposed, ~7-9 points spanning "no noise" to "near-total scrambling"), conc = the 6 anchor rows, budgets = anchor-matched (BUDGET_GRID), 12 seeds under the L2 seed discipline. Both arms share seeds, type vector, u-block, and (where V2 adopted) noise stream. Report mean paired gain ± seed-level bootstrap CI (10k resamples, named seed), Kendall τ_K per run, and the matched-budget accepted/verified ratio per point.

**Outputs** (`results/sweep_m2.md`): gain-vs-τ curves per conc (per generator variant); the τ_K-vs-τ calibration curve; per-conc thresholds — zero-crossing τ* and half-retention τ½ (where mean gain drops below half its τ = 0 value) — each stated as criterion + flip condition; V1-vs-V2 comparison on the ratio diagnostic. Any non-monotonicity is a finding, recorded, not smoothed (N8).

**Definition of Done:** sweep tables committed; thresholds tabulated with their flip conditions; suite green; runtime within the §7.2-style budget (suite < 2 min, sweep minutes); ledger entry complete.

### 5.3 Lane M2-L3 — Auditable bound + M2 report

`results/m2_report.md`, five sections:
1. Question and mechanism recap (M1 headline → ranking quality as the binding variable).
2. Machinery and window-length semantics, with upstream citations and the τ = 0 pin evidence.
3. Curves and thresholds — per-row criteria and flip conditions; counts derivable from the committed tables (ruling-1 discipline from M1: no untraceable counts).
4. **M1-tension diagnosis:** with ideal ranking (τ = 0) under V2 (if adopted), what accepted/verified ratio is achieved — does ranking quality close the 0.9722-vs-0.8198 gap, or does rate separation bind? Quantified, with the residual attributed.
5. What this does NOT prove: the auditable-bound statement in its measured scope ("under this generator family and cost slice, retaining ≥ 50% of the ideal-band gain at conc c requires Kendall τ_K ≥ t(c)"); ideal-information upper-bound framing carries over (user ruling 4 of the L2 triage); no fresh-sequence dynamics, no ZOS staleness, no cost-family re-fitting. Milestone-3 proposal (FDR-style selective verification on top of the ranking results; ZOS staleness as the alternative) — proposal only, not started.

**Definition of Done:** report committed; suite green; ledger outcome entry; Milestone 2 closes.

## 6. M2 non-goals

No FDR-wrapper implementation; no ZOS-staleness modeling; no cost-family re-fitting (the t_pow bands are frozen M1 results); no upstream-clone edits or PR work; no new dependencies without §10 approval.

## 7. Open questions the M2-L1 triage must answer

1. Which noise mechanism (i)/(ii)/(iii), and why — with its failure modes named.
2. The exact V2 candidate levels (or a reasoned rejection of the V2 path), with the Ruling-D diagnostics pre-computed as evidence.
3. The τ grid endpoints and the definition of "near-total scrambling" for this batch geometry.
4. How the noise stream seeds compose with the frozen seed discipline (named, separate, paired).
