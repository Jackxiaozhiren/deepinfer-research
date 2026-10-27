# MASTER PROMPT — Milestone 3: selective-verification error control (FDR-style)

> **How to use this document.** Open it from inside a coding agent working in
> the repository at `/Users/jackson/deepinfer-research`. Protocol is
> HANDOFF_PROMPT.md Sections 4-6 and 8-10 (absolute, referenced not copied);
> HANDOFF_PROMPT_MU.md documents the upstream lane (closed — PR #236, no
> follow-up in this milestone). One lane family, three lanes, sequential. Do
> not paste any handoff document into a chat window — read from disk.

## 1. How to use this document

Ledger numbering continues at **entry 28** (entries 1-27 are complete; the
research repo is pushed through entry 27 at `cadb036`). The §6 triage stop
applies per lane. Non-negotiables N1-N10 and §10 are absolute. Milestones
1/2/S/R/U are CLOSED.

## 2. Mission

The project's very first code-level finding: the engine fuses
`|actual_accept_prob − predicted_confidence|` into a `batch_confidence_error`
statistic (asc/sampler.py:556-558 → sequence.py:248 → stats_logger.py:236) —
**a calibration monitor that never feeds the allocation policy**. Milestone 3
builds the missing loop in simulation and measures its price:

> Can the allocator bound its wasted verification (verified-but-rejected
> draft positions) — the "false discoveries" of this setting — without
> giving back the gains Milestones 1-2 mapped?

**Waste semantics (Ruling A, pinned):** per step, W = rejected_draft_positions
/ verified_draft_positions. Draft positions only; the fresh base row is not a
verification and stays out of both numerator and denominator. Reference
points from the anchor table (published claims, N8): fixed-5 W ≈ 34%
(5.00 verified, 3.30 accepted); dynamic at conc 16 W ≈ 27% (4.44/3.24).

## 3. Verified state (reference)

Research repo HEAD `cadb036` (pushed), ledger entries 1-27, guards 75, fast
suite 113. Frozen surfaces: WRITEUP.md, both reports, both sweep tables,
convention.md, ledger history, LICENSE, pytest.ini — untouched by M3. The
machinery to reuse: `dflash_alloc.allocate_seqlens_q` (the allocation
primitive), `sim/rates.py` (calibrated generator, V1 + V2E levels),
`sim/rank_noise.py` (the σ noise axis + paired-pipeline harness),
`sim/cost_model.py` (cost slices), `results/sweep_m2.md` (the M2 frontier:
gain vs τ_K per conc/regime). Upstream PR #236 is OPEN — this lane does not
touch it and does not reference it in any published surface (no self-linking).

## 4. The design-space structure (hand-derived; verify empirically before building on it)

A tension discovered while planning this milestone defines its shape — verify
each claim with an assertion before relying on it (Rulings B/C):

1. **Rank-only allocation is calibration-invariant.** The M1/M2 allocator
   (global top-k on cumulative products) depends only on the ORDER of the
   cumprods. Any monotone transform that commutes with cumprod — e.g.
   `r' = r^γ` at the rate level (for rates in (0,1), x^γ preserves order and
   `(cumprod r)^γ = cumprod(r^γ)`) — leaves the allocation bit-identical.
   **Consequence:** M2's σ axis (value noise) corrupts ranking; a *global*
   calibration bias γ cannot corrupt the top-k allocation at all.
   **Verify:** a γ-commutation invariant test on `allocate_seqlens_q`
   (Ruling C). If it fails, this paragraph is wrong and that is a finding.
2. **Value-reading rules are calibration-sensitive.** A threshold rule
   ("verify position j only if predicted cumprod ≥ c") reads VALUES, and
   `cumprod^γ ≥ c ⟺ cumprod ≥ c^(1/γ)` — the same γ that leaves the top-k
   untouched reinterprets the threshold. **This is exactly why error control
   is possible:** a waste-aware rule must read values, and reading values is
   precisely what miscalibration attacks.
3. **Per-type calibration bias breaks cross-sequence comparability.** A global
   γ preserves within-sequence order AND cross-sequence cumprod order; the
   realistic miscalibration is type-dependent (`γ_i` per request type — the
   head over/under-confident on some request types), which reshuffles
   cross-sequence rankings while preserving within-sequence order. This is
   the miscalibration axis for M3 (distinct from M2's σ), and it is what a
   per-position |error| monitor like upstream's would detect.

So the three-regime design space M3 maps: **(a) pure top-k** (rank-only,
calibration-invariant, no waste control — the M1/M2 regime), **(b) pure
threshold** (value-reading, waste-controllable, γ-vulnerable), **(c) hybrid**
(e.g., top-k within threshold-eligible candidates). The deliverable is the
measured map of realized waste and gain retention across this space under
(σ, γ_type, conc) — not a theorem.

## 5. Lane rulings

- **Ruling A — waste semantics** as in §2; accounting implemented once,
  unit-tested against hand-derived cases (criterion + flip).
- **Ruling B — the calibrated-identity anchor is built and verified FIRST.**
  Under exact calibration, a threshold at predicted-cumprod ≥ c must yield
  realized waste ≈ 1−c in expectation (E[accept | predicted = c] = c). The
  harness verifies this identity on the uncorrupted generator; a failure is
  an accounting bug, never a finding.
- **Ruling C — the γ-commutation invariant** (§4.1) is asserted in the test
  suite; failure = a finding about §4, recorded per N8.
- **Ruling D — miscalibration axis:** per-type γ on the 3-type mixture
  (declared per-run); global-γ effects apply only to threshold/hybrid rules —
  the sweep must not conflate the two axes.
- **Ruling E — no guarantee language.** BH-style procedures may be implemented
  as candidate threshold families, but classic FDR guarantees assume
  calibrated, (PRDS-)dependent p-values — assumptions this generator only
  partially satisfies. Deliverables are measured per-(conc, regime) maps with
  declared caveats (the M2 single-noise-realization conditioning carries
  over). "Controls FDR" is never written; "realized waste ≤ α on this
  family" is.
- **Ruling F — `results/m3_report.md` registers as a guarded prose surface**
  (locator builder + G0-G4) before content lands, same as MR did for README.
- **Ruling G — frozen surfaces untouched** (§3 list); no upstream-PR
  follow-up (PR #236 stands alone); no WRITEUP edits in-lane (a close-out
  pointer line may be proposed separately).
- **Ruling H — budget discipline:** fast suite < 2 min (slow pin run
  separately and recorded); sweeps backgrounded with declared grids; numpy
  only.

## 6. Lanes

### 6.1 Lane M3-L1 — waste accounting + three-regime machinery

`sim/selective.py` (+ tests): waste accounting per Ruling A; the three rule
families — (a) the unchanged top-k; (b) threshold rules (fixed-c; per-batch
quantile; BH-style on 1−cumprod as the candidate p-value analog — dependence
caveats declared); (c) one hybrid (top-k within eligible set). The two
invariant/anchor tests (Rulings B, C). Calibration axis: per-type γ
(Ruling D). **DoD:** machinery + anchors green; ledger entry; no sweep yet.

### 6.2 Lane M3-L2 — the waste/gain sweep

Grid: rules × γ_type ∈ {1.0, 1.5, 2.0 (and/or 0.5)} × σ ∈ {0, 1.6} (the
M2-calibrated pair: pure ranking vs plateau) × 6 concs × anchor-matched
budgets × 12 seeds, V1 + V2E. Outputs (`results/sweep_m3.md`): realized-W
maps; gain retention vs the M2 frontier at matched settings; the Pareto map
(waste ≤ α ⇒ retained gain ≥ β%); the identity-anchor's drift under γ
(how fast "waste ≈ 1−c" degrades as calibration corrupts). Every
non-monotonicity a finding. **DoD:** tables committed; ledger entry; suites.

### 6.3 Lane M3-L3 — m3_report.md + program close-out

Five sections mirroring the house template: question/semantics; machinery +
the §4 design-space verification results; the Pareto maps and operating
points ("waste ≤ α at conc c costs β% of the ideal gain, under γ ≤ δ");
the upstream connection (batch_confidence_error as the production substrate —
stated as observation, no self-linking to PR #236); What this does NOT prove
+ program close-out (the three-path program's final state; what a Milestone 4
or an upstream follow-up would need). **DoD:** report committed as a guarded
surface; suites; ledger; tree clean.

## 7. Non-goals

No upstream-PR follow-up or self-linking; no production claims; no new
dependencies (numpy only); frozen surfaces untouched (§3); no BH-theorem
claims (E); no WRITEUP edits in-lane; Milestone-3 does not monitor PR #236.

## 8. Definition of Done (program level)

Three lanes closed with their ledger entries; `results/m3_report.md` a
guarded surface; the operating-point map stated per-(conc, regime) with the
declared caveats; the project's three-path program (research → writeup →
upstream contribution) closed with this third gap documented and mapped.

## 9. Open questions the M3-L1 triage must answer

1. The exact threshold families and their parameter grids (b), and the hybrid
   rule's precise shape (c).
2. The γ_type parameterization (per-type exponents vs additive biases) and
   how it interacts with mean-matching (does fixed-5 calibration stay in
   band under γ ≠ 1? If the band breaks, the re-matching rule must be
   declared — the L1 precedent).
3. Whether V2E is re-derived for M3 or the M2-frozen V2E carries over
   (recommendation: carry over; V2E is a frozen generator, not a fitting
   target).
4. The waste denominator at budget edges (does the fresh row count when it
   IS the verification target at wins = 0? — semantics edge case, decide
   with a hand-derived fixture).
5. Suite/sweep budgets (Ruling H).
