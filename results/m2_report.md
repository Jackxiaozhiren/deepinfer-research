# Milestone 2 report — rank sufficiency of the confidence-scheduled allocator (2026-10-24)

- Sources: results/sweep_m2.md (committed M2-L2 artifact, 1440 paired sims); frozen constants
  from ledger entries 3/5/7/10; every count and range below is parsed from the committed tables.
- Anchor-table values are published claims (N8), not ground truth.

## 1. Question and mechanism recap

Milestone 1's headline finding made ranking quality the load-bearing variable: the anchor's
conc-16 magnitude (-1.5%) and its scoring-row signs are jointly irreproducible in the t_pow
family, and the binding discrepancy is the dynamic accepted length. Milestone 2 quantifies the
consequence: **how good must the ConfidenceHead's ranking be for the allocation to retain its
benefit?** The machinery perturbs ONLY the allocator's ranking input (window lengths), never
acceptance outcomes; the deliverable is an auditable bound of the form "retaining >= X% of the
ideal-band gain at concurrency c requires Kendall tau_K >= t(c)".

## 2. Machinery, window-length semantics, and the reachable ranking-quality band

Per-cell multiplicative lognormal noise in RATE space (r' = clip(r * exp(sigma * z))); the
allocator primitive consumes rates and re-cumprods internally, so the cumprod ordering the
global top-k ranks carries the noise; acceptance always samples the TRUE rates over the first
wins' positions; k_dyn = min(k_true_prefix, wins') (Ruling B, verified elementwise by test).
Upstream justification: the allocator emits a length per sequence (seqlens_q = 1 + wins,
schedule_mtp_varlen.py:98) and the verifier accepts a prefix (effective_accept,
tilelang_ops/sampler.py:246, :778). The sigma = 0 pin reproduces the committed (465.0, 1.1)
sign-map row bit-for-bit.

**Rate space is the more faithful model, not a reparameterization.** A production
ConfidenceHead outputs probabilities in (0, 1); its predicted per-position rates therefore
produce a cumprod that is monotonically non-increasing WITHIN each sequence, so the predicted
ordering and the true ordering are never discordant on within-sequence pairs. The tau_K floor
(plateau ~0.19-0.29 across generators and concurrencies) is consequently a REAL property of
any probability-constrained head under this metric, not an artifact of the noise mechanism.
The C-space shorthand approved in the M2-L1 triage would model an impossible head (predicted
cumprods that can be non-monotone within a sequence). **Reachable ranking-quality band =
[per-conc plateau value, 1]**: V1 plateaus at 0.232-0.287, V2E at 0.190-0.243 (parsed plateau
table below); the sharper-separation generator has a lower reachable floor (more extreme
clipping, heavier ties).

## 3. Curves and thresholds (all counts parsed from results/sweep_m2.md)

Non-monotonicity record: **24 FINDING cells (18 V1 + 6 V2E)**, every one at
sigma 6.4 -> 12.8 (all carry the 6.4->12.8 signature), with
slice-invariant per-conc magnitudes — the cost-model ratio cancels in paired differences, so
this is ONE mechanism-level observation replicated across slices by construction: the
sigma=12.8 noise realization accepts more than sigma=6.4's (aggressive clipping pins high-type
rates at 1-eps, raising within-window acceptance). Recorded, not smoothed (N8).

Half-retention threshold tau-half (the auditable bound's core), stated per regime:

- Stable band over concs 48-96 (c = 1.1 slices) and 48-72 (all slices): **V1 [0.441, 0.453]**;
  **V2E [0.392, 0.404]** over the same scope (the conc-48-96 c=1.1 cells alone give
  V2E [0.392, 0.404]).
- Out-of-band cells, named with the mechanism (larger ideal gain => more noise tolerance):
  - V1 conc-32 (primary/band-edge): tau_half = 0.734 — DEGENERATE: ideal gain +0.16% halves within the first noise step — measures the CI noise floor, not retention
  - V2E conc-32 (primary/band-edge): tau_half = 0.545 — larger ideal gain +4.38% tolerates more noise; this is the ruling's stated V2E upper endpoint
  - V2E conc-16 (all slices): tau_half = 0.444 — moderate ideal gain +1.90%
  - V1 conc-96 (second regime): tau_half = 0.331 — larger ideal gain +14.64% tolerates more noise
  - V2E conc-96 (second regime): tau_half = 0.284 — largest ideal gain +20.25% tolerates most noise
    (The ruling's stated V2E upper endpoint 0.545 IS the V2E conc-32 exception cell.)
- Zero-crossing tau*: V1 sigma 0.1-1.6 (tau_K 0.271-0.734); V2E sigma 0.8-3.2 (tau_K 0.224-0.444);
  rows with sigma=0 gain <= 0 are n/a (negative at sigma=0), never interpolated (Ruling 5).
  Second-regime robustness: V1 conc-96 at (1550.0, 1.3) has NO zero-crossing in the grid
  (plateau gain +3.55 to +4.42) — at c = 1.3/conc 96 even scrambled-window dynamic retains a
  positive gain over fixed-5.
- The full per-sigma tables (2 generators x 6 concs x 10 sigma, three slices) live in
  results/sweep_m2.md; this section states only the derived thresholds and findings.

Ideal-gain multipliers (V2E/V1 at sigma = 0, both-positive rows, stated row-by-row per
Ruling 1c — only the two conc-32 rows reach 6.7-27.4x; every other row is 1.4-2.1x):

- primary conc 32: V2E +4.38% vs V1 +0.16% = 27.4x
- primary conc 48: V2E +7.63% vs V1 +3.63% = 2.1x
- primary conc 60: V2E +9.42% vs V1 +4.99% = 1.9x
- primary conc 72: V2E +11.29% vs V1 +6.09% = 1.9x
- primary conc 96: V2E +14.51% vs V1 +9.17% = 1.6x
- band_edge conc 48: V2E +7.49% vs V1 +3.50% = 2.1x
- band_edge conc 60: V2E +9.23% vs V1 +4.81% = 1.9x
- band_edge conc 72: V2E +11.04% vs V1 +5.85% = 1.9x
- band_edge conc 96: V2E +14.27% vs V1 +8.94% = 1.6x
- second conc 32: V2E +4.99% vs V1 +0.74% = 6.7x
- second conc 48: V2E +9.10% vs V1 +5.05% = 1.8x
- second conc 60: V2E +12.16% vs V1 +7.62% = 1.6x
- second conc 72: V2E +15.59% vs V1 +10.19% = 1.5x
- second conc 96: V2E +20.25% vs V1 +14.64% = 1.4x

## 4. M1-tension diagnosis — does ranking quality close the gap?

At ideal ranking (sigma = 0, conc 96, matched budget): anchor accepted/verified =
3.15/3.24 = 0.9722; V1 = 0.8198; V2E = 0.8747. The V2E adoption (Ruling D, entry 10)
closed 0.0549 of the 0.1524 gap = **36.0%**; the residual 64.0% is attributed
to rate separation and the depth profile — **ranking quality is necessary but not sufficient**:
even a PERFECT ranking (tau_K = 1) under the sharper-separation generator leaves the dynamic
windows accepting at ~0.87 of the verified length, because the generator's rates are not
bimodal enough that verified prefixes are near-certainly accepted.

Conc-16 sign region (Ruling 3 of the M2-L2 acceptance): the anchor's -1.5% sits BELOW both
variants at the primary slice — V1 -0.05% and V2E +1.90% — so the anchor's conc-16
magnitude is reproducible (if at all) only in a band whose conc-16 gain is smaller in magnitude
than the anchor's, and the sharper the separation the FURTHER the sign region moves (V2E's
conc-16 ideal gain is positive). The M1 joint-irreproducibility persists under both variants.

## 5. What this does NOT prove

- The auditable bound is a MEASURED statement per (concurrency, regime): "under this generator
  family and cost slice, retaining >= 50% of the ideal-band gain at conc c requires Kendall
  tau_K >= t(c)" with t(c) from the threshold tables. It is conditioned on ONE noise realization
  per batch geometry (the noise stream is shared across seeds): the seed-level CIs capture
  acceptance-stream variance only, NOT noise-realization variance. It is not a distributional
  guarantee.
- IDEAL-INFORMATION UPPER BOUND carries over: ideal confidence ranking, steady batch, no
  arrivals/terminations; staleness and ranking errors can only reduce gains.
- The tau_K floor (~0.19-0.29) bounds the reachable degradation: thresholds near the floor
  cannot be crossed within this mechanism — "scrambled" does not mean tau_K = 0.
- No FDR-wrapper implementation, no ZOS-staleness modeling, no cost-family re-fitting, no
  upstream-clone edits (M2 non-goals). The anchor table remains published claims (N8).
- Milestone-3 PROPOSAL (not started): (a) FDR-style selective verification on top of the
  ranking-noise pipeline — treat verified-but-rejected slots as false discoveries and bound the
  wasted-verification fraction, using the tau thresholds as operating points; (b) ZOS staleness
  regret as the alternative. Recommendation: (a) first — it reuses this machinery directly and
  the M2 thresholds give its operating points; (b) second, as it needs remap-semantics work.
