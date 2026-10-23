# Milestone 1 report — anchor-table replay (Lane L3, 2026-10-23)

- Frozen slice (ledger entry 7): (a/b = 464.15888336127773, c = 0.9); commit at freeze `c842e85`.
- Seeds: L2_SWEEP_SEED + i, i in [0, 12), shared by both arms; bootstrap seed 20261101, 10000 resamples; horizon 6000 steps.
- Anchor-table values are published claims (N8), not ground truth.

## 1. Convention derivation summary

The engine emits **k+1** tokens per verification step (accepted prefix plus the final
fresh row): decisive sites are the emitted-stop-token docstring "(accepted prefix plus
final row)" (tilelang_ops/sampler.py:466-467), the finalize-path comment +
`emitted_rows = (effective_accept | is_final) & stop_hits` (tilelang_ops/sampler.py:376-379),
and the KV advance `seqlens_k += seqlens_q - (nextn - accepted)` (compute_seqlens_k_zos.py:22-23)
which returns k+1 under full-depth verification; fresh sequences pack exactly 1 row
(schedule_mtp_varlen.py:98). `EMIT_BONUS = 1` in sim/rates.py encodes this. The Q&A
"accepted length" statistic is frozen as the mean accepted draft count k per step
(excluding the bonus; num_generated_tokens / num_accepted_tokens_gpu semantics); the
k+1 reading is the declared first re-freeze candidate (sim/convention.md).

## 2. Calibration table

Fixed-5 accepted length k per concurrency row (frozen generator constants, seed 20261022,
horizon 6000):

| conc | fixed-5 accepted k |
|---:|---:|
| 16 | 3.3058 |
| 32 | 3.3007 |
| 48 | 3.3050 |
| 60 | 3.3036 |
| 72 | 3.3062 |
| 96 | 3.3040 |

Criterion: mean across rows in [3.29, 3.32] and every row in [3.25, 3.32] — SATISFIED; mean = 3.3042. Flip: any row or the mean leaving the band.

## 3. Replay table (frozen slice)

| conc | anchor gain | anchor sign | model gain (mean [CI]) | model sign | sign match | model verified_dyn | model accepted dyn/fix |
|---:|---:|:---:|---:|:---:|:---:|---:|---|
| 16 | -1.5% | - | -1.60% [-1.62, -1.59] | - | YES | 4.438 | 3.196 / 3.307 |
| 32 | +1.6% | + | -6.10% [-6.12, -6.08] | - | NO | 3.656 | 2.882 / 3.307 |
| 48 | +3.6% | + | -2.73% [-2.74, -2.72] | - | NO | 3.938 | 3.014 / 3.307 |
| 60 | +5.4% | + | -4.50% [-4.51, -4.49] | - | NO | 3.583 | 2.848 / 3.308 |
| 72 | +6.7% | + | -6.74% [-6.76, -6.71] | - | NO | 3.236 | 2.654 / 3.307 |
| 96 | +11.0% | + | -5.00% [-5.01, -4.99] | - | NO | 3.240 | 2.656 / 3.307 |

Per-row criteria and flip conditions (N6):

- conc 16 (selection row): criterion — the model's paired gain is negative with the CI wholly
  below zero, matching the anchor's -1.5% sign; magnitude closeness was the declared argmin
  constraint at freeze time. Flip: CI crossing zero or a positive mean.
- conc 32 (scoring row): criterion — the model's paired-gain sign matches the anchor's + sign; flip — the CI classifies the opposite sign or is not significant.
- conc 48 (scoring row): criterion — the model's paired-gain sign matches the anchor's + sign; flip — the CI classifies the opposite sign or is not significant.
- conc 60 (scoring row): criterion — the model's paired-gain sign matches the anchor's + sign; flip — the CI classifies the opposite sign or is not significant.
- conc 72 (scoring row): criterion — the model's paired-gain sign matches the anchor's + sign; flip — the CI classifies the opposite sign or is not significant.
- conc 96 (scoring row): criterion — the model's paired-gain sign matches the anchor's + sign; flip — the CI classifies the opposite sign or is not significant.

Accepted/verified tension, exact construction (ruling: never as bare 0.97-vs-0.82): the anchor's
conc-96 dynamic accepted/verified = 3.15/3.24 = 0.9722, while the generator at the matched
budget yields 2.656/3.240 = 0.8198.
The anchor implies near-perfect ranking over sharply separated rates; this generator's
realized dynamic acceptance is the binding discrepancy behind every scoring-row mismatch above.

## 4. Sensitivity summary — where the family reproduces the anchor's sign flip

Structural argument (a) — exponents c = 0.9 and 1.0, machine-precision zero-gain curves from
the fresh paired sims:

| c | flip_16 (a/b) | flip_32 (a/b) | flip_32 < flip_16 | anchor pattern possible? |
|---:|---:|---:|:---:|:---:|
| 0.9 | 139.7 | 121.2 | yes | no — structurally impossible |
| 1.0 | 254.0 | 243.8 | yes | no — structurally impossible |

Structural argument (b) — conc-16 gain evaluated AT a/b = flip_32(c) exactly (three new cells;
the conc-32 boundary is the best case for the anchor pattern at that exponent):

| c | flip_32 (a/b, exact) | conc-16 gain at that a/b | interpolation expectation | consistent? |
|---:|---:|---:|---:|:---:|
| 1.1 | 476.45 | -0.097% | -0.06% | yes |
| 1.3 | 1716.04 | -0.414% | -0.50% | yes |
| 1.5 | 5870.10 | -0.688% | -0.55% | yes |

Reading: at every exponent where the anchor-pattern band is non-empty, the conc-16 magnitude at the
band's conc-32 edge stays far above (smaller than) the anchor's -1.5% — so the anchor's conc-16
magnitude and its scoring-row signs are jointly irreproducible within the t_pow family under this
generator, independent of the argmin slice. Where the pattern IS reproducible ((465.0, 1.1),
(1550.0, 1.3), (4900.0, 1.5) in the committed map), conc-16 lands at -0.05% to -0.37%.

Committed-artifact counts (ruling 1): the L2 sign map evaluates 75 t_pow cost points
(15 a/b values x 5 exponents) and 30 saturation points (10 x 3); the saturation
family reproduces no anchor-pattern point (entry-5 FINDING) and is excluded from slice selection.

## 5. What this does NOT prove

- IDEAL-INFORMATION UPPER BOUND: the allocator reads current-step true rates (idealized
  ConfidenceHead); the batch is steady (identity remap, no arrivals/terminations). All dynamic
  gains above are upper bounds under ideal confidence; staleness and ranking errors can only
  reduce them (user ruling 4 of the L2 triage).
- The cost families and the throughput identity num_seqs * E[emitted] / T(v) are inference
  assumptions (cost_model.py docstring), not code facts; this report maps which parameter
  regions reproduce the anchor's signs — it does not fit production hardware.
- The anchor table is a set of published claims (N8). A scoring-row mismatch is a finding about
  this model family under these assumptions, not a bug to tune away; reproducing a sign is not
  verification of production truth.
- NON-BLIND: the committed L2 sign map was in view at freeze time (entry 7); the freeze rule is
  executable from selection-set information alone, but the scoring-row outcomes were already
  public in the repo when the slice was frozen.
- Declared modeling choices: horizon 6000 steps; stratified type composition; the progress term
  (design-driven, no empirical meaning); no fresh-sequence dynamics, no request termination, no
  zero-rate pathways, no per-step budget adaptation (sim/convention.md).
