# A published speculative-decoding speedup, audited on a laptop

*An independent reproduction study of DeepSeek's DeepInfer / DSpark serving claims —
numpy on an Apple-silicon laptop, no GPU, no model weights, no tokenizer.*

## 1. What was audited, and with what

DeepInfer is DeepSeek's production inference framework, open-sourced 2026-10-16. Its
deployment Q&A (`doc/inference_deployment_qa.md`) publishes serving numbers for
DeepSeek-V4.1-Flash, including a fixed-5 versus dynamic speculative-decoding comparison
across six concurrency levels. As published claims, dynamic verification is said to lose
-1.5% at concurrency 16 per rank and to climb monotonically through the ladder to +11.0%
at concurrency 96.

We audited that table with a seeded, CPU-only simulator of the allocator the engine
actually ships, driven by an acceptance-rate generator calibrated to the table's own
fixed-5 column. Three results survived the audit:

1. **The published sign pattern is jointly irreproducible.** Within the cost family we
   studied, no parameter slice reproduces the anchor's concurrency-16 magnitude *and* its
   higher-concurrency signs at the same time. This is structural — a property of the
   geometry of the sign-flip curves, not of a badly tuned slice.
2. **The ranking requirement is bounded.** Retaining >= 50% of the ideal-band gain from a
   perfect ranking requires the head's window-length concordance to stay above a measured
   threshold, and a probability-constrained head cannot fall below a reachable floor.
3. **Ranking quality is necessary but not sufficient.** Sharpening the ranking to a
   near-ideal head closes 36.0% of the discrepancy between our generator and the published
   table; the residual 64.0% lives in the acceptance rates themselves.

Everything below is a statement about the *internal consistency* of published claims under
stated assumptions. A model family that fails to reproduce a published number has told us
something about the number's assumption set — it has not told us the number is wrong on
production hardware.

## 2. The mechanism, as read from the open-source engine

DSpark is a DFlash draft engine plus confidence-scheduled variable-length verification.
The `ConfidenceHead` (`deepinfer/model_executor/models/mtp/dflash.py:46-79`) predicts a
per-draft-token acceptance probability. The allocator
(`deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py:35-98`) ranks
cumulative products of those predictions across the whole batch and spends a hard,
batch-level token budget `total_q` on the best prefixes. Because the values are exact
cumulative products and the costs are per-token, that is a greedy knapsack: exchange-
optimal when the predictions are exact, and the research questions all sit where that
premise breaks.

Two conventions decide every number downstream, and both were adjudicated from code rather
than from the speculative-decoding literature.

**Emission is k+1.** When k of n drafts are accepted, the step emits k+1 tokens for that
sequence — the accepted prefix plus the final fresh row: the emitted-stop-token docstring
"(accepted prefix plus final row)" (`deepinfer/model_executor/layers/tilelang_ops/sampler.py:466-467`),
the finalize path `emitted_rows = (effective_accept | is_final) & stop_hits`
(`deepinfer/model_executor/layers/tilelang_ops/sampler.py:376-379`), and the KV advance
`seqlens_k += seqlens_q - (nextn - accepted)`
(`deepinfer/model_executor/layers/tilelang_ops/compute_seqlens_k_zos.py:22-23`), which
returns k+1 under full-depth verification. Fresh sequences pack exactly one row
(`schedule_mtp_varlen.py:98`). The simulator encodes the finding as a single constant,
`EMIT_BONUS = 1`. The Q&A's "accepted length" statistic is frozen as the mean accepted
draft count k per step, excluding the bonus.

**Window length is a budget, not a verdict.** The allocator emits a length per sequence
(`seqlens_q = 1 + wins`); the verifier accepts a *prefix* of it, so the accepted draft
count is `min(k_true_prefix, wins')`. Verification budget can therefore be spent on tokens
that are rejected, which is exactly the quantity the audit tracks.

## 3. Milestone 1 — replaying the anchor table

The generator is heterogeneous across sequences by construction (a small mixture of request
types with distinct acceptance levels, per-position decay, a concurrency knob). Heterogeneity
is load-bearing: a homogeneous population makes dynamic allocation equal to fixed-5 by
construction, so it cannot produce the anchor's gain ladder at all.

Calibration lands on the table's own fixed-5 column: the accepted length k falls in
[3.29, 3.32] at every concurrency row, mean 3.3042.

| conc | anchor gain | model gain (mean [CI]) | sign match |
|---:|---:|---:|:---:|
| 16 | -1.5% | -1.60% [-1.62, -1.59] | YES |
| 32 | +1.6% | -6.10% | NO |
| 48 | +3.6% | -2.73% | NO |
| 60 | +5.4% | -4.50% | NO |
| 72 | +6.7% | -6.74% | NO |
| 96 | +11.0% | -5.00% | NO |

The selection row reproduces, including its magnitude. All five scoring rows fail, and they
fail in the same direction. At concurrency 16 the mechanism is visible in the raw lengths:
dynamic verifies 4.438 tokens per sequence and accepts 3.196 of them, while fixed-5 accepts
3.307 — over-verification at low concurrency spends budget on drafts that get rejected,
which is what makes the dynamic arm lose there. The discrepancy above that row is not
diffuse: it is concentrated in one
quantity. At concurrency 96, under the matched budget, the published pair is
3.15/3.24 = 0.9722, while our calibrated generator reaches
2.656/3.240 = 0.8198. Near-perfect ranking over sharply separated rates is what the
published ratio implies; that is the binding constraint on every row above.

The sign-flip geometry turns the local failure into a structural one. The committed L2 sign
map evaluates 75 t_pow cost points (15 a/b values x 5 exponents) and 30 saturation points
(10 x 3). For every exponent where the anchor's pattern is even reachable, the concurrency-16
zero-crossing sits *above* the concurrency-32 one: a model whose dynamic arm loses at low
concurrency and wins at higher concurrency requires the flip point to fall as concurrency
rises, and within this family it rises. Where the pattern is reproducible — at
(465.0, 1.1), (1550.0, 1.3), (4900.0, 1.5) — concurrency 16 lands at -0.05% to -0.37%,
nowhere near the published -1.5%. The magnitude and the signs are jointly irreproducible.

The saturation cost family reproduces no anchor-pattern point at all, and is excluded from
slice selection rather than from the report.

## 4. Milestone 2 — how good must the ranking be?

Milestone 1 makes ranking quality the load-bearing variable. Milestone 2 measures it. The
machinery perturbs only the allocator's *ranking input* — multiplicative lognormal noise in
rate space, `r' = clip(r * exp(sigma * z))` — while acceptance always samples the true rates
over the winning prefix. 1440 paired sims = 2 generators x 6 concurrencies x 10 noise
levels x 12 seeds; the three cost slices are analytic transforms of that one shared run
set, not additional runs.

Two things fall out that a laptop audit could not have asked for in advance.

**The ranking quality has a reachable floor, not a zero.** A production head emits
probabilities in (0, 1), so its predicted cumulative products are monotonically
non-increasing within each sequence, and predicted and true orderings can never disagree on
within-sequence pairs. Under a Kendall tau_K measured on window lengths, that bounds
degradation from below: the plateau is 0.232-0.287 for the base generator (V1) and
0.190-0.243 for the sharper-separation variant (V2E), i.e. a floor around ~0.19-0.29.
"Scrambled ranking" does not mean tau_K = 0. Reachable ranking quality is the band
[per-concurrency plateau, 1].

**The auditable bound.** Retaining >= 50% of the ideal-band gain requires, in the stable
regime (concurrency 48-96 at exponent c = 1.1, and 48-72 across slices):

- V1 [0.441, 0.453]
- V2E [0.392, 0.404]

with named exceptions, each driven by a larger ideal gain to halve from: V1 at
concurrency 32 needs 0.734 and is degenerate (its ideal gain +0.16% halves within the first
noise step, so that threshold measures the CI noise floor); V2E at concurrency 32 needs
0.545 (ideal gain +4.38%); V2E at concurrency 16 needs 0.444; in the second cost regime
concurrency 96 needs 0.331 for V1 and 0.284 for V2E. The sharper generator tolerates less
concordance because it has more to lose. The generator comparison is not uniform: only the
two concurrency-32 rows reach 6.7-27.4x on ideal gain, every other positive row is
1.4-2.1x.

One non-monotonicity is recorded rather than smoothed away: 24 FINDING cells (18 V1 + 6 V2E),
all of them at sigma 6.4 -> 12.8, with slice-invariant magnitudes. Aggressive clipping pins
high-type rates near 1 at that noise level, which *raises* within-window acceptance — one
mechanism-level observation, replicated across slices by construction.

**The attribution.** Re-running the Milestone-1 tension under a near-ideal head: anchor
0.9722, V1 0.8198, V2E 0.8747. Adopting the sharper-separation generator closes 0.0549 of
the 0.1524 gap = 36.0%. The residual 64.0% is attributed to rate separation and the depth
profile: even a perfect ranking under the sharper generator leaves verified windows
accepting at ~0.87 of their length, because the rates are not bimodal enough to make
verified prefixes near-certain accepts. Concurrency 16 stays unresolved in the same
direction: the anchor's -1.5% sits below both variants (-0.05% for V1, +1.90% for V2E).
Ranking quality is necessary but not sufficient.

## 5. What this does NOT prove

- **Published claims are not ground truth.** The Q&A table is a set of self-reported
  serving numbers. Reproducing a sign is not verification of production truth; failing to
  reproduce one is a finding about this model family under these assumptions, not a bug to
  tune away.
- **Ideal information.** The allocator reads current-step true rates through an idealized
  head, on a steady batch with no arrivals, terminations, or per-step budget adaptation. All
  reported dynamic gains are upper bounds: staleness and ranking error can only reduce them.
- **One noise realization.** The threshold table is conditioned on a single noise stream per
  batch geometry; seed-level bootstrap CIs capture acceptance-stream variance, not
  noise-realization variance. The bound is a measured per-(concurrency, regime) statement,
  not a distributional guarantee.
- **Self-consistency only.** The anchor audit is an internal-consistency check on published
  numbers. It says nothing about the quality of DeepSeek's engineering or of hardware we
  never touched.
- **The cost model is an assumption**, linear plus per-step overhead with a saturation
  family as a second option; the deliverable is a sensitivity sweep, not a point claim about
  production hardware.

## 6. Reproduce it yourself

Environment: Python 3.12.14, numpy 2.5.4, pytest 9.1.1, in the repo's own `.venv`; no
torch, no downloads, no network in the experiments. Every stochastic step takes an explicit
seed: generator seed 20261022, horizon 6000, paired seeds `L2_SWEEP_SEED + i` for i in
[0, 12), bootstrap seed 20261101 with 10000 resamples.

```bash
git clone https://github.com/Jackxiaozhiren/deepinfer-research.git deepinfer-research
cd deepinfer-research
python3 -m venv .venv && .venv/bin/python -m pip install numpy==2.5.4 pytest==9.1.1
.venv/bin/python dflash_alloc.py            # allocator port + hand-derived fixtures
.venv/bin/python -m sim.sweep_l2            # sign map (75 t_pow + 30 saturation points)
.venv/bin/python -m sim.sweep_m2            # ranking-noise sweep (1440 paired sims)
.venv/bin/python -m pytest                  # fast suite
.venv/bin/python -m pytest -m slow           # full-scale pins
```

Committed artifacts: `results/m1_report.md`, `results/m2_report.md`, `results/sweep_l2.md`,
`results/sweep_m2.md`, `sim/convention.md`, and the session ledger `RESULTS_LEDGER.md`.
Milestone 1 was frozen and scored at commit `c842e85`; Milestone 2 closed at `be55c16`;
every number in this writeup is regex-captured from those artifacts by
`sim/build_writeup_numbers.py`, and `tests/test_writeup_numbers.py` fails if a number in
prose is not one of them.

## 7. Attribution

DeepInfer and DSpark facts are read from the public repository
([deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)) at commit `9bd9cf2`,
Apache-2.0 licensed. The anchor table is quoted as published claims from
`doc/inference_deployment_qa.md`. This analysis is independent and unaffiliated with
DeepSeek; it neither received nor sought their input, and it does not claim their production
numbers are wrong.
