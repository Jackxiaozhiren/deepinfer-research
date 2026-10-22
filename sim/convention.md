# Emission convention adjudication (Lane L1)

Frozen 2026-10-22, lane L1. All sources read from `/Users/jackson/DeepInfer`
@ `9bd9cf2` (read-only evidence, rule N1). Every file:line below was opened
and read in the adjudicating session.

## Question 1 — does a verification step emit k or k+1 tokens?

**Answer: k+1 (`EMIT_BONUS = 1`).** When k of the verified draft positions are
accepted, the engine emits for that sequence in that step the k accepted draft
tokens plus one bonus token sampled at the sequence's final fresh-sampling row.

### Evidence chain

1. **Row layout.** The allocator gives every sequence `seqlens_q = 1 + wins`
   ([schedule_mtp_varlen.py:72](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L72),
   [schedule_mtp_varlen.py:98](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L98)):
   one base row beyond the won (verified) draft positions. A fresh sequence
   (`remap == -1`) has zero rates, `wins = 0`, and therefore exactly 1 row —
   the fresh row — and emits 1 token per step.
2. **Every request has a final fresh-sampling row.** `is_verify` is True for
   verification rows and False exactly at "each request's final fresh-sampling
   row" ([tilelang_ops/sampler.py:442-443](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L442)).
3. **The fresh row follows the accepted prefix.** `last_valid_offset` is the
   "Request-local offset of the first rejected row, or of the final fresh row
   when all drafts pass"
   ([tilelang_ops/sampler.py:458-459](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L458)).
4. **Stop-token scan covers the emitted stream.** `emitted_stop_token` is True
   when "the request's emitted tokens this step **(accepted prefix plus final
   row)** contain any of its `stop_token_ids`"
   ([tilelang_ops/sampler.py:466-467](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L466)).
5. **Torch finalize path states it in a comment and encodes it.** "Emitted
   tokens per request are the accepted prefix plus the final row";
   `emitted_rows = (effective_accept | is_final) & stop_hits`
   ([tilelang_ops/sampler.py:376-379](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L376))
   — an independent site from the kernel docstrings.
6. **The generated-token counters count accepted draft rows only.**
   `effective_accept = alive & is_verify & accept`
   ([tilelang_ops/sampler.py:246](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L246))
   excludes the fresh row; the CUDA kernel increments
   `num_generated_tokens` only `if accepted and step < num_steps`
   ([tilelang_ops/cuda/sampler.py:366-367](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/cuda/sampler.py#L366));
   the Ascend kernel casts `generated = accepted and step < num_steps`
   ([asc/sampler.py:544](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py#L544),
   aggregated at
   [:587](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py#L587)).
   The fresh row's emission is unconditional; it is just not counted by this
   counter.
7. **KV bookkeeping advances the context by k+1.** ZOS correction:
   `seqlens_k[i] += seqlens_q[i]; seqlens_k[i] -= nextn - prev_num_accepted_tokens[j]`
   ([compute_seqlens_k_zos.py:22-23](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/compute_seqlens_k_zos.py#L22),
   reference implementation at
   [:44-52](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/compute_seqlens_k_zos.py#L44)).
   Full-depth verification packs `nextn + 1` rows (nextn draft rows + 1 fresh
   row), so the advance is `nextn + 1 - (nextn - k) = k + 1`. A fresh sequence
   packs 1 row with no adjustment and advances exactly 1.
8. **Per-sequence accepted count.** `num_accepted_tokens_gpu = last_valid_offset`
   ([layers/sampler.py:289](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/sampler.py#L289)),
   i.e. k per sequence in `[0, nextn]`, matching the field docstring at
   [sequence.py:265-266](file:///Users/jackson/DeepInfer/deepinfer/sequence.py#L265).

**Why not k:** if the engine emitted only the k accepted drafts, the KV
advance for full-depth verification would be k, but the code advances k+1
(evidence 7) — the extra slot holds the bonus token. The stop-token scan also
covers the final row (evidence 4-5), so the bonus token is part of the emitted
stream, not bookkeeping only.

**Correction note (2026-10-22, user review of the triage):** the triage chat
message misattributed `effective_accept` to `layers/sampler.py:246`; the
correct location is
[tilelang_ops/sampler.py:246](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L246),
as recorded here.

## Question 2 — which statistic does the Q&A "accepted length" report?

**Frozen choice: the mean accepted draft count k per step per sequence
(excludes the bonus token).** Rationale: the engine counters that match the
table's two columns both count k —

- "accepted length" ↔ `num_generated_tokens`, documented as "Accepted draft
  count at each speculative step"
  ([tilelang_ops/sampler.py:460-461](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L460));
- "verified length" ↔ `num_verified_tokens`, "Verification-row count at each
  speculative step"
  ([tilelang_ops/sampler.py:462-463](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L462)),
  which is 5.00 under fixed-5 only if the fresh row is excluded from the
  count.

The Q&A states its numbers are computed from EP64/DP64 benchmark data
([inference_deployment_qa.md:243-247](file:///Users/jackson/DeepInfer/doc/inference_deployment_qa.md#L243);
anchor table at
[:249-256](file:///Users/jackson/DeepInfer/doc/inference_deployment_qa.md#L249),
re-read this session).

**Declared alternative (not frozen):** in the speculative-decoding literature
"accepted length" often includes the bonus token (the k+1 reading). Under that
reading every calibration target shifts down by exactly 1 and `EMIT_BONUS`
enters the accepted-length statistic directly. If lane L3's frozen slice fails
scoring rows, this reading is the first re-freeze candidate, via a new dated
ledger entry (rules N7/N8) — never a silent retune.

## Zero-rate production pathways (evidence for research gap 2)

Zero-confidence rates reach the allocator through two independent production
pathways:

1. **Fresh sequences:** `remap == -1` rows have their gathered rates overridden
   to 0.0 before the cumulative product
   ([schedule_mtp_varlen.py:55-65](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L55)).
2. **Post-stop requests:** after a request's emitted tokens hit one of its
   stop tokens, the NEXT round's draft confidences are masked to 0.0:
   `predict_tokens.confidence_rates.masked_fill_(emitted_stop_token.unsqueeze(1), 0.0)`
   ([layers/sampler.py:252-254](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/sampler.py#L252));
   the flag is computed from the emitted stream at
   [tilelang_ops/sampler.py:376-382](file:///Users/jackson/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L376).

So the zero-rate-wins-under-slack edge case
([dflash_alloc.py, fixture 3](file:///Users/jackson/deepinfer-research/dflash_alloc.py),
lines 82-89) is not a cold-start-only artifact: any request that just emitted
EOS re-enters the next batch with all-zero confidence rates and can win
verification budget whenever `total_q - num_seqs` exceeds the number of
positive marginal values.

## Simulation conventions carried by sim/rates.py

- `EMIT_BONUS = 1` — the single constant encoding Question 1.
- `NEXTN = 5` — production draft length (`DEEPINFER_USE_NEXTN_DECODE` default).
- Fixed-5 arm (the only arm in lane L1): every sequence verifies all 5 draft
  positions each step; k is the accepted prefix length; emitted tokens per
  step are `k + EMIT_BONUS`; the sequence's progress counter G advances by
  `k + EMIT_BONUS` per step.
- Accepted-length statistic (the calibration target): mean k per
  (sequence, step) under the fixed-5 arm.
- ZOS staleness, the allocator, fresh-sequence dynamics, and any cost model
  are out of L1 scope (lanes L2/L3).

## Rate generator family (sim/rates.py)

Per sequence i and draft position j (1-based):

```
r_ij = clip( level_type(i) * depth_decay^(j-1) * exp(-progress_decay * G_i), eps, 1-eps )
```

with `type(i)` assigned by a fixed 3-type mixture (cross-sequence
heterogeneity), `depth_decay` the within-step draft-depth decay, `G_i` the
tokens sequence i has already emitted, and `progress_decay` the
request-progress term. Acceptance is a sequential per-position Bernoulli
process; k is the accepted prefix length.

**Type composition is stratified, not i.i.d.** `stratified_types` assigns
types by largest remainder so the batch composition equals the mixture
proportions exactly at every anchor concurrency (the frozen weights
(0.5, 0.25, 0.25) are divisible by 1/4 = gcd(16, 32, 48, 60, 72, 96), so the
remainder is always zero). Rationale: the engine's benchmark statistics
average over a long window of request arrivals, so the composition noise of
i.i.d. type draws (about ±1.4/sqrt(conc) tokens on the accepted length —
observed as ±0.15 row swings in the first calibration attempt, e.g. a 3.42
row next to a 3.13 row) is not part of the modeled quantity. Stratification
removes that noise while keeping the population heterogeneous, and it makes
the base and mutant arms share the identical type vector (tighter pairing).

**Declared modeling assumptions:**

1. **Cross-sequence heterogeneity is the load-bearing assumption.** A
   homogeneous population makes dynamic ≈ fixed by construction (no room for
   a global top-k allocator to gain). The type mixture is what the whole
   study rests on.
2. **Depth decay** `depth_decay^(j-1)`: acceptance decays with draft depth
   within a step (deeper drafts are conditioned on more unverified
   predictions).
3. **Progress decay is design-driven, not empirical.** The `progress_decay`
   parameter exists so that the emission convention is observable in the
   accepted-length statistic: it makes the N5 mutant test able to go red
   (EMIT_BONUS=0 slows progress accumulation, raises acceptance, and pushes
   the statistic out of the calibration band). No traffic evidence supports
   it; no calibration conclusion may rely on its empirical meaning. Recorded
   per user note, 2026-10-22. The accepted-length statistic is defined at the
   frozen horizon `CALIB_STEPS = 6000`; with the frozen `progress_decay` the
   per-sequence accepted length drifts by a few tenths of a token between a
   fresh request and a tens-of-thousands-token-old request, and the
   statistic would shift if the horizon changed.

## Calibration criteria (rule N6)

- **Criterion:** with the frozen parameters and seeds, the fixed-5 accepted
  length k averaged over the six concurrency rows lies in [3.29, 3.32]; every
  row lies in [3.25, 3.32] (the anchor table's fixed-5 accepted min-max).
- **Why it holds:** the parameters were selected by a deterministic, seeded
  search that imposes exactly these constraints plus the mutant constraint
  below; the frozen constants reproduce the search output bit-for-bit.
- **Flip conditions:** any change to the generator's draw order, RNG seeding,
  frozen constants, or numpy's random stream behavior moves the statistic out
  of the band and the calibration test goes red.
- **Mutant constraint:** with `EMIT_BONUS` forced to 0 and everything else
  frozen, the statistic must leave the band (direction: above 3.32 — slower
  progress means less decay and higher acceptance). Enforced by
  `test_mutant_emit_bonus_red`; the paired per-row direction is asserted.
- **Per-row spread:** the generator's population is i.i.d. across
  concurrency, so the fixed-5 accepted length is concurrency-flat by
  construction; the anchor table's per-row variation (3.25-3.32,
  non-monotone) is treated as benchmark noise. Declared position, falsifiable
  at L3 if the sign-flip location turns out to depend on it.

## What this document does NOT pin

- The exact provenance of the Q&A table's statistic (which counter, which
  averaging window): inferred from counter semantics, not documented
  upstream.
- Request lifetime and termination beyond the progress term; L1 has no
  allocator, no fresh-sequence dynamics, no cost model, no ZOS staleness
  model.
