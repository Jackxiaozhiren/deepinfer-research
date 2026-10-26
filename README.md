# A published speculative-decoding speedup, audited on a laptop

An independent, CPU-only audit of the serving numbers DeepSeek publishes for
DeepInfer's DSpark speculative decoding — numpy on Apple silicon: no GPU, no model
weights, no tokenizer, no cluster.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12.14](https://img.shields.io/badge/python-3.12.14-blue.svg)](WRITEUP.md)
[![tests](https://github.com/Jackxiaozhiren/deepinfer-research/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Jackxiaozhiren/deepinfer-research/actions/workflows/tests.yml)

Read the audit in [WRITEUP.md](WRITEUP.md) · the code-level reading of the engine in
[research_DeepInfer_20261022.md](research_DeepInfer_20261022.md) · every session's
baseline, frozen parameters and findings in
[RESULTS_LEDGER.md](RESULTS_LEDGER.md).

## What this is, and what it is not

> [!IMPORTANT]
> **It is:** a seeded simulation of the allocator the open-source engine actually
> ships, calibrated to the published table's own fixed-5 column, and a machine-checked
> claim that every number in this repository's prose was captured from a committed
> artifact by regex — a drifted figure reddens CI.
>
> **It is not:** a verification of production numbers, a hardware benchmark, or a claim
> that DeepSeek's engineering is wrong. Reproducing a sign is not verification of
> production truth; failing to reproduce one is a finding about *this model family under
> these assumptions*, not a bug to tune away.

The analysis is independent and unaffiliated. Engine facts are read from the public
[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) repository at commit
`9bd9cf2` (Apache-2.0); the anchor table is quoted as published claims from its
`doc/inference_deployment_qa.md`.

## Headline results

- **The published gain ladder is not jointly reproducible.** The Q&A table has dynamic
  verification losing `-1.5%` at concurrency 16 and climbing monotonically to `+11.0%` at
  concurrency 96. Within the cost family studied, no parameter slice reproduces that
  low-concurrency magnitude *and* those higher-concurrency signs at once: where the
  anchor's sign pattern is reachable at all, this family's concurrency 16 row lands
  between `-0.05%` and `-0.37%`. That is structural — a property of the geometry of the
  zero-crossings, not of a badly tuned slice.
- **Ranking quality has a reachable floor and an auditable threshold.** Retaining `>= 50%`
  of the ideal-band gain requires the head's window-length concordance (`tau_half`) to
  stay in `[0.441, 0.453]` for the base generator (V1) and `[0.392, 0.404]` for the
  sharper-separation variant (V2E) in the stable regime. A production head emitting
  probabilities cannot scramble below that plateau: the reachable floor is `~0.19-0.29`,
  not zero.
- **Sharpening the ranking is necessary, not sufficient.** Re-running the discrepancy
  attribution under a near-ideal head (V2E) closes `36.0%` of the gap between the
  calibrated generator and the published table; the residual `64.0%` lives in the
  acceptance rates themselves, so no ranking fix alone reaches the published numbers.

Sources: [results/m1_report.md](results/m1_report.md) (replay table, sensitivity
summary), [results/m2_report.md](results/m2_report.md) (rank thresholds, attribution),
[results/sweep_l2.md](results/sweep_l2.md) and
[results/sweep_m2.md](results/sweep_m2.md) (the sweeps themselves).

## Quickstart

Environment: Python 3.12.14, numpy 2.5.4, pytest 9.1.1. No torch, no downloads, no
network in the experiments. Every stochastic step takes an explicit seed: generator
seed 20261022, horizon 6000, paired seeds `L2_SWEEP_SEED + i` for i in [0, 12),
bootstrap seed 20261101 with 10000 resamples.

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

Committed artifacts: `results/m1_report.md`, `results/m2_report.md`,
`results/sweep_l2.md`, `results/sweep_m2.md`, `sim/convention.md`, and the session
ledger `RESULTS_LEDGER.md`.

## Repository map

| Path | What it is |
|---|---|
| `dflash_alloc.py` | numpy port of the engine's allocator, with hand-derived fixtures and a self-test that prints one `PASS` line per fixture |
| `sim/convention.md` | the emission-convention finding (`k+1` tokens per step), adjudicated from upstream code with `file:line` citations |
| `sim/rates.py` | calibrated acceptance-rate generator; cross-sequence heterogeneity is the load-bearing modeling assumption, stated in its docstring |
| `sim/allocator.py` | dynamic (confidence-ranked) versus fixed-budget policies, on a per-step batch token budget |
| `sim/cost_model.py` | per-step time `T(v) = a + b·v` plus an optional saturation family as a second model |
| `sim/rank_noise.py` | lognormal noise applied only to the allocator's ranking input, plus Kendall tau_K on window lengths |
| `sim/sweep_l2.py`, `sim/sweep_m2.py` | the two sweeps; each rewrites its committed results table |
| `sim/replay_qa.py` | frozen-slice replay of the anchor table → `results/m1_report.md` |
| `sim/build_writeup_numbers.py` | the locator builder: regex-captures every published number from a committed artifact, refuses to emit on a locator that stops matching, and renders the release body |
| `tests/` | seeded tests; `tests/test_writeup_numbers.py` is the numeric census over the prose surfaces |
| `WRITEUP.md` | the long-form writeup: mechanism, both milestones, evidence boundary, reproduce path |
| `RESULTS_LEDGER.md` | append-only audit trail — dated entries, never edited in place |

## Reproduce and cite

- **Cite:** [CITATION.cff](CITATION.cff), or the `v1.0.0` release page.
- **How the numbers are welded:** `sim/build_writeup_numbers.py` captures each published
  figure from a committed artifact and writes `results/writeup_numbers.json`.
  `tests/test_writeup_numbers.py` then fails if a numeric token in `README.md`,
  `WRITEUP.md`, the report's Outcomes section, the release body, or `CITATION.cff` is not
  one of them, and fails again if a real number sits next to the entity it does not
  belong to. Editing a number in an artifact reddens the freshness guard before any prose
  can drift.
- **Pins:** Milestone 1 was frozen and scored at commit `c842e85`; Milestone 2 closed at
  `be55c16`; this front door is tagged `v1.0.0`. A fresh public clone on the pinned
  interpreter re-ran the sign-map sweep, regenerated `results/sweep_l2.md` bit-identically,
  and left the working tree clean — the datum and its reproduction live in
  [RESULTS_LEDGER.md](RESULTS_LEDGER.md), not in a chat log.
- **License:** MIT ([LICENSE](LICENSE)). Engine sources remain Apache-2.0 by
  [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer).
