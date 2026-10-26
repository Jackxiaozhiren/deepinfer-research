## v1.0.0 — the audited front door

This release is the repository's front door: `README.md`, `CITATION.cff`, the CI
guard lane, and the About/topics metadata. The research content did not change -
the committed artifacts and the commits they were scored at are the same ones
Milestone 1 and Milestone 2 closed against.

### What the audit found

- The published gain ladder (-1.5% at the anchor's concurrency 16 row,
  climbing to +11.0% at concurrency 96) is not jointly reproducible inside
  the cost family studied. Where the anchor's sign pattern is reachable at all,
  this family's concurrency 16 row lands at -0.05% to
  -0.37% - magnitude and signs cannot both be matched.
- Ranking quality has a reachable floor, not a zero. Retaining >= 50% of
  the ideal-band gain needs window-length concordance tau_half in
  [0.441, 0.453] for the base generator (V1) and
  [0.392, 0.404] for the sharper-separation variant (V2E),
  while the plateau band itself is ~0.19-0.29.
- A near-ideal head (V2E) closes 36.0% of the generator-versus-table
  discrepancy; the residual 64.0% lives in the acceptance rates themselves,
  not in the ranking.

### Reproduce

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

### Guarded prose surfaces

`WRITEUP.md`, the Outcomes section of `research_DeepInfer_20261022.md`,
`README.md`, this file, and `CITATION.cff` are census-tested by
`tests/test_writeup_numbers.py`: a numeric token that is not captured from a
committed artifact reddens the suite. This body was rendered by
`sim/build_writeup_numbers.py` from `results/writeup_numbers.json`.
