"""Emission-convention tests (Lane L1).

Each test follows HANDOFF_PROMPT.md §5 N6: criterion, why it holds, and what
flips it red.
"""

from pathlib import Path

import numpy as np

from sim import rates
from sim.rates import EMIT_BONUS, NEXTN

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_emit_bonus_frozen_to_one():
    """Criterion: EMIT_BONUS == 1 and NEXTN == 5.

    Why: three independent upstream sites state the engine emits the accepted
    prefix plus the final fresh row (tilelang_ops/sampler.py:376-379,
    :466-467) and the KV advance is k+1 (compute_seqlens_k_zos.py:22-23);
    the production draft length is 5.
    Flip: a re-adjudication of convention Question 1 changes the constant and
    sim/convention.md together.
    """
    assert EMIT_BONUS == 1
    assert NEXTN == 5


def test_emission_accounting_near_certain_rates():
    """Criterion: near-certain rates give k = 5 and emitted = 6 every step.

    Why: a step emits the accepted prefix (all 5 drafts) plus the fresh row
    (EMIT_BONUS). Flip: the simulator counts only draft tokens or drops the
    fresh row.
    """
    k, emitted = rates.simulate_fixed5(
        num_seqs=4,
        steps=20,
        seed=7,
        levels=(1.0, 1.0, 1.0),
        weights=(1 / 3, 1 / 3, 1 / 3),
        depth_decay=1.0,
        progress_decay=0.0,
    )
    assert int(k.min()) == NEXTN and int(k.max()) == NEXTN
    assert int(emitted.min()) == NEXTN + EMIT_BONUS
    assert int(emitted.max()) == NEXTN + EMIT_BONUS


def test_emission_accounting_near_zero_rates():
    """Criterion: near-zero rates give k = 0 and emitted = 1 every step.

    Why: zero-confidence sequences still emit their fresh row's token — the
    engine's one-token-per-step behavior for zero-rate sequences
    (schedule_mtp_varlen.py:55-65, :98). Flip: emission modeled as k only
    (zero tokens per step).
    """
    k, emitted = rates.simulate_fixed5(
        num_seqs=4,
        steps=20,
        seed=7,
        levels=(0.0, 0.0, 0.0),
        weights=(1 / 3, 1 / 3, 1 / 3),
        depth_decay=1.0,
        progress_decay=0.0,
    )
    assert int(k.max()) == 0
    assert int(emitted.min()) == EMIT_BONUS
    assert int(emitted.max()) == EMIT_BONUS


def test_emit_bonus_flip_changes_emission(monkeypatch):
    """N5 mutant: with EMIT_BONUS=0 the near-certain-rate emission drops from
    6 to 5 tokens per step — the constant is observable, not decorative.

    Flip: if the flip left emission unchanged, EMIT_BONUS would not be
    encoded in the simulator.
    """
    kwargs = dict(
        levels=(1.0, 1.0, 1.0),
        weights=(1 / 3, 1 / 3, 1 / 3),
        depth_decay=1.0,
        progress_decay=0.0,
    )
    _, emitted_base = rates.simulate_fixed5(num_seqs=4, steps=20, seed=7, **kwargs)
    assert int(emitted_base.min()) == NEXTN + 1
    monkeypatch.setattr(rates, "EMIT_BONUS", 0)
    k_mut, emitted_mut = rates.simulate_fixed5(num_seqs=4, steps=20, seed=7, **kwargs)
    assert int(k_mut.max()) == NEXTN
    assert int(emitted_mut.min()) == NEXTN
    assert int(emitted_mut.max()) == NEXTN


def test_rates_matrix_bounded():
    """Criterion: rates land in the open interval (0, 1), shape [num_seqs, NEXTN].

    Why: r_ij are conditional Bernoulli probabilities and the clip enforces
    the open interval even for degenerate parameters. Flip: a parametrization
    change that lets rates reach 0 or 1.
    """
    types = rates.stratified_types(6, (0.5, 0.25, 0.25))
    r = rates.rates_matrix((0.97, 0.85, 0.7), 0.99, 1e-6, types, np.zeros(6))
    assert r.shape == (6, NEXTN)
    assert bool(np.all(r > 0.0)) and bool(np.all(r < 1.0))
    extreme = rates.rates_matrix((1.0, 1.0, 1.0), 1.0, 0.0, types, np.zeros(6))
    assert bool(np.all(extreme > 0.0)) and bool(np.all(extreme < 1.0))


def test_stratified_types_exact_proportions():
    """Criterion: every anchor concurrency gets exactly the mixture proportions.

    Why: the frozen weights (0.5, 0.25, 0.25) are divisible by 1/4 = gcd of
    the anchor concurrencies, so largest-remainder rounding leaves no
    residual and composition noise cannot re-enter row statistics.
    Flip: weights change or the stratification logic drifts.
    """
    for conc in rates.CONCURRENCY_ROWS:
        types = rates.stratified_types(conc, rates.TYPE_WEIGHTS)
        counts = np.bincount(types, minlength=3)
        assert int(counts.sum()) == conc
        assert int(counts[0]) == conc // 2
        assert int(counts[1]) == conc // 4
        assert int(counts[2]) == conc // 4


def test_convention_doc_pins_citations():
    """Criterion: convention.md carries the corrected citation set and the
    design-driven progress-decay note (user notes 1-3 on the triage,
    2026-10-22). Flip: any pinned marker is removed from the document.
    """
    text = (REPO_ROOT / "sim" / "convention.md").read_text(encoding="utf-8")
    for marker in (
        "tilelang_ops/sampler.py:246",
        "tilelang_ops/sampler.py:376",
        "tilelang_ops/sampler.py:442",
        "tilelang_ops/sampler.py:458",
        "tilelang_ops/sampler.py:466",
        "layers/sampler.py:252",
        "layers/sampler.py:289",
        "compute_seqlens_k_zos.py:22",
        "schedule_mtp_varlen.py:98",
        "sequence.py:265",
        "EMIT_BONUS = 1",
        "design-driven, not empirical",
    ):
        assert marker in text, f"sim/convention.md lost marker: {marker}"


def test_progress_decay_documented_as_design_driven():
    """Criterion: the rates module docstring declares the progress term
    design-driven (user note 3: calibration conclusions must not rely on its
    empirical meaning). Flip: the declaration is removed from the docstring.

    The match runs on whitespace-normalized text so rewrapping the docstring
    cannot blind the guard.
    """
    assert rates.__doc__ is not None
    normalized = " ".join(rates.__doc__.split())
    assert "DESIGN-DRIVEN, NOT EMPIRICAL" in normalized
