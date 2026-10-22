"""Numpy port of DeepInfer's confidence-scheduled verification allocator.

Ported line-by-line from
deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py
::compute_seqlens_q_by_confidence_rates_ref (torch), commit 9bd9cf2.
Equivalence to the torch reference is asserted on hand-derived fixtures
below; a GPU box is needed to check against the torch code directly.

Algorithm: the batch has a fixed verification budget `total_q`. Every
(sequence, draft-position) pair contributes a marginal value equal to the
cumulative product of per-position predicted acceptance rates (cumprod
along nextn). The top `total_q - num_seqs` marginal values win one extra
verified token each; every sequence keeps one base token.

Known edge case (found by fixture 3): sequences whose rates are all zero
(fresh sequences, remap == -1) are documented to "stay at 0.0" but still
participate in the ranking with a stable tie-break that favours the
lowest flat index. When `total_q - num_seqs` exceeds the number of
positive marginal values, zero-confidence sequences win slots too.
"""

from __future__ import annotations

import numpy as np


def allocate_seqlens_q(
    total_q: int,
    remap: np.ndarray,
    prev_confidence_rates: np.ndarray,
) -> np.ndarray:
    """Return per-sequence query lengths (int32, sums to total_q).

    remap: [num_seqs] int, previous-batch index or -1 for fresh sequences.
    prev_confidence_rates: [n_prev_seq, nextn] float, per-position
        predicted acceptance rates from the previous MTP step.
    """
    num_seqs = remap.shape[0]
    nextn = prev_confidence_rates.shape[1]

    mask = remap != -1
    safe_remap = np.clip(remap, 0, None)
    gathered_rates = prev_confidence_rates[safe_remap]
    gathered_rates = np.where(mask[:, None], gathered_rates, np.float32(0.0))

    cum_rates = np.cumprod(gathered_rates, axis=1)

    extra_budget = total_q - num_seqs
    assert extra_budget >= 0, f"total_q={total_q} < num_seqs={num_seqs}"

    flat_rates = cum_rates.reshape(-1)
    sorted_indices = np.argsort(-flat_rates, kind="stable")
    ranks = np.empty(len(flat_rates), dtype=np.int32)
    ranks[sorted_indices] = np.arange(len(flat_rates), dtype=np.int32)
    ranks = ranks.reshape(num_seqs, nextn)

    wins = (ranks < extra_budget).sum(axis=1, dtype=np.int32)
    return (1 + wins).astype(np.int32)


def _check(name: str, got: np.ndarray, expected: list[int]) -> None:
    expected_arr = np.array(expected, dtype=np.int32)
    assert np.array_equal(got, expected_arr), f"{name}: got {got.tolist()}, expected {expected}"
    print(f"PASS {name}: seqlens_q={got.tolist()} (sum={int(got.sum())})")


if __name__ == "__main__":
    # Fixture 1: monotone prefixes win in cumprod order.
    # seq0 cum = [0.9, 0.72], seq1 cum = [0.5, 0.2]; extra=2 -> both slots to seq0.
    rates = np.array([[0.9, 0.8], [0.5, 0.4]], dtype=np.float32)
    remap = np.array([0, 1], dtype=np.int32)
    got = allocate_seqlens_q(total_q=4, remap=remap, prev_confidence_rates=rates)
    _check("fixture1 prefix-strict", got, [3, 1])

    # Fixture 2: fresh sequence (remap=-1) with zero rates stays at 1 token
    # while a confident sequence takes every extra slot.
    rates = np.array([[0.5, 0.2]], dtype=np.float32)
    remap = np.array([-1, 0], dtype=np.int32)
    got = allocate_seqlens_q(total_q=4, remap=remap, prev_confidence_rates=rates)
    _check("fixture2 fresh-gets-one", got, [1, 3])

    # Fixture 3 (edge case): budget slack exceeds positive marginal values,
    # so the zero-rate fresh sequence wins a slot via stable tie-break.
    # flat = [0.0, 0.0, 0.5, 0.2]; extra=3 -> winners: idx2 (0.5), idx3 (0.2), then
    # the first zero (idx0, seq0). seq0 = 1+1, seq1 = 1+2.
    rates = np.array([[0.5, 0.2]], dtype=np.float32)
    remap = np.array([-1, 0], dtype=np.int32)
    got = allocate_seqlens_q(total_q=5, remap=remap, prev_confidence_rates=rates)
    _check("fixture3 zero-rate-wins-under-slack", got, [2, 3])

    # Fixture 4: total_q == num_seqs -> no extra budget, everyone gets 1.
    rates = np.array([[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]], dtype=np.float32)
    remap = np.array([0, 1, 2], dtype=np.int32)
    got = allocate_seqlens_q(total_q=3, remap=remap, prev_confidence_rates=rates)
    _check("fixture4 no-slack", got, [1, 1, 1])

    print("all fixtures passed")
