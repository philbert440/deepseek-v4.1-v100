# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark adaptive draft length for the DeepSeek-V4.1 port (VLLM_DS41_DSPARK_PMIN / VLLM_DS41_DSPARK_MAX_ROWS).

The drafter still proposes k tokens per request (one parallel block); the knobs only choose how many of them the
target verifies. Off (both 0, the default) must be today's fixed-k path: no lengths tensor at all."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import math

import pytest
import torch

from vllm.v1.spec_decode import dspark as dspark_mod
from vllm.v1.spec_decode.dspark import DSparkProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

K = 5
V = 8


class _Model:
    vocab_size = V

    def markov_embed(self, token_ids):
        return token_ids.unsqueeze(-1).float()

    def markov_bias(self, embeds):
        return torch.zeros(embeds.shape[0], V)

    @staticmethod
    def map_draft_to_target(ids):
        return ids


def _proposer(pmin=0.0, max_rows=0, cap_enabled=False, cap=K, max_batch=0):
    p = object.__new__(DSparkProposer)
    p.num_speculative_tokens = K
    p.model = _Model()
    p._enable_probabilistic_draft_probs = False
    p._static_draft_vocab = None
    p.collect_confidence_logits = False
    p.confidence_scheduling_enabled = cap_enabled
    p._max_verification_tokens = cap
    p._ds41_pmin = pmin
    p._ds41_max_rows = max_rows
    p._ds41_pmin_max_batch = max_batch
    return p


def _logits_for(top1_probs: list[list[float]]) -> torch.Tensor:
    """[B*K, V] logits whose softmax max at (b, j) is top1_probs[b][j] (token j+1 wins)."""
    rows = []
    for probs in top1_probs:
        for j, q in enumerate(probs):
            # one winner with prob q, the rest uniform: logit_w - logit_o = log(q (V-1) / (1-q))
            row = torch.zeros(V)
            row[(j + 1) % V] = math.log(q * (V - 1) / (1.0 - q))
            rows.append(row)
    return torch.stack(rows)


def _run(p, top1_probs):
    b = len(top1_probs)
    p.input_ids = torch.zeros(b * K, dtype=torch.int32)
    p._anchor_indices = torch.arange(b, dtype=torch.int64) * K
    logits = _logits_for(top1_probs)
    tokens, probs = p._sample_draft_tokens(torch.zeros(b * K, 2), sampling_metadata=None, logits=logits)
    assert probs is None
    return tokens.view(b, K).tolist(), p.take_last_verification_lengths()


PROBS = [
    [0.9, 0.8, 0.4, 0.9, 0.9],   # stops at the 3rd position -> 2
    [0.3, 0.9, 0.9, 0.9, 0.9],   # unsure at once -> 0
    [0.95, 0.9, 0.85, 0.8, 0.76],  # confident throughout -> 5
]


def test_off_is_fixed_k_and_has_no_lengths():
    tokens, lengths = _run(_proposer(), PROBS)
    assert lengths is None                       # the runner keeps the plain fixed-k path
    assert tokens == [[1, 2, 3, 4, 5]] * 3


def test_pmin_truncates_at_first_unsure_position_and_keeps_the_tokens():
    tokens, lengths = _run(_proposer(pmin=0.75), PROBS)
    assert tokens == [[1, 2, 3, 4, 5]] * 3       # the proposals themselves do not change
    assert lengths.dtype == torch.int32
    assert lengths.tolist() == [2, 0, 5]
    _, lengths = _run(_proposer(pmin=0.5), PROBS)
    assert lengths.tolist() == [2, 0, 5]
    _, lengths = _run(_proposer(pmin=0.35), PROBS)
    assert lengths.tolist() == [5, 0, 5]
    _, lengths = _run(_proposer(pmin=0.25), PROBS)
    assert lengths.tolist() == [5, 5, 5]


def test_row_ceiling_per_batch_size():
    # exllamav3 85cae3e7: draft = min(k, max_rows // B - 1)
    _, lengths = _run(_proposer(max_rows=8), PROBS)            # B=3: 8 // 3 - 1 = 1
    assert lengths.tolist() == [1, 1, 1]
    _, lengths = _run(_proposer(max_rows=8), PROBS[:2])        # B=2: 3
    assert lengths.tolist() == [3, 3]
    _, lengths = _run(_proposer(max_rows=8), PROBS[:1])        # B=1: min(5, 7)
    assert lengths.tolist() == [5]
    _, lengths = _run(_proposer(max_rows=2), PROBS)            # B=3: 2 // 3 - 1 < 0 -> 0 drafts
    assert lengths.tolist() == [0, 0, 0]
    _, lengths = _run(_proposer(pmin=0.75, max_rows=8), PROBS[:2])
    assert lengths.tolist() == [2, 0]


def test_combines_with_the_static_cap():
    _, lengths = _run(_proposer(pmin=0.75, cap_enabled=True, cap=3), PROBS)
    assert lengths.tolist() == [2, 0, 3]


def test_knob_parsing(monkeypatch):
    monkeypatch.delenv(dspark_mod.PMIN_KNOB, raising=False)
    monkeypatch.delenv(dspark_mod.MAX_ROWS_KNOB, raising=False)
    assert dspark_mod.ds41_dspark_adaptive_knobs() == (0.0, 0)
    assert not dspark_mod.ds41_dspark_adaptive_enabled()
    monkeypatch.setenv(dspark_mod.PMIN_KNOB, "0.75")
    assert dspark_mod.ds41_dspark_adaptive_knobs() == (0.75, 0)
    assert dspark_mod.ds41_dspark_adaptive_enabled()
    monkeypatch.setenv(dspark_mod.PMIN_KNOB, "1.5")
    with pytest.raises(ValueError):
        dspark_mod.ds41_dspark_adaptive_knobs()
    monkeypatch.setenv(dspark_mod.PMIN_KNOB, "nan")
    with pytest.raises(ValueError):
        dspark_mod.ds41_dspark_adaptive_knobs()
    monkeypatch.setenv(dspark_mod.PMIN_KNOB, "0")
    monkeypatch.setenv(dspark_mod.MAX_ROWS_KNOB, "8")
    assert dspark_mod.ds41_dspark_adaptive_knobs() == (0.0, 8)
    assert dspark_mod.ds41_dspark_adaptive_enabled()


def test_runner_hands_the_scheduler_the_truncated_drafts():
    """take_draft_token_ids -> _get_draft_token_ids_cpu applies the per-request lengths (0..k)."""
    runner = SimpleNamespace(
        _draft_token_ids=torch.zeros(3, K, dtype=torch.int32),
        _draft_token_req_ids=["a", "b", "c"],
        draft_token_ids_event=MagicMock(),
        draft_token_ids_cpu=torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
        dspark_confidence_scheduling=True,
        _dspark_verification_lengths_cpu=torch.tensor([2, 0, 5], dtype=torch.int32),
    )
    drafts, req_ids = GPUModelRunner._get_draft_token_ids_cpu(runner)
    assert req_ids == ["a", "b", "c"]
    assert drafts == [[1, 2], [], [11, 12, 13, 14, 15]]


def test_pmin_max_batch_gates_truncation():
    """VLLM_DS41_DSPARK_PMIN_MAX_BATCH=N: p_min truncates only batches of <= N requests; above it full k drafts
    (the verify step keeps a precaptured 1+k shape per request)."""
    _, lengths = _run(_proposer(pmin=0.75, max_batch=1), PROBS[:1])          # B=1 <= 1: truncated
    assert lengths.tolist() == [2]
    _, lengths = _run(_proposer(pmin=0.75, max_batch=1), PROBS[:2])          # B=2 > 1: full k
    assert lengths.tolist() == [K, K]
    _, lengths = _run(_proposer(pmin=0.75, max_batch=2), PROBS[:2])          # B=2 <= 2: truncated
    assert lengths.tolist() == [2, 0]
    _, lengths = _run(_proposer(pmin=0.75, max_batch=0), PROBS)              # 0 = any batch size (default)
    assert lengths.tolist() == [2, 0, 5]
    _, lengths = _run(_proposer(pmin=0.75, max_batch=1, max_rows=8), PROBS[:2])   # the row ceiling still applies
    assert lengths.tolist() == [3, 3]
    # the live request count comes from the runner when present (drafter rows may be graph-padded)
    p = _proposer(pmin=0.75, max_batch=1)
    p._ds41_runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=1), requests={})
    _, lengths = _run(p, PROBS[:2])
    assert lengths.tolist() == [2, 0]


def _req(prompt, computed):
    return SimpleNamespace(num_prompt_tokens=prompt, num_computed_tokens=computed)


def test_pmin_max_batch_counts_decoders_outside_this_step():
    """PP2 batch queue: two decoders ride alternate batches (R=1 per step). The gate counts every decoding request
    the runner holds, so B=2 keeps full k even though this step has one request."""
    p = _proposer(pmin=0.75, max_batch=1)
    p._ds41_runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=1),
                                     requests={"a": _req(40, 52), "b": _req(40, 47)})
    _, lengths = _run(p, PROBS[:1])
    assert lengths.tolist() == [K]
    # the other request is still prefilling: only one decoder -> truncate
    p._ds41_runner.requests = {"a": _req(40, 52), "b": _req(3000, 1024)}
    _, lengths = _run(p, PROBS[:1])
    assert lengths.tolist() == [2]
    # the other request finished (dropped from runner.requests) -> truncate again
    p._ds41_runner.requests = {"a": _req(40, 60)}
    _, lengths = _run(p, PROBS[:1])
    assert lengths.tolist() == [2]
    # max_batch=2 with two decoders: truncate
    p._ds41_pmin_max_batch = 2
    p._ds41_runner.requests = {"a": _req(40, 52), "b": _req(40, 47)}
    _, lengths = _run(p, PROBS[:1])
    assert lengths.tolist() == [2]


def test_pmin_max_batch_knob(monkeypatch):
    monkeypatch.delenv(dspark_mod.PMIN_MAX_BATCH_KNOB, raising=False)
    assert dspark_mod.ds41_dspark_pmin_max_batch() == 0
    monkeypatch.setenv(dspark_mod.PMIN_MAX_BATCH_KNOB, "1")
    assert dspark_mod.ds41_dspark_pmin_max_batch() == 1
    monkeypatch.setenv(dspark_mod.PMIN_MAX_BATCH_KNOB, "-1")
    with pytest.raises(ValueError):
        dspark_mod.ds41_dspark_pmin_max_batch()
