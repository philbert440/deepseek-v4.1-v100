# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_INDEXER_FUSED_DECODE on CPU: the torch models of the decode kernels (block maxima -> top-2048 blocks ->
rescored top-512) equal the reference full scan (causal top-512, ties -> lowest position) and its candidate blocks
(select_candidate_blocks semantics: newest block pinned, -1 pad, CAND_ALL at <= 2048 blocks); topk_sel's rule; the
split heuristic; and the indexer's dispatch (decode-sized steps only, never attention parts, layer 20 capturable
only with the knob, prefill unchanged). GPU kernels + graphs: test_indexer_fused_decode_gpu.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41 import indexer as ix
from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks_torch
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS
from vllm.models.deepseek_v41.sm70 import indexer_fused as fx


def _data(T, n, seed, exact=True):
    g = torch.Generator().manual_seed(seed)
    if exact:
        q = (torch.randint(-4, 5, (T, IDX_HEADS, IDX_DIM), generator=g).float() / 4).half()
        keys = (torch.randint(-4, 5, (n, IDX_DIM), generator=g).float() / 4).half()
    else:
        q = (torch.randn(T, IDX_HEADS, IDX_DIM, generator=g) * 0.5).half()
        keys = (torch.randn(n, IDX_DIM, generator=g) * 0.5).half()
    w = torch.rand(T, IDX_HEADS, generator=g)
    return q, w, keys


def _scores(q, w, keys):
    lg = torch.einsum("thd,nd->thn", q.float(), keys.float())
    s = torch.zeros(q.shape[0], keys.shape[0])
    for h in range(IDX_HEADS):
        s = s + lg[:, h].relu() * w[:, h:h + 1]
    return s


def _ref_top512(s, ends):
    cols = torch.arange(s.shape[1])
    return fx.topk_sel_emulate(s.masked_fill(cols[None, :] >= ends[:, None], float("-inf")), 512)


@pytest.mark.parametrize("n,ends_list", [(20000, [20000, 19990, 17000, 9, 600, 1]), (16000, [16000, 15000, 8000])])
def test_decode_scan_model_equals_full_scan(n, ends_list) -> None:
    T = len(ends_list)
    q, w, keys = _data(T, n, seed=n + T)
    ends = torch.tensor(ends_list, dtype=torch.int64)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3))
    stored = torch.empty_like(keys)
    stored[perm] = keys
    rows_of = lambda pos: torch.where(pos >= 0, perm[pos.clamp(min=0)], torch.full_like(pos, -1))
    top, cand = fx.fused_decode_scan_emulate(q, w, ends, stored, rows_of)
    s = _scores(q, w, keys)
    assert torch.equal(top, _ref_top512(s, ends))
    for t in range(T):
        reach = (int(ends[t]) + CAND_BLOCK - 1) // CAND_BLOCK
        if reach <= CAND_TOPK_BLOCKS:
            assert int(cand[t, 0]) == CAND_ALL
        else:
            ids = cand[t][cand[t] >= 0]
            assert ids.numel() == CAND_TOPK_BLOCKS and bool((ids[1:] > ids[:-1]).all())
            assert int((int(ends[t]) - 1) // CAND_BLOCK) in ids.tolist()          # newest block pinned


def test_decode_candidate_blocks_match_select_reference() -> None:
    """Real-valued scores (no ties): the decode model's blocks == select_candidate_blocks_torch's kept blocks."""
    T, n = 3, 24000
    q, w, keys = _data(T, n, seed=5, exact=False)
    ends = torch.tensor([24000, 23000, 20000], dtype=torch.int64)
    _, cand = fx.fused_decode_scan_emulate(q, w, ends, keys, lambda pos: pos)
    keep = select_candidate_blocks_torch(_scores(q, w, keys), ends, CAND_TOPK_BLOCKS, CAND_BLOCK)
    nb = (n + CAND_BLOCK - 1) // CAND_BLOCK
    kb = torch.nn.functional.pad(keep, (0, nb * CAND_BLOCK - n)).view(T, nb, CAND_BLOCK).any(-1)
    for t in range(T):
        assert set(cand[t][cand[t] >= 0].tolist()) == set(kb[t].nonzero().squeeze(1).tolist())


def test_topk_sel_rule_and_splits() -> None:
    s = torch.tensor([[1.0, 3.0, 3.0, float("-inf"), 2.0, 3.0]])
    assert fx.topk_sel_emulate(s, 2)[0, :2].tolist() == [1, 2]                  # ties -> lowest columns
    assert fx.topk_sel_emulate(s, 2, torch.tensor([3]))[0].tolist() == [1, 2]
    assert fx.topk_sel_emulate(s, 4, torch.tensor([4]))[0].tolist() == [0, 1, 2, -1]   # -inf never picked
    assert [fx.decode_slots_per_cta(T) for T in (1, 2, 3, 6, 12, 16, 24)] == [256, 256, 512, 1024, 2048, 2048, 2048]


def _fake(fused_decode=True, uses_candidates=True, source=False):
    x = ix.DeepseekV41Indexer.__new__(ix.DeepseekV41Indexer)
    for k in ("_parameters", "_buffers", "_modules"):
        object.__setattr__(x, k, {})
    x.__dict__.update(fused_decode=fused_decode, _dec_max_n=1 << 20, candidate_scoring=True,
                      topo=SimpleNamespace(uses_candidates=uses_candidates, is_candidate_source=source))
    return x


def test_gates(monkeypatch) -> None:
    monkeypatch.setattr(ix, "ATTN_GRAPH", True)
    assert _fake(True, False, True).graph_capable()                  # layer 20: capturable only with the knob
    assert not _fake(False, False, True).graph_capable()
    assert _fake(False, True).graph_capable()                        # reindex: today's rule
    assert not _fake(True, False, True)._cand_device()
    x = _fake(True)
    assert x.fused_decode_active(1) and x.fused_decode_active(24) and not x.fused_decode_active(25)
    assert not _fake(False).fused_decode_active(6)
    monkeypatch.setenv(fx.FUSED_DECODE_ENV, "1")
    assert fx.fused_decode_enabled()
    b = fx.decode_buffers(torch.device("cpu"), 1 << 20)
    assert tuple(b["blk"].shape) == (fx.DECODE_SCAN_ROWS, (1 << 20) // 8) and tuple(b["scores"].shape) == (16, 16384)
    assert fx.decode_buffers(torch.device("cpu"), 1 << 20) is b
    mib = fx.decode_buffer_bytes() / 2**20
    assert 3.0 < mib < 3.3, mib                                   # 2.0 + 1.0 + 0.125 MiB at 1M (was 9.1)


def test_forward_dispatch(monkeypatch) -> None:
    """Decode-sized forward of a source / reindex layer calls the fused decode entry points with the step's rows;
    attention parts and T > 24 never do."""
    import numpy as np

    calls = []
    monkeypatch.setattr(fx, "fused_decode_scan", lambda *a: calls.append(("scan", a[0].shape[0], a[-2] is not None)))
    monkeypatch.setattr(fx, "fused_decode_candidates", lambda *a: calls.append(("cand", a[0].shape[0])))
    monkeypatch.setattr(fx, "decode_buffers", lambda dev, n: {"lazy": n})
    monkeypatch.setattr(ix, "mm_fp32_full", lambda a, w: torch.zeros(a.shape[0], IDX_HEADS * IDX_DIM))
    monkeypatch.setattr(ix, "index_q_rope_qat", lambda q, pos, rope, impl=None: q.half())
    monkeypatch.setattr(ix, "mm_fp32", lambda a, w: torch.zeros(a.shape[0], IDX_HEADS))
    for uses, src, want in ((False, True, ("scan", 6, True)), (False, False, ("scan", 6, False)),
                            (True, False, ("cand", 6))):
        x = _fake(True, uses, src)
        T = 6
        md = SimpleNamespace(num_actual_tokens=T, max_visible=40000, num_visible=torch.full((T,), 40000),
                             token_to_req_indices=torch.zeros(T, dtype=torch.int32), block_table=None,
                             storage_block_size=64, query_start_loc_cpu=np.array([0, T]), num_reqs=1,
                             positions_cpu=np.arange(39994, 40000))
        x.__dict__.update(shared=SimpleNamespace(topk_indices=torch.zeros(T, 512, dtype=torch.int32),
                                                 candidate_blocks=torch.zeros(T, 2048, dtype=torch.int32)),
                          softmax_scale=1.0, wq_b=None, weights_proj=None, cand_prefill="off", ratio=1,
                          score_budget=128 << 20, qtile_cap=0, cand_scan_max=49152, fused=False)
        x.__dict__["_metadata"] = lambda: md
        x.__dict__["_key_rows"] = lambda: torch.zeros(1, 128, dtype=torch.float16)
        x.__dict__["_rope"] = lambda: None
        x.__dict__["_w"] = lambda lin, shape: lin
        calls.clear()
        x.forward(torch.zeros(T, 1), torch.zeros(T, 1), torch.arange(T))
        assert calls == [want]
        calls.clear()
        monkeypatch.setattr(ix.DeepseekV41Indexer, "scan_tasks", lambda *a, **k: ([], {}))
        x.forward(torch.zeros(T, 1), torch.zeros(T, 1), torch.arange(T), tiles=(0, 1))   # attention part: never
        assert calls == []
        monkeypatch.undo() if False else None
