# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_INDEXER_CAND_PREFILL on the live indexer: the mode gate and the per-request scan / candidate choice
(auto = a pure size switch on n, deterministic per (n, T)), and candidate_topk with CAND_ALL rows expanded equals the
reference full scan + mask (CPU, exact logits: q, k quarter-integers, GEMMs wrapped to FP32). GPU equality + timing:
scripts/ds41_cand_prefill_bench.py on rivet/cand-score (identical rows at every n / drift, 20261024T211909Z)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41 import indexer as ix
from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks_torch
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS
from vllm.models.deepseek_v41.sm70.indexer_kernels import topk_sorted

_bmm = torch.bmm


def _fake(mode: str, scan_max: int = 49152, candidate_scoring: bool = True, uses_candidates: bool = True):
    x = ix.DeepseekV41Indexer.__new__(ix.DeepseekV41Indexer)
    object.__setattr__(x, "_parameters", {}); object.__setattr__(x, "_buffers", {})
    object.__setattr__(x, "_modules", {})
    x.__dict__.update(cand_prefill=mode, cand_scan_max=scan_max, candidate_scoring=candidate_scoring,
                      topo=SimpleNamespace(uses_candidates=uses_candidates, is_candidate_source=False))
    return x


def test_mode_gate_and_kind() -> None:
    plain = torch.zeros(8, 4, dtype=torch.int32)
    some_all = plain.clone(); some_all[3, 0] = CAND_ALL
    off = _fake("off")
    assert off.cand_prefill_mode(1024) == "off"
    assert off.cand_kind("off", 100_000, plain) == "cand" and off.cand_kind("off", 100_000, some_all) == "scan"
    auto = _fake("auto", scan_max=49152)
    assert auto.cand_prefill_mode(16) == "off" and auto.cand_prefill_mode(17) == "auto"
    assert auto.cand_kind("auto", 49151, plain) == "scan" and auto.cand_kind("auto", 49152, plain) == "cand"
    assert auto.cand_kind("auto", 49152, some_all) == "cand"          # CAND_ALL rows: expanded in forward
    assert _fake("scan").cand_kind("scan", 10 ** 6, plain) == "scan"
    assert _fake("auto", uses_candidates=False).cand_prefill_mode(1024) == "off"
    assert _fake("auto", candidate_scoring=False).cand_kind("auto", 10 ** 6, plain) == "scan"
    with pytest.raises(ValueError):
        ix.knobs.env_str("X_UNSET_KNOB_FOR_TEST", "union", choices=ix.CAND_PREFILL_MODES)


def test_expanded_cand_all_equals_scan_mask(monkeypatch) -> None:
    monkeypatch.setattr(torch, "bmm", lambda a, b, out_dtype=None: _bmm(a.float(), b.float()))
    T, n = 40, 16400
    g = torch.Generator().manual_seed(3)
    base = torch.randint(-4, 5, (1, IDX_HEADS, IDX_DIM), generator=g).float()
    q = (torch.round(base + 0.3 * torch.randint(-4, 5, (T, IDX_HEADS, IDX_DIM), generator=g).float()) / 4).half()
    keys = (torch.randint(-4, 5, (n, IDX_DIM), generator=g).float() / 4).half()
    w = torch.rand(T, IDX_HEADS, generator=g)
    ends = torch.arange(16384 - 20 + 1, 16384 + 21, dtype=torch.int64)        # rows 0..19 see <= 16,384: CAND_ALL
    scores = (torch.einsum("thd,nd->thn", q.float(), keys.float()).relu_() * w[:, :, None]).sum(dim=1)
    keep = select_candidate_blocks_torch(scores, ends, CAND_TOPK_BLOCKS, CAND_BLOCK)
    nb = (n + CAND_BLOCK - 1) // CAND_BLOCK
    kb = torch.nn.functional.pad(keep, (0, nb * CAND_BLOCK - n)).view(T, nb, CAND_BLOCK).any(-1)
    cand = torch.full((T, CAND_TOPK_BLOCKS), -1, dtype=torch.int32)
    for t in range(T):
        ids = kb[t].nonzero().squeeze(1)
        cand[t, :ids.numel()] = ids.to(torch.int32)
        if (ends[t] + CAND_BLOCK - 1) // CAND_BLOCK <= CAND_TOPK_BLOCKS:
            cand[t, 0] = CAND_ALL
    assert int((cand[:, 0] == CAND_ALL).sum()) == 20
    ref = topk_sorted(scores.masked_fill(~keep, float("-inf")), ends, 512)    # full scan + candidate mask
    blocks = torch.arange(CAND_TOPK_BLOCKS, dtype=cand.dtype)[None, :]
    c = torch.where(cand[:, :1] == CAND_ALL, blocks, cand)
    got = torch.cat([ix.candidate_topk(q[a:a + 16], w[a:a + 16], c[a:a + 16], ends[a:a + 16], keys, lambda p: p)
                     for a in range(0, T, 16)])
    assert torch.equal(got, ref)
