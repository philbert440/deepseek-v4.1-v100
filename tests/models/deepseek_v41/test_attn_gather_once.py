# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_ATTN_GATHER_ONCE: layer-group sharing of the compressed row ids + the direct tile gather.

CPU (no GPU):        pytest tests/models/deepseek_v41/test_attn_gather_once.py -m "not sm70"
CPU kernel bits:     CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 pytest ... -k tile_bits
GPU (sm70) equality: pytest ... -m sm70   (sparse_attention knob on == knob off, bit for bit)
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.common.topology import layer_topology
from vllm.models.deepseek_v41.sm70 import sparse_kernels as sk
from vllm.models.deepseek_v41.sm70.gather_once import GroupRowsCache, ends_group

RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
GROUPS = {(2, 2): range(2, 8), (8, 8): range(8, 14), (14, 14): range(14, 20), (20, 20): range(20, 24),
          (20, 24): range(24, 28), (20, 28): range(28, 32), (20, 32): range(32, 36), (20, 36): range(36, 40)}


def _config() -> SimpleNamespace:
    return SimpleNamespace(compress_ratios=list(RATIOS), num_hidden_layers=40, kv_source_layer_ids=[2, 8, 14, 20],
                           index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36], engram_layer_ids=[1, 14],
                           candidate_source_layer_id=20)


def test_groups_from_topology() -> None:
    """Exactly 8 (kv source, index source) groups over the 38 compressed layers; ends_group marks each group's end."""
    cfg = _config()
    got: dict[tuple, list[int]] = {}
    for layer in range(40):
        t = layer_topology(cfg, layer)
        if t.compress_ratio == 0:
            continue
        got.setdefault((t.kv_source, t.index_source), []).append(layer)
    assert {k: list(v) for k, v in GROUPS.items()} == got
    assert sum(len(v) for v in got.values()) == 38
    ends = sorted(layer for layer in range(2, 40) if ends_group(layer))
    assert ends == [max(v) for v in got.values()]


def _run_stage(cache: GroupRowsCache, layers: range, md_for, topk: torch.Tensor, T: int, calls: list) -> dict:
    cfg = _config()
    out = {}
    for layer in layers:
        t = layer_topology(cfg, layer)
        if t.compress_ratio == 0:
            continue
        md = md_for(layer)

        def compute(layer=layer, md=md):
            calls.append(layer)
            return ("ids", md["tag"], layer)

        out[layer] = cache.get(layer_id=layer, owns_indexer=t.owns_indexer, kv_source=t.kv_source,
                               index_source=t.index_source, src_md=md, topk=topk, num_tokens=T, rows_ptr=1234,
                               compute=compute)
    return out


def test_group_cache_one_build_per_group() -> None:
    topk = torch.full((64, 512), -1, dtype=torch.int32)
    mds = {s: {"tag": f"md{s}"} for s in (2, 8, 14, 20)}
    src = lambda layer: mds[layer_topology(_config(), layer).kv_source]
    cache, calls = GroupRowsCache(), []
    pp0 = _run_stage(cache, range(0, 20), src, topk, 64, calls)
    assert calls == [2, 8, 14] and not cache.live
    pp1 = _run_stage(cache, range(20, 40), src, topk, 64, calls)
    assert calls == [2, 8, 14, 20, 24, 28, 32, 36] and not cache.live
    res = {**pp0, **pp1}
    for (_, _), layers in GROUPS.items():                 # every layer of a group got its group's first build
        assert all(res[l] == res[layers[0]] for l in layers)
        assert res[layers[0]][2] == layers[0]


def test_group_cache_new_step_and_replay_compaction() -> None:
    topk = torch.full((64, 512), -1, dtype=torch.int32)
    cache, calls = GroupRowsCache(), []
    full20, kept20 = {"tag": "full"}, {"tag": "kept"}   # replay: layers 21-39 see a compacted copy of 20's metadata
    _run_stage(cache, range(20, 40), lambda l: full20 if l == 20 else kept20, topk, 64, calls)
    assert calls == [20, 21, 24, 28, 32, 36]             # 21 rebuilds for the kept rows, 22-23 reuse it
    # a later step with equal-looking metadata (new object) and the same buffers never reuses a stale entry
    calls.clear()
    step2 = {"tag": "full"}
    _run_stage(cache, range(20, 40), lambda l: step2, topk, 64, calls)
    assert calls == [20, 24, 28, 32, 36]


def test_group_cache_key_mismatch_rebuilds() -> None:
    topk = torch.zeros((64, 512), dtype=torch.int32)
    cache, calls = GroupRowsCache(), []
    md = {"tag": "x"}
    common = dict(kv_source=2, index_source=2, src_md=md, topk=topk, rows_ptr=7)
    cache.get(layer_id=2, owns_indexer=True, num_tokens=64, compute=lambda: calls.append(2), **common)
    cache.get(layer_id=3, owns_indexer=False, num_tokens=32, compute=lambda: calls.append(3), **common)
    cache.get(layer_id=4, owns_indexer=False, num_tokens=32, compute=lambda: calls.append(4), **common)
    assert calls == [2, 3]


# ----------------------------------------------------------------------------- direct tile gather
def _interp() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _sources(dev: str, T: int = 40, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    win = torch.randn(300, 512, generator=g).half()
    win[7] = -0.0                                        # signed zeros must survive
    lat = torch.randn(900, 512, generator=g) * 3
    lat[11, :64] = -0.0
    ckv = qat.fp4_e4m3_pack_torch(lat)                   # packed [900, 288]
    wi = torch.randint(0, 300, (T, 128), generator=g)
    ci = torch.randint(0, 900, (T, 512), generator=g)
    wi[:, 100:] = -1
    ci[:, ::3] = -1
    wi[5], ci[5] = -1, -1                                # all-invalid row
    ci[6, :] = 11
    return [(win.to(dev), wi.to(dev)), (ckv.to(dev), ci.to(dev))]


def _reference_tile(sources, t0, t1, impl):
    return torch.cat([sk._gather(rows, idx[t0:t1], impl) for rows, idx in sources], dim=1)


@pytest.mark.skipif(not (_interp() or torch.cuda.is_available()), reason="needs TRITON_INTERPRET=1 or a GPU")
def test_gather_tile_bits() -> None:
    dev = "cuda" if torch.cuda.is_available() and not _interp() else "cpu"
    sources = _sources(dev)
    assert sk.gather_tile_ok(sources, 512)
    for t0, t1 in ((0, 16), (16, 40), (3, 9)):
        got = sk.gather_tile(sources, t0, t1, 512)
        for impl in (("torch", "sm70") if dev == "cuda" else ("torch",)):
            ref = _reference_tile(sources, t0, t1, impl)
            assert got.shape == ref.shape and got.dtype == ref.dtype
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), impl


@pytest.mark.sm70
@pytest.mark.parametrize("T,tile", [(17, 256), (600, 256), (1024, 256), (333, 64)])
def test_sparse_attention_knob_bit_identical(monkeypatch, T: int, tile: int) -> None:
    torch.manual_seed(T)
    sources = _sources("cuda", T=T, seed=T)
    q = (torch.randn(T, 16, 512, device="cuda") * 0.05).half()
    sink = torch.randn(16, device="cuda")
    outs = []
    for on in (False, True):
        monkeypatch.setattr(sk, "GATHER_ONCE", on)
        out = torch.empty_like(q)
        sk.sparse_attention(q, sources, sink, 512 ** -0.5, out, impl="sm70", tile=tile)
        outs.append(out)
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
