# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder replay / prefill row trim (sm70/replay.py, VLLM_DS41_REPLAY_ROWS): host plan, metadata compaction, zero
window records, and a toy SWA stack that checks the receptive-field argument end to end over chunked steps. CPU only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.sm70 import replay
from vllm.models.deepseek_v41.sm70.replay import CompactRows, compact_metadata, plan_rows, zero_window_rows
from vllm.models.deepseek_v41.sm70.sparse import DS41CompressedMetadata, DS41SWAMetadata


def _positions(qsl, seq):
    qlen = np.diff(qsl)
    starts = np.asarray(seq) - qlen
    return np.concatenate([np.arange(s, s + q) for s, q in zip(starts, qlen)]).astype(np.int64)


def test_knob(monkeypatch):
    monkeypatch.delenv(replay.REPLAY_ROWS_KNOB, raising=False)
    assert replay.replay_rows_k() == 0
    monkeypatch.setenv(replay.REPLAY_ROWS_KNOB, "3072")
    assert replay.replay_rows_k() == 3072
    monkeypatch.setenv(replay.REPLAY_ROWS_KNOB, "64")
    with pytest.raises(ValueError):
        replay.replay_rows_k()


def test_plan_kinds():
    K = 3072
    # middle chunk of a 100K prompt: nothing kept
    qsl, seq, n = np.array([0, 1024]), np.array([10240]), np.array([100_000])
    p = plan_rows(qsl, _positions(qsl, seq), n, K)
    assert p.kind == "skip" and p.kept.size == 0 and p.dropped.size == 1024
    # decode row of request 0 + a chunk of request 1 that straddles the cut 100_000 - 3072 = 96_928
    qsl, seq, n = np.array([0, 1, 1025]), np.array([5001, 97_000]), np.array([5001, 100_000])
    pos = _positions(qsl, seq)
    p = plan_rows(qsl, pos, n, K)
    assert p.kind == "compact"
    assert p.kept[0] == 0                                     # the decode row
    assert np.all(pos[p.kept[1:]] >= 96_928) and np.all(pos[p.dropped] < 96_928)
    assert p.kept.size == 1 + 72 and list(p.query_start_loc) == [0, 1, 73]
    assert np.all(np.diff(p.kept) > 0)
    # last chunk + spec-verify rows: every row kept -> "none"
    qsl, seq, n = np.array([0, 6, 1030]), np.array([5006, 100_000]), np.array([5001, 100_000])
    assert plan_rows(qsl, _positions(qsl, seq), n, K).kind == "none"
    # prompt logprobs: bound with n_req 0
    qsl, seq, n = np.array([0, 1024]), np.array([10240]), np.array([0])
    assert plan_rows(qsl, _positions(qsl, seq), n, K).kind == "none"


def test_plan_empty_request_segment():
    qsl = np.array([0, 4, 4, 8])
    pos = np.array([0, 1, 2, 3, 96, 97, 98, 99], dtype=np.int64)
    p = plan_rows(qsl, pos, np.array([10_000, 7, 100]), 2)
    assert list(p.kept) == [6, 7] and list(p.query_start_loc) == [0, 0, 0, 2]


def _rows(plan, pos):
    kept = torch.from_numpy(plan.kept)
    tok2req = np.repeat(np.arange(plan.query_start_loc.size - 1), np.diff(plan.query_start_loc))
    return CompactRows(kept=kept, dropped=torch.from_numpy(plan.dropped), positions=torch.from_numpy(pos[plan.kept]),
                       tok2req=torch.from_numpy(tok2req).to(torch.int32),
                       query_start_loc=torch.from_numpy(plan.query_start_loc).to(torch.int32),
                       kept_cpu=plan.kept, positions_cpu=pos[plan.kept], query_start_loc_cpu=plan.query_start_loc)


def _base(T, R, qsl, seq, pos):
    return dict(num_reqs=R, num_actual_tokens=T, query_start_loc=torch.from_numpy(qsl).to(torch.int32),
                query_start_loc_cpu=qsl, seq_lens_cpu=seq, token_to_req_indices=torch.from_numpy(
                    np.repeat(np.arange(R), np.diff(qsl))).to(torch.int32),
                positions=torch.from_numpy(pos), positions_cpu=pos, block_table=torch.arange(R * 4).view(R, 4),
                block_size=32, slot_mapping=torch.arange(T, dtype=torch.int64) + 1000)


def test_compact_metadata():
    qsl, seq = np.array([0, 1, 9]), np.array([50, 300])
    pos = _positions(qsl, seq)
    T, R = 9, 2
    plan = plan_rows(qsl, pos, np.array([50, 430]), 135)             # req 1 cut 295: keep 295..299
    rows = _rows(plan, pos)
    win = torch.arange(T * 128, dtype=torch.int64).view(T, 128)
    swa = DS41SWAMetadata(**_base(T, R, qsl, seq, pos), window_slots=win)
    c = compact_metadata(swa, rows)
    assert c.replay_compacted and not swa.replay_compacted
    assert c.num_actual_tokens == plan.kept.size == 6
    assert torch.equal(c.window_slots, win[torch.from_numpy(plan.kept)])
    assert torch.equal(c.slot_mapping, swa.slot_mapping[torch.from_numpy(plan.kept)])
    assert c.num_reqs == R and c.block_table is swa.block_table and list(c.query_start_loc_cpu) == [0, 1, 6]
    assert list(c.positions_cpu) == [49, 295, 296, 297, 298, 299]
    nvis = torch.from_numpy((pos + 1) // 2)
    cmd = DS41CompressedMetadata(**_base(T, R, qsl, seq, pos), compress_ratio=2, storage_block_size=16,
                                 num_latents=T, latent_token_idx=torch.arange(T), latent_pos=torch.from_numpy(pos),
                                 latent_slots=torch.arange(T), num_visible=nvis, max_visible=150)
    cc = compact_metadata(cmd, rows)
    assert cc.num_latents == 0 and cc.latent_slots.numel() == 0
    assert torch.equal(cc.num_visible, nvis[torch.from_numpy(plan.kept)]) and cc.max_visible == 150


def test_zero_window_rows():
    rows = torch.ones(8, 4)
    zero_window_rows(rows, torch.tensor([3, -1, 5]))
    assert rows[3].sum() == 0 and rows[5].sum() == 0 and rows[0].sum() == 0 and rows[1:3].sum() == 8


def test_toy_stack_exact_over_chunks():
    """A 19-layer causal sliding-window stack (window W, KV = function of the layer input) prefilled in chunks with
    the trim: every layer's KV over the last W positions and the last-row output equal the untrimmed run whenever
    K >= W * 19 + W -- the receptive field of the design note, with dropped rows' records zeroed."""
    torch.manual_seed(0)
    L, W, P, chunk = 19, 4, 200, 37
    K = (W - 1) * (L + 1) + 8
    mix = [torch.randn(3, 3) * 0.5 for _ in range(L)]

    def run(trim: bool):
        h0 = torch.randn(P, 3, generator=torch.Generator().manual_seed(1))
        cache = torch.full((L, P, 3), float("nan"))             # recycled garbage until written
        out = torch.zeros(P, 3)
        for c0 in range(0, P, chunk):
            rows = np.arange(c0, min(P, c0 + chunk))
            plan = plan_rows(np.array([0, rows.size]), rows, np.array([P]), K if trim else P)
            keep = rows[plan.kept] if plan.kind != "none" else rows
            drop = rows[plan.dropped] if plan.kind != "none" else rows[:0]
            x = h0[keep]
            for layer in range(L):
                cache[layer, torch.from_numpy(drop)] = 0.0       # zero window records of the dropped rows
                cache[layer, torch.from_numpy(keep)] = torch.tanh(x @ mix[layer])
                y = torch.stack([cache[layer, max(0, p - W + 1):p + 1].mean(0) for p in keep]) if keep.size else x
                x = x + y
            out[torch.from_numpy(keep)] = x
        return cache, out

    c_ref, o_ref = run(False)
    c_trim, o_trim = run(True)
    assert torch.equal(o_trim[P - 1], o_ref[P - 1])
    assert torch.equal(c_trim[:, P - W:], c_ref[:, P - W:])
    assert not torch.isnan(c_trim[:, P - K:]).any()


def test_controller_step(monkeypatch):
    """bind -> begin -> compact_step / zero_dropped with stand-in layers (layers 21-22 share one SWA metadata, layer
    22 reads layer 20's compressed + index-K metadata)."""
    from types import SimpleNamespace

    import vllm.forward_context as fc

    monkeypatch.setattr(fc, "get_forward_context", lambda: SimpleNamespace(is_dummy_run=False))
    qsl, seq = np.array([0, 1, 9]), np.array([50, 300])
    pos = _positions(qsl, seq)
    T, R = 9, 2
    swa = DS41SWAMetadata(**_base(T, R, qsl, seq, pos), window_slots=torch.zeros(T, 128, dtype=torch.int64))
    cmd = DS41CompressedMetadata(**_base(T, R, qsl, seq, pos), compress_ratio=1, storage_block_size=32,
                                 num_latents=T, latent_token_idx=torch.arange(T), latent_pos=torch.from_numpy(pos),
                                 latent_slots=torch.arange(T), num_visible=torch.from_numpy(pos + 1), max_visible=300)
    caches = {n: torch.ones(2048, 4) for n in ("c21", "c22")}

    def layer(name, kv_source=None, k_cache=None):
        cache = SimpleNamespace(layer_name=f"{name}.swa", rows=lambda: caches[name])
        indexer = SimpleNamespace(k_cache_name=k_cache) if k_cache else None
        return SimpleNamespace(attn=SimpleNamespace(swa_cache=cache, kv_source_name=kv_source, indexer=indexer))

    layers = [(21, layer("c21", "L20")), (22, layer("c22", "L20", "L20.k"))]
    ctl = replay.ReplayController(135, 21, layers, 64, torch.device("cpu"))
    md_all = {"c21.swa": swa, "c22.swa": swa, "L20": cmd, "L20.k": cmd, "L5.swa": "untouched"}
    # no binding (dummy / profile / graph-replayed step) -> today's path
    assert ctl.begin(md_all) is None
    ctl.bind(R, T, qsl, np.array([50, 430]))
    plan = ctl.begin(md_all)
    assert plan is not None and plan.kind == "compact" and list(plan.kept) == [0, 4, 5, 6, 7, 8]
    assert ctl.begin(md_all) is None                          # the binding is consumed
    shared = SimpleNamespace(topk_indices=torch.arange(64 * 3).view(64, 3).int(),
                             candidate_blocks=torch.arange(64 * 2).view(64, 2).int())
    ref_topk = shared.topk_indices[torch.from_numpy(plan.kept)].clone()
    out, rows = ctl.compact_step(md_all, plan, shared)
    assert out["L5.swa"] == "untouched" and out["c21.swa"] is out["c22.swa"] and out["L20"] is out["L20.k"]
    assert out["c21.swa"].num_actual_tokens == 6 and torch.equal(shared.topk_indices[:6], ref_topk)
    assert list(out["L20"].num_visible) == [50, 296, 297, 298, 299, 300]
    assert replay.drafter_rows()[0] == T and torch.equal(replay.drafter_rows()[1], rows.kept)
    ctl.zero_dropped(md_all, rows.dropped)
    for c in caches.values():
        zeroed = (c.sum(1) == 0).nonzero().flatten().tolist()
        assert zeroed == [1001, 1002, 1003]                   # slots 1000 + dropped rows 1..3
    # a forward that is not the bound step never trims
    ctl.bind(R, T, np.array([0, 2, 9]), np.array([50, 430]))
    assert ctl.begin(md_all) is None
    # decode-like step: every first position clears its cut -> None without row arrays
    ctl.bind(R, T, qsl, np.array([50, 300]))
    assert ctl.begin(md_all) is None
