# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_ATTN_PRECAPTURE / VLLM_DS41_ATTN_NO_LAZY_CAPTURE (sm70/precapture.py): the startup key plan, the
no-lazy eager path of the per-layer attention graphs and of the metadata-builder graphs.

CPU only:  CUDA_VISIBLE_DEVICES= pytest tests/models/deepseek_v41/test_attn_precapture.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v41 import attention as attn
from vllm.models.deepseek_v41.sm70 import precapture as pc
from vllm.models.deepseek_v41.sm70 import sparse

SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 24]   # launcher capture sizes (dspark-adapt p0-ext)


@pytest.fixture(autouse=True)
def _fresh_state():
    pc.reset_for_tests()
    yield
    pc.reset_for_tests()


# ------------------------------------------------------------------------------------------------ the plan
def test_decode_plan_b2_dspark() -> None:
    """max_num_seqs 2, k = 5: every B=1 verify length 1..6, every B=2 length 2..12 (adaptive drafts), and the
    16-row capture size (R=2); 24 is past GRAPH_MAX_T and never gets per-layer graphs."""
    pairs = pc.batch_pairs(SIZES, max_num_seqs=2, query_len=6, graph_max_t=16, max_num_tokens=1024, shapes="decode")
    assert pairs == [(t, 1) for t in range(1, 7)] + [(t, 2) for t in range(2, 13)] + [(16, 2)]
    batches = pc.plan_batches(SIZES, 2, 6, 16, max_model_len=1 << 20, max_num_tokens=1024, shapes="decode")
    assert len(batches) == 2 * len(pairs)
    for b in batches:
        assert b.num_reqs <= 2 and b.seq_len >= max(b.query_lens)
        if (b.num_tokens, b.num_reqs) != (16, 2):
            assert max(b.query_lens) <= 6              # a real verify step: <= 1 + k rows per request
    short = {(b.num_tokens, b.num_reqs) for b in batches if b.regime == "short"}
    long_ = {(b.num_tokens, b.num_reqs) for b in batches if b.regime == "long"}
    assert short == long_ == set(pairs)
    for b in batches:
        if b.regime == "short":
            assert b.seq_len == max(b.query_lens)
        else:   # every compressed source (ratio <= 2) sees more than IDX_TOPK entries
            assert b.seq_len // pc.MAX_COMPRESS_RATIO > pc.IDX_TOPK
    assert {(6, 1), (12, 2), (7, 2), (1, 1)} <= set(pairs)


def test_all_plan_and_limits() -> None:
    pairs = pc.batch_pairs(SIZES, 2, 6, 16, 1024, shapes="all")
    assert len(pairs) == 16 + 15 and (16, 1) in pairs and (13, 2) in pairs
    assert max(t for t, _ in pairs) == 16
    # the graph limit is also bounded by the largest capture size and the token budget
    assert max(t for t, _ in pc.batch_pairs([1, 2, 4, 8], 2, 6, 16, 1024, shapes="all")) == 8
    assert max(t for t, _ in pc.batch_pairs(SIZES, 2, 6, 16, 5, shapes="all")) == 5
    assert pc.batch_pairs([], 2, 6, 16, 1024) == []
    with pytest.raises(ValueError):
        pc.batch_pairs(SIZES, 2, 6, 16, 1024, shapes="bogus")


def test_no_spec_plan() -> None:
    """Plain decode (query_len 1): R rows of 1, plus (decode) the capture sizes the batch can fill."""
    pairs = pc.batch_pairs([1, 2, 4, 8], max_num_seqs=4, query_len=1, graph_max_t=16, max_num_tokens=1024,
                           shapes="decode")
    assert {(1, 1), (2, 2), (3, 3), (4, 4), (8, 4)} == set(pairs)
    assert pc.batch_pairs([1, 2, 4, 8], 4, 1, 16, 1024) == [(1, 1), (2, 2), (3, 3), (4, 4)]


def test_verify_plan_default() -> None:
    """Default (verify): each request 1 or 1+k rows -> B=1: T 1, 6; B=2: T 2, 7, 12. 10 batches, not 36."""
    assert pc.precapture_shapes() == "verify"
    pairs = pc.batch_pairs(SIZES, 2, 6, 16, 1024)
    assert pairs == [(1, 1), (6, 1), (2, 2), (7, 2), (12, 2)]
    batches = pc.plan_batches(SIZES, 2, 6, 16, 1 << 20, 1024)
    assert len(batches) == 10
    assert {b.query_lens for b in batches} == {(1,), (6,), (1, 1), (6, 1), (6, 6)}
    # B=4 verify passes GRAPH_MAX_T beyond 2 full verifies: 4,9,14 only
    assert [t for t, r in pc.batch_pairs(SIZES, 4, 6, 16, 1024) if r == 4] == [4, 9, 14]


def test_verify_b1all_plan(monkeypatch) -> None:
    """verify_b1all (DSpark p_min only at B=1): R=1 every T 1..6, R=2 only the verify shapes 2, 7, 12.
    18 batches (9 shapes x long/short): 4 R=1 shapes, 8 batches more than verify; largest-first order kept."""
    pairs = pc.batch_pairs(SIZES, 2, 6, 16, 1024, shapes="verify_b1all")
    assert pairs == [(t, 1) for t in range(1, 7)] + [(2, 2), (7, 2), (12, 2)]
    assert set(pairs) - set(pc.batch_pairs(SIZES, 2, 6, 16, 1024)) == {(2, 1), (3, 1), (4, 1), (5, 1)}
    batches = pc.plan_batches(SIZES, 2, 6, 16, 1 << 20, 1024, shapes="verify_b1all")
    assert len(batches) == 18
    assert {b.query_lens for b in batches if b.num_reqs == 1} == {(t,) for t in range(1, 7)}
    assert {b.query_lens for b in batches if b.num_reqs == 2} == {(1, 1), (6, 1), (6, 6)}
    for reg in ("long", "short"):
        ts = [b.num_tokens for b in batches if b.regime == reg]
        assert ts == sorted(ts, reverse=True) and ts[0] == 12
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE_SHAPES", "verify_b1all")
    assert pc.precapture_shapes() == "verify_b1all"
    # plain decode (k = 0): identical to verify
    assert pc.batch_pairs([1, 2, 4, 8], 4, 1, 16, 1024, shapes="verify_b1all") == pc.batch_pairs([1, 2, 4, 8], 4, 1, 16,
                                                                                              1024)


def test_capture_order_largest_first() -> None:
    """Long regime first, then short; each by decreasing T (the pool reuses the largest capture's blocks)."""
    batches = pc.plan_batches(SIZES, 2, 6, 16, 1 << 20, 1024, shapes="decode")
    regimes = [b.regime for b in batches]
    assert regimes == sorted(regimes)            # 'long' < 'short'
    for reg in ("long", "short"):
        ts = [b.num_tokens for b in batches if b.regime == reg]
        assert ts == sorted(ts, reverse=True) and ts[0] == 16


def test_long_regime_needs_the_context() -> None:
    short_model = pc.plan_batches(SIZES, 2, 6, 16, max_model_len=1000, max_num_tokens=1024)
    assert {b.regime for b in short_model} == {"short"}
    b = pc.plan_batches([1], 1, 1, 16, 1 << 20, 1024, long_context=10)
    assert [x.seq_len for x in b if x.regime == "long"] == [pc.min_long_seq_len()]


def test_split_and_describe() -> None:
    assert pc.split_rows(12, 2) == (6, 6) and pc.split_rows(7, 2) == (4, 3) and pc.split_rows(5, 1) == (5,)
    with pytest.raises(ValueError):
        pc.split_rows(1, 2)
    descs = {(t, t, 1, "short") for t in range(1, 7)} | {(t, t, 2, "long") for t in (2, 3, 4, 12)} | {(13, 16, 2, "-")}
    assert pc.describe_pairs(descs) == "R=1: T 1-6; R=2: T 2-4,12-13"
    assert pc.describe_pairs(set()) == "none"


def test_knobs(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_DS41_ATTN_PRECAPTURE", raising=False)
    monkeypatch.delenv("VLLM_DS41_ATTN_NO_LAZY_CAPTURE", raising=False)
    assert not pc.precapture_enabled() and not pc.no_lazy_capture() and pc.precapture_reserve_bytes() == 0
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE", "1")
    assert pc.precapture_reserve_bytes() == 256 << 20
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE_RESERVE_MIB", "64")
    assert pc.precapture_reserve_bytes() == 64 << 20
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE_SHAPES", "nope")
    with pytest.raises(ValueError):
        pc.precapture_shapes()
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE_LONG_SEQ", "100")
    assert pc.long_seq_len() == pc.min_long_seq_len()


def test_regime_keyed_only_for_graph_capable_indexers(monkeypatch) -> None:
    layer = _layer(("k",))
    layer.indexer = None
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE", "0")
    assert attn.DeepseekV41Attention._regime_keyed(layer)                 # knob off: the old key
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE", "1")
    assert not attn.DeepseekV41Attention._regime_keyed(layer)             # reuse layer: regime only steers eager code
    layer.indexer = SimpleNamespace(graph_capable=lambda: False)
    assert not attn.DeepseekV41Attention._regime_keyed(layer)             # full-scan indexer: eager segment
    layer.indexer = SimpleNamespace(graph_capable=lambda: True)
    assert attn.DeepseekV41Attention._regime_keyed(layer)                 # candidate scoring inside the graph


def test_key_capacity() -> None:
    old = attn.GRAPH_MAX_KEYS
    try:
        assert attn.ensure_graph_key_capacity(10) == old          # never lowered
        assert attn.ensure_graph_key_capacity(old + 40) == old + 40
    finally:
        attn.GRAPH_MAX_KEYS = old


# ------------------------------------------------------------------------- per-layer attention graphs
class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def _layer(key: tuple) -> attn.DeepseekV41Attention:
    layer = attn.DeepseekV41Attention.__new__(attn.DeepseekV41Attention)
    nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.attn.ds41_attention"
    layer.topo = SimpleNamespace(compress_ratio=0)
    layer._graph_key = lambda *a: key            # instance attribute shadows the method (pointers are CUDA-only)
    return layer


def _step(layer, T: int = 6, padded: int = 6, R: int = 1) -> list[str]:
    ran: list[str] = []
    plan = lambda st: [(lambda: ran.append("project"), True), (lambda: ran.append("attend"), True)]
    swa_md = SimpleNamespace(block_table=torch.zeros((R, 4), dtype=torch.int32))
    out = torch.zeros((padded, 2, 2))
    attn.DeepseekV41Attention._graph_step(layer, plan, None, None, out, {}, swa_md, T)
    return ran


def test_no_lazy_unseen_key_runs_eagerly() -> None:
    pc.finish(freeze=True)
    layer = _layer(("k12",))
    for _ in range(3):                        # never warms, never captures, never stores an entry
        assert _step(layer, T=12, padded=12, R=2) == ["project", "attend"]
    assert layer.__dict__.get("_decode_graphs", {}) == {}
    assert pc.STATE.eager_steps == 3
    assert ("eager", (12, 12, 2, "-")) in pc.STATE.unseen_logged


def test_no_lazy_warm_only_key_stays_eager() -> None:
    layer = _layer(("k",))
    assert _step(layer) == ["project", "attend"]          # lazy allowed: warm run, entry created
    assert layer._decode_graphs[("k",)]["runs"] == 1
    pc.finish(freeze=True)
    assert _step(layer) == ["project", "attend"]          # frozen before its capture: eager, still not captured
    assert layer._decode_graphs[("k",)]["graphs"] is None and pc.STATE.eager_steps == 1


def test_no_lazy_captured_key_replays() -> None:
    layer = _layer(("k",))
    g = _FakeGraph()
    layer._decode_graphs = {("k",): {"runs": 1, "graphs": [g], "st": {}}}
    pc.finish(freeze=True)
    assert _step(layer) == []                 # one capturable group -> one replay, no eager segment
    assert g.replays == 1 and pc.STATE.eager_steps == 0


def test_default_path_unchanged() -> None:
    """Knobs off: first use of a key is the warm run (eager) exactly as before."""
    assert pc.capture_allowed() and not pc.STATE.done
    layer = _layer(("k",))
    assert _step(layer) == ["project", "attend"]
    assert layer._decode_graphs[("k",)]["runs"] == 1


# -------------------------------------------------------------------------------- builder graphs
def _builder(cls):
    b = cls.__new__(cls)
    b.__dict__["_bgraphs"] = {}
    return b


def _kernels(b, monkeypatch) -> int:
    calls = []
    monkeypatch.setattr(sparse, "BUILDER_GRAPHS", True)
    monkeypatch.setattr(sparse, "token_meta_kernel", lambda *a: calls.append("meta"))
    t = torch.zeros(4, dtype=torch.int64)
    cm = SimpleNamespace(query_start_loc=t, seq_lens=t, block_table_tensor=t, slot_mapping=t)
    c = dict(num_actual_tokens=2, num_reqs=1, token_to_req_indices=t, positions=t, slot_mapping=t, block_table=t)
    sparse._DS41BuilderBase._kernels(b, cm, c, lambda: calls.append("fn"))
    return len(calls)


def test_builder_no_lazy(monkeypatch) -> None:
    pc.finish(freeze=True)
    b = _builder(sparse.DS41SWAMetadataBuilder)
    assert _kernels(b, monkeypatch) == 2 and b._bgraphs == {}      # eager, no entry
    d = _builder(sparse.DS41DraftSWAMetadataBuilder)                 # the drafter's builders stay lazy
    assert _kernels(d, monkeypatch) == 2 and len(d._bgraphs) == 1
    assert next(iter(d._bgraphs.values()))["runs"] == 1


def test_builder_shared_pool_key(monkeypatch) -> None:
    pools = {}
    monkeypatch.setattr(sparse, "_BUILDER_POOLS", pools)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: object())
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE", "1")
    assert sparse._builder_pool((1, 1)) is sparse._builder_pool((2, 2)) and list(pools) == [()]
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE", "0")
    assert sparse._builder_pool((1, 1)) is not sparse._builder_pool((2, 2))


# ------------------------------------------------------------------------- runner orchestration
def _fake_runner(monkeypatch, calls: list):
    from vllm.v1.worker import gpu_model_runner as gmr

    runner = gmr.GPUModelRunner.__new__(gmr.GPUModelRunner)
    layer = _layer(("k",))
    runner.compilation_config = SimpleNamespace(cudagraph_capture_sizes=list(SIZES),
                                                static_forward_context={layer.layer_name: layer})
    runner.scheduler_config = SimpleNamespace(max_num_seqs=2)
    runner.uniform_decode_query_len = 6
    runner.max_model_len = 1 << 20
    runner.max_num_tokens = 1024
    runner.requests = {}
    runner._ds41_mapped_blocks = {0}

    def dummy_run(num_tokens, **kw):
        lens, seq = kw["ds41_batch"]
        assert sum(lens) == num_tokens and kw["force_attention"] and pc.STATE.precapturing
        calls.append((tuple(lens), seq))

    runner._dummy_run = dummy_run
    monkeypatch.setattr(attn, "layer_graphs_active", lambda: (True, ""))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a: None, raising=False)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda *a: None, raising=False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1 << 30, 2 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    return runner


@pytest.mark.parametrize("no_lazy", ["0", "1"])
def test_runner_precapture_drives_every_shape_twice(monkeypatch, no_lazy) -> None:
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE", "1")
    monkeypatch.setenv("VLLM_DS41_ATTN_NO_LAZY_CAPTURE", no_lazy)
    old_cap = attn.GRAPH_MAX_KEYS
    calls: list = []
    try:
        runner = _fake_runner(monkeypatch, calls)
        assert runner._ds41_precapture_attention_graphs() == 0
        plan = pc.plan_batches(SIZES, 2, 6, attn.GRAPH_MAX_T, 1 << 20, 1024)
        runs = attn.GRAPH_WARMUP_STEPS + 1
        assert calls == [(b.query_lens, b.seq_len) for b in plan for _ in range(runs)]
        assert attn.GRAPH_MAX_KEYS >= len(plan)
        assert pc.STATE.done and not pc.STATE.precapturing and pc.STATE.frozen == (no_lazy == "1")
    finally:
        attn.GRAPH_MAX_KEYS = old_cap


def test_runner_precapture_off_is_a_noop(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_DS41_ATTN_PRECAPTURE", raising=False)
    monkeypatch.delenv("VLLM_DS41_ATTN_NO_LAZY_CAPTURE", raising=False)
    calls: list = []
    runner = _fake_runner(monkeypatch, calls)
    assert runner._ds41_precapture_attention_graphs() == 0
    assert calls == [] and not pc.STATE.done and pc.capture_allowed()


def test_runner_refuses_live_requests(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ATTN_PRECAPTURE", "1")
    runner = _fake_runner(monkeypatch, [])
    runner.requests = {"r": object()}
    with pytest.raises(RuntimeError):
        runner._ds41_precapture_attention_graphs()


# ------------------------------------------- reindex candidate scoring scratch (VLLM_DS41_ATTN_CAND_DECODE_ROWS)
def _cand_inputs(Q: int, device: str, seed: int = 0):
    from vllm.models.deepseek_v41.common.contracts import CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_rows = CAND_TOPK_BLOCKS * CAND_BLOCK + 64
    q = torch.randn((Q, IDX_HEADS, IDX_DIM), generator=g).half().to(device)
    w = torch.randn((Q, IDX_HEADS), generator=g).float().to(device)
    cand = torch.sort(torch.randperm(4 * CAND_TOPK_BLOCKS, generator=g)[:CAND_TOPK_BLOCKS]).values
    cand = cand[None, :].repeat(Q, 1).int()
    cand[:, -5:] = -1                                                # -1 padding at the end
    ends = torch.full((Q,), 4 * CAND_TOPK_BLOCKS * CAND_BLOCK - 3, dtype=torch.int64)
    keys_all = torch.randn((n_rows, IDX_DIM), generator=g).half().to(device)
    return q, w, cand.to(device), ends.to(device), keys_all, (lambda pos: torch.where(pos >= 0, pos % n_rows, -1))


def _candidate_topk_before(q, w, cand, ends, keys_all, rows_of):
    """The pre-change body (out-of-place weighting, keys kept alive), for the bit-equality check."""
    from vllm.models.deepseek_v41.common.contracts import CAND_BLOCK, IDX_TOPK
    Q = q.shape[0]
    dev = q.device
    pos = (cand.to(torch.int64)[:, :, None] * CAND_BLOCK + torch.arange(CAND_BLOCK, device=dev)[None, None, :]).view(Q, -1)
    valid = (cand >= 0)[:, :, None].expand(-1, -1, CAND_BLOCK).reshape(Q, -1) & (pos < ends[:, None])
    pos = torch.where(valid, pos, torch.full_like(pos, -1))
    rows = rows_of(pos)
    keys = keys_all.index_select(0, rows.clamp(min=0).view(-1)).view(Q, -1, keys_all.shape[1])
    logits = torch.bmm(q, keys.transpose(1, 2), out_dtype=torch.float32)
    scores = (logits.relu_() * w.float()[:, :, None]).sum(dim=1)
    scores.masked_fill_(~valid, float("-inf"))
    k = min(IDX_TOPK, scores.shape[1])
    vals, idx = scores.topk(k, dim=-1, sorted=False)
    picked = pos.gather(1, idx)
    picked = torch.where(vals > float("-inf"), picked, torch.full_like(picked, torch.iinfo(torch.int64).max))
    picked = picked.sort(dim=-1).values
    picked = torch.where(picked == torch.iinfo(torch.int64).max, torch.full_like(picked, -1), picked)
    out = torch.full((Q, IDX_TOPK), -1, dtype=torch.int32, device=dev)
    out[:, :k] = picked.to(torch.int32)
    return out


def _bmm_fp32_out_supported(device: str) -> bool:
    try:
        a = torch.zeros((1, 2, 2), dtype=torch.float16, device=device)
        torch.bmm(a, a, out_dtype=torch.float32)
        return True
    except (RuntimeError, TypeError, NotImplementedError):
        return False


def _check_candidate_topk(device: str) -> None:
    from vllm.models.deepseek_v41.indexer import candidate_topk
    q, w, cand, ends, keys_all, rows_of = _cand_inputs(12, device)
    whole = candidate_topk(q, w, cand, ends, keys_all, rows_of)
    assert torch.equal(whole, _candidate_topk_before(q, w, cand, ends, keys_all, rows_of))   # in-place: same bits
    for tile in (1, 2, 4):     # the decode-row cap: per-tile calls must give the one-tile result exactly
        parts = [candidate_topk(q[a:a + tile], w[a:a + tile], cand[a:a + tile], ends[a:a + tile], keys_all, rows_of)
                 for a in range(0, 12, tile)]
        assert torch.equal(torch.cat(parts), whole), f"tile {tile} differs from the one-tile result"


def test_candidate_topk_exact_cpu(monkeypatch) -> None:
    """CPU: the in-place weighting and the row tiling give the same bits (fp32 bmm stand-in when the CPU build has
    no fp16 bmm with fp32 output)."""
    if not _bmm_fp32_out_supported("cpu"):
        bmm = torch.bmm
        monkeypatch.setattr(torch, "bmm", lambda a, b, out_dtype=None: bmm(a.float(), b.float()))
    _check_candidate_topk("cpu")


@pytest.mark.sm70
def test_candidate_topk_exact_sm70() -> None:
    """GPU gate for VLLM_DS41_ATTN_CAND_DECODE_ROWS: bit-identical top-k at tiles 1/2/4 vs one 12-row tile (if
    this passes, the knob changes no decode output and needs no greedy re-check)."""
    _check_candidate_topk("cuda")
