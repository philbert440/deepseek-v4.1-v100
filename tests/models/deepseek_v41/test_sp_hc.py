# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_SP_HC (sm70/sp_hc.py) on CPU with a 4-rank fake TP group (one thread per rank, barrier collectives):

* the REAL sp_block / _run_layers_sp == today's full-row block math bit for bit on exact inputs (quarter-integer
  stream / fn / partial maps, so every GEMM is exact in any summation order -- the two documented order-change sites,
  hc_mixes' SGEMM at M = S and the 4-way fp32 sum, cannot show by construction; on the GPU they are gated by oracle KL);
* routing stays TP-consistent: the all-gathered x every rank hands to attention / MoE is byte-identical across ranks;
* layers as one range == the same layers as two slices with a gather / re-shard between (layer-sliced prefill), bit
  for bit; uneven shards (22 rows / 4 ranks); Engram via reduce-scatter; the model-level gate (sp_active).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import HC, HIDDEN
from vllm.models.deepseek_v41.sm70 import sp_hc

# 32 rows over 4 ranks (shards of 8). CPU elementwise kernels (sigmoid / exp in the Sinkhorn split) take a vectorised
# path plus a scalar tail whose rounding differs by 1 ulp, so a shard whose [S, 4] mixes are not a multiple of the
# vector width would differ from the full-row run by CPU artefacts, not SP math; the GPU kernels are per-element.
# Uneven shards (22 / 4 -> 6 6 5 5) are covered by the shape / Engram tests below.
T, R = 32, 4
TU = 22


class FakeTP:
    """Barrier-synchronised fake TP group: all_gatherv concatenates the ranks' shards (every rank gets the same
    tensor), reduce_scatterv sums the ranks' [T, ...] inputs in rank order and hands each rank its rows."""

    def __init__(self, size: int):
        self.size = size
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(size, timeout=60)
        self.slots: list = [None] * size
        self.result = None
        self.gathered: list[list[torch.Tensor]] = []        # per all-gather: what each rank received

    def _collective(self, rank, x, sizes, kind):
        self.slots[rank] = x.clone()
        self.barrier.wait()
        if rank == 0:
            parts = self.slots
            if kind == "ag":
                self.result = torch.cat(parts, 0)
            else:
                total = parts[0].clone()
                for p in parts[1:]:
                    total = total + p
                b = [0]
                for s in sizes:
                    b.append(b[-1] + s)
                self.result = [total[b[r]:b[r + 1]].clone() for r in range(self.size)]
        self.barrier.wait()
        out = self.result.clone() if kind == "ag" else self.result[rank]
        if kind == "ag":
            with self.lock:
                if rank == 0:
                    self.gathered.append([None] * self.size)
        self.barrier.wait()
        if kind == "ag":
            self.gathered[-1][rank] = out
        self.barrier.wait()
        return out

    def group(self, rank: int) -> sp_hc.SPGroup:
        return sp_hc.SPGroup(rank, self.size, lambda x, sizes: self._collective(rank, x, sizes, "ag"),
                             lambda x, sizes: self._collective(rank, x, sizes, "rs"))


def run_ranks(fn, size=R):
    """fn(rank) in ``size`` threads; returns the per-rank results, re-raising any rank's exception."""
    results, errors = [None] * size, [None] * size

    def body(r):
        try:
            results[r] = fn(r)
        except BaseException as e:          # noqa: BLE001
            errors[r] = e
    ts = [threading.Thread(target=body, args=(r,)) for r in range(size)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    for e in errors:
        if e is not None:
            raise e
    return results


def _exact(shape, g, scale=4.0, lo=-4, hi=5):
    return torch.randint(lo, hi, shape, generator=g).float() / scale


def _params(g):
    return SimpleNamespace(
        hc_attn_fn=_exact((24, HC * HIDDEN), g, 64.0, -2, 3), hc_attn_scale=torch.tensor([0.5, 0.5, 0.25]),
        hc_attn_base=_exact((24,), g, 4.0), attn_norm_weight=_exact((HIDDEN,), g, 2.0, 1, 4),
        hc_ffn_fn=_exact((24, HC * HIDDEN), g, 64.0, -2, 3), hc_ffn_scale=torch.tensor([0.5, 0.5, 0.25]),
        hc_ffn_base=_exact((24,), g, 4.0), ffn_norm_weight=_exact((HIDDEN,), g, 2.0, 1, 4))


def _sparse_map(g):
    w = torch.zeros(HIDDEN, HIDDEN)
    idx = torch.randint(0, HIDDEN, (HIDDEN, 8), generator=g)
    val = torch.randint(-2, 3, (HIDDEN, 8), generator=g).float() / 4
    return w.scatter_(1, idx, val)


def _partials(g, n_layers):
    """Per layer: rank partial maps (attention, MoE), exact; the all-reduced output is their sum."""
    return [([_sparse_map(g) for _ in range(R)], [_sparse_map(g) for _ in range(R)]) for _ in range(n_layers)]


class FakeLayer:
    """A decoder layer for _run_layers_sp: forward_sp = the real sp_block with partial linear maps."""

    def __init__(self, params, wa, wm, seen_x, L):
        self.p, self.wa, self.wm, self.seen_x, self.L = params, wa, wm, seen_x, L

    def forward_sp(self, st, positions):
        r = st.group.rank

        def attn_partial(x):
            self.seen_x.setdefault(("attn", self.L), {})[r] = x.clone()
            return x.float() @ self.wa[r].t()

        def moe_partial(x):
            self.seen_x.setdefault(("moe", self.L), {})[r] = x.clone()
            return x.float() @ self.wm[r].t()
        sp_hc.sp_block(st, self.p, attn_partial, moe_partial)


def _model(layers):
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41Model

    m = DeepseekV41Model.__new__(DeepseekV41Model)
    m.__dict__.update(layers=layers, _sp_hc=True, pp_is_first=True)
    return m


@pytest.fixture
def order_invariant_mixes_gemm(monkeypatch):
    """hc_mixes' FP32 GEMM (K = 20480) is the one op whose reduction order depends on M (cuBLAS / MKL pick the
    schedule from M): the documented order-change site of VLLM_DS41_SP_HC. For the plumbing tests it is replaced by
    an fp64-accumulated version (products exact; the fp32 result independent of M), so SP vs the full-row reference
    must be bit for bit. test_mixes_gemm_order_site measures the real site."""
    import torch.nn.functional as F
    orig = F.linear

    def linear64(x, w, bias=None):
        if bias is None and x.dtype == torch.float32 and w.shape[-1] == HC * HIDDEN:
            return (x.double() @ w.double().t()).float()
        return orig(x, w, bias)
    monkeypatch.setattr(F, "linear", linear64)
    monkeypatch.setattr(torch.nn.functional, "linear", linear64)


def _setup(n_layers, seed=7):
    g = torch.Generator().manual_seed(seed)
    stream = _exact((T, HC, HIDDEN), g).to(torch.bfloat16)
    pre = _exact((T, HC), g, 8.0, 1, 5)
    params = [_params(g) for _ in range(n_layers)]
    parts = _partials(g, n_layers)
    s, p = stream.clone(), pre.clone()
    for L in range(n_layers):                            # today's math, all-reduced outputs = sum of partials
        wa, wm = parts[L]
        s, p = sp_hc.reference_block(s, p, params[L], lambda x, wa=wa: sum(x.float() @ w.t() for w in wa),
                                     lambda x, wm=wm: sum(x.float() @ w.t() for w in wm))
    return stream, pre, params, parts, s, p


def _run_sp(stream, pre, params, parts, ranges, seen_x=None):
    seen_x = {} if seen_x is None else seen_x
    tp = FakeTP(R)
    model = _model([FakeLayer(params[L], *parts[L], seen_x, L) for L in range(len(params))])
    positions = torch.arange(T)

    def rank_fn(r):
        ss, pp = stream.clone(), pre.clone()
        for a, b in ranges:                              # a slice boundary gathers and re-shards (full rows between)
            ss, pp = model._run_layers_sp(ss, pp, positions, a, b, group=tp.group(r))
        return ss, pp
    return run_ranks(rank_fn), tp, seen_x


@pytest.mark.parametrize("ranges", [((0, 3),), ((0, 1), (1, 3)), ((0, 2), (2, 3)), ((0, 1), (1, 2), (2, 3))])
def test_sp_equals_reference_bit_for_bit(order_invariant_mixes_gemm, ranges) -> None:
    """With an order-invariant mixes GEMM every other op is per-row, so the shard run (as one range or as slices
    with a gather / re-shard between, as layer-sliced prefill does) equals the full-row run bit for bit."""
    stream, pre, params, parts, s, p = _setup(3)
    results, tp, seen_x = _run_sp(stream, pre, params, parts, ranges)
    for r, (ss, pp) in enumerate(results):
        assert torch.equal(ss, s), f"rank {r}: stream differs from the full-row reference"
        assert torch.equal(pp, p), f"rank {r}: pre differs"
    for key, per_rank in seen_x.items():                 # TP-consistent routing input, identical on every rank
        xs = [per_rank[r] for r in range(R)]
        assert all(torch.equal(xs[0], x) for x in xs[1:]), f"gathered x differs across ranks at {key}"
        assert xs[0].shape == (T, HIDDEN) and xs[0].dtype == torch.float16
    for outs in tp.gathered:
        assert all(torch.equal(outs[0], o) for o in outs[1:])


@pytest.mark.parametrize("ranges", [((0, 1), (1, 3)), ((0, 1), (1, 2), (2, 3))])
def test_real_gemm_slices_equal_unsliced_and_track_reference(ranges) -> None:
    """With the real (M-dependent) mixes GEMM: sliced SP == unsliced SP bit for bit (same shards, same kernels; the
    gather / re-shard is a copy) -- the SLICE_VERIFY contract -- and SP tracks the full-row reference to bf16
    rounding (the order-change site, gated on the GPU by oracle KL)."""
    stream, pre, params, parts, s, p = _setup(3)
    whole, _, _ = _run_sp(stream, pre, params, parts, ((0, 3),))
    sliced, _, seen_x = _run_sp(stream, pre, params, parts, ranges)
    for r in range(R):
        assert torch.equal(whole[r][0], sliced[r][0]) and torch.equal(whole[r][1], sliced[r][1]), f"rank {r}"
        assert torch.equal(whole[r][0], whole[0][0])
    torch.testing.assert_close(whole[0][0].float(), s.float(), rtol=2e-2, atol=0.1)
    torch.testing.assert_close(whole[0][1], p, rtol=1e-2, atol=1e-2)
    frac = float((whole[0][0] != s).float().mean())
    print(f"\nSP vs reference after 3 layers (real GEMM): {100 * frac:.3f}% of bf16 stream elements differ")
    for key, per_rank in seen_x.items():
        xs = [per_rank[r] for r in range(R)]
        assert all(torch.equal(xs[0], x) for x in xs[1:]), key


def test_mixes_gemm_order_site() -> None:
    """The documented order-change site: hc_mixes' FP32 GEMM at M = S vs M = T on an arbitrary bf16 stream differs
    by reduction order only (<= 1e-5 relative here); every other HC op is per-row and identical."""
    from vllm.models.deepseek_v41.common.hc import hc_mixes, hc_post, hc_pre, rmsnorm_to_act

    g = torch.Generator().manual_seed(11)
    p = _params(g)
    stream = (torch.randn(T, HC, HIDDEN, generator=g) * 2).to(torch.bfloat16)
    pre = torch.rand(T, HC, generator=g)
    b = sp_hc.shard_bounds(T, R)
    sh = [stream[b[r]:b[r + 1]] for r in range(R)]
    mx_full = torch.nn.functional.linear(stream.flatten(1).float(), p.hc_attn_fn)
    mx_sh = torch.cat([torch.nn.functional.linear(x.flatten(1).float(), p.hc_attn_fn) for x in sh])
    rel = float(((mx_full - mx_sh).abs() / mx_full.abs().clamp(min=1e-3)).max())
    print(f"\nhc_mixes SGEMM M={T} vs M={T // R}: max rel diff {rel:.2e}")
    assert rel < 1e-5
    hp = hc_pre(stream, pre)
    assert torch.equal(hp, torch.cat([hc_pre(x, pre[b[r]:b[r + 1]]) for r, x in enumerate(sh)]))
    x = rmsnorm_to_act(hp, p.attn_norm_weight)
    assert torch.equal(x, torch.cat([rmsnorm_to_act(hp[b[r]:b[r + 1]], p.attn_norm_weight) for r in range(R)]))
    _, post, comb = hc_mixes(stream, p.hc_attn_fn, p.hc_attn_scale, p.hc_attn_base)
    y = torch.randn(T, HIDDEN, generator=g)
    out = hc_post(y, stream, post, comb)
    assert torch.equal(out, torch.cat([hc_post(y[b[r]:b[r + 1]], sh[r], post[b[r]:b[r + 1]], comb[b[r]:b[r + 1]])
                                       for r in range(R)]))


def test_shard_bounds_and_state() -> None:
    assert sp_hc.shard_bounds(1016, 4) == [0, 254, 508, 762, 1016]
    assert sp_hc.shard_bounds(999, 4) == [0, 250, 500, 750, 999]
    assert sp_hc.shard_bounds(3, 4) == [0, 1, 2, 3, 3] and sp_hc.shard_bounds(TU, 4) == [0, 6, 12, 17, 22]
    tp = FakeTP(4)
    st = sp_hc.shard(tp.group(2), torch.zeros(10, HC, HIDDEN, dtype=torch.bfloat16), torch.zeros(10, HC))
    assert (st.lo, st.hi) == (6, 8) and st.sizes == [3, 3, 2, 2] and st.stream.shape[0] == 2
    with pytest.raises(ValueError):
        st.scatter_sum(torch.zeros(9, HIDDEN))
    with pytest.raises(ValueError):
        sp_hc.check_shapes(torch.zeros(4, HC, HIDDEN), torch.zeros(4, HC))      # fp32 stream


def test_engram_via_reduce_scatter() -> None:
    """engram_rows returns un-reduced [T, D] rows; the shard gate receives the SUMMED rows of its own shard."""
    g = torch.Generator().manual_seed(3)
    tp = FakeTP(R)
    stream = _exact((TU, HC, HIDDEN), g).to(torch.bfloat16)
    pre = _exact((TU, HC), g, 8.0, 1, 5)
    rows = [_exact((TU, 16), g) for _ in range(R)]
    total = sum(rows)
    b = sp_hc.shard_bounds(TU, R)

    def rank_fn(r):
        st = sp_hc.shard(tp.group(r), stream, pre)
        got = {}
        sp_hc.sp_block(st, _params(torch.Generator().manual_seed(1)), lambda x: x.float(), lambda x: x.float(),
                       engram_rows=lambda n: rows[r], engram_gate=lambda sh, kv: got.setdefault("kv", kv.clone()))
        return got["kv"]
    for r, kv in enumerate(run_ranks(rank_fn)):
        assert torch.equal(kv, total[b[r]:b[r + 1]])


def test_model_gate(monkeypatch) -> None:
    import vllm.distributed as dist

    m = _model([])
    monkeypatch.setattr(dist, "get_tensor_model_parallel_world_size", lambda: 4)
    assert m.sp_active(1016)
    assert not m.sp_active(16)                                       # decode-sized
    assert not m.sp_active(1016, plan=object())                      # replay plan
    assert not m.sp_active(1016, aux_layers=(5,))                    # DSpark taps
    m.__dict__["pp_is_first"] = False
    assert not m.sp_active(1016)                                     # stage 1 unchanged
    m.__dict__.update(pp_is_first=True, _sp_hc=False)
    assert not m.sp_active(1016)
    monkeypatch.setattr(dist, "get_tensor_model_parallel_world_size", lambda: 1)
    m.__dict__["_sp_hc"] = True
    assert not m.sp_active(1016)                                     # TP 1: nothing to shard
    monkeypatch.setenv(sp_hc.SP_HC_ENV, "1")
    assert sp_hc.sp_hc_enabled()


def test_rank_timer_cpu(monkeypatch) -> None:
    """MoERankTimer bookkeeping with fake events (VLLM_DS41_MOE_RANK_TIMING)."""
    from vllm.models.deepseek_v41.sm70 import moe as moe_mod

    clock = [0.0]

    class Ev:
        def __init__(self):
            self.t = clock[0]

        def elapsed_time(self, other):
            return (other.t - self.t) * 1000.0

        def query(self):
            return True
    lines = []
    monkeypatch.setattr(moe_mod, "get_tensor_model_parallel_rank", lambda: 2, raising=False)
    monkeypatch.setattr("vllm.distributed.get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(moe_mod.MoERankTimer, "_ev", staticmethod(lambda: Ev()))
    monkeypatch.setattr(moe_mod.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(moe_mod.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    t = moe_mod.MoERankTimer(7, every=2)
    for dt in (0.004, 0.006):
        t.begin(1016); clock[0] += dt; t.mark("experts"); clock[0] += 0.001; t.mark("ar"); t.end()
    assert len(lines) == 1 and lines[0].startswith("DS41 moe rank timing layer 7 tp_rank 2 (2 calls, mean T 1016): "
                                                   "experts 4.00/6.00 ms, ar 1.00/1.00 ms"), lines
    t.begin(6)                                                       # decode-sized: not timed
    assert t.cur is None
