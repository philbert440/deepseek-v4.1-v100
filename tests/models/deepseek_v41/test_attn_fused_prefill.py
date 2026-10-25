# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_ATTN_FUSED_PREFILL: fused sparse prefill attention (csrc/ds41_sparse_prefill_sm70.cu).

CPU (algorithm model vs oracle):  pytest tests/models/deepseek_v41/test_attn_fused_prefill.py -m "not sm70"
GPU (one V100, prebuilt .so):     CUDA_VISIBLE_DEVICES=0 pytest tests/models/deepseek_v41/test_attn_fused_prefill.py \
                                      -m sm70 -s     (-s prints the error table)

Accuracy bar = the fused DECODE kernel's class: every path is scored against an FP64 reference in fp16 ulps of
max(|ref|, row RMS) (see _ulp; fp16 output rounding alone is <= 0.5). The fused prefill kernel with P split must stay within
1 ulp of the decode kernel's worst case; without the split it must be no worse than today's bmm path (+1 ulp).
"""

from __future__ import annotations

import pytest
import torch

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.sm70 import sparse_kernels as sk

SCALE = 512 ** -0.5


def _sources(dev: str, T: int, seed: int, nw: int = 128, nc: int = 512, n_win: int = 4096, n_ckv: int = 8192,
             qstd: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    win = (torch.randn(n_win, 512, generator=g) * 0.5).half()
    win[7] = -0.0
    lat = torch.randn(n_ckv, 512, generator=g) * 2
    lat[11, :64] = -0.0
    ckv = qat.fp4_e4m3_pack_torch(lat)
    wi = torch.randint(0, n_win, (T, nw), generator=g)
    wi[: T // 4, : nw // 2] = -1                       # early queries: window not full yet (whole blocks skipped)
    wi[:, 100:] = torch.where(torch.rand(T, nw - 100, generator=g) < 0.3, -1, wi[:, 100:])
    srcs = [(win.to(dev), wi.to(dev))]
    if nc:
        ci = torch.sort(torch.randint(0, n_ckv, (T, nc), generator=g), dim=1).values
        ci[:, ::3] = -1
        ci[: T // 8, 300:] = -1                        # short context: tail of the top-k empty
        ci[min(5, T - 1)] = -1
        srcs.append((ckv.to(dev), ci.to(dev)))
    wi[min(5, T - 1)] = -1                             # one query with no valid key at all -> output 0
    srcs[0] = (srcs[0][0], wi.to(dev))
    q = (torch.randn(T, 16, 512, generator=g) * qstd).half().to(dev)
    sink = torch.randn(16, generator=g).to(dev)
    return q, srcs, sink


def _ref64(q, sources, sink, scale=SCALE) -> torch.Tensor:
    """FP64 reference of the reference math (ref:k.py:310-403)."""
    kv = torch.cat([sk._gather(rows, idx, "torch") for rows, idx in sources], dim=1).double()
    valid = torch.cat([idx >= 0 for _, idx in sources], dim=1)
    s = torch.einsum("thd,tkd->thk", q.double(), kv) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    m = s.amax(dim=-1).clamp(min=-1e30)
    p = torch.exp(s - m[..., None])
    den = p.sum(-1) + torch.exp(sink.double()[None, : q.shape[1]] - m)
    return torch.einsum("thk,tkd->thd", p, kv) / den[..., None]


def _ulp(ref: torch.Tensor) -> torch.Tensor:
    """fp16 ulp of max(|ref|, RMS of its (token, head) output row): outputs are signed averages, so single components
    sit arbitrarily close to 0 and a pure per-element ulp would measure cancellation, not the kernel. Output rounding
    alone stays <= 0.5 of this unit."""
    rms = ref.pow(2).mean(dim=-1, keepdim=True).sqrt()
    a = torch.maximum(ref.abs(), rms).clamp(min=2.0 ** -14)
    return torch.exp2(torch.floor(torch.log2(a)) - 10)


def _err(out: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (out.double() - ref).abs()
    u = d / _ulp(ref)
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "max_ulp": u.max().item(),
            "mean_ulp": u.mean().item(), "rel_rms": (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()}


# ----------------------------------------------------------------------------- CPU: algorithm model
@pytest.mark.parametrize("T,nc", [(17, 512), (40, 512), (24, 0)])
def test_emulation_matches_reference(T: int, nc: int) -> None:
    """Blockwise skip + online softmax + floor + sink + P split reproduce the reference (CPU, FP64 scored)."""
    q, sources, sink = _sources("cpu", T, seed=T + nc, nc=nc)
    ref = _ref64(q, sources, sink)
    for p_split, bound in ((True, 1.0), (False, 2.0)):
        e = _err(sk.fused_prefill_emulate(q, sources, sink, SCALE, p_split=p_split), ref)
        assert e["max_ulp"] <= bound, (p_split, e)
    out = sk.fused_prefill_emulate(q, sources, sink, SCALE)
    assert torch.equal(out[5], torch.zeros_like(out[5]))                  # no valid key -> exactly 0


def test_fused_prefill_ok_shapes() -> None:
    q, sources, _ = _sources("cpu", 20, seed=1)
    assert not sk.fused_prefill_ok(q, sources, q)                          # CPU tensors never take the kernel
    if torch.cuda.is_available():
        qc = q.cuda()
        cs = [(r.cuda(), i.cuda()) for r, i in sources]
        assert sk.fused_prefill_ok(qc, cs, qc)
        assert sk.fused_prefill_ok(qc, cs[:1], qc)
        assert not sk.fused_prefill_ok(qc[:, :8], cs, qc[:, :8])           # 8 heads: not a WMMA M tile
        assert not sk.fused_prefill_ok(qc, cs + [cs[0]], qc)               # extra fp16 source -> tiled path


# ----------------------------------------------------------------------------- GPU: the kernel
def _have_ext() -> bool:
    try:
        sk.load_prefill_ext()
        return True
    except ImportError:
        return False


sm70 = pytest.mark.skipif(not (torch.cuda.is_available() and _have_ext()), reason="needs a GPU and the prebuilt .so")


def _decode_kernel(q, sources, sink) -> torch.Tensor:
    """FUSED_DECODE (the accuracy bar) on the same rows, 16 tokens per call (its workspace is T x 40 x H x D fp32)."""
    out = torch.empty_like(q)
    for t0 in range(0, q.shape[0], 16):
        t1 = min(q.shape[0], t0 + 16)
        sk.fused_decode_attention(q[t0:t1], [(r, i[t0:t1]) for r, i in sources], sink.float(), SCALE, out[t0:t1])
    return out


def _bmm_path(q, sources, sink, monkeypatch) -> torch.Tensor:
    monkeypatch.setattr(sk, "FUSED_PREFILL", False)
    out = torch.empty_like(q)
    sk.sparse_attention(q, sources, sink, SCALE, out, impl="sm70", tile=256)
    return out


@pytest.mark.sm70
@sm70
@pytest.mark.parametrize("T,nc,qstd", [(17, 512, 0.05), (300, 512, 0.05), (1024, 512, 0.05), (96, 0, 0.05),
                                       (256, 512, 1.0), (64, 512, 4.0)])
def test_fused_prefill_accuracy(monkeypatch, T: int, nc: int, qstd: float) -> None:
    q, sources, sink = _sources("cuda", T, seed=T + nc, nc=nc, qstd=qstd)
    ref = _ref64(q, sources, sink)
    errs = {"bmm (today)": _err(_bmm_path(q, sources, sink, monkeypatch), ref)}
    if nc:                                       # the decode kernel needs NW, NC multiples of 16 (true here)
        errs["fused decode"] = _err(_decode_kernel(q, sources, sink), ref)
    outs = {}
    for p_split in (True, False):
        out = torch.full_like(q, float("nan"))
        sk.fused_prefill_attention(q, sources, sink, SCALE, out, p_split=p_split)
        outs[p_split] = out
        errs[f"fused prefill psplit={int(p_split)}"] = _err(out, ref)
    print(f"\nT={T} nc={nc} qstd={qstd}")
    for k, e in errs.items():
        print(f"  {k:26s} max_ulp {e['max_ulp']:6.2f} mean_ulp {e['mean_ulp']:.3f} max_abs {e['max_abs']:.3e} "
              f"rel_rms {e['rel_rms']:.2e}")
    for out in outs.values():
        assert torch.isfinite(out).all()
        assert torch.equal(out[5], torch.zeros_like(out[5]))
    bar = errs.get("fused decode", errs["bmm (today)"])
    fs, fn, bmm = errs["fused prefill psplit=1"], errs["fused prefill psplit=0"], errs["bmm (today)"]
    assert fs["max_ulp"] <= max(bar["max_ulp"], 1.0) + 1.0, fs
    assert fs["mean_ulp"] <= 1.25 * bar["mean_ulp"] + 0.02, fs
    assert fn["max_ulp"] <= bmm["max_ulp"] + 1.0, fn
    assert fn["mean_ulp"] <= 1.25 * bmm["mean_ulp"] + 0.02, fn


@pytest.mark.sm70
@sm70
def test_fused_prefill_dispatch_strided_and_deterministic(monkeypatch) -> None:
    """Knob path through sparse_attention: out is a row slice of a padded buffer, ids are column slices of wider
    tensors (non-unit row stride), two runs are bitwise equal (fixed reduction order, no atomics)."""
    T = 200
    q, sources, sink = _sources("cuda", T, seed=7)
    wide = [(r, torch.cat([i, i[:, :32]], dim=1)[:, : i.shape[1]]) for r, i in sources]
    buf = torch.zeros((T + 24, 16, 512), dtype=torch.float16, device="cuda")
    monkeypatch.setattr(sk, "FUSED_PREFILL", True)
    sk.sparse_attention(q, wide, sink, SCALE, buf[:T], impl="sm70")
    again = torch.empty_like(q)
    sk.sparse_attention(q, sources, sink, SCALE, again, impl="sm70")
    assert torch.equal(buf[:T].view(torch.int16), again.view(torch.int16))
    assert torch.equal(buf[T:], torch.zeros_like(buf[T:]))
    ref = sk.fused_prefill_emulate(q, sources, sink, SCALE)
    assert _err(again, ref.double())["max_ulp"] <= 2.0                     # kernel == its algorithm model


@pytest.mark.sm70
@sm70
def test_fused_prefill_graph_capture() -> None:
    """No allocation inside: capturable, replay == eager bitwise."""
    T = 64
    q, sources, sink = _sources("cuda", T, seed=11)
    eager = torch.empty_like(q)
    sk.fused_prefill_attention(q, sources, sink, SCALE, eager)
    out = torch.empty_like(q)
    sinkf = sink.float().contiguous()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sk.fused_prefill_attention(q, sources, sinkf, SCALE, out)          # warm-up (loads the module)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sk.fused_prefill_attention(q, sources, sinkf, SCALE, out)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
