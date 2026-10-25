# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_INDEXER_FUSED: fused lightning-indexer scoring + top-512 on SM70 (csrc/ds41_indexer_fused_sm70.cu).

* ``fused_candidate_topk`` replaces ``indexer.candidate_topk`` (reindex layers 24-36): keys are gathered in-kernel
  from the index-K cache through the block table, scores [rows, 16,384 slots] come from one WMMA kernel with the
  relu*w head sum in its epilogue, then a radix-select top-512 kernel. No [T, 16K, 128] gather, no [T, 32, 16K]
  FP32 logits, no torch topk / sort.
* ``fused_index_scores`` replaces ``index_scores`` (layer 20 and the scan + mask path): same [Q, n] FP32 output.
* ``fused_topk_sorted`` replaces ``topk_sorted`` where every pick that matters is finite (scores already -inf where
  masked; a row never has fewer than min(512, end) finite entries -- true for the causal bound and for candidate
  masks, which keep >= 2048 visible positions).

Numerics: HMMA FP32 accumulation and an in-order RN head sum (no FMA) -- not bit-identical to cuBLAS, so a near-tie
at rank 512 can swap. Ties at the threshold go to the lowest position (torch.topk's tie order is unspecified).
Gate: tests/models/deepseek_v41/test_indexer_fused_gpu.py (identical top-512 SETS + drift report) and the KL panel.
The extension is PREBUILT (scripts/build_ds41_indexer_fused.py); knob on + missing .so raises at the first call.
.so lookup: VLLM_DS41_INDEXER_FUSED_SO if set, else the serving tree's own build (<tree>/build/indexer-fused/
ds41_indexer_fused.so: ``build_ds41_indexer_fused.py --out <tree>/build/indexer-fused``; build/ is gitignored), else
<vllm package dir>/ds41_indexer_fused.so (the release wheel).
Applies to prefill-sized steps only (T > 16, not capturing); decode-sized steps use the graph-capturable decode
kernels below when VLLM_DS41_INDEXER_FUSED_DECODE=1, else today's path.
"""

from __future__ import annotations

import importlib.util
import os

import torch

from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_HEADS, IDX_TOPK

FUSED_ENV = "VLLM_DS41_INDEXER_FUSED"
FUSED_SO_ENV = "VLLM_DS41_INDEXER_FUSED_SO"
# VLLM_DS41_INDEXER_TOPK_V2 (needs FUSED=1 and the v2 .so): the scan path's top-512 by topk_sel -- 1024 threads,
# warp-aggregated radix histograms, the causal bound (ends) applied IN the kernel, so the [Q, n] masked_fill of
# non-candidate layers is skipped. Same selection rule as topk512 (ties -> lowest position).
TOPK_V2_ENV = "VLLM_DS41_INDEXER_TOPK_V2"
FUSED_SO_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "ds41_indexer_fused.so")
TREE_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 4), "build", "indexer-fused",
                       "ds41_indexer_fused.so")
CAND_SLOTS = CAND_TOPK_BLOCKS * CAND_BLOCK          # 16,384
SLOTS_PER_CTA = 2048
DENSE_KEYS_PER_CTA = 2048
ROWS_PER_CALL = 1024                                 # candidate scratch: 1024 x 16K x 4 B = 64 MiB
_EXT: list = []


def fused_enabled() -> bool:
    return knobs.env_bool(FUSED_ENV, False)


def so_path() -> str:
    explicit = os.environ.get(FUSED_SO_ENV, "").strip()
    if explicit:
        return explicit
    tree = os.path.normpath(TREE_SO)
    return tree if os.path.isfile(tree) else FUSED_SO_DEFAULT


def load_ext():
    if _EXT:
        return _EXT[0]
    path = so_path()
    if not os.path.isfile(path):
        raise ImportError(f"{FUSED_ENV}=1 but the extension is not built at {path} "
                          f"(run scripts/build_ds41_indexer_fused.py, or set {FUSED_SO_ENV})")
    spec = importlib.util.spec_from_file_location("ds41_indexer_fused", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXT.append(mod)
    return mod


def _q_ok(q: torch.Tensor) -> torch.Tensor:
    if q.stride(-1) == 1 and q.stride(1) % 16 == 0 and q.stride(0) % 16 == 0 and q.data_ptr() % 32 == 0:
        return q
    return q.contiguous()


def fused_candidate_topk(q: torch.Tensor, w: torch.Tensor, cand: torch.Tensor, ends: torch.Tensor,
                         keys_all: torch.Tensor, tok2req: torch.Tensor, block_table: torch.Tensor,
                         storage_block_size: int) -> torch.Tensor:
    """``candidate_topk`` contract: q [Q, 32, 128] fp16, w [Q, 32], cand [Q, 2048] int32 (ascending, -1 pad; CAND_ALL
    rows allowed = blocks 0..2047), ends [Q], keys_all [rows, 128] fp16, tok2req [Q] (request of each row),
    block_table [R, W] -> [Q, 512] int32 positions ascending, -1 padding."""
    ext = load_ext()
    Q = q.shape[0]
    out = torch.empty((Q, IDX_TOPK), dtype=torch.int32, device=q.device)
    if Q == 0:
        return out
    q = _q_ok(q)
    w = w.float().contiguous()
    cand = cand if (cand.dtype == torch.int32 and cand.stride(-1) == 1) else cand.to(torch.int32).contiguous()
    ends = ends.to(torch.int64).contiguous()
    tok2req = tok2req.to(torch.int32).contiguous()
    bt = block_table if (block_table.dtype == torch.int32 and block_table.stride(-1) == 1) else \
        block_table.to(torch.int32).contiguous()
    keys_all = keys_all.contiguous()
    rows = min(Q, ROWS_PER_CALL)
    scores = torch.empty((rows, CAND_SLOTS), dtype=torch.float32, device=q.device)
    for a in range(0, Q, rows):
        b = min(Q, a + rows)
        s = scores[:b - a]
        ext.cand_scores(q[a:b], w[a:b], cand[a:b], ends[a:b], keys_all, tok2req[a:b], bt, int(storage_block_size),
                        s, SLOTS_PER_CTA)
        ext.topk512(s, CAND_SLOTS, cand[a:b], out[a:b])
    return out


def fused_index_scores(q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``index_scores`` contract: out[:, :n] = sum_h relu(q . k) * w (FP32)."""
    ext = load_ext()
    n = keys.shape[0]
    ext.dense_scores(_q_ok(q), weights.float().contiguous(), keys.contiguous(), out, DENSE_KEYS_PER_CTA)
    return out[:, :n]


def topk_v2_enabled() -> bool:
    return fused_enabled() and knobs.env_bool(TOPK_V2_ENV, False)


def fused_topk_sorted_v2(scores: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """``topk_sorted`` with the causal bound in-kernel: scores [Q, n] fp32 (-inf where otherwise masked), ends [Q]
    -> top-512 of each row's first ``end`` columns, ascending, -1 padded."""
    ext = load_ext()
    if int(getattr(ext, "version", 1)) < 2:
        raise ImportError(f"{TOPK_V2_ENV}=1 needs ds41_indexer_fused v2 (topk_sel); the .so at {so_path()} is "
                          f"v{ext.version} -- rebuild with scripts/build_ds41_indexer_fused.py")
    Q, n = scores.shape
    out = torch.empty((Q, IDX_TOPK), dtype=torch.int32, device=scores.device)
    if Q:
        s = scores if scores.stride(-1) == 1 else scores.contiguous()
        ext.topk_sel(s, n, ends.to(torch.int64).contiguous(), 1, None, out, IDX_TOPK, 0)
    return out


def topk_sel_emulate(scores: torch.Tensor, k: int, lens: torch.Tensor | None = None,
                     pos: torch.Tensor | None = None) -> torch.Tensor:
    """topk_sel's rule in torch: the k largest finite of row[:len], ties -> lowest column, ascending (pos[col] or
    col), -1 pad."""
    Q, L = scores.shape
    out = torch.full((Q, k), -1, dtype=torch.int32)
    for t in range(Q):
        n = L if lens is None else min(L, int(lens[t]))
        s = scores[t, :n]
        fin = (s != float("-inf")).nonzero().squeeze(1).tolist()
        order = sorted(fin, key=lambda j: (-float(s[j]), j))[:k]
        order.sort()
        p = torch.tensor(order, dtype=torch.int64)
        if pos is not None and p.numel():
            p = pos[t][p]
        out[t, :p.numel()] = p.to(torch.int32)
    return out


def fused_topk_sorted(scores: torch.Tensor, ends: torch.Tensor, k: int = IDX_TOPK) -> torch.Tensor:
    """``topk_sorted`` for k = 512 when the scores are -inf wherever masked (see module doc)."""
    if k != IDX_TOPK:
        raise ValueError(f"fused top-k is fixed at {IDX_TOPK}")
    ext = load_ext()
    Q, n = scores.shape
    out = torch.empty((Q, IDX_TOPK), dtype=torch.int32, device=scores.device)
    if Q:
        s = scores if scores.stride(-1) == 1 else scores.contiguous()
        ext.topk512(s, n, None, out)
    return out


# ----------------------------------------------------------------------------- torch model (CPU tests)
def topk512_emulate(scores: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
    """The kernel's selection rule in torch: the 512 largest finite scores, ties at the threshold to the lowest
    column, output = their positions ascending (pos[col] or col), -1 padding. Implemented as the kernel does it
    (order keys, radix select 4 x 8 bits, ordered compaction) so the CPU test checks the algorithm."""
    Q, L = scores.shape
    sc = torch.where(scores == 0, torch.zeros_like(scores), scores)          # -0 == +0
    b = sc.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = torch.where(b >= 0x80000000, (~b) & 0xFFFFFFFF, b | 0x80000000)
    key = torch.where(scores == float("-inf"), torch.zeros_like(key), key)
    out = torch.full((Q, IDX_TOPK), -1, dtype=torch.int32)
    for t in range(Q):
        kr = key[t]
        thr, need = 0, 0
        if L > IDX_TOPK:
            prefix, mask, k = 0, 0, IDX_TOPK
            for shift in (24, 16, 8, 0):
                m = (kr & mask) == prefix
                hist = torch.bincount(((kr[m] >> shift) & 255), minlength=256)
                cum, bb = 0, 255
                while bb > 0:
                    if cum + int(hist[bb]) >= k:
                        break
                    cum += int(hist[bb]); bb -= 1
                prefix |= bb << shift
                k -= cum
                mask |= 0xFF << shift
            thr, need = prefix, (0 if prefix == 0 else k)
        gt = kr > thr
        eq = (kr == thr) & (thr != 0)
        eq_rank = torch.cumsum(eq.to(torch.int64), 0) - eq.to(torch.int64)
        sel = gt | (eq & (eq_rank < need))
        cols = sel.nonzero().squeeze(1)[:IDX_TOPK]
        p = cols if pos is None else pos[t][cols]
        out[t, :cols.numel()] = p.to(torch.int32)
    return out


def cand_positions(cand: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Slot -> position map of the candidate kernel: [Q, 16384] int64, -1 for invalid slots."""
    blocks = torch.arange(CAND_TOPK_BLOCKS, dtype=cand.dtype, device=cand.device)[None, :]
    c = torch.where(cand[:, :1] == CAND_ALL, blocks, cand).to(torch.int64)
    pos = (c[:, :, None] * CAND_BLOCK + torch.arange(CAND_BLOCK, device=cand.device)[None, None, :]).view(
        cand.shape[0], -1)
    ok = (c >= 0)[:, :, None].expand(-1, -1, CAND_BLOCK).reshape(cand.shape[0], -1) & (pos < ends[:, None])
    return torch.where(ok, pos, torch.full_like(pos, -1))


def fused_candidate_emulate(q, w, cand, ends, keys_all, rows_of) -> torch.Tensor:
    """Torch model of fused_candidate_topk (FP32 math; the kernel's head sum is in-order RN, no FMA)."""
    pos = cand_positions(cand, ends)
    rows = rows_of(pos)
    k = keys_all.float().index_select(0, rows.clamp(min=0).view(-1)).view(pos.shape[0], -1, keys_all.shape[1])
    lg = torch.einsum("thd,tkd->thk", q.float(), k)
    s = torch.zeros(pos.shape, dtype=torch.float32)
    for h in range(IDX_HEADS):
        s = s + lg[:, h].relu() * w.float()[:, h:h + 1]
    s = s.masked_fill(pos < 0, float("-inf"))
    return topk512_emulate(s, pos)


# ----------------------------------------------------------------------------- decode (VLLM_DS41_INDEXER_FUSED_DECODE)
# Graph-capturable small-T path (T <= DECODE_MAX_T rows per step; per-layer attention graphs capture T <= 16): no host
# decision, no allocation after init (static buffers per device), so a reindex layer's candidate scoring AND an index
# source's full scan (layers 2 / 8 / 14 / 20) run inside the decode graphs.
#   reindex : cand_scores (2048..256 slots per CTA by T, ~160 CTAs) -> topk_sel<512> over the 16,384 slots
#   source  : block_max (all visible keys via the block table, per 8-block max, newest block pinned +inf) ->
#             topk_sel<2048> over the row's ceil(end / 8) blocks (= layer 20's candidate blocks: ascending, -1 pad,
#             CAND_ALL when <= 2048 blocks) -> cand_scores over those blocks -> topk_sel<512>. The top-512 positions
#             lie in the top-512 blocks by max, so this equals the full scan's top-512 (ties: lowest position).
FUSED_DECODE_ENV = "VLLM_DS41_INDEXER_FUSED_DECODE"
DECODE_MAX_T = 24                  # steps up to the largest decode capture size
DECODE_ROWS = 16                   # rows per candidate-scoring call (slot scores: 16 x 16K fp32 = 1 MiB)
DECODE_SCAN_ROWS_ENV = "VLLM_DS41_INDEXER_FUSED_DECODE_SCAN_ROWS"
DECODE_SCAN_ROWS = knobs.env_int(DECODE_SCAN_ROWS_ENV, 4, minimum=1, maximum=DECODE_ROWS)   # block-max rows per call
DECODE_KEYS_PER_CTA = 4096
_DEC_BUFS: dict = {}
# Memory (gate 20261025T042540Z): 16 block-max rows at 1M = 8 MiB static per GPU, allocated at model init on both
# stages -> counted as model memory -> the 1M KV pool no longer fit (1,047,296 < 1,048,576). Now: block maxima for
# DECODE_SCAN_ROWS (4) rows at a time (2 MiB at 1M; the kernels loop), and every buffer is allocated LAZILY at the first
# decode call outside any capture -- i.e. after KV sizing, from the VLLM_DS41_ATTN_PRECAPTURE reserve (160 MiB, measured
# growth +32..58 MiB on live), never from the KV budget. Per GPU: blk 2.0 + slot scores 1.0 + blocks 0.125 = 3.1 MiB.


def fused_decode_enabled() -> bool:
    return knobs.env_bool(FUSED_DECODE_ENV, False)


def decode_slots_per_cta(T: int) -> int:
    """Candidate slots per CTA for T rows: ~160 CTAs (2 per SM) -> 16,384 / splits, splits in {8 .. 64}."""
    splits = max(8, min(64, 160 // max(T, 1)))
    p = 8
    while p * 2 <= splits:
        p *= 2
    return CAND_SLOTS // p


def decode_buffers(device: torch.device, max_n: int) -> dict:
    """Static per-device decode scratch, allocated LAZILY at the first use outside a capture (the per-layer attention
    graphs warm up eagerly before capturing, so the first call is eager; a first use under capture raises instead of
    pinning the scratch inside one graph's pool): block maxima [DECODE_SCAN_ROWS, ceil(max_n / 8)] fp32, slot scores
    [16, 16384] fp32, candidate blocks [16, 2048] int32. One set per device for the process lifetime (graphs replay
    with these addresses)."""
    key = (str(device), int(max_n))
    bufs = _DEC_BUFS.get(key)
    if bufs is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{FUSED_DECODE_ENV}=1: the decode scratch was first needed inside a CUDA-graph capture; "
                               "it must be allocated by an eager (warm-up) call first")
        nb = (int(max_n) + CAND_BLOCK - 1) // CAND_BLOCK
        bufs = {"blk": torch.empty((DECODE_SCAN_ROWS, nb), dtype=torch.float32, device=device),
                "scores": torch.empty((DECODE_ROWS, CAND_SLOTS), dtype=torch.float32, device=device),
                "cand": torch.empty((DECODE_ROWS, CAND_TOPK_BLOCKS), dtype=torch.int32, device=device),
                "max_n": int(max_n)}
        _DEC_BUFS[key] = bufs
        from vllm.logger import init_logger

        init_logger(__name__).info("DS41 fused decode scratch on %s: %.2f MiB allocated lazily (block maxima %d rows x "
                                   "%d blocks, slot scores %d x %d, blocks %d x %d)", device,
                                   decode_buffer_bytes() / 2**20, DECODE_SCAN_ROWS, nb, DECODE_ROWS, CAND_SLOTS,
                                   DECODE_ROWS, CAND_TOPK_BLOCKS)
    return bufs


def decode_buffer_bytes() -> int:
    return sum(t.numel() * t.element_size() for b in _DEC_BUFS.values() for t in b.values()
               if isinstance(t, torch.Tensor))


def _dec_ext():
    ext = load_ext()
    if int(getattr(ext, "version", 1)) < 2:
        raise ImportError(f"{FUSED_DECODE_ENV}=1 needs ds41_indexer_fused v2 (block_max / topk_sel); the .so at "
                          f"{so_path()} is v{ext.version} -- rebuild with scripts/build_ds41_indexer_fused.py")
    return ext


def _dec_inputs(q, w, ends, tok2req, bt):
    return (_q_ok(q), w if (w.dtype == torch.float32 and w.is_contiguous()) else w.float().contiguous(),
            ends if (ends.dtype == torch.int64 and ends.is_contiguous()) else ends.to(torch.int64).contiguous(),
            tok2req if (tok2req.dtype == torch.int32 and tok2req.is_contiguous()) else
            tok2req.to(torch.int32).contiguous(),
            bt if (bt.dtype == torch.int32 and bt.stride(-1) == 1) else bt.to(torch.int32).contiguous())


def fused_decode_candidates(q, w, cand, ends, keys_all, tok2req, block_table, storage_block_size, out, bufs) -> None:
    """Reindex decode: out [T, 512] int32 <- top-512 over each row's candidate slots (CAND_ALL in-kernel)."""
    ext = _dec_ext()
    T = q.shape[0]
    q, w, ends, tok2req, bt = _dec_inputs(q, w, ends, tok2req, block_table)
    cand = cand if (cand.dtype == torch.int32 and cand.stride(-1) == 1) else cand.to(torch.int32).contiguous()
    spc = decode_slots_per_cta(T)
    for a in range(0, T, DECODE_ROWS):
        b = min(T, a + DECODE_ROWS)
        s = bufs["scores"][:b - a]
        ext.cand_scores(q[a:b], w[a:b], cand[a:b], ends[a:b], keys_all, tok2req[a:b], bt, int(storage_block_size),
                        s, spc)
        ext.topk_sel(s, CAND_SLOTS, None, 0, cand[a:b], out[a:b], IDX_TOPK, 0)


def fused_decode_scan(q, w, ends, keys_all, tok2req, block_table, storage_block_size, out, cand_out, bufs) -> None:
    """Index-source decode (full scan): out [T, 512] int32 <- top-512 over each row's visible positions; cand_out
    [T, 2048] int32 (layer 20) <- its candidate blocks, or None (layers 2 / 8 / 14: scratch)."""
    ext = _dec_ext()
    T = q.shape[0]
    q, w, ends, tok2req, bt = _dec_inputs(q, w, ends, tok2req, block_table)
    blk, max_n = bufs["blk"], bufs["max_n"]
    R = blk.shape[0]
    spc = decode_slots_per_cta(T)
    for a in range(0, T, DECODE_ROWS):
        b = min(T, a + DECODE_ROWS)
        c = cand_out[a:b] if cand_out is not None else bufs["cand"][:b - a]
        for a2 in range(a, b, R):                        # block maxima R rows at a time (the scratch is R x n/8)
            b2 = min(b, a2 + R)
            ext.block_max(q[a2:b2], w[a2:b2], ends[a2:b2], keys_all, tok2req[a2:b2], bt, int(storage_block_size),
                          blk, max_n, DECODE_KEYS_PER_CTA)
            ext.topk_sel(blk[:b2 - a2], blk.shape[1], ends[a2:b2], CAND_BLOCK, None, c[a2 - a:b2 - a],
                         CAND_TOPK_BLOCKS, CAND_TOPK_BLOCKS)
        s = bufs["scores"][:b - a]
        ext.cand_scores(q[a:b], w[a:b], c, ends[a:b], keys_all, tok2req[a:b], bt, int(storage_block_size), s, spc)
        ext.topk_sel(s, CAND_SLOTS, None, 0, c, out[a:b], IDX_TOPK, 0)


# ---- torch models of the decode kernels (CPU tests)
def fused_decode_scan_emulate(q, w, ends, keys_all, rows_of):
    """Torch model of fused_decode_scan -> (top-512 [T, 512], candidate blocks [T, 2048])."""
    T = q.shape[0]
    n = int(ends.max())
    j = torch.arange(n)
    rows = rows_of(j[None, :].expand(T, -1))
    k = keys_all.float()[rows.clamp(min=0)]                                    # [T, n, 128]
    lg = torch.einsum("thd,tnd->thn", q.float(), k)
    s = torch.zeros((T, n), dtype=torch.float32)
    for h in range(IDX_HEADS):
        s = s + lg[:, h].relu() * w.float()[:, h:h + 1]
    s = s.masked_fill(j[None, :] >= ends[:, None], float("-inf"))
    nb = (n + CAND_BLOCK - 1) // CAND_BLOCK
    blk = torch.nn.functional.pad(s, (0, nb * CAND_BLOCK - n), value=float("-inf")).view(T, nb, CAND_BLOCK).amax(-1)
    reach = (ends + CAND_BLOCK - 1) // CAND_BLOCK
    for t in range(T):
        if int(ends[t]) > 0:
            blk[t, (int(ends[t]) - 1) // CAND_BLOCK] = float("inf")
    cand = topk_sel_emulate(blk, CAND_TOPK_BLOCKS, reach)
    cand[reach <= CAND_TOPK_BLOCKS, 0] = CAND_ALL
    pos = cand_positions(cand, ends)
    slot = torch.where(pos >= 0, s.gather(1, pos.clamp(min=0, max=n - 1)), torch.full(pos.shape, float("-inf")))
    return topk_sel_emulate(slot, IDX_TOPK, None, pos), cand
