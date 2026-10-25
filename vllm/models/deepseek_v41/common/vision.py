# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 vision tower for SM70: ViT + aligner with host-resident weights (reference inference/vision.py).

The tower (32 blocks, dim 1024, 2D RoPE, full bidirectional attention over one image; aligner = 3x3 unfold ->
w1 -> GELU -> w2 into the 5120 hidden) is ~0.49B parameters. The V100 stages have no room for it next to a 1M-token
KV pool, so its weights stay in pinned host memory as FP16 and stream through two ~26 MB device buffers per encode
(piece i + 1 copies on a side stream while piece i computes). Only TP rank 0 of the first PP stage encodes; the
span embeddings are broadcast to the other TP ranks.

Numerics: FP16 GEMM operands, FP32 outputs (biases, norms, RoPE and the residual stream in FP32). Inputs that are not
bounded by a norm (the MLP's SwiGLU product, the aligner's GELU output) are scaled by a power of two when their amax
exceeds FP16 range (exact). The reference runs the whole tower in BF16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_PIECE_MAX_NUMEL = 13 << 20          # FP16 elements per streamed piece (26 MiB)
_ROW_CHUNK = 2048                    # token rows per GEMM chunk inside a block (bounds the FP32 temporaries)
_ATTN_SCORE_BYTES = 128 << 20         # per query chunk: SDPA takes the math path on SM70 (full score matrix)
_FP16_SAFE = 32768.0                 # GEMM inputs above this amax are scaled by 2^-k first
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, ...]:
    """ref:vision.py get_vision_cos_sin: [N, 1, dim] FP32."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ref:vision.py apply_rotary on FP32 [N, H, D]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def mm16(x16: torch.Tensor, w16: torch.Tensor) -> torch.Tensor:
    """x16 [M, K] @ w16 [N, K]^T -> FP32 [M, N]: FP16 operands, FP32 accumulate and output."""
    if x16.is_cuda:
        return torch.mm(x16, w16.t(), out_dtype=torch.float32)
    return x16.float() @ w16.float().t()


def mm16_scaled(x: torch.Tensor, w16: torch.Tensor) -> torch.Tensor:
    """FP32 x of unbounded range: divide by 2^k (k >= 0, on device, no host sync) so it fits FP16, multiply back."""
    k = torch.clamp(torch.ceil(torch.log2(x.abs().amax().clamp_min(1.0) / _FP16_SAFE)), min=0.0)
    scale = torch.exp2(k)
    return mm16((x / scale).half(), w16).mul_(scale)


@dataclass
class _Piece:
    """One streamed slice of FP16 weights: ``names`` -> (offset, shape) inside a flat host buffer."""
    host: torch.Tensor
    layout: dict[str, tuple[int, tuple[int, ...]]]


class _Streamer:
    """Double-buffered host->device copies of the pieces, in order, on a side stream."""

    def __init__(self, pieces: list[_Piece], device: torch.device) -> None:
        self.pieces = pieces
        self.device = device
        n = max(p.host.numel() for p in pieces)
        self.bufs = [torch.empty(n, dtype=torch.float16, device=device) for _ in range(2)]
        self.copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.ready = [None, None]
        self.free = [None, None]

    def _issue(self, i: int) -> None:
        slot, piece = i % 2, self.pieces[i]
        dst = self.bufs[slot][: piece.host.numel()]
        if self.copy_stream is None:
            dst.copy_(piece.host)
            return
        with torch.cuda.stream(self.copy_stream):
            if self.free[slot] is not None:
                self.copy_stream.wait_event(self.free[slot])     # compute finished reading this slot
            dst.copy_(piece.host, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
            self.ready[slot] = ev

    def __iter__(self):
        if self.pieces:
            self._issue(0)
        for i, piece in enumerate(self.pieces):
            if i + 1 < len(self.pieces):
                self._issue(i + 1)
            slot = i % 2
            if self.copy_stream is not None:
                torch.cuda.current_stream(self.device).wait_event(self.ready[slot])
            flat = self.bufs[slot]
            yield {name: flat[off: off + _numel(shape)].view(shape) for name, (off, shape) in piece.layout.items()}
            if self.copy_stream is not None:
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(self.device))
                self.free[slot] = ev


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


class DeepseekV41VisionTower:
    """Host-resident ViT + aligner. ``load(name, tensor)`` takes the checkpoint's ``vision.*`` / ``aligner.*`` /
    ``image_*`` tensors; ``finalize()`` packs them; ``encode(patches, n_vit_h, n_vit_w)`` -> aligner rows
    [n_llm_h * n_llm_w, hidden] FP32; ``span(patches, n_vit_h, n_vit_w, types)`` -> the image's whole span."""

    def __init__(self, config) -> None:
        self.dim = int(config.vision_dim)
        self.n_heads = int(config.vision_n_heads)
        self.head_dim = self.dim // self.n_heads
        self.inter = int(config.vision_inter_dim)
        self.n_layers = int(config.vision_n_layers)
        self.patch = int(config.vision_patch_size)
        self.ratio = int(config.vision_downsample_ratio)
        self.rope_dim = self.head_dim // 2
        self.rope_theta = float(config.vision_rope_theta)
        self.hidden = int(config.hidden_size)
        self._raw: dict[str, torch.Tensor] = {}
        self.small: dict[str, torch.Tensor] = {}       # FP32 biases / norms, host until first encode
        self.pieces: list[_Piece] = []
        self._dev_small: dict[str, torch.Tensor] | None = None
        self._dev_key: torch.device | None = None

    # ---- loading ----
    def expected_names(self) -> set[str]:
        names = {"vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", "vision.norm.weight",
                 "aligner.w1.weight", "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias",
                 "image_start", "image_end", "image_newline"}
        for i in range(self.n_layers):
            b = f"vision.blocks.{i}."
            names |= {b + s for s in ("norm1.weight", "norm2.weight", "attn.wqkv.weight", "attn.wqkv.bias",
                                      "attn.wo.weight", "attn.wo.bias", "mlp.w1.weight", "mlp.w2.weight")}
        return names

    def load(self, name: str, tensor: torch.Tensor) -> None:
        if name not in self.expected_names():
            raise KeyError(f"unexpected vision tensor {name!r}")
        self._raw[name] = tensor.detach().to("cpu", copy=True)

    def finalize(self) -> None:
        missing = sorted(self.expected_names() - set(self._raw))
        if missing:
            raise RuntimeError(f"vision tower: {len(missing)} checkpoint tensor(s) missing, e.g. {missing[:4]}")
        pin = torch.cuda.is_available()
        raw = self._raw
        for name, t in raw.items():
            if name.endswith(".bias") or "norm" in name or name.startswith("image_"):
                self.small[name] = t.float().contiguous()

        def piece(entries: list[tuple[str, torch.Tensor]]) -> _Piece:
            total = sum(t.numel() for _, t in entries)
            host = torch.empty(total, dtype=torch.float16, pin_memory=pin)
            layout, off = {}, 0
            for key, t in entries:
                t32 = t.float()
                if not bool(torch.isfinite(t32).all()) or float(t32.abs().max()) >= 65504.0:
                    raise ValueError(f"vision weight {key} does not fit FP16")
                host[off: off + t.numel()].copy_(t32.reshape(-1).half())
                layout[key] = (off, tuple(t.shape))
                off += t.numel()
            return _Piece(host, layout)

        self.pieces.append(piece([("patch", raw["vision.patch_embed.proj.weight"])]))
        for i in range(self.n_layers):
            b = f"vision.blocks.{i}."
            self.pieces.append(piece([("wqkv", raw[b + "attn.wqkv.weight"]), ("wo", raw[b + "attn.wo.weight"]),
                                      ("w1", raw[b + "mlp.w1.weight"]), ("w2", raw[b + "mlp.w2.weight"])]))
        for name in ("aligner.w1.weight", "aligner.w2.weight"):   # split by output rows to stay under a piece
            w = raw[name]
            rows = max(1, _PIECE_MAX_NUMEL // w.shape[1])
            for r0 in range(0, w.shape[0], rows):
                self.pieces.append(piece([(f"{name}:{r0}", w[r0: r0 + rows])]))
        self._raw = {}

    def host_bytes(self) -> int:
        return sum(p.host.numel() * 2 for p in self.pieces) + sum(t.numel() * 4 for t in self.small.values())

    def _small_on(self, device: torch.device) -> dict[str, torch.Tensor]:
        if self._dev_small is None or self._dev_key != device:
            self._dev_small = {k: v.to(device) for k, v in self.small.items()}
            self._dev_key = device
        return self._dev_small

    # ---- forward ----
    @torch.inference_mode()
    def encode(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """patches [N, 3, p, p] (N = n_vit_h * n_vit_w) -> aligner rows [ceil(h/3) * ceil(w/3), hidden] FP32."""
        device = patches.device
        small = self._small_on(device)
        n = n_vit_h * n_vit_w
        if patches.shape[0] != n:
            raise ValueError(f"vision: {patches.shape[0]} patches for a {n_vit_h}x{n_vit_w} grid")
        cos, sin = vision_cos_sin(n_vit_h, n_vit_w, self.rope_dim, self.rope_theta, device)
        pieces = iter(_Streamer(self.pieces, device))
        w = next(pieces)
        x = mm16(patches.reshape(n, -1).half(), w["patch"]).add_(small["vision.patch_embed.proj.bias"])
        H, D = self.n_heads, self.head_dim
        for i in range(self.n_layers):
            w = next(pieces)
            b = f"vision.blocks.{i}."
            q = torch.empty((n, H, D), dtype=torch.float16, device=device)
            k = torch.empty_like(q)
            v = torch.empty_like(q)
            bqkv = small[b + "attn.wqkv.bias"]
            for r0 in range(0, n, _ROW_CHUNK):
                r1 = min(n, r0 + _ROW_CHUNK)
                h = rmsnorm(x[r0:r1], small[b + "norm1.weight"]).half()
                qkv = mm16(h, w["wqkv"]).add_(bqkv)
                qc, kc, vc = (t.view(r1 - r0, H, D) for t in qkv.chunk(3, dim=-1))
                q[r0:r1] = apply_rotary(qc, cos[r0:r1], sin[r0:r1]).half()
                k[r0:r1] = apply_rotary(kc, cos[r0:r1], sin[r0:r1]).half()
                v[r0:r1] = vc.half()
            qt, kt, vt = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            o = torch.empty((n, H, D), dtype=torch.float16, device=device)
            qc = max(64, _ATTN_SCORE_BYTES // (H * n * 6))   # FP16 scores + FP32 softmax temporaries
            for r0 in range(0, n, qc):
                r1 = min(n, r0 + qc)
                o[r0:r1] = F.scaled_dot_product_attention(qt[:, r0:r1], kt, vt).transpose(0, 1)
            del q, k, v, qt, kt, vt
            o = o.reshape(n, self.dim)
            for r0 in range(0, n, _ROW_CHUNK):
                r1 = min(n, r0 + _ROW_CHUNK)
                x[r0:r1] += mm16(o[r0:r1].contiguous(), w["wo"]).add_(small[b + "attn.wo.bias"])
            del o
            for r0 in range(0, n, _ROW_CHUNK):
                r1 = min(n, r0 + _ROW_CHUNK)
                h = rmsnorm(x[r0:r1], small[b + "norm2.weight"]).half()
                gate, up = mm16(h, w["w1"]).chunk(2, dim=-1)
                x[r0:r1] += mm16_scaled(F.silu(gate) * up, w["w2"])
        x = rmsnorm(x, small["vision.norm.weight"])
        # aligner (ref:vision.py Aligner): [h, w, C] -> pad to a multiple of r -> r x r unfold -> [L, C * r * r]
        r = self.ratio
        g = x.view(n_vit_h, n_vit_w, -1).permute(2, 0, 1)
        g = F.pad(g, (0, -n_vit_w % r, 0, -n_vit_h % r))
        g = F.unfold(g.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1).contiguous()
        del x
        g16 = g.half()
        hid = torch.empty((g.shape[0], self.hidden), dtype=torch.float32, device=device)
        out = torch.empty_like(hid)
        for stage, src, dst in (("aligner.w1", g16, hid), ("aligner.w2", None, out)):
            if src is None:
                hid = F.gelu(hid.add_(small["aligner.w1.bias"]))
                k = torch.clamp(torch.ceil(torch.log2(hid.abs().amax().clamp_min(1.0) / _FP16_SAFE)), min=0.0)
                scale = torch.exp2(k)
                src = (hid / scale).half()
            else:
                scale = None
            r0 = 0
            while r0 < dst.shape[1]:
                w = next(pieces)
                ((key, wt),) = w.items()
                if not key.startswith(stage + ".weight:"):
                    raise RuntimeError(f"vision: streamed piece {key!r} out of order (expected {stage})")
                rows = wt.shape[0]
                part = mm16(src, wt)
                dst[:, r0: r0 + rows] = part if scale is None else part.mul_(scale)
                r0 += rows
        return out.add_(small["aligner.w2.bias"])

    def span(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int, types: torch.Tensor) -> torch.Tensor:
        small = self._small_on(patches.device)
        return build_image_span(self.encode(patches, n_vit_h, n_vit_w), types, small["image_start"],
                                small["image_end"], small["image_newline"])


def build_image_span(embeds: torch.Tensor, types: torch.Tensor, image_start: torch.Tensor, image_end: torch.Tensor,
                     image_newline: torch.Tensor) -> torch.Tensor:
    """[span, hidden]: IMAGE slots take the aligner rows in reading order, the delimiters the learned vectors
    (ref:model.py merge_image_embeddings)."""
    types = types.to(embeds.device)
    span = torch.empty((types.numel(), embeds.shape[1]), dtype=embeds.dtype, device=embeds.device)
    is_image = types == IMAGE
    if int(is_image.sum()) != embeds.shape[0]:
        raise ValueError(f"vision: {embeds.shape[0]} aligner rows for {int(is_image.sum())} IMAGE slots")
    span[is_image] = embeds
    span[types == IMAGE_START] = image_start.to(embeds.dtype)
    span[types == IMAGE_END] = image_end.to(embeds.dtype)
    span[types == IMAGE_NEW_LINE] = image_newline.to(embeds.dtype)
    return span
