# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 layer-sliced prefill, model/worker half (VLLM_DS41_SLICED_PREFILL=1; hol-design.md §2a + Appendix A).

While a request decodes, the scheduler stages a prefill chunk and emits ``SliceDirective`` s: each engine step a
stage runs layers [layer_start, layer_end) of its staged chunk AFTER the step's decode/verify forward and its PP isend
(the decode batch keeps its CUDA graphs; slices are eager). This module holds the per-stage state and runs the
directives:

* ``SlicedPrefillState`` (one slot per stage): allocated at load_model (before the memory profile), sized so the KV
  pool keeps 1M tokens on the binding stage (stage 1):
  - stage 0: private stream [T_max,4,5120] bf16 / pre / positions / input ids and a private ``topk_indices``
    (``candidate_blocks`` is never touched by layers < 20, so the model's own buffer is shared).
  - later stages: the staged stream/pre LIVE IN the runner's persistent PP receive buffers (``intermediate_tensors``)
    and the chunk uses the model's own SharedAttnBuffers. No private copies.
  - "low rows": a decode/verify step between two slices writes only rows [0, T_dec) of those per-step buffers (PP
    receive copy, shared top-k / candidate rows, the SWA builders' window_slots), T_dec <= the largest decode capture
    size. ``pre_step`` (top of the worker step, before the decode forward) saves rows [0, R0) of every such tensor the
    held chunk uses (R0 = VLLM_DS41_SLICE_SAVE_ROWS, default 32) and ``run_step`` restores them before the next slice. A
    main batch larger than R0 while a chunk is held evacuates the whole extents into a transient copy (warning).
  - every other metadata tensor (~8 KiB each: some builder buffers are rewritten whole every build, e.g.
    get_compressed_slot_mapping fills its out buffer with -1) is copied into a small metadata arena.
* ``first``: demand-KV commit of the chunk's blocks, chunk-only attention metadata built once through the runner's
  own builders (num_reqs = 1, the directive's block table) and cloned into the arena, Engram ``begin_slice`` on the
  stage that owns Engram layers. Stage 0 embeds the chunk's ids; later stages wait the handoff irecv.
* every slice: ``set_forward_context(chunk metadata, NONE)`` + ``DeepseekV41Model.forward_slice``.
* handoff (stage k ``last`` -> stage k+1): one static p2p message after the decode payload, in directive order:
  stream, pre, positions (+ kv20 ckv/ik/cand(/crc) when the stage exports source 20). Stage k+1 posts the matching
  irecv when it meets stage k's ``last`` directive in the same step (both stages see every directive), into a free
  slot; its own ``first`` (same or a later step) waits that receive.
* last stage ``last``: (req_id, chunk_id) -> ModelRunnerOutput.ds41_slice_done.
* aborts (SchedulerOutput.ds41_slice_aborts): drain pending p2p, end the Engram slice, free the slot.

VLLM_DS41_SLICE_VERIFY=1 (debug, GPU gate): at ``last`` re-run the whole chunk unsliced on this stage from a copy of its
initial state and compare the chunk's KV blocks + final stream/pre bit for bit; mismatches log
``DS41 slice verify MISMATCH ...``.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1 import ds41_slices
from vllm.v1.worker.ds41_slice_feedback import SliceFeedback

logger = init_logger(__name__)

SLICED_KNOB = "VLLM_DS41_SLICED_PREFILL"
VERIFY_KNOB = "VLLM_DS41_SLICE_VERIFY"
SLOTS_KNOB = "VLLM_DS41_SLICE_S1_SLOTS"              # slots on non-first stages (> 1: extra slots own a 40 MiB stream)
ARENA_KNOB = "VLLM_DS41_SLICE_MD_ARENA_MIB"          # device arena for one chunk's small metadata tensors
SAVE_ROWS_KNOB = "VLLM_DS41_SLICE_SAVE_ROWS"         # R0: low rows saved across a decode step (>= largest decode T)
SAVE_KNOB = "VLLM_DS41_SLICE_SAVE_MIB"               # device buffer for the saved low rows
DEFAULT_ARENA_MIB = 3
DEFAULT_SAVE_ROWS = 32
DEFAULT_SAVE_MIB = 3
TIMING_KNOB = "VLLM_DS41_SLICE_TIMING_EVERY"         # log CUDA-event slice timings every N slices (TP rank 0), 0 = off
SEGMENT_KNOB = "VLLM_DS41_SLICE_SEGMENT_TIMING"      # log per-segment CUDA-event timings every N slices, 0 = off
VERIFY_LOG_EVERY = 16                                # "verify ok" summary line every N verified chunks

HANDOFF_POSITIONS_KEY = "ds41_slice_positions"

# graphs of the attention / builder kernels bind decode-sized batches only (attention.GRAPH_MAX_T): a staged chunk
# must be larger so its eager forward never keys or lazily captures a decode graph
MIN_SLICE_TOKENS = 17


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    return int(raw) if raw.strip() else default


def sliced_prefill_enabled() -> bool:
    from vllm.v1.ds41_slices import sliced_prefill_enabled as _enabled   # the knob parser the scheduler uses
    return _enabled()


def _h2d(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host -> device without a stream sync (pinned staging from the caching host allocator, which keeps the block
    alive until the async copy is done); the slice's work is queued behind the decode forward, so a pageable copy's
    implicit sync would stall the CPU for the whole decode step."""
    if device.type != "cuda":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


# ------------------------------------------------------------------------------------------------ device arena
class DeviceArena:
    """Bump allocator over one byte buffer allocated at init (``reset()`` per use). Exceeding it raises a clean error
    naming ``knob``: these buffers are part of the profiled budget, a silent fallback would allocate outside it."""

    ALIGN = 256

    def __init__(self, nbytes: int, device: torch.device, knob: str = ARENA_KNOB, what: str = "metadata arena") -> None:
        self.buf = torch.empty((max(int(nbytes), self.ALIGN),), dtype=torch.uint8, device=device)
        self.off = 0
        self.high = 0
        self.knob, self.what = knob, what

    @property
    def nbytes(self) -> int:
        return self.buf.numel()

    def reset(self) -> None:
        self.off = 0

    def owns(self, t: torch.Tensor) -> bool:
        if t.device != self.buf.device or t.numel() == 0:
            return False
        lo = self.buf.data_ptr()
        p = t.data_ptr()
        return lo <= p < lo + self.buf.numel()

    def take(self, t: torch.Tensor) -> torch.Tensor:
        """A contiguous arena copy of ``t`` (same shape/dtype), issued on the current stream."""
        n = t.numel() * t.element_size()
        start = (self.off + self.ALIGN - 1) // self.ALIGN * self.ALIGN
        if n == 0:
            return t.new_empty(t.shape)
        if t.device != self.buf.device:
            raise RuntimeError(f"DS41 sliced prefill {self.what}: tensor on {t.device}, arena on {self.buf.device}")
        if start + n > self.buf.numel():
            raise RuntimeError(
                f"DS41 sliced prefill: {self.what} of {self.buf.numel() / 2**20:.2f} MiB exceeded (needs > "
                f"{(start + n) / 2**20:.2f} MiB so far); raise {self.knob} (it is allocated before the KV profile)")
        self.off = start + n
        self.high = max(self.high, self.off)
        out = self.buf[start:start + n].view(t.dtype).view(t.shape)
        out.copy_(t)
        return out


LOW_ROW_FIELDS = ("window_slots",)   # builder buffers written only in rows [0, T) per build (rows_of_positions grid)


def clone_metadata(md: Any, arena: DeviceArena, memo: dict[int, tuple[Any, Any]], lowrows: list | None = None) -> Any:
    """Copy of one attention-metadata object: device tensors copied into the arena (tensors already there are kept),
    numpy arrays copied; with ``lowrows`` given, LOW_ROW_FIELDS stay in their builder buffer and are appended to it
    (saved/restored per step instead of copied). Dataclasses only (the DS41 builders return dataclasses)."""
    # memo values hold the SOURCE object too: builders return a fresh metadata object per KV group, and once the caller
    # drops it its id() can be reused by the next group's object; without the reference that hit would hand the SWA
    # layers the compressed group's clone (2026-10-24 GPU gate: DS41CompressedMetadata has no window_slots)
    key = id(md)
    hit = memo.get(key)
    if hit is not None and hit[0] is md:
        return hit[1]
    if not dataclasses.is_dataclass(md):
        raise TypeError(f"sliced prefill: cannot clone attention metadata of type {type(md).__name__}")
    fields = {}
    for f in dataclasses.fields(md):
        if not f.init:
            continue
        v = getattr(md, f.name)
        if isinstance(v, torch.Tensor):
            if v.device == arena.buf.device:
                if arena.owns(v):
                    pass
                elif lowrows is not None and f.name in LOW_ROW_FIELDS:
                    lowrows.append(v)
                else:
                    v = arena.take(v)
            else:
                v = v.clone()                 # host-side tensors (e.g. *_cpu) on a GPU arena
        elif isinstance(v, np.ndarray):
            v = v.copy()
        fields[f.name] = v
    out = dataclasses.replace(md, **fields)
    memo[key] = (md, out)
    return out


# ------------------------------------------------------------------------------------------------ per-slot state
class SlicedPrefillState:
    """One staged chunk's persisted state on this stage (``DeepseekV41Model.forward_slice`` reads/writes it).

    own_stream: allocate stream/pre (else bound to the runner's PP receive buffers by the executor). own_shared:
    a full private SharedAttnBuffers (verify reference); else ``private_topk`` decides whether topk_indices is
    private, every other field is the model's own buffer."""

    def __init__(self, slot_id: int, max_tokens: int, device: torch.device, stage: Any, *, model_shared: Any,
                 own_stream: bool, own_shared: bool, private_topk: bool, has_ids: bool, arena_bytes: int,
                 save_bytes: int) -> None:
        from vllm.models.deepseek_v41.common import contracts as C

        self.slot_id = slot_id
        self.max_tokens = max_tokens
        self.device = device
        self.own_stream = own_stream
        self.stream: torch.Tensor | None = None
        self.pre: torch.Tensor | None = None
        if own_stream:
            self.stream = torch.zeros((max_tokens, C.HC, C.HIDDEN), dtype=C.STREAM_DTYPE, device=device)
            self.pre = torch.zeros((max_tokens, C.HC), dtype=C.MIX_DTYPE, device=device)
        self.positions = torch.zeros((max_tokens,), dtype=torch.int64, device=device)
        self.input_ids_dev = torch.zeros((max_tokens,), dtype=torch.int32, device=device) if has_ids else None
        self._owned: list[torch.Tensor] = []
        if own_shared:
            self.shared = C.allocate_shared_attn_buffers(max_tokens, stage, device)
            self._owned += [self.shared.topk_indices, self.shared.candidate_blocks]
            self._owned += [t for t in (self.shared.export_ckv, self.shared.export_ik) if t is not None]
        else:
            topk = model_shared.topk_indices
            if private_topk:
                topk = torch.full_like(topk, -1)
                self._owned.append(topk)
            self.shared = C.SharedAttnBuffers(topk_indices=topk, candidate_blocks=model_shared.candidate_blocks,
                                              export_ckv=model_shared.export_ckv, export_ik=model_shared.export_ik)
        self.private_topk = private_topk or own_shared
        self.arena = DeviceArena(arena_bytes, device)
        self.save = DeviceArena(save_bytes, device, SAVE_KNOB, "low-row save buffer") if save_bytes > 0 else None
        # receive buffers for the kv-source-20 records (stages mirroring 20 only; not under PP2 20/20)
        self.kv20: dict[str, torch.Tensor] = {}
        if C.CAND_SOURCE in stage.mirrored_kv_sources:
            self.kv20 = {
                C.PP_KEY_KV20_CKV: torch.zeros((max_tokens, C.CKV_PACKED_DIM), dtype=C.CKV_RECORD_DTYPE, device=device),
                C.PP_KEY_KV20_IK: torch.zeros((max_tokens, C.IK_RECORD_DIM), dtype=C.KV_RECORD_DTYPE, device=device),
                C.PP_KEY_CAND20: torch.zeros((max_tokens, C.CAND_TOPK_BLOCKS), dtype=torch.int32, device=device),
            }
        self.verify_ref: SlicedPrefillState | None = None   # VLLM_DS41_SLICE_VERIFY scratch (set by the executor)
        self.verify_block_ids: tuple | None = None
        self.recv_handles: list = []
        self.send_handles: list = []
        self.lowrows: list[torch.Tensor] = []
        self.saved: list = []
        self.clear()

    def nbytes(self) -> int:
        ts = [self.positions, self.arena.buf, *self._owned, *self.kv20.values()]
        ts += [t for t in (self.input_ids_dev, self.save.buf if self.save is not None else None) if t is not None]
        if self.own_stream:
            ts += [self.stream, self.pre]
        return sum(t.numel() * t.element_size() for t in ts)

    def register_lowrows(self, ts: list[torch.Tensor]) -> None:
        seen = {(t.data_ptr(), tuple(t.shape)) for t in self.lowrows}
        for t in ts:
            if t is None or t.dim() == 0 or t.numel() == 0:
                continue
            k = (t.data_ptr(), tuple(t.shape))
            if k not in seen:
                seen.add(k)
                self.lowrows.append(t)

    def lowrow_bytes(self, rows: int) -> int:
        a = DeviceArena.ALIGN
        return sum((min(rows, t.shape[0]) * (t.numel() // t.shape[0]) * t.element_size() + a - 1) // a * a
                   for t in self.lowrows)

    def clear(self) -> None:
        """Drop the chunk (buffers stay allocated). Pending send handles are kept: the next ``first`` waits them."""
        self.busy = False
        self.req_id: str | None = None
        self.chunk_id = -1
        self.start_pos = 0
        self.num_tokens = 0
        self.rows = 0
        self.n_req = 0
        self.md: dict[str, Any] | None = None
        self.md_active: dict[str, Any] | None = None
        self.needs_init = False
        self.input_ids: torch.Tensor | None = None
        self.handoff: Any = None
        self.replay_done = False
        self.replay_skip = False
        self.engram_handle: int | None = None
        self.engram_service: Any = None
        self.next_layer = -1
        self.attn_part = 0            # sub-layer slices: next attention part of layer next_layer
        self.attn_parts = 0
        self.attn_done = False        # the attention half of next_layer ran: its MoE part comes next
        self.started = False          # first directive ran (metadata built)
        self.verify_block_ids = None
        self.lowrows = []
        self.saved = []

    @property
    def key(self) -> tuple[str | None, int]:
        return (self.req_id, self.chunk_id)


# ------------------------------------------------------------------------------------------------ handoff schema
def handoff_schema(model: Any, num_tokens: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Ordered keys/shapes/dtypes of the staged-chunk handoff from ``model``'s stage to the next one: stream, pre,
    positions, then the kv-source-20 records when this boundary carries them (same order on both sides)."""
    from vllm.models.deepseek_v41.common import contracts as C

    base = model._schema(num_tokens, False)
    out = {C.PP_KEY_HIDDEN: base[C.PP_KEY_HIDDEN], C.PP_KEY_PRE_MIX: base[C.PP_KEY_PRE_MIX],
           HANDOFF_POSITIONS_KEY: ((num_tokens,), torch.int64)}
    return out


def _with_kv20(model: Any, num_tokens: int, schema: dict, with_kv20: bool) -> dict:
    if not with_kv20:
        return schema
    full = model._schema(num_tokens, True)
    out = dict(schema)
    for k, v in full.items():
        if k not in out:
            out[k] = v
    return out


def _with_feedback(schema: dict) -> dict:
    """VLLM_DS41_SLICE_POLICY=adaptive: stage k's slice / decode cost samples ride the handoff (last key, both sides
    read the same environment)."""
    if not ds41_slices.adaptive_feedback_enabled():
        return schema
    return {**schema, ds41_slices.SLICE_FB_KEY: ((ds41_slices.SLICE_FB_LEN,), torch.float32)}


def send_schema(model: Any, num_tokens: int) -> dict:
    from vllm.models.deepseek_v41.common import contracts as C
    return _with_feedback(_with_kv20(model, num_tokens, handoff_schema(model, num_tokens),
                                     C.CAND_SOURCE in model.stage.exports_kv_sources))


def recv_schema(model: Any, num_tokens: int) -> dict:
    from vllm.models.deepseek_v41.common import contracts as C
    return _with_feedback(_with_kv20(model, num_tokens, handoff_schema(model, num_tokens),
                                     C.CAND_SOURCE in model.stage.mirrored_kv_sources))


def attach_feedback(output: Any, slice_ms: Sequence[tuple], decode_ms: Sequence[tuple]) -> Any:
    """``output`` with ds41_slice_ms / ds41_decode_ms extended (never mutates the shared EMPTY constant)."""
    if output is None or not (slice_ms or decode_ms):
        return output
    upd = {"ds41_slice_ms": tuple(getattr(output, "ds41_slice_ms", ()) or ()) + tuple(slice_ms),
           "ds41_decode_ms": tuple(getattr(output, "ds41_decode_ms", ()) or ()) + tuple(decode_ms)}
    if dataclasses.is_dataclass(output) and all(any(f.name == k for f in dataclasses.fields(output)) for k in upd):
        return dataclasses.replace(output, **upd)
    out = copy.copy(output)
    for k, v in upd.items():
        setattr(out, k, v)
    return out


# ------------------------------------------------------------------------------------------------ chunk metadata
def chunk_slot_mapping(block_ids: Sequence[int], block_size: int, start_pos: int, num_tokens: int) -> np.ndarray:
    """[T] int64 rows written by positions start..start+T-1 (BlockTable.compute_slot_mapping for one request)."""
    pos = np.arange(start_pos, start_pos + num_tokens, dtype=np.int64)
    bt = np.asarray(block_ids, dtype=np.int64)
    need = int((start_pos + num_tokens - 1) // block_size) + 1
    if bt.shape[0] < need:
        raise ValueError(f"sliced prefill: {bt.shape[0]} blocks of {block_size} cover < {start_pos + num_tokens} tokens")
    return bt[pos // block_size] * block_size + pos % block_size


def _group_block_size(runner: Any, gid: int) -> int:
    try:
        return int(runner.input_batch.block_table[gid].block_size)
    except Exception:  # noqa: BLE001 - tests / runners without an input batch
        return int(runner.kv_cache_config.kv_cache_groups[gid].kv_cache_spec.block_size)


def _ds41_builder(builder: Any) -> bool:
    from vllm.models.deepseek_v41.sm70 import sparse
    return (isinstance(builder, sparse._DS41BuilderBase)
            and not isinstance(builder, sparse.DS41DraftSWAMetadataBuilder))


@contextlib.contextmanager
def _builders_eager():
    """The chunk's builds run their kernels eagerly: no builder-graph key / lazy capture for slice-owned inputs."""
    from vllm.models.deepseek_v41.sm70 import sparse
    saved = sparse.BUILDER_GRAPHS
    sparse.BUILDER_GRAPHS = False
    try:
        yield
    finally:
        sparse.BUILDER_GRAPHS = saved


def build_chunk_metadata(runner: Any, d: Any, arena: DeviceArena, positions: torch.Tensor,
                         lowrows: list | None = None,
                         is_builder: Callable[[Any], bool] = _ds41_builder) -> dict[str, Any]:
    """Chunk-only per-layer attention metadata (one request: [start_pos, start_pos + T)) built with the runner's own
    metadata builders, every device tensor owned by ``arena``. ``positions`` [T] int64 (device, slice-owned)."""
    from vllm.v1.attention.backend import CommonAttentionMetadata

    arena.reset()
    T, s = int(d.num_tokens), int(d.start_pos)
    seq = s + T
    groups = runner.kv_cache_config.kv_cache_groups
    if len(d.block_ids) != len(groups):
        raise ValueError(f"sliced prefill {d.req_id}/{d.chunk_id}: {len(d.block_ids)} block-id groups, runner has "
                         f"{len(groups)} KV-cache groups")
    dev = arena.buf.device
    qsl_cpu = torch.tensor([0, T], dtype=torch.int32)
    seq_cpu = torch.tensor([seq], dtype=torch.int32)
    qsl = arena.take(_h2d(qsl_cpu, dev))
    seq_lens = arena.take(_h2d(seq_cpu, dev))
    out: dict[str, Any] = {}
    memo: dict[int, tuple[Any, Any]] = {}
    with _builders_eager():
        for gid, group in enumerate(groups):
            bs = _group_block_size(runner, gid)
            ids = d.block_ids[gid]
            slots = chunk_slot_mapping(ids, bs, s, T)
            bt = arena.take(_h2d(torch.from_numpy(np.asarray(ids, dtype=np.int32).reshape(1, -1)), dev))
            sm = arena.take(_h2d(torch.from_numpy(slots), dev))
            cm = CommonAttentionMetadata(
                query_start_loc=qsl, query_start_loc_cpu=qsl_cpu, seq_lens=seq_lens, num_reqs=1,
                num_actual_tokens=T, max_query_len=T, max_seq_len=seq, block_table_tensor=bt, slot_mapping=sm,
                causal=True, positions=positions[:T], is_prefilling=torch.tensor([True]),
                seq_lens_cpu_upper_bound=seq_cpu, _seq_lens_cpu=seq_cpu,
                _num_computed_tokens_cpu=torch.tensor([s], dtype=torch.int32))
            for attn_group in runner.attn_groups[gid]:
                builder = attn_group.get_metadata_builder(0)
                if not is_builder(builder):
                    continue
                md = builder.build(common_prefix_len=0, common_attn_metadata=cm)
                md = clone_metadata(md, arena, memo, lowrows)
                for name in attn_group.layer_names:
                    out[name] = md
    return out


# ------------------------------------------------------------------------------------------------ output plumbing
def attach_slice_done(output: Any, done: Sequence[tuple[str, int]]) -> Any:
    """``output`` (a ModelRunnerOutput, possibly the shared EMPTY constant) with ``ds41_slice_done`` extended."""
    if not done or output is None:
        return output
    prev = tuple(getattr(output, "ds41_slice_done", ()) or ())
    value = prev + tuple((str(r), int(c)) for r, c in done)
    if dataclasses.is_dataclass(output) and any(f.name == "ds41_slice_done" for f in dataclasses.fields(output)):
        return dataclasses.replace(output, ds41_slice_done=value)
    out = copy.copy(output)                  # never mutate EMPTY_MODEL_RUNNER_OUTPUT
    out.ds41_slice_done = value              # pickled with the instance dict (multiproc executor)
    return out


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a, b = a.contiguous(), b.contiguous()
    return bool(torch.equal(a.view(-1).view(torch.uint8), b.view(-1).view(torch.uint8)))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    d = (a.float() - b.float()).abs()
    return float(torch.nan_to_num(d, nan=float("inf")).max()) if d.numel() else 0.0


def _layer_of(name: str) -> str:
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return parts[i + 1]
    return "?"


# ------------------------------------------------------------------------------------------------ segment timer
class SegmentTimer:
    """VLLM_DS41_SLICE_SEGMENT_TIMING=N: CUDA-event (CPU: host clock) timings of the named segments of every slice
    (attention: project / reproject / indexer (idx_proj, idx_gather, idx_scan, idx_select, idx_topk) / attend /
    oproj, plus "slice" = the whole forward_slice), aggregated per (stage, layers, part) and logged every N slices."""

    def __init__(self, device: torch.device, every: int, stage: int) -> None:
        self.device, self.every, self.stage = device, int(every), int(stage)
        self.cur: dict | None = None
        self.pending: list[dict] = []
        self.agg: dict[tuple, dict] = {}
        self.count = 0

    def begin(self, label: tuple, T: int, n_ctx: int) -> None:
        self.cur = {"label": label, "T": T, "n": n_ctx, "events": []}

    @contextlib.contextmanager
    def segment(self, name: str):
        if self.cur is None:
            yield
            return
        if self.device.type == "cuda":
            a = torch.cuda.Event(enable_timing=True)
            a.record()
            try:
                yield
            finally:
                b = torch.cuda.Event(enable_timing=True)
                b.record()
                self.cur["events"].append((name, a, b))
        else:
            import time
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.cur["events"].append((name, t0, time.perf_counter()))

    def end(self) -> None:
        if self.cur is not None:
            self.pending.append(self.cur)
        self.cur = None

    def drain(self) -> list[tuple[tuple, dict[str, float]]]:
        """Completed records -> [(label, {segment: ms})]; logs the aggregate every ``every`` records."""
        done, keep = [], []
        for rec in self.pending:
            evs = rec["events"]
            if evs and not isinstance(evs[-1][2], float) and not all(e[2].query() for e in evs):
                keep.append(rec)
                continue
            ms: dict[str, float] = {}
            for name, a, b in evs:
                ms[name] = ms.get(name, 0.0) + ((b - a) * 1000.0 if isinstance(a, float) else a.elapsed_time(b))
            done.append((rec["label"], ms))
            g = self.agg.setdefault(rec["label"], {"n": 0, "T": 0, "ctx": 0, "ms": {}})
            g["n"] += 1
            g["T"] += rec["T"]
            g["ctx"] += rec["n"]
            for k, v in ms.items():
                g["ms"][k] = g["ms"].get(k, 0.0) + v
            self.count += 1
        self.pending = keep
        if self.every and self.count >= self.every:
            for (a, b, part, pi, pc), g in sorted(self.agg.items()):
                n = g["n"]
                segs = " ".join(f"{k} {v / n:.2f}" for k, v in sorted(g["ms"].items(), key=lambda kv: -kv[1]))
                kind = {0: "layers", 1: f"attn part {pi}/{pc}", 2: "moe+layers"}.get(part, str(part))
                logger.info("DS41 slice segments stage %d [%d,%d) %s: %d slices, mean T %d n_ctx %d, ms: %s",
                            self.stage, a, b, kind, n, g["T"] // n, g["ctx"] // n, segs)
            self.agg, self.count = {}, 0
        return done


# ------------------------------------------------------------------------------------------------ executor
class SlicedPrefillExecutor:
    """Runs this stage's slice directives inside the worker step. ``model`` = DeepseekV41Model (has forward_slice)."""

    def __init__(self, runner: Any, model: Any, *, pp_rank: int, pp_size: int, max_tokens: int,
                 device: torch.device, comm: Any = None, vllm_config: Any = None,
                 build_metadata: Callable[..., dict] | None = None) -> None:
        self.runner = runner
        self.model = model
        self.pp_rank, self.pp_size = int(pp_rank), int(pp_size)
        self.is_first, self.is_last = self.pp_rank == 0, self.pp_rank == self.pp_size - 1
        self.max_tokens = int(max_tokens)
        self.device = device
        self.comm = comm                              # PP GroupCoordinator (isend/irecv_tensor_dict_static)
        self.vllm_config = vllm_config
        self.build_metadata = build_metadata or build_chunk_metadata
        self.verify = _env_bool(VERIFY_KNOB)
        n_slots = 1 if self.is_first else max(1, _env_int(SLOTS_KNOB, 1))
        arena = int(float(os.environ.get(ARENA_KNOB, "") or DEFAULT_ARENA_MIB) * 2**20)
        save = int(float(os.environ.get(SAVE_KNOB, "") or DEFAULT_SAVE_MIB) * 2**20)
        self.save_rows = _env_int(SAVE_ROWS_KNOB, DEFAULT_SAVE_ROWS)
        self._check_save_rows(vllm_config)
        self.stage_first = int(model.start_layer)
        self.stage_end = int(model.end_layer)
        self.engram_layers = tuple(getattr(model.stage, "engram_layers", ()) or ())
        from vllm.models.deepseek_v41.common import contracts as C
        # candidate rows are touched (written by layer 20 / the mirror, read by later reindex layers) only from 20 on
        self.uses_cand = int(model.stage.last_layer) >= C.CAND_SOURCE or bool(model.stage.mirrored_kv_sources)
        self.slots = [SlicedPrefillState(i, self.max_tokens, device, model.stage, model_shared=model.shared,
                                         own_stream=self.is_first or i > 0, own_shared=False,
                                         private_topk=self.is_first, has_ids=self.is_first, arena_bytes=arena,
                                         save_bytes=save)
                      for i in range(n_slots)]
        if self.verify:      # debug only: a full private reference state per slot (not part of the 1M budget)
            for st in self.slots:
                st.verify_ref = SlicedPrefillState(-1, self.max_tokens, device, model.stage, model_shared=model.shared,
                                                   own_stream=True, own_shared=True, private_topk=True,
                                                   has_ids=self.is_first, arena_bytes=0, save_bytes=0)
        self._evac_warned = 0
        self._save_high = 0
        self.on_staged: Callable[[str, int, np.ndarray], None] | None = None   # stage 0: Engram planner history
        self.pending_done: list[tuple[str, int]] = []
        self.stats = {"slices": 0, "chunks": 0, "aborts": 0, "verified": 0, "verify_fail": 0}
        self.timing_every = _env_int(TIMING_KNOB, 0)
        seg_every = _env_int(SEGMENT_KNOB, 0)
        self.segments = SegmentTimer(device, seg_every, self.pp_rank) if seg_every > 0 else None
        self._attn_acc: dict[tuple, float] = {}
        # sub-layer planner on: stage-1 slice samples carry their cost units (ds41_slices.COST_KIND_LAYERS)
        self.s1_costed = ds41_slices.slice_s1_attn_parts() != 0
        self._timing: list = []
        self._timing_ms: list[float] = []
        # adaptive slice sizing: cost samples (ds41_slice_feedback.py); stage 0 ships them in its handoff
        self.feedback = SliceFeedback(self.pp_rank) if ds41_slices.adaptive_feedback_enabled() else None
        self._fb_pending: list = []

    def _check_save_rows(self, vllm_config: Any) -> None:
        """R0 must cover every decode/verify step's padded token count (the largest CUDA-graph capture size)."""
        cc = getattr(vllm_config, "compilation_config", None)
        sizes = list(getattr(cc, "cudagraph_capture_sizes", None) or ())
        biggest = max(sizes) if sizes else int(getattr(cc, "max_cudagraph_capture_size", 0) or 0)
        if biggest > self.save_rows:
            raise ValueError(f"{SAVE_ROWS_KNOB}={self.save_rows} < the largest decode capture size {biggest}: a decode "
                             f"step between two slices would overwrite unsaved rows; raise {SAVE_ROWS_KNOB}")

    def _bind_stream(self, st: SlicedPrefillState, T: int) -> None:
        """Later stages: the slot's stream/pre are the runner's persistent PP receive buffers (decode rows use only
        their low rows while a chunk is held; those rows are saved/restored around each decode step)."""
        if st.own_stream:
            return
        from vllm.models.deepseek_v41.common import contracts as C
        buffers = getattr(self.runner, "intermediate_tensors", None)
        if buffers is None:
            raise RuntimeError("DS41 sliced prefill: the runner's persistent PP receive buffers do not exist yet")
        stream, pre = buffers[C.PP_KEY_HIDDEN], buffers[C.PP_KEY_PRE_MIX]
        if stream.shape[0] < T or tuple(stream.shape[1:]) != (C.HC, C.HIDDEN) or pre.shape[0] < T:
            raise RuntimeError(f"DS41 sliced prefill: PP receive buffers {tuple(stream.shape)} / {tuple(pre.shape)} "
                               f"cannot hold a {T}-token chunk")
        st.stream, st.pre = stream, pre

    def _register_chunk(self, st: SlicedPrefillState, md_lowrows: list[torch.Tensor]) -> None:
        T = st.num_tokens
        ts = list(md_lowrows)
        if not st.own_stream:
            ts += [st.stream[:T], st.pre[:T]]
        if not st.private_topk:
            ts.append(st.shared.topk_indices[:T])
            if self.uses_cand:
                ts.append(st.shared.candidate_blocks[:T])
            ts += [t[:T] for t in (st.shared.export_ckv, st.shared.export_ik) if t is not None]
        st.register_lowrows(ts)
        need = st.lowrow_bytes(self.save_rows)
        if st.save is None or need > st.save.nbytes:
            raise RuntimeError(
                f"DS41 sliced prefill: stage {self.pp_rank} needs {need / 2**20:.2f} MiB to save {self.save_rows} low "
                f"rows of {len(st.lowrows)} tensors, buffer {0 if st.save is None else st.save.nbytes / 2**20:.2f} MiB; "
                f"raise {SAVE_KNOB}")
        if need > self._save_high:
            self._save_high = need
            logger.info("DS41 sliced prefill: stage %d saves %d low rows of %d tensors = %.2f MiB per step (buffer "
                        "%.2f MiB); metadata arena %.2f of %.2f MiB", self.pp_rank, self.save_rows, len(st.lowrows),
                        need / 2**20, st.save.nbytes / 2**20, st.arena.high / 2**20, st.arena.nbytes / 2**20)

    def pre_step(self, scheduler_output: Any) -> None:
        """Top of the worker step, BEFORE the decode receive / forward: save the held chunk's low rows of every shared
        per-step buffer. A main batch larger than R0 evacuates the whole extents (transient copy, logged)."""
        n_main = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        for st in self.slots:
            if not st.busy:
                continue
            self._wait(st.recv_handles)       # the handoff lands before the decode copy reuses these rows
            if not st.lowrows or n_main == 0:
                continue
            evac = n_main > self.save_rows
            if evac:
                self._evac_warned += 1
                if self._evac_warned in (1, 10, 100) or self._evac_warned % 1000 == 0:
                    logger.warning("DS41 sliced prefill: stage %d main batch of %d tokens (> %s=%d) while chunk %s is "
                                   "held: evacuating %.1f MiB transiently (%d times); the scheduler should keep held "
                                   "steps decode-only", self.pp_rank, n_main, SAVE_ROWS_KNOB, self.save_rows, st.key,
                                   st.lowrow_bytes(self.max_tokens) / 2**20, self._evac_warned)
            st.save.reset()
            st.saved = []
            for t in st.lowrows:
                n = t.shape[0] if evac else min(self.save_rows, t.shape[0])
                src = t[:n]
                st.saved.append((t, src.clone() if evac else st.save.take(src), n))

    def _restore(self) -> None:
        for st in self.slots:
            for t, c, n in st.saved:
                t[:n].copy_(c)
            st.saved = []

    # ---- sizes ----
    def nbytes(self) -> int:
        n = sum(st.nbytes() for st in self.slots)
        if self.verify:
            n += sum(st.verify_ref.nbytes() for st in self.slots if st.verify_ref is not None)
        return n

    # ---- slot lookup ----
    def _slot_for(self, key: tuple[str, int]) -> SlicedPrefillState | None:
        for st in self.slots:
            if st.busy and st.key == key:
                return st
        return None

    def _free_slot(self) -> SlicedPrefillState:
        for st in self.slots:
            if not st.busy:
                return st
        raise RuntimeError(
            f"DS41 sliced prefill: stage {self.pp_rank} has no free slot ({len(self.slots)} slot(s), busy with "
            f"{[st.key for st in self.slots]}); the scheduler must not hand a chunk to a stage still holding one "
            f"(or set {SLOTS_KNOB}=2)")

    @staticmethod
    def _wait(handles: list) -> None:
        for h in handles:
            h.wait()
        handles.clear()

    # ---- the step ----
    def run_step(self, scheduler_output: Any) -> list[tuple[str, int]]:
        """Aborts, then every directive in order: this stage's slices, and on later stages the handoff irecv for the
        previous stage's ``last`` directives. Returns the chunks this (last) stage completed."""
        done: list[tuple[str, int]] = []
        self._drain_timing()
        self._restore()                       # the decode forward of this step is issued: put the held rows back
        for req_id, chunk_id in getattr(scheduler_output, "ds41_slice_aborts", ()) or ():
            self.abort(str(req_id), int(chunk_id))
        # handoff receives wait here (in order) until a slot is free: within the step nothing else crosses this
        # stage pair after the decode payload, so a receive may be posted after this stage's own earlier directives,
        # but never in a later step (the next step's decode payload would overtake it)
        pending: list = []
        for d in getattr(scheduler_output, "ds41_slices", ()) or ():
            if d.stage == self.pp_rank - 1 and d.last:
                pending.append(d)
            elif d.stage == self.pp_rank:
                if d.first and not self.is_first:
                    while pending:          # this chunk's handoff (and any before it) must be posted first
                        self._post_handoff_recv(pending.pop(0))
                if self._run_directive(d):
                    done.append((d.req_id, int(d.chunk_id)))
            else:
                continue
            while pending and any(not st.busy for st in self.slots):
                self._post_handoff_recv(pending.pop(0))
        if pending:
            raise RuntimeError(
                f"DS41 sliced prefill: stage {self.pp_rank} cannot receive the handoff of {[(d.req_id, d.chunk_id) for d in pending]} "
                f"this step: {len(self.slots)} slot(s) busy with {[st.key for st in self.slots]} (the scheduler must "
                f"finish this stage's chunk first, or set {SLOTS_KNOB}=2)")
        return done

    def _check_tokens(self, d: Any) -> None:
        T = int(d.num_tokens)
        if not MIN_SLICE_TOKENS <= T <= self.max_tokens:
            raise ValueError(f"DS41 sliced prefill {d.req_id}/{d.chunk_id}: chunk of {T} tokens outside "
                             f"[{MIN_SLICE_TOKENS}, {self.max_tokens}]")

    def _post_handoff_recv(self, d: Any) -> None:
        """Stage k+1 meets stage k's ``last`` directive: post the static irecv of the chunk's handoff into a slot."""
        self._check_tokens(d)
        key = (d.req_id, int(d.chunk_id))
        if self._slot_for(key) is not None:
            raise RuntimeError(f"DS41 sliced prefill: handoff {key} received twice")
        st = self._free_slot()
        self._wait(st.send_handles)
        st.clear()
        st.busy = True
        st.req_id, st.chunk_id = d.req_id, int(d.chunk_id)
        st.start_pos, st.num_tokens, st.rows = int(d.start_pos), int(d.num_tokens), int(d.num_tokens)
        T = st.num_tokens
        self._bind_stream(st, T)
        views = self._handoff_views(st, recv_schema(self.model, T), T)
        st.handoff = views
        if not st.own_stream:
            st.register_lowrows([st.stream[:T], st.pre[:T]])
        st.recv_handles = list(self.comm.irecv_tensor_dict_static(views)) if self.comm is not None else []

    def _handoff_views(self, st: SlicedPrefillState, schema: dict, T: int) -> dict[str, torch.Tensor]:
        from vllm.models.deepseek_v41.common import contracts as C
        src = {C.PP_KEY_HIDDEN: st.stream, C.PP_KEY_PRE_MIX: st.pre, HANDOFF_POSITIONS_KEY: st.positions, **st.kv20}
        views = {}
        for k, (shape, dtype) in schema.items():
            if k == ds41_slices.SLICE_FB_KEY:            # fixed-size, not per token
                views[k] = self._fb_buffer(st)
                continue
            if k not in src:
                raise RuntimeError(f"DS41 sliced prefill: no receive buffer for handoff key {k!r}")
            v = src[k][:T]
            if tuple(v.shape) != tuple(shape) or v.dtype != dtype:
                raise RuntimeError(f"DS41 sliced prefill: handoff {k} buffer {tuple(v.shape)} {v.dtype} != schema "
                                   f"{tuple(shape)} {dtype}")
            views[k] = v
        return views

    def _fb_buffer(self, st: SlicedPrefillState) -> torch.Tensor:
        fb = getattr(st, "fb", None)
        if fb is None:
            fb = st.fb = torch.zeros((ds41_slices.SLICE_FB_LEN,), dtype=torch.float32, device=self.device)
        return fb

    def _begin_chunk(self, d: Any) -> SlicedPrefillState:
        self._check_tokens(d)
        key = (d.req_id, int(d.chunk_id))
        T, s = int(d.num_tokens), int(d.start_pos)
        if self.is_first:
            if self._slot_for(key) is not None:
                raise RuntimeError(f"DS41 sliced prefill: chunk {key} started twice on stage 0")
            st = self._free_slot()
            self._wait(st.send_handles)       # the previous chunk's handoff still reads these buffers
            st.clear()
            st.busy = True
            st.req_id, st.chunk_id, st.start_pos, st.num_tokens, st.rows = d.req_id, int(d.chunk_id), s, T, T
            ids = np.asarray(d.token_ids, dtype=np.int32)
            if ids.shape != (T,):
                raise ValueError(f"DS41 sliced prefill {key}: {ids.shape[0]} token ids for a {T}-token chunk")
            st.input_ids_dev[:T].copy_(_h2d(torch.from_numpy(ids), self.device))
            st.positions[:T].copy_(_h2d(torch.arange(s, s + T, dtype=torch.int64), self.device))
            st.input_ids = st.input_ids_dev[:T]
            st.handoff = None
        else:
            st = self._slot_for(key)
            if st is None:
                raise RuntimeError(f"DS41 sliced prefill: stage {self.pp_rank} got 'first' for {key} without its "
                                   f"handoff (stage {self.pp_rank - 1}'s 'last' directive must come first)")
            if st.start_pos != s or st.num_tokens != T:
                raise RuntimeError(f"DS41 sliced prefill {key}: handoff was [{st.start_pos}+{st.num_tokens}), "
                                   f"directive [{s}+{T})")
            self._wait(st.recv_handles)       # the current stream waits for the handoff
            st.input_ids = None
            if self.feedback is not None and st.handoff is not None and ds41_slices.SLICE_FB_KEY in st.handoff:
                self._fb_receive(st.handoff[ds41_slices.SLICE_FB_KEY])
        st.n_req = int(d.n_req)
        st.needs_init = True
        st.next_layer = self.stage_first
        # KV pages of the chunk's blocks (demand KV), then the chunk-only metadata
        commit = getattr(self.runner, "_ds41_demand_kv_commit", None)
        if commit is not None:
            commit([list(g) for g in d.block_ids], None)
        md_lowrows: list[torch.Tensor] = []
        st.md = self.build_metadata(self.runner, d, st.arena, st.positions, md_lowrows)
        st.md_active = st.md
        self._register_chunk(st, md_lowrows)
        svc = getattr(self.model, "engram_service", None)
        if svc is not None and self.engram_layers:
            ids = np.asarray(d.token_ids, dtype=np.int32)
            st.engram_service = svc
            st.engram_handle = svc.begin_slice(d.req_id, s, T, num_tokens_padded=T, token_ids=ids)
        if self.on_staged is not None and len(d.token_ids):
            self.on_staged(d.req_id, s, np.asarray(d.token_ids, dtype=np.int32))
        if self.verify and st.verify_ref is not None:
            self._verify_snapshot(st)
            st.verify_block_ids = tuple(tuple(int(x) for x in g) for g in d.block_ids)
        st.started = True
        self.stats["chunks"] += 1
        return st

    def _forward_context(self, st: SlicedPrefillState):
        """Forward context of a staged slice (and of its SLICE_VERIFY re-run): the chunk metadata, no CUDA graphs, and
        ds41_slices.slice_forward() -- the indexer's scan applies its query-row tile cap only inside it, so a slice and
        its verify reference share the capped grid while normal-path prefill keeps today's grid."""
        stack = contextlib.ExitStack()
        stack.enter_context(ds41_slices.slice_forward())
        if self.vllm_config is None:
            return stack
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        stack.enter_context(set_forward_context(st.md_active, self.vllm_config, num_tokens=st.rows,
                                                cudagraph_runtime_mode=CUDAGraphMode.NONE))
        return stack

    def _run_directive(self, d: Any) -> bool:
        """One slice of this stage. True when it completed the chunk on the last stage (slice_done)."""
        key = (d.req_id, int(d.chunk_id))
        st = self._begin_chunk(d) if d.first else self._slot_for(key)
        if st is None or not st.started:
            raise RuntimeError(f"DS41 sliced prefill: stage {self.pp_rank} has no staged chunk {key} for "
                               f"[{d.layer_start},{d.layer_end})")
        a, b = int(d.layer_start), int(d.layer_end)
        part = int(getattr(d, "layer_part", 0) or 0)
        pi, pc = int(getattr(d, "part_index", 0) or 0), int(getattr(d, "part_count", 1) or 1)
        if a != st.next_layer or not a < b <= self.stage_end:
            raise RuntimeError(f"DS41 sliced prefill {key}: slice [{a},{b}) but the chunk is at layer {st.next_layer} "
                               f"(stage layers [{self.stage_first},{self.stage_end}))")
        self._check_part(key, st, part, pi, pc, a, b, bool(d.last))
        rows_before = st.rows
        tag = None
        if part == 1:
            if a == self.stage_first:      # the stage's first layer (20): full-context scan, work T * n_ctx / 1e6
                tag = ((key, a), pi == pc - 1, st.num_tokens * (st.start_pos + st.num_tokens) / 1e6,
                       ds41_slices.COST_KIND_SCAN)
            else:                          # a reindex layer's scoring on the kept rows; work = its prior ms
                tag = ((key, a), pi == pc - 1,
                       ds41_slices.reindex_attn_prior_ms(st.rows, st.start_pos + st.num_tokens),
                       ds41_slices.COST_KIND_REINDEX_ATTN)
        ev = self._timing_begin()
        seg = self.segments
        if seg is not None:
            from vllm.models.deepseek_v41 import attention as _attn_mod
            seg.begin((a, b, part, pi, pc), st.num_tokens, st.start_pos + st.num_tokens)
            _attn_mod.SEGMENT_TIMER = seg
        try:
            with self._forward_context(st), (seg.segment("slice") if seg is not None else contextlib.nullcontext()):
                self.model.forward_slice(st, a, b, part, pi, pc)
        finally:
            if seg is not None:
                _attn_mod.SEGMENT_TIMER = None
                seg.end()
        if tag is None and self.feedback is not None and not self.is_first and self.s1_costed:
            tag = self._layers_tag(st, part, a, b, rows_before)
        self._timing_end(ev, b - a, bool(d.first), tag)
        if part == 1:
            st.attn_part, st.attn_parts = pi + 1, pc
            st.attn_done = pi == pc - 1
        else:
            st.next_layer = b
            st.attn_part, st.attn_parts, st.attn_done = 0, 0, False
        self.stats["slices"] += 1
        if (part != 1 and st.engram_handle is not None and not self.verify and self.engram_layers
                and b > max(self.engram_layers)):
            st.engram_service.end_slice(st.engram_handle)
            st.engram_handle = None
        if not d.last:
            return False
        if b != self.stage_end:
            raise RuntimeError(f"DS41 sliced prefill {key}: 'last' slice ends at {b}, stage ends at {self.stage_end}")
        if self.verify and st.verify_ref is not None:
            self._verify_chunk(st)
        if st.engram_handle is not None:
            st.engram_service.end_slice(st.engram_handle)
            st.engram_handle = None
        if not self.is_last:
            self._send_handoff(st)
            st.clear()                         # send handles stay on the slot until its next chunk
            return False
        st.clear()
        return True

    def _layers_tag(self, st: SlicedPrefillState, part: int, a: int, b: int, rows_before: int) -> tuple:
        """Stage-1 whole-layer / MoE slice cost units for the adaptive policy: plain-layer work = (whole non-reindex
        layers + MOE_SHARE for a MoE half) x rows / 1024 (the MoE half of the first layer on all rows, layers past the
        replay cut on the kept rows; 0 rows after a replay skip), plus how many whole reindex layers it held."""
        rows_after = 0 if st.replay_skip else st.rows
        work, n_reindex = 0.0, 0
        for idx in range(a, b):
            rows = rows_before if idx == self.stage_first else rows_after
            if idx == a and part == 2:
                work += ds41_slices.MOE_SHARE * rows / 1024.0
            elif ds41_slices.is_reindex_layer(idx) and rows:
                n_reindex += 1
            else:
                work += rows / 1024.0
        return (None, True, (work, n_reindex), ds41_slices.COST_KIND_LAYERS)

    def _check_part(self, key, st: SlicedPrefillState, part: int, pi: int, pc: int, a: int, b: int,
                    last: bool) -> None:
        """Sub-layer slice order: attention parts 0..n-1 of layer a, then its MoE part; whole-layer slices only
        between layers."""
        where = f"DS41 sliced prefill {key}: slice [{a},{b}) part {part} ({pi}/{pc})"
        if part == 0:
            if st.attn_part or st.attn_done:
                raise RuntimeError(f"{where}: layer {a} is mid-way (attention part {st.attn_part}/{st.attn_parts}, "
                                   f"done={st.attn_done})")
        elif part == 1:
            if b != a + 1 or last or not 0 <= pi < pc or st.attn_done or pi != st.attn_part or (
                    pi and pc != st.attn_parts):
                raise RuntimeError(f"{where}: expected attention part {st.attn_part}"
                                   f"{'/' + str(st.attn_parts) if st.attn_parts else ''} of layer {a}")
        elif part == 2:
            if not st.attn_done:
                raise RuntimeError(f"{where}: the MoE part needs every attention part of layer {a} first")
        else:
            raise RuntimeError(f"{where}: unknown layer part")

    def _send_handoff(self, st: SlicedPrefillState) -> None:
        T = st.num_tokens
        schema = send_schema(self.model, T)
        payload = dict(self.model.slice_handoff(st))
        payload[HANDOFF_POSITIONS_KEY] = st.positions[:T]
        if ds41_slices.SLICE_FB_KEY in schema:
            fb = self._fb_buffer(st)
            if self.feedback is not None:
                self._drain_timing()
                fb.copy_(torch.tensor(self.feedback.pack(), dtype=torch.float32))
            payload[ds41_slices.SLICE_FB_KEY] = fb
        send = {}
        for k, (shape, dtype) in schema.items():
            t = payload[k]
            if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
                raise RuntimeError(f"DS41 sliced prefill handoff {k}: {tuple(t.shape)} {t.dtype} vs schema "
                                   f"{tuple(shape)} {dtype}")
            send[k] = t
        st.send_handles = list(self.comm.isend_tensor_dict_static(send)) if self.comm is not None else []

    def abort(self, req_id: str, chunk_id: int) -> None:
        for k in [k for k in self._attn_acc if k[0] == (req_id, chunk_id)]:
            self._attn_acc.pop(k, None)
        st = self._slot_for((req_id, chunk_id))
        if st is None:
            return
        # a posted irecv cannot be cancelled: the previous stage sent it (its 'last' directive was emitted)
        self._wait(st.recv_handles)
        if st.engram_handle is not None:
            st.engram_service.end_slice(st.engram_handle)
        st.clear()
        self.stats["aborts"] += 1

    # ---- VLLM_DS41_SLICE_VERIFY ----
    def _chunk_kv_views(self, st: SlicedPrefillState, d_block_ids: Sequence[Sequence[int]]) -> list:
        """[(cache layer name, cache tensor, block index tensor)] for every KV cache of this stage's layers, blocks
        covering [start_pos, start_pos + T)."""
        runner = self.runner
        ctx = runner.compilation_config.static_forward_context
        out = []
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            bs = _group_block_size(runner, gid)
            lo, hi = st.start_pos // bs, (st.start_pos + st.num_tokens - 1) // bs + 1
            ids = torch.tensor(list(d_block_ids[gid])[lo:hi], dtype=torch.long, device=self.device)
            for name in group.layer_names:
                layer = ctx.get(name)
                kv = getattr(layer, "kv_cache", None)
                if isinstance(kv, (list, tuple)):
                    kv = kv[0] if kv else None
                if not isinstance(kv, torch.Tensor) or kv.numel() == 0:
                    continue
                out.append((name, kv, ids))
        return out

    def _verify_snapshot(self, st: SlicedPrefillState) -> None:
        """At ``first`` (handoff received, before any slice): copy the chunk's stage inputs into the reference slot."""
        ref = st.verify_ref
        T = st.num_tokens
        ref.positions[:T].copy_(st.positions[:T])
        if self.is_first:
            ref.input_ids_dev[:T].copy_(st.input_ids_dev[:T])
        else:
            ref.stream[:T].copy_(st.stream[:T])
            ref.pre[:T].copy_(st.pre[:T])
            for k, t in st.kv20.items():
                ref.kv20[k][:T].copy_(t[:T])
        st.verify_block_ids = None

    def _verify_chunk(self, st: SlicedPrefillState, block_ids: Sequence[Sequence[int]] | None = None) -> bool:
        """Re-run the chunk unsliced on this stage (layers [stage_first, stage_end) in one forward_slice from the
        snapshot) and compare the chunk's KV blocks and final stream/pre/rows bit for bit with the sliced run."""
        block_ids = block_ids if block_ids is not None else st.verify_block_ids
        T = st.num_tokens
        views = self._chunk_kv_views(st, block_ids) if block_ids is not None else []
        sliced_kv = [(name, kv.index_select(0, ids).clone()) for name, kv, ids in views]
        n = st.rows
        sliced = {"stream": st.stream[:n].clone(), "pre": st.pre[:n].clone()}
        ref = st.verify_ref
        ref.clear()
        ref.busy = True
        ref.req_id, ref.chunk_id, ref.start_pos, ref.num_tokens, ref.rows = st.req_id, st.chunk_id, st.start_pos, T, T
        ref.n_req = st.n_req
        ref.md = ref.md_active = st.md
        ref.needs_init = True
        ref.engram_handle, ref.engram_service = st.engram_handle, st.engram_service
        if self.is_first:
            ref.input_ids = ref.input_ids_dev[:T]
        else:
            ref.handoff = self._handoff_views(ref, recv_schema(self.model, T), T)
        with self._forward_context(ref):
            self.model.forward_slice(ref, self.stage_first, self.stage_end)
        bad: list[str] = []
        if ref.rows != n:
            bad.append(f"rows sliced={n} unsliced={ref.rows}")
        for name, t in (("stream", ref.stream[:ref.rows]), ("pre", ref.pre[:ref.rows])):
            if ref.rows == n and not _bit_equal(sliced[name], t):
                bad.append(f"layer={self.stage_end - 1} tensor={name} max_abs={_max_abs(sliced[name], t):.6g}")
        for (name, kv, ids), (_, a) in zip(views, sliced_kv):
            b = kv.index_select(0, ids)
            if not _bit_equal(a, b):
                bad.append(f"layer={_layer_of(name)} tensor={name} max_abs={_max_abs(a, b):.6g}")
        ref.clear()
        self.stats["verified"] += 1
        if bad:
            self.stats["verify_fail"] += 1
            for line in bad:
                logger.error("DS41 slice verify MISMATCH stage=%d req=%s chunk=%d %s", self.pp_rank, st.req_id,
                             st.chunk_id, line)
            return False
        ok = self.stats["verified"] - self.stats["verify_fail"]
        if ok % VERIFY_LOG_EVERY == 1 or VERIFY_LOG_EVERY == 1:
            logger.info("DS41 slice verify ok: stage %d, %d chunks bit-identical (%d mismatched), last req=%s chunk=%d "
                        "(%d KV tensors + stream/pre)", self.pp_rank, ok, self.stats["verify_fail"], st.req_id,
                        st.chunk_id, len(views))
        return True

    # ---- timing ----
    def _timing_begin(self):
        if not self.timing_every and self.feedback is None and self.segments is None:
            return None
        if self.device.type != "cuda":
            import time
            return time.perf_counter() if self.feedback is not None else None   # CPU: eager, the host clock is exact
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _timing_end(self, ev, n_layers: int, first: bool = False, tag: tuple | None = None) -> None:
        """``tag`` = (chunk key, last attention part, T * n_ctx / 1e6) for attention parts: their times are summed per
        chunk and reported once as the layer's attention-half cost (layers -1) with its work units."""
        if ev is None:
            return
        if isinstance(ev, float):
            import time
            self._timing.append((ev, time.perf_counter(), n_layers, first, tag))
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._timing.append((ev, e, n_layers, first, tag))

    def _fb_receive(self, fb_dev: torch.Tensor) -> None:
        """Stage k+1: the previous stage's samples arrived with the handoff (the stream already waits on it); copy
        them to the host without synchronizing and unpack once the copy completed (``_drain_timing``)."""
        if fb_dev.device.type != "cuda":
            self.feedback.unpack(fb_dev.tolist(), stage=self.pp_rank - 1)
            return
        host = torch.empty(fb_dev.shape, dtype=fb_dev.dtype, pin_memory=True)
        host.copy_(fb_dev, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._fb_pending.append((host, ev))

    def _drain_timing(self) -> None:
        keep = []
        for a, b, n, first, tag in self._timing:
            if isinstance(a, float):
                ms = (b - a) * 1000.0
            elif b.query():
                ms = a.elapsed_time(b)
            else:
                keep.append((a, b, n, first, tag))
                continue
            if tag is not None and tag[3] == ds41_slices.COST_KIND_LAYERS:    # stage-1 whole layers / MoE part
                if self.timing_every:
                    self._timing_ms.append(ms / max(n, 1))
                if self.feedback is not None:
                    work, n_reindex = tag[2]
                    self.feedback.note_slice(n, ms, first, extra=(work, ds41_slices.COST_KIND_LAYERS, n_reindex))
                continue
            if tag is not None:          # attention part: one sample per (chunk, layer) (sum of its parts)
                key, last_part, work, kind = tag
                total = self._attn_acc.pop(key, 0.0) + ms
                if not last_part:
                    self._attn_acc[key] = total
                    continue
                if self.timing_every:
                    self._timing_ms.append(total)
                if self.feedback is not None:
                    self.feedback.note_attn(total, work, kind)
                continue
            if self.timing_every:
                self._timing_ms.append(ms / max(n, 1))
            if self.feedback is not None:
                self.feedback.note_slice(n, ms, first)
        self._timing = keep
        if self.segments is not None:
            self.segments.drain()
        if self._fb_pending:
            pending = []
            for host, ev in self._fb_pending:
                if ev.query():
                    self.feedback.unpack(host.tolist(), stage=self.pp_rank - 1)
                else:
                    pending.append((host, ev))
            self._fb_pending = pending
        if self.timing_every and len(self._timing_ms) >= self.timing_every:
            ms = np.array(self._timing_ms)
            logger.info("DS41 slice timing stage %d: %d slices, per-layer ms p50 %.2f p90 %.2f max %.2f",
                        self.pp_rank, ms.shape[0], np.percentile(ms, 50), np.percentile(ms, 90), ms.max())
            self._timing_ms = []

    def take_feedback(self) -> tuple[tuple, tuple]:
        """Last stage: every completed sample (own + the previous stage's) for the step's ModelRunnerOutput."""
        if self.feedback is None:
            return (), ()
        self._drain_timing()
        return self.feedback.take()


# ------------------------------------------------------------------------------------------------ construction
def create_executor(runner: Any) -> SlicedPrefillExecutor | None:
    """GPUModelRunner.load_model hook: the stage's executor with its buffers allocated (before the memory profile),
    or None when VLLM_DS41_SLICED_PREFILL is off / the model is not DeepSeek-V4.1."""
    if not sliced_prefill_enabled():
        return None
    model = runner.get_model()
    inner = getattr(model, "model", None)
    if inner is None or not callable(getattr(inner, "forward_slice", None)):
        logger.warning("%s=1 ignored: %s has no forward_slice (DeepSeek-V4.1 only)", SLICED_KNOB,
                       type(model).__name__)
        return None
    cfg = runner.vllm_config
    if cfg.scheduler_config.async_scheduling:
        raise NotImplementedError(f"{SLICED_KNOB}=1 needs synchronous scheduling (exact host token counts)")
    from vllm.models.deepseek_v41.common import dump as ds41_dump
    if ds41_dump.ENABLED:
        raise ValueError(f"{SLICED_KNOB}=1 cannot run with the golden dump ({ds41_dump.DIR_KNOB})")
    from vllm.distributed import get_pp_group

    pp = get_pp_group()
    free0 = torch.cuda.mem_get_info()[0] if runner.device.type == "cuda" else 0
    ex = SlicedPrefillExecutor(runner, inner, pp_rank=pp.rank_in_group, pp_size=pp.world_size,
                               max_tokens=cfg.scheduler_config.max_num_batched_tokens, device=runner.device,
                               comm=pp if pp.world_size > 1 else None, vllm_config=cfg)
    st = ex.slots[0]
    mib = lambda n: n / 2**20
    nb = lambda t: 0 if t is None else t.numel() * t.element_size()
    logger.info(
        "DS41 sliced prefill: stage %d/%d layers [%d,%d), %d slot(s): stream %s %.1f MiB, private topk %.1f, md arena "
        "%.1f, low-row save %.1f (R0=%d), positions/ids %.2f, kv20 rx %.1f%s => %.1f MiB per GPU (driver-free drop "
        "%.1f MiB); T_max %d",
        ex.pp_rank, ex.pp_size, ex.stage_first, ex.stage_end, len(ex.slots),
        "private" if st.own_stream else "= PP recv buffer", mib(nb(st.stream) if st.own_stream else 0),
        mib(nb(st.shared.topk_indices) if st.private_topk else 0), mib(st.arena.nbytes),
        mib(st.save.nbytes if st.save is not None else 0), ex.save_rows,
        mib(nb(st.positions) + nb(st.input_ids_dev)), mib(sum(nb(t) for t in st.kv20.values())),
        " + verify reference (debug)" if ex.verify else "", mib(ex.nbytes()),
        mib(free0 - torch.cuda.mem_get_info()[0]) if runner.device.type == "cuda" else 0.0, ex.max_tokens)
    return ex
