# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder replay / prefill row trim for layers 21-39 (``VLLM_DS41_REPLAY_ROWS``; default 0 = off).

Past the candidate source (layer 20) the only
per-token state layers 21-39 keep is each layer's SWA window (W = 128): compressed KV and index keys come from the kv
sources (2, 8, 14, 20), all computed for every row; the reindex layers 24/28/32/36 score layer 20's keys with their own
queries (``indexer.owns_k`` is False there) and keep nothing; HC, norms and MoE are per row. So layers 21-39 only need
the rows whose results something will read: per request ``keep(row) = position >= N_req - K``, N_req = the request's
total token count this step works toward (prompt + generated). Decode / spec-verify rows always satisfy it (K >= W).
With K = 3072 every layer's KV is exact from N_req - 786 and the logits rows are exact (see the design note).

Per step (host-only decision, no device sync):

* the runner binds ``(num_reqs, T, query_start_loc, N_req)`` right before the forward (``ReplayController.bind``);
* the model plans the rows from that binding + the step's DS41 metadata (exact host positions, synchronous scheduling)
  -> ``none`` (every row kept: today's path, untouched -- decode / verify steps, including their CUDA graphs),
  ``skip`` (no row kept: most chunks of a long prompt; layers 21-39 do not run) or ``compact`` (mixed);
* dropped rows get a ZERO record in the layer 21-39 SWA cache slots (design option b): a kept row near the cut reads
  window slots of rows that were never computed, and a recycled block can hold FP8-QAT garbage incl. NaN/Inf;
* ``compact``: gather stream/pre/positions to the kept rows after layer 20, swap in compacted attention metadata for
  layers 21-39 (and layer 20's compressed / index-K metadata that the reuse / reindex layers read), compact layer 20's
  published top-k and candidate blocks, run 21-39 on the compacted batch, scatter hidden + DSpark aux back to [T].

Rows nobody keeps return zeros: the runner samples them only for partial-prefill requests, whose token is discarded.
Requests with prompt logprobs are bound with N_req = 0 (all rows kept). Never under graph capture, dumps or dummy runs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import WINDOW

logger = init_logger(__name__)

REPLAY_ROWS_KNOB = "VLLM_DS41_REPLAY_ROWS"
# 1 = one log line per bound step on TP rank 0 (plan kind, T, kept rows, per request qlen/first position/N_req): step
# composition is timing-dependent under concurrency (arrival order, PP batch queue), so run-to-run differences in mixed
# gates must be read against it. With K >= every N_req the plan is always "none" = today's numerics, so
# VLLM_DS41_REPLAY_ROWS=1048576 + this trace is an OFF-equivalent run with the same composition log.
REPLAY_TRACE_KNOB = "VLLM_DS41_REPLAY_TRACE"
EXACT_DEFAULT_K = 3072          # 128 x 20 + 512 margin (design note); 128 = the approximate vLLM-style trim


def replay_rows_k() -> int:
    """K of the trim, 0 = off. 1..WINDOW-1 is refused: a decode row reads its window p-127..p, and with K < W the
    rows below N_req - K inside that window were dropped by the prompt's last chunk (never written)."""
    k = knobs.env_int(REPLAY_ROWS_KNOB, 0, minimum=0)
    if 0 < k < WINDOW:
        raise ValueError(f"{REPLAY_ROWS_KNOB}={k}: must be 0 (off) or >= the SWA window {WINDOW}; "
                         f"{EXACT_DEFAULT_K} is the exact setting")
    return k


# ------------------------------------------------------------------------------------------- host plan (numpy only)
@dataclass(frozen=True)
class RowPlan:
    kind: str                     # "none" | "skip" | "compact"
    num_tokens: int               # T (real rows; padding rows are never kept)
    kept: np.ndarray              # [Tk] int64 row ids, ascending (a suffix of every request's rows)
    dropped: np.ndarray           # [T - Tk] int64 row ids, ascending
    query_start_loc: np.ndarray   # [R+1] int64 of the compacted batch (requests keep their index; may be empty)


def plan_rows(query_start_loc: np.ndarray, positions: np.ndarray, n_req: np.ndarray, k: int) -> RowPlan:
    """Rows of a step that layers 21-39 must process: ``positions[t] >= n_req[req(t)] - k``.

    query_start_loc [R+1], positions [T] (exact host positions), n_req [R] (0 = keep every row of the request)."""
    qsl = np.asarray(query_start_loc, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.int64)
    n = np.asarray(n_req, dtype=np.int64)
    T = int(qsl[-1])
    if pos.shape[0] != T or n.shape[0] != qsl.shape[0] - 1:
        raise ValueError(f"replay plan: {pos.shape[0]} positions / {n.shape[0]} requests vs query_start_loc {qsl}")
    qlen = np.diff(qsl)
    cut = np.repeat(n - k, qlen)
    keep = pos >= cut
    kept = np.flatnonzero(keep).astype(np.int64)
    if kept.shape[0] == T:
        kind = "none"
    elif kept.shape[0] == 0:
        kind = "skip"
    else:
        kind = "compact"
    counts = np.add.reduceat(keep.astype(np.int64), qsl[:-1]) if T else np.zeros_like(qlen)
    counts = np.where(qlen > 0, counts, 0)          # reduceat reads the next segment for an empty request
    qsl_k = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return RowPlan(kind=kind, num_tokens=T, kept=kept, dropped=np.flatnonzero(~keep).astype(np.int64),
                   query_start_loc=qsl_k)


# ------------------------------------------------------------------------------------------- metadata compaction
@dataclass
class CompactRows:
    """Device side of a ``compact`` plan (one packed H2D per step)."""
    kept: torch.Tensor            # [Tk] int64
    dropped: torch.Tensor         # [Td] int64
    positions: torch.Tensor       # [Tk] int64
    tok2req: torch.Tensor         # [Tk] int32
    query_start_loc: torch.Tensor  # [R+1] int32
    kept_cpu: np.ndarray
    positions_cpu: np.ndarray
    query_start_loc_cpu: np.ndarray


def compact_metadata(md, rows: CompactRows):
    """A copy of one DS41 cache metadata for the kept rows. Per-request fields (block table, seq lens, num_reqs)
    stay; per-token fields are gathered. Per-latent fields are emptied: layers 21-39 own no compressor, and
    ``write_keys`` / the latent stores raise on a latent-count mismatch rather than write a wrong row."""
    from vllm.models.deepseek_v41.sm70.sparse import (
        DS41BatchMetadata,
        DS41CompressedMetadata,
        DS41StateMetadata,
        DS41SWAMetadata,
    )

    if not isinstance(md, DS41BatchMetadata):
        raise TypeError(f"replay: cannot compact {type(md).__name__} (layers 21-39 read only DS41 cache metadata)")
    if md.num_actual_tokens != int(rows.kept_cpu.shape[0] + rows.dropped.shape[0]):
        raise RuntimeError(f"replay: metadata has {md.num_actual_tokens} rows, plan {rows.kept_cpu.shape[0]} + "
                           f"{rows.dropped.shape[0]}")
    kept = rows.kept
    tk = int(rows.kept_cpu.shape[0])
    fields = dict(
        num_actual_tokens=tk, query_start_loc=rows.query_start_loc, query_start_loc_cpu=rows.query_start_loc_cpu,
        token_to_req_indices=rows.tok2req, positions=rows.positions, positions_cpu=rows.positions_cpu,
        slot_mapping=md.slot_mapping[: md.num_actual_tokens].index_select(0, kept), replay_compacted=True)
    if isinstance(md, DS41SWAMetadata):
        fields["window_slots"] = md.window_slots[: md.num_actual_tokens].index_select(0, kept)
    elif isinstance(md, DS41CompressedMetadata):
        r = int(md.compress_ratio)
        fields["num_visible"] = md.num_visible[: md.num_actual_tokens].index_select(0, kept)
        fields["max_visible"] = int(((rows.positions_cpu + 1) // r).max()) if tk else 0
        fields["num_latents"] = 0
        for name in ("latent_token_idx", "latent_pos", "latent_slots"):
            t = getattr(md, name)
            fields[name] = t[:0] if isinstance(t, torch.Tensor) else t
    elif isinstance(md, DS41StateMetadata):
        fields["prev_slot"] = md.prev_slot[: md.num_actual_tokens].index_select(0, kept)
    return dataclasses.replace(md, **fields)


def zero_window_rows(cache_rows: torch.Tensor, slots: torch.Tensor) -> None:
    """Write a zero record into ``cache_rows`` at ``slots``. Slot -1 (a token whose SWA block vLLM skipped) is
    redirected to row 0, which belongs to the null block (physical block 0, never handed to a request, always mapped
    under demand KV) -- a masked write without a host sync."""
    if slots.numel() == 0:
        return
    cache_rows.index_fill_(0, torch.where(slots >= 0, slots, torch.zeros_like(slots)), 0)


# ------------------------------------------------------------------------------------------- the per-stage controller
def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001 - no TP group (CPU unit tests)
        return 0


_LAST_DRAFT_ROWS: list = [None]   # (T, kept, dropped or None = all) of the latest real target forward, for DSpark


def drafter_rows() -> tuple[int, torch.Tensor, torch.Tensor | None] | None:
    """Kept / dropped target rows of the latest real target forward when it trimmed (``skip`` or ``compact``), else
    None. The DSpark context pass (combine_hidden_states + window KV store) of the same step applies it when its row
    count is that forward's T (row i of the context = target row i)."""
    return _LAST_DRAFT_ROWS[0]


class ReplayController:
    """Owned by DeepseekV41Model on the stage that holds the candidate source and is the last stage."""

    def __init__(self, k: int, first_layer: int, layers: list, max_tokens: int, device: torch.device) -> None:
        self.k = k
        self.first_layer = first_layer          # first trimmed layer (CAND_SOURCE + 1)
        self.layers = layers                    # [(layer_id, decoder layer)] of the trimmed range
        self.device = device
        names: list[str] = []
        for _, layer in layers:
            attn = layer.attn
            names.append(attn.swa_cache.layer_name)
            if attn.kv_source_name is not None:
                names.append(attn.kv_source_name)
            if attn.indexer is not None:
                names.append(attn.indexer.k_cache_name)
        self.md_names = tuple(dict.fromkeys(names))
        self.swa_names = tuple(layer.attn.swa_cache.layer_name for _, layer in layers)
        self._binding: tuple | None = None
        self._pinned = torch.empty((4 * max_tokens + 1024,), dtype=torch.int64, device="cpu",
                                   pin_memory=device.type == "cuda")   # model init runs under torch.device("cuda")
        self._h2d_done: torch.cuda.Event | None = None
        self._logged: set[str] = set()
        self.stats = {"none": 0, "skip": 0, "compact": 0, "rows_in": 0, "rows_kept": 0}
        self.trace = knobs.env_bool(REPLAY_TRACE_KNOB, False) and _tp_rank() == 0
        self._step = 0

    # ---- runner side ----
    def bind(self, num_reqs: int, num_tokens: int, query_start_loc: np.ndarray, n_req: np.ndarray) -> None:
        # a step whose forward replays a captured graph never reaches begin(): drop the previous step's drafter rows
        _LAST_DRAFT_ROWS[0] = None
        self._binding = (int(num_reqs), int(num_tokens), np.array(query_start_loc[: num_reqs + 1], dtype=np.int64),
                         np.array(n_req[:num_reqs], dtype=np.int64))

    def _once(self, key: str, msg: str, *args) -> None:
        if key not in self._logged:
            self._logged.add(key)
            logger.warning(msg, *args)

    # ---- model side ----
    def begin(self, md_all) -> RowPlan | None:
        """Plan this forward (consumes the binding). None = run every layer on every row (today's path)."""
        binding, self._binding = self._binding, None
        _LAST_DRAFT_ROWS[0] = None
        if binding is None or not isinstance(md_all, dict) or (
                self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            return None
        from vllm.forward_context import get_forward_context
        if get_forward_context().is_dummy_run:
            return None
        num_reqs, num_tokens, qsl, n_req = binding
        swa = md_all.get(self.swa_names[0])
        if swa is None:
            return None
        if swa.num_reqs != num_reqs or swa.num_actual_tokens != num_tokens or not np.array_equal(
                swa.query_start_loc_cpu[: num_reqs + 1], qsl):
            # a forward that is not the step the runner bound (should not happen; never trim on a mismatch)
            self._once("mismatch", "DS41 replay: forward does not match the bound step (R %d/%d, T %d/%d); not "
                       "trimming", swa.num_reqs, num_reqs, swa.num_actual_tokens, num_tokens)
            return None
        # every request's first scheduled position already clears its cut (decode / verify / last chunks): no row
        # arrays, today's path
        first_pos = np.asarray(swa.seq_lens_cpu[:num_reqs], dtype=np.int64) - np.diff(qsl)
        self._step += 1
        if bool((first_pos >= n_req - self.k).all()):
            self.stats["none"] += 1
            if self.trace:
                self._trace("none", num_tokens, num_tokens, qsl, first_pos, n_req)
            return None
        plan = plan_rows(qsl, swa.positions_cpu, n_req, self.k)
        if self.trace:
            self._trace(plan.kind, num_tokens, int(plan.kept.shape[0]), qsl, first_pos, n_req)
        self.stats[plan.kind] += 1
        self.stats["rows_in"] += plan.num_tokens
        self.stats["rows_kept"] += int(plan.kept.shape[0])
        if plan.kind != "none":
            n = self.stats["skip"] + self.stats["compact"]
            if n in (1, 10, 100) or n % 1000 == 0:
                logger.info("DS41 replay (K=%d): %d skip / %d compact / %d full steps; layers %d+ ran %d of %d rows",
                            self.k, self.stats["skip"], self.stats["compact"], self.stats["none"], self.first_layer,
                            self.stats["rows_kept"], self.stats["rows_in"])
        return None if plan.kind == "none" else plan

    def _trace(self, kind: str, T: int, kept: int, qsl: np.ndarray, first_pos: np.ndarray, n_req: np.ndarray) -> None:
        reqs = " ".join(f"{int(q)}@{int(p)}/{int(n)}" for q, p, n in zip(np.diff(qsl), first_pos, n_req))
        logger.info("DS41 replay trace step %d: %s T=%d kept=%d reqs(qlen@first_pos/N_req)=[%s]", self._step, kind, T,
                    kept, reqs)

    def upload(self, plan: RowPlan, positions_cpu: np.ndarray) -> CompactRows:
        """One pinned H2D of kept | dropped | positions | tok2req | qsl (all int64)."""
        tk, td, r1 = plan.kept.shape[0], plan.dropped.shape[0], plan.query_start_loc.shape[0]
        need = 3 * tk + td + r1
        if need > self._pinned.shape[0]:
            self._pinned = torch.empty((need,), dtype=torch.int64, device="cpu",
                                       pin_memory=self.device.type == "cuda")
        elif self._h2d_done is not None:
            self._h2d_done.synchronize()        # the previous step's copy out of this buffer (long done in practice)
        host = self._pinned.numpy()
        pos_k = np.asarray(positions_cpu, dtype=np.int64)[plan.kept]
        tok2req = np.repeat(np.arange(r1 - 1, dtype=np.int64), np.diff(plan.query_start_loc))
        o = 0
        for part in (plan.kept, plan.dropped, pos_k, tok2req, plan.query_start_loc):
            host[o:o + part.shape[0]] = part
            o += part.shape[0]
        dev = self._pinned[:need].to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            self._h2d_done = torch.cuda.Event()
            self._h2d_done.record()
        return CompactRows(kept=dev[:tk], dropped=dev[tk:tk + td], positions=dev[tk + td:2 * tk + td],
                           tok2req=dev[2 * tk + td:3 * tk + td].to(torch.int32),
                           query_start_loc=dev[3 * tk + td:need].to(torch.int32), kept_cpu=plan.kept,
                           positions_cpu=pos_k, query_start_loc_cpu=plan.query_start_loc)

    def zero_dropped(self, md_all, dropped: torch.Tensor | None) -> None:
        """Zero records into the trimmed layers' SWA slots of the dropped rows (``None`` = every real row)."""
        done: dict[tuple[int, int], torch.Tensor] = {}
        for (_, layer), name in zip(self.layers, self.swa_names):
            md = md_all[name]
            slot = md.slot_mapping[: md.num_actual_tokens]
            key = (slot.data_ptr(), slot.shape[0])
            s = done.get(key)
            if s is None:
                s = done[key] = slot if dropped is None else slot.index_select(0, dropped)
            zero_window_rows(layer.attn.swa_cache.rows(), s)

    def compact_step(self, md_all, plan: RowPlan, shared) -> tuple[dict, CompactRows]:
        """Compacted metadata dict for layers 21-39 + layer 20's published top-k / candidate blocks compacted in
        place (the reuse layers 21-23 read layer 20's top-k; the reindex layers mask to its candidate blocks)."""
        rows = self.upload(plan, md_all[self.swa_names[0]].positions_cpu)
        memo: dict[int, object] = {}
        out = dict(md_all)
        for name in self.md_names:
            md = md_all[name]
            c = memo.get(id(md))
            if c is None:
                c = memo[id(md)] = compact_metadata(md, rows)
            out[name] = c
        tk = int(plan.kept.shape[0])
        # index_select makes the gathered copy first: kept[i] >= i, but a parallel in-place gather could still read a
        # row another thread already overwrote
        shared.topk_indices[:tk].copy_(shared.topk_indices.index_select(0, rows.kept))
        shared.candidate_blocks[:tk].copy_(shared.candidate_blocks.index_select(0, rows.kept))
        _LAST_DRAFT_ROWS[0] = (plan.num_tokens, rows.kept, rows.dropped)
        return out, rows

    def note_skip(self, plan: RowPlan) -> None:
        """A ``skip`` step: tell DSpark that no context row is kept (dropped None = every row)."""
        empty = torch.empty((0,), dtype=torch.int64, device=self.device)
        _LAST_DRAFT_ROWS[0] = (plan.num_tokens, empty, None)
