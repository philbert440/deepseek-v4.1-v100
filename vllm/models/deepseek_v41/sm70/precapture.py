# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup capture of the DS41 per-layer decode/verify attention graphs (VLLM_DS41_ATTN_PRECAPTURE).

Without it, each per-layer attention graph (attention.py ``_graph_step``) and each metadata-builder graph
(sparse.py ``_kernels``) is captured the first time a live step hits its key. That capture runs inside a timed
step (a B=2 DSpark bench lost ~30% to the T=12 first use) and allocates outside the profiled budget (the
2026-10-23 23:40 OOM). With the knob on, ``GPUModelRunner.capture_model`` drives one real-metadata dummy step per
key shape below, twice (warm + capture), after vLLM's own graph capture and before the workspace lock.

The dummy steps write no KV: the runner fills every slot mapping with -1 (all store kernels skip slots < 0), the
block-table rows are zero (every read goes to the null block 0, which demand KV keeps mapped), no request exists,
so the scheduler / prefix cache and the demand-KV map are untouched.

Knobs (read at call time):

=====================================  =========  ==================================================================
VLLM_DS41_ATTN_PRECAPTURE              0          1 = capture every key shape below at startup
VLLM_DS41_ATTN_NO_LAZY_CAPTURE         0          1 = after startup, a key that was not captured runs eagerly for
                                                  that step (attention + main-model builders); never captures
VLLM_DS41_ATTN_PRECAPTURE_SHAPES       verify     verify: each request has 1 or 1+k rows (plain decode / a full
                                                  DSpark verify): T = a(1+k) + (R-a), a = 0..R. B=2, k=5:
                                                  T 1,6 (R=1) and 2,7,12 (R=2). Truncated drafts run eagerly.
                                                  verify_b1all: R=1 every T 1..1+k, R>=2 as verify (for
                                                  VLLM_DS41_DSPARK_PMIN with _PMIN_MAX_BATCH=1: only B=1 steps
                                                  truncate). B=2, k=5: T 1-6 (R=1), 2,7,12 (R=2).
                                                  decode: every R..R*(1+k) row count per batch size plus each
                                                  capture size (adaptive drafts; ~3.5x the keys of verify);
                                                  all: every (T rows, R requests) with R <= max_num_seqs
VLLM_DS41_ATTN_PRECAPTURE_LONG_SEQ     2048       context length of the "long" regime batches (> 512 visible
                                                  compressed entries; raised to the minimum that crosses it)
VLLM_DS41_ATTN_PRECAPTURE_RESERVE_MIB  256        taken off the KV budget at memory profiling (the capture runs
                                                  after KV sizing, so this is how its memory is counted)
=====================================  =========  ==================================================================
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from vllm.models.deepseek_v41.common.contracts import IDX_TOPK

MAX_COMPRESS_RATIO = 2           # compressed caches have ratio 1 or 2 (sparse._check_mla_block)
SHAPES = ("verify", "verify_b1all", "decode", "all")


def precapture_enabled() -> bool:
    return os.environ.get("VLLM_DS41_ATTN_PRECAPTURE", "0") == "1"


def no_lazy_capture() -> bool:
    return os.environ.get("VLLM_DS41_ATTN_NO_LAZY_CAPTURE", "0") == "1"


def precapture_shapes() -> str:
    v = os.environ.get("VLLM_DS41_ATTN_PRECAPTURE_SHAPES", "verify").strip().lower()
    if v not in SHAPES:
        raise ValueError(f"VLLM_DS41_ATTN_PRECAPTURE_SHAPES={v!r}: expected one of {SHAPES}")
    return v


def min_long_seq_len() -> int:
    """Shortest context whose last token sees more than IDX_TOPK compressed entries at every ratio."""
    return MAX_COMPRESS_RATIO * (IDX_TOPK + 1)


def long_seq_len() -> int:
    v = int(os.environ.get("VLLM_DS41_ATTN_PRECAPTURE_LONG_SEQ", "2048"))
    return max(v, min_long_seq_len())


def precapture_reserve_bytes() -> int:
    if not precapture_enabled():
        return 0
    mib = int(os.environ.get("VLLM_DS41_ATTN_PRECAPTURE_RESERVE_MIB", "256"))
    return max(0, mib) * (1 << 20)


@dataclass(frozen=True)
class PrecaptureBatch:
    """One dummy step: ``query_lens[r]`` rows for request r, every request at context ``seq_len``."""
    query_lens: tuple[int, ...]
    seq_len: int
    regime: str                  # "short" (<= IDX_TOPK visible compressed entries) | "long"

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def num_reqs(self) -> int:
        return len(self.query_lens)


def split_rows(num_tokens: int, num_reqs: int) -> tuple[int, ...]:
    """``num_tokens`` rows over ``num_reqs`` requests as evenly as possible (the larger ones first)."""
    if not 1 <= num_reqs <= num_tokens:
        raise ValueError(f"cannot split {num_tokens} rows over {num_reqs} requests")
    base, extra = divmod(num_tokens, num_reqs)
    return tuple(base + 1 if r < extra else base for r in range(num_reqs))


def max_rows(capture_sizes: list[int], graph_max_t: int, max_num_tokens: int) -> int:
    """Largest row count with a per-layer graph: inside an outer capture size (else the eager-break arguments are
    transient tensors and the key would never repeat) and at most attention.GRAPH_MAX_T."""
    if not capture_sizes:
        return 0
    return min(graph_max_t, max(capture_sizes), max_num_tokens)


def batch_pairs(capture_sizes: list[int], max_num_seqs: int, query_len: int, graph_max_t: int,
                max_num_tokens: int, shapes: str = "verify") -> list[tuple[int, int]]:
    """Sorted (T rows, R requests) pairs whose attention graphs are captured.

    The per-layer graph key holds T, the padded size, R (block-table rows) and the visible-entry regime; the
    per-request split and the context length are device data. ``decode``: a step of R decoding requests has
    R..R*query_len rows (adaptive DSpark drafts any length up to k), plus one pair per capture size (R = the
    fewest requests that fill it) so each outer graph size has its attention graphs. ``all``: every T <= the
    graph limit with every R <= min(T, max_num_seqs) (short prefill tails, mixed decode + prefill steps).
    ``verify``: only steps whose requests each carry 1 row (no drafts) or 1 + k rows (a full verify).
    ``verify_b1all``: ``verify``, plus every R=1 row count 1..query_len (a lone request's p_min-truncated drafts)."""
    if shapes not in SHAPES:
        raise ValueError(f"shapes={shapes!r}: expected one of {SHAPES}")
    t_max = max_rows(capture_sizes, graph_max_t, max_num_tokens)
    q = max(1, int(query_len))
    pairs: set[tuple[int, int]] = set()
    for r in range(1, max(0, int(max_num_seqs)) + 1):
        if shapes == "verify_b1all" and r == 1:
            for t in range(1, min(q, t_max) + 1):
                pairs.add((t, 1))
            continue
        if shapes in ("verify", "verify_b1all"):
            for a in range(r + 1):
                t = a * q + (r - a)
                if t <= t_max:
                    pairs.add((t, r))
            continue
        hi = t_max if shapes == "all" else min(r * q, t_max)
        for t in range(r, hi + 1):
            pairs.add((t, r))
    if shapes == "decode" and max_num_seqs >= 1:
        for s in capture_sizes:
            if 1 <= s <= t_max:
                pairs.add((s, min(int(max_num_seqs), max(1, math.ceil(s / q)))))
    return sorted(pairs, key=lambda p: (p[1], p[0]))


def plan_batches(capture_sizes: list[int], max_num_seqs: int, query_len: int, graph_max_t: int,
                 max_model_len: int, max_num_tokens: int, shapes: str = "verify",
                 long_context: int | None = None) -> list[PrecaptureBatch]:
    """Every dummy step of the startup capture: each (T, R) pair in the short regime (context = the longest query,
    as vLLM's dummy runs) and, when the model can reach it, the long regime (more than IDX_TOPK visible compressed
    entries; the graph key changes there).

    Order: the long regime first, then the short one, each by decreasing T. The shared private pool can split a
    free block for a smaller request but never grow one for a larger request, so captures in increasing T left
    one block per distinct T of every large temporary (reindex candidate scoring is ~6 MiB per row): 2026-10-24
    GPU run, PP1 pool +622 MiB. Largest first, every later capture reuses the first one's blocks."""
    long_ctx = long_seq_len() if long_context is None else max(int(long_context), min_long_seq_len())
    pairs = batch_pairs(capture_sizes, max_num_seqs, query_len, graph_max_t, max_num_tokens, shapes)
    pairs = sorted(pairs, key=lambda p: (-p[0], p[1]))
    batches: list[PrecaptureBatch] = []
    if long_ctx <= max_model_len:
        batches += [PrecaptureBatch(_rows(t, r, query_len), long_ctx, "long") for t, r in pairs]
    for t, r in pairs:
        lens = _rows(t, r, query_len)
        batches.append(PrecaptureBatch(lens, max(lens), "short"))
    return batches


def _rows(t: int, r: int, query_len: int) -> tuple[int, ...]:
    """Rows per request: a verify-shaped split (each request 1 or 1 + k rows) when T has one, else even."""
    q = max(1, int(query_len))
    if q > 1 and (t - r) % (q - 1) == 0 and 0 <= (t - r) // (q - 1) <= r:
        a = (t - r) // (q - 1)
        return tuple([q] * a + [1] * (r - a))
    return split_rows(t, r)


# ------------------------------------------------------------------------------------------- process state
@dataclass
class _State:
    precapturing: bool = False   # inside the startup capture (per-layer logs suppressed, keys recorded)
    done: bool = False           # the startup capture ran (a capture after this is a mid-serve lazy capture)
    frozen: bool = False         # VLLM_DS41_ATTN_NO_LAZY_CAPTURE: no capture or warm-up of a new key
    captured: set = field(default_factory=set)       # key descriptions captured during the startup capture
    unseen_logged: set = field(default_factory=set)  # key descriptions already reported as eager / lazy
    eager_steps: int = 0         # layer steps that ran eagerly because their key was not captured
    lazy_captures: int = 0       # per-layer captures after the startup capture


STATE = _State()


def begin() -> None:
    STATE.precapturing = True
    STATE.captured.clear()


def finish(freeze: bool) -> None:
    STATE.precapturing = False
    STATE.done = True
    STATE.frozen = bool(freeze)


def capture_allowed() -> bool:
    return not STATE.frozen


def first_report(kind: str, desc: tuple) -> bool:
    """True the first time ``(kind, desc)`` is reported (one log line per key shape, not per layer)."""
    k = (kind, desc)
    if k in STATE.unseen_logged:
        return False
    STATE.unseen_logged.add(k)
    return True


def reset_for_tests() -> None:
    STATE.__init__()


def describe_pairs(descs: set) -> str:
    """'R=1: T 1-6; R=2: T 2-12,16' from {(T, padded, R, regime)}."""
    by_r: dict[int, set[int]] = {}
    for d in descs:
        by_r.setdefault(d[2], set()).add(d[0])
    parts = []
    for r in sorted(by_r):
        ts = sorted(by_r[r])
        runs, start, prev = [], ts[0], ts[0]
        for t in ts[1:] + [None]:
            if t is not None and t == prev + 1:
                prev = t
                continue
            runs.append(f"{start}" if start == prev else f"{start}-{prev}")
            if t is not None:
                start = prev = t
        parts.append(f"R={r}: T {','.join(runs)}")
    return "; ".join(parts) if parts else "none"
