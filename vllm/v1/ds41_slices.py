# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 layer-sliced prefill: types shared by the scheduler (rivet/hol-sched) and the workers (rivet/hol-model).

hol-design.md Appendix A (Phase 1 interface contract). While a request decodes, the scheduler stages a non-final
prefill chunk of another request and, in every engine step, sends each PP stage a few layers of it to run after the
step's decode/verify batch ("slices"). The scheduler emits the SAME directive tuple to both stages; each stage runs
the directives whose ``stage`` equals its PP rank, in tuple order:

  (1) the decode/verify batch exactly as today; (2) PP isend of the decode payload; (3) for each own directive in
  order: stage 1 with ``first`` -> irecv the handoff for (req_id, chunk_id) FIRST; ``forward_slice(layer_start,
  layer_end)``; stage 0 with ``last`` -> isend the handoff (after the decode payload).

The scheduler guarantees that stage 0's ``last`` directive of a chunk and stage 1's ``first`` directive of the same
chunk are in the SAME step (so every handoff send has its receive in that step, in the same order on both sides),
that at most one chunk per stage is in flight, and that a chunk's stage-1 ``last`` directive is the only one whose
completion is reported back (``ModelRunnerOutput.ds41_slice_done``, last stage only). ``ds41_slice_aborts`` tells
both stages to drop the per-chunk state of (req_id, chunk_id); KV blocks are owned and freed by the scheduler.

Knobs (engine and workers read the same environment; default off = byte-identical to today):
  VLLM_DS41_SLICED_PREFILL=1    turn the lane on (needs PP2, sync scheduling, VLLM_DS41_FOLD_DRAFTS=1)
  VLLM_DS41_SLICE_LAYERS_S0=N   stage-0 layers per slice (default 3)
  VLLM_DS41_SLICE_LAYERS_S1=N   stage-1 layers per slice; unset/0 = the stage's first layer alone, then the rest in
                                one slice (layer 20, then 21-39 under replay)
  VLLM_DS41_SLICE_POLICY        fixed (default: the two knobs above) | adaptive: the scheduler sizes each slice from
                                the workers' measured costs (ModelRunnerOutput.ds41_slice_ms / ds41_decode_ms) so it
                                fits the stage's idle gap in the decode cycle, stretched by DECODE_KEEP:
                                budget = EMA(decode span) / KEEP - EMA(stage decode fwd) - MARGIN
  VLLM_DS41_SLICE_DECODE_KEEP   adaptive: fraction of the decode-only cycle rate kept while slicing (default 0.75;
                                1.0 = slices only fill the idle gap)
  VLLM_DS41_SLICE_MARGIN_MS     adaptive: safety margin per slice budget (default 3)
  VLLM_DS41_SLICE_S1_ATTN_PARTS sub-layer slices for stage 1's first layer (layer 20: full-context indexer scan +
                                candidate selection, O(T x n_ctx)): 0 = off (default, today), N = always N attention
                                parts, auto = parts from the cost model A + B * T * n_ctx (B learned from the measured
                                attention-half cost under the adaptive policy) and the stage-1 budget. The layer then
                                runs as attention parts 0..N-1 (layer_part 1: the scan's query-row tiles split across
                                them) and a MoE part (layer_part 2: the layer's MoE half, then whole layers up to
                                layer_end). Needs VLLM_DS41_INDEXER_QTILE_MAX (default 128 when this knob is on).
  VLLM_DS41_SLICE_S1_PART_MS    target ms per stage-1 sub-layer slice with the fixed policy (default 45)
  VLLM_DS41_SLICE_S1_ATTN_FIXED_MS / _SCAN_MS_PER_MTOK2   auto cost-model priors: attention half minus the scan
                                (default 12) and scan ms per million (T x n_ctx) token pairs (default 1.5)

Adaptive feedback path (workers -> scheduler): every stage times its slices with CUDA events; its decode steps are
timed by the stage step timer (ds41_stage_timing, built on TP rank 0 with logging off when only this policy needs it).
The last stage reports its own samples in each ModelRunnerOutput. Stage 0's samples ride the chunk handoff as one
extra static f32[SLICE_FB_LEN] tensor (SLICE_FB_KEY, only when the policy is adaptive, on both sides), so they reach
the scheduler with up to one chunk of lag; no new p2p channel and no extra message per step.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.models.deepseek_v41 import knobs

SLICED_PREFILL_KNOB = "VLLM_DS41_SLICED_PREFILL"
SLICE_LAYERS_S0_KNOB = "VLLM_DS41_SLICE_LAYERS_S0"
SLICE_LAYERS_S1_KNOB = "VLLM_DS41_SLICE_LAYERS_S1"
DEFAULT_SLICE_LAYERS_S0 = 3
SLICE_POLICY_KNOB = "VLLM_DS41_SLICE_POLICY"
SLICE_DECODE_KEEP_KNOB = "VLLM_DS41_SLICE_DECODE_KEEP"
SLICE_MARGIN_KNOB = "VLLM_DS41_SLICE_MARGIN_MS"
SLICE_FB_KEY = "ds41_slice_fb"
SLICE_FB_LEN = 32
SLICE_S1_ATTN_PARTS_KNOB = "VLLM_DS41_SLICE_S1_ATTN_PARTS"
SLICE_S1_PART_MS_KNOB = "VLLM_DS41_SLICE_S1_PART_MS"
SLICE_S1_ATTN_FIXED_MS_KNOB = "VLLM_DS41_SLICE_S1_ATTN_FIXED_MS"
SLICE_S1_SCAN_MS_KNOB = "VLLM_DS41_SLICE_S1_SCAN_MS_PER_MTOK2"
INDEXER_QTILE_MAX_KNOB = "VLLM_DS41_INDEXER_QTILE_MAX"
DEFAULT_SPLIT_QTILE_MAX = 128          # query-row tile cap of the full indexer scan when stage-1 parts are on

# stage-1 cost model (sub-layer planner): layers past the candidate source that own an indexer score candidates
# (reindex: 24, 28, 32, 36 under the V4.1 topology); their attention half dominates when the chunk's rows are kept
# by the decoder replay (the prompt's last VLLM_DS41_REPLAY_ROWS rows). Feedback sample kinds (ModelRunnerOutput
# .ds41_slice_ms tails): first-layer scan, reindex attention half, whole layers (plain work units + reindex count).
COST_KIND_SCAN = 1
COST_KIND_REINDEX_ATTN = 2
COST_KIND_LAYERS = 3
MOE_SHARE = 0.6                       # MoE half of a plain layer's cost (attention ~6 / MoE ~8 ms at 1024 rows)
REPLAY_ROWS_KNOB = "VLLM_DS41_REPLAY_ROWS"
SLICE_S1_LAYER_MS_KNOB = "VLLM_DS41_SLICE_S1_LAYER_MS"          # plain layer, 1024 kept rows (prior 14)
SLICE_S1_REINDEX_MS_KNOB = "VLLM_DS41_SLICE_S1_REINDEX_ATTN_MS"  # reindex attention half, 1024 kept rows (prior 74)


def reindex_layers() -> tuple[int, ...]:
    from vllm.models.deepseek_v41.common.contracts import CAND_SOURCE, INDEX_SOURCES
    return tuple(s for s in INDEX_SOURCES if s > CAND_SOURCE)


def is_reindex_layer(layer: int) -> bool:
    return layer in reindex_layers()


def replay_first_layer() -> int:
    from vllm.models.deepseek_v41.common.contracts import CAND_SOURCE
    return CAND_SOURCE + 1


def replay_rows_k() -> int:
    """Decoder replay K (0 = off: layers 21-39 run every row of every chunk)."""
    return knobs.env_int(REPLAY_ROWS_KNOB, 0, minimum=0)


def slice_s1_layer_ms() -> float:
    return knobs.env_float(SLICE_S1_LAYER_MS_KNOB, 14.0, minimum=0.0)


def indexer_fused_enabled() -> bool:
    """VLLM_DS41_INDEXER_FUSED (indexer.py / sm70/indexer_fused.py): prefill scoring + top-512 by the fused kernels."""
    return knobs.env_bool("VLLM_DS41_INDEXER_FUSED", False)


def slice_s1_reindex_attn_ms() -> float:
    """Reindex attention half on 1024 kept rows when it scores candidates: 74 ms with candidate_topk; with the fused
    kernels the plain attention share + ~8.5 ms (fused candidate scoring is ~5-9 ms flat, GPU 20261024T221236Z)."""
    default = (1.0 - MOE_SHARE) * slice_s1_layer_ms() + REINDEX_CAND_FUSED_MS if indexer_fused_enabled() else 74.0
    return knobs.env_float(SLICE_S1_REINDEX_MS_KNOB, default, minimum=0.0)


# reindex scoring by VLLM_DS41_INDEXER_CAND_PREFILL (indexer.py; rivet/cand-live): full scan + candidate mask costs
# ~2.5 + 1.12 ms per 1K visible keys at 1024 rows (GPU bench 20261024T211909Z), candidate_topk ~66-68 ms flat
REINDEX_SCAN_FIXED_MS = 2.5
REINDEX_SCAN_MS_PER_KKEY = 1.12
# with VLLM_DS41_INDEXER_FUSED (GPU bench 20261024T221236Z, T=1024): scan + mask 2.1 / 4.1 / 14.3 / 54.4 ms at n = 8K /
# 17K / 64K / 256K (~2.5 + 0.2 ms per 1K keys), fused candidates 5.2-9.2 ms flat -> crossover ~36K
REINDEX_SCAN_FUSED_FIXED_MS = 2.5
REINDEX_SCAN_FUSED_MS_PER_KKEY = 0.2
REINDEX_CAND_FUSED_MS = 8.5
CAND_SCAN_MAX_DEFAULT = 49152
CAND_SCAN_MAX_FUSED_DEFAULT = 36864


def cand_scan_max() -> int:
    """VLLM_DS41_INDEXER_CAND_SCAN_MAX (auto: full scan + mask below this n); the default follows the scoring kernels."""
    default = CAND_SCAN_MAX_FUSED_DEFAULT if indexer_fused_enabled() else CAND_SCAN_MAX_DEFAULT
    return knobs.env_int("VLLM_DS41_INDEXER_CAND_SCAN_MAX", default, minimum=0)


def reindex_scores_by_scan(n_visible: int) -> bool:
    """Whether a reindex layer scores a prefill request with n visible entries by the full scan + mask (vs its
    candidates): candidate scoring off, or VLLM_DS41_INDEXER_CAND_PREFILL scan / auto below CAND_SCAN_MAX."""
    if not knobs.env_bool("VLLM_DS41_INDEXER_CANDIDATE_SCORING", False):
        return True
    mode = knobs.env_str("VLLM_DS41_INDEXER_CAND_PREFILL", "off").strip().lower()
    if mode == "scan":
        return True
    if mode == "auto":
        return n_visible < cand_scan_max()
    return False


def reindex_attn_prior_ms(rows: int, n_ctx: int) -> float:
    """Prior cost of a reindex layer's attention half on ``rows`` rows of a chunk ending at n_ctx (compress ratio 1:
    n visible ~ n_ctx): the layer's plain attention share + its scoring (scan ~ n, candidates flat)."""
    base = (1.0 - MOE_SHARE) * slice_s1_layer_ms()
    if reindex_scores_by_scan(n_ctx):
        if indexer_fused_enabled():
            idx = REINDEX_SCAN_FUSED_FIXED_MS + REINDEX_SCAN_FUSED_MS_PER_KKEY * n_ctx / 1000.0
        else:
            idx = REINDEX_SCAN_FIXED_MS + REINDEX_SCAN_MS_PER_KKEY * n_ctx / 1000.0
    else:
        idx = max(slice_s1_reindex_attn_ms() - base, 0.0)
    return (base + idx) * rows / 1024.0


# SliceDirective.layer_part
PART_WHOLE = 0       # whole layers [layer_start, layer_end)
PART_ATTN = 1        # attention half of layer_start (layer_end == layer_start + 1), part part_index of part_count
PART_MOE = 2         # MoE half of layer_start, then whole layers layer_start + 1 .. layer_end - 1


@dataclass(frozen=True)
class SliceDirective:
    stage: int                 # PP rank that runs it
    req_id: str
    chunk_id: int              # monotonically increasing per request
    start_pos: int             # first token position of the chunk
    num_tokens: int            # T (<= max_num_batched_tokens)
    token_ids: tuple[int, ...]  # stage 0 only (embed); () on stage 1
    block_ids: tuple[tuple[int, ...], ...]   # per KV-cache group: the request's FULL block table rows needed for [0, start_pos+T)
    layer_start: int           # absolute layer ids, within the stage's range
    layer_end: int             # exclusive
    first: bool                # first slice of this chunk on this stage (build metadata / allocate state / Engram begin_slice)
    last: bool                 # last slice on this stage (stage 0: send handoff; stage 1: report slice_done)
    n_req: int                 # request's total token count (decoder-replay N_req for this chunk)
    # sub-layer slices (VLLM_DS41_SLICE_S1_ATTN_PARTS; defaults = whole layers, today's directives)
    layer_part: int = PART_WHOLE   # PART_WHOLE | PART_ATTN | PART_MOE
    part_index: int = 0            # PART_ATTN: this part (0 = first: KV / compressor stores; last: attend + wo + hc)
    part_count: int = 1            # PART_ATTN: number of attention parts of the layer
    # stage-0 first slice of a request the workers have never seen whose first chunk starts after a prefix-cache hit:
    # token ids [0, start_pos), so the Engram history (hashed n-grams reach back before the chunk) exists; () otherwise
    prefix_token_ids: tuple[int, ...] = ()

    @property
    def key(self) -> tuple[str, int]:
        """(req_id, chunk_id): the id used by ds41_slice_done / ds41_slice_aborts."""
        return (self.req_id, self.chunk_id)

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.num_tokens


# SchedulerOutput additions:  ds41_slices: tuple[SliceDirective, ...] = ()
#                             ds41_slice_aborts: tuple[tuple[str, int], ...] = ()   # (req_id, chunk_id): drop state, free nothing
# ModelRunnerOutput addition: ds41_slice_done: tuple[tuple[str, int], ...] = ()     # from the LAST stage only


def sliced_prefill_enabled() -> bool:
    """VLLM_DS41_SLICED_PREFILL (strict boolean; unset = off)."""
    return knobs.env_bool(SLICED_PREFILL_KNOB, False)


def slice_layers_s0() -> int:
    return knobs.env_int(SLICE_LAYERS_S0_KNOB, DEFAULT_SLICE_LAYERS_S0, minimum=1)


def slice_layers_s1() -> int:
    """0 = default plan (first layer alone, then the rest)."""
    return knobs.env_int(SLICE_LAYERS_S1_KNOB, 0, minimum=0)


def slice_policy() -> str:
    return knobs.env_str(SLICE_POLICY_KNOB, "fixed", choices=("fixed", "adaptive"))


def adaptive_feedback_enabled() -> bool:
    """Workers measure and report slice / decode costs (both stages must agree: it changes the handoff schema)."""
    return sliced_prefill_enabled() and slice_policy() == "adaptive"


def slice_decode_keep() -> float:
    return knobs.env_float(SLICE_DECODE_KEEP_KNOB, 0.75, minimum=0.05, maximum=1.0)


def slice_margin_ms() -> float:
    return knobs.env_float(SLICE_MARGIN_KNOB, 3.0, minimum=0.0)


def slice_s1_attn_parts() -> int | str:
    """0 = off, N >= 1 fixed parts, "auto"."""
    raw = knobs.env_str(SLICE_S1_ATTN_PARTS_KNOB, "0").strip().lower()
    if raw == "auto":
        return "auto"
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(f"{SLICE_S1_ATTN_PARTS_KNOB}={raw!r}: expected 0, a part count or 'auto'") from exc
    if n < 0:
        raise ValueError(f"{SLICE_S1_ATTN_PARTS_KNOB}={n} must be >= 0")
    return n


def slice_s1_part_ms() -> float:
    return knobs.env_float(SLICE_S1_PART_MS_KNOB, 45.0, minimum=1.0)


def slice_s1_attn_fixed_ms() -> float:
    return knobs.env_float(SLICE_S1_ATTN_FIXED_MS_KNOB, 12.0, minimum=0.0)


def slice_s1_scan_ms_per_mtok2() -> float:
    """Layer-20 scan prior per million (T x n_ctx) pairs: 1.5 ms today; ~0.4 with VLLM_DS41_INDEXER_FUSED (score kernel
    21.5 -> 2.7 ms at 1024 x 17K, plus candidate-block selection and the fused top-512)."""
    return knobs.env_float(SLICE_S1_SCAN_MS_KNOB, 0.4 if indexer_fused_enabled() else 1.5, minimum=0.0)


def indexer_qtile_max() -> int:
    """Query-row tile cap of the indexer's full scan: 0 (none, today) unless sliced prefill with stage-1 attention
    parts is on -- only the parts need it (they split the scan's tiles; sliced and unsliced runs then share the capped
    grid and stay bit-identical); it costs ~4 % of solo prefill at 128K, so it never applies otherwise.
    VLLM_DS41_INDEXER_QTILE_MAX sets the cap while the parts are on (default DEFAULT_SPLIT_QTILE_MAX; 0 = none)."""
    if not (sliced_prefill_enabled() and slice_s1_attn_parts() != 0):
        return 0
    return split_qtile_cap()


# Staged-slice scope: the scan's query-row tile cap applies only to indexer calls inside a staged slice
# (DS41SlicedPrefill: every forward_slice, the SLICE_VERIFY reference re-run included), never to normal-path prefill.
_SLICE_FORWARD = [0]


@contextlib.contextmanager
def slice_forward():
    """Marks a staged slice's forward (worker ds41_sliced_prefill._forward_context)."""
    _SLICE_FORWARD[0] += 1
    try:
        yield
    finally:
        _SLICE_FORWARD[0] -= 1


def slice_forward_active() -> bool:
    return _SLICE_FORWARD[0] > 0


def split_qtile_cap() -> int:
    """The scan's query-row tile cap while stage-1 attention parts are on (VLLM_DS41_INDEXER_QTILE_MAX, default
    DEFAULT_SPLIT_QTILE_MAX); the planner sizes parts with it."""
    raw = knobs.env_str(INDEXER_QTILE_MAX_KNOB, "").strip()
    if raw:
        n = int(raw)
        if n < 0:
            raise ValueError(f"{INDEXER_QTILE_MAX_KNOB}={n} must be >= 0")
        return n
    return DEFAULT_SPLIT_QTILE_MAX


def split_range(n_items: int, part_index: int, part_count: int) -> tuple[int, int]:
    """[lo, hi) of part ``part_index`` when ``n_items`` consecutive items are split into ``part_count`` near-equal
    parts (earlier parts never larger than later ones by more than one item; empty parts allowed)."""
    if not 0 <= part_index < part_count:
        raise ValueError(f"part {part_index} of {part_count}")
    return (n_items * part_index // part_count, n_items * (part_index + 1) // part_count)


def plan_layer_slices(layer_start: int, layer_end: int, per_slice: int) -> tuple[tuple[int, int], ...]:
    """Split a stage's layer range [layer_start, layer_end) into consecutive slices. ``per_slice`` > 0: slices of that
    many layers (the last may be shorter). ``per_slice`` == 0: the first layer alone, then the rest in one slice."""
    if layer_end <= layer_start:
        raise ValueError(f"empty stage layer range [{layer_start}, {layer_end})")
    if per_slice < 0:
        raise ValueError(f"per_slice must be >= 0, got {per_slice}")
    if per_slice == 0:
        if layer_end - layer_start == 1:
            return ((layer_start, layer_end),)
        return ((layer_start, layer_start + 1), (layer_start + 1, layer_end))
    return tuple((a, min(a + per_slice, layer_end)) for a in range(layer_start, layer_end, per_slice))


def for_stage(slices: Iterable[SliceDirective], stage: int) -> list[SliceDirective]:
    """The directives a PP rank runs, in emission order."""
    return [d for d in slices if d.stage == stage]

