# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 decoder replay (prefill row trim): prefix-cache hit validity.

With ``VLLM_DS41_REPLAY_ROWS=K``
> 0 the model runs layers 21-39 (and the DSpark drafter) only on rows with
``position >= N - K``, where N is the request's ``num_tokens`` (prompt + output
tokens) when its prefill pass is scheduled. Rows below ``S = N - K`` never write
their layer 21-39 SWA window slots, and the inexactness that starts at S reaches
``WINDOW - 1`` rows further per layer, so every layer 21-39 SWA row and the
layer-39 output (DSpark aux / drafter window KV) are exact only from
``S + REPLAY_EXACT_MARGIN``.

Scheduler side (this module, ``KVCacheBlock.replay_deep``,
``BlockPool.cache_full_blocks``, ``KVCacheManager.get_computed_blocks``):

* each request carries ``replay_exact_from``, the first position whose freshly
  computed KV is exact in every layer, set when its pass starts (prefix lookup);
* a block is marked ``replay_deep`` when it is hashed, iff its first position is
  >= that value (decode-produced blocks always are: the value never exceeds N);
* a hit of H tokens with ``H > N - K`` reads its window rows
  ``[H - (W - 1), H)`` from the cache, so every sliding-window group's hit blocks
  covering them must be deep; otherwise the lookup is redone with
  ``max_length = N - K``, which the coordinator aligns for every KV cache group.
  With ``H <= N - K`` any hit is fine (layers 21-39 restart at S regardless).

K = 0 (the default) is a strict no-op: every block stays deep and the check never
runs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs

REPLAY_ROWS_ENV = "VLLM_DS41_REPLAY_ROWS"
DS41_MODEL_VERSION = "deepseek_v41"
# Layers 21..39 keep each row's last WINDOW rows as their only per-token state.
REPLAY_WINDOW = 128
REPLAY_NUM_LAYERS = 19
# 127 * 19 = 2413: exact SWA KV in every replayed layer and an exact layer-39
# output from S + 2413 on.
REPLAY_EXACT_MARGIN = (REPLAY_WINDOW - 1) * REPLAY_NUM_LAYERS


def read_replay_rows() -> int:
    """``VLLM_DS41_REPLAY_ROWS`` via the model's strict knob parser (unset -> 0)."""
    try:
        from vllm.models.deepseek_v41.knobs import env_int
    except ImportError:
        env_int = None
    if env_int is not None:
        return env_int(REPLAY_ROWS_ENV, 0, minimum=0)
    value = (os.environ.get(REPLAY_ROWS_ENV) or "").strip()
    rows = int(value, 10) if value else 0
    if rows < 0:
        raise ValueError(f"{REPLAY_ROWS_ENV}={rows} is below the minimum 0")
    return rows


def is_ds41_kv_cache_config(kv_cache_config: KVCacheConfig) -> bool:
    """True when any KV cache group holds DeepSeek-V4.1 caches."""
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        inner = (
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        if any(getattr(s, "model_version", None) == DS41_MODEL_VERSION for s in inner):
            return True
    return False


def replay_exact_from(num_tokens: int, num_hit_tokens: int, replay_rows: int) -> int:
    """First position whose KV this request's pass computes exactly in all layers.

    ``num_hit_tokens`` is the (already validated) prefix-cache hit the pass
    starts from. Every row from the hit on is exact when the hit is above
    ``S = num_tokens - replay_rows`` (its window was checked deep) or nothing is
    trimmed (``S <= 0``). Otherwise exactness returns REPLAY_EXACT_MARGIN rows
    past S, capped at ``num_tokens`` so decode-produced blocks are always deep.
    """
    if replay_rows <= 0:
        return 0
    replay_start = num_tokens - replay_rows
    if replay_start <= 0 or num_hit_tokens > replay_start:
        return num_hit_tokens
    return min(replay_start + REPLAY_EXACT_MARGIN, num_tokens)


def window_tail_is_deep(
    hit_blocks: Sequence[Sequence[KVCacheBlock]],
    window_groups: Sequence[tuple[int, int, int]],
    num_hit_tokens: int,
) -> bool:
    """Whether each sliding-window group's hit blocks covering
    ``[H - (window - 1), H)`` are all deep.

    ``window_groups`` lists ``(group_id, block_size, sliding_window)``. A null
    block inside the window counts as not deep (the hit cannot vouch for it).
    """
    for group_id, block_size, sliding_window in window_groups:
        blocks = hit_blocks[group_id]
        if not blocks:
            continue
        first = max(0, num_hit_tokens - (sliding_window - 1)) // block_size
        last = min(len(blocks), cdiv(num_hit_tokens, block_size))
        for block in blocks[first:last]:
            if block.is_null or not block.replay_deep:
                return False
    return True
