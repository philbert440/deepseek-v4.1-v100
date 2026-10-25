# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-group sharing of the compressed-row ids in prefill (``VLLM_DS41_ATTN_GATHER_ONCE``; default 0 = off).

Which layers attend over the same compressed rows (common/topology.py): a compressed layer L reads the cache of
``kv_source = max(KV_SOURCES <= L)`` and the top-k published by ``index_source = max(INDEX_SOURCES <= L)`` into
``shared.topk_indices``; reuse layers never rewrite it. The cache records are RoPE'd and FP4-packed once at write
time by the kv source (``ckv_rope_qat_store``), so they hold no per-layer state. Hence every layer of a
(kv_source, index_source) group selects the same rows with the same bytes:

    (2, 2): 2-7    (8, 8): 8-13    (14, 14): 14-19    (20, 20): 20-23
    (20, 24): 24-27    (20, 28): 28-31    (20, 32): 32-35    (20, 36): 36-39

38 compressed layers, 8 groups. The per-layer parts are the query (own wq_b, TP-sharded heads), the SWA window
rows (own swa cache) and the sink -- the gathered compressed rows are not.

What is shared here is the row-id tensor ([T, 512] int64, 4 KiB per token): computed by the group's first layer,
reused by the others, released after the group's last layer. The FP16 gathered rows themselves are NOT held across
layers: per query they are 512 x 1 KiB (512 MiB per 1024-token chunk), more than the whole attention transient of
today, and layers are sequential (layer L+1's queries need layer L's output). Instead the tile gather reads the cache
by row id in one pass (``sparse_kernels.gather_tile``), so holding a gathered copy would save nothing but a re-read
of the same 288-byte packed records.

Validity: an entry matches only the same source metadata OBJECT (held, so identity cannot be recycled), the same
top-k buffer, T and source rows. Index-source layers always rebuild (they just rewrote the top-k). Decoder replay
(sm70/replay.py) swaps in compacted metadata for layers 21-39 and compacts the top-k in place: the new metadata
object misses, so layer 21 rebuilds for the kept rows and 22-23 reuse that.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.models.deepseek_v41.common.contracts import INDEX_SOURCES

N_BACKBONE_LAYERS = 40


def ends_group(layer_id: int) -> bool:
    """Last compressed layer of its (kv source, index source) group: the next layer starts a new group (an index
    source) or the backbone ends."""
    return layer_id + 1 in INDEX_SOURCES or layer_id + 1 >= N_BACKBONE_LAYERS


class GroupRowsCache:
    """One live entry per stage (groups are consecutive layers): (key, held metadata, row ids)."""

    def __init__(self) -> None:
        self._entry: tuple[tuple, Any, Any] | None = None
        self.stats = {"built": 0, "reused": 0}

    def get(self, *, layer_id: int, owns_indexer: bool, kv_source: int, index_source: int, src_md: Any,
            topk: Any, num_tokens: int, rows_ptr: int, compute: Callable[[], Any]) -> Any:
        key = (kv_source, index_source, num_tokens, int(topk.data_ptr()), rows_ptr)
        e = self._entry
        if not owns_indexer and e is not None and e[0] == key and e[1] is src_md:
            self.stats["reused"] += 1
            idx = e[2]
        else:
            self._entry = None                  # drop the previous group's ids before building the next
            idx = compute()
            self.stats["built"] += 1
            self._entry = (key, src_md, idx)
        if ends_group(layer_id):
            self._entry = None
        return idx

    def clear(self) -> None:
        self._entry = None

    @property
    def live(self) -> bool:
        return self._entry is not None
