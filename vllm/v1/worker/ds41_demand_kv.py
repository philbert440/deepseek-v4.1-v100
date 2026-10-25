# DS41 demand KV. The scheduler still sizes the pool for the full 1M admission
# cap. Physical GPU memory is committed only for chunks that cover a block the
# worker must keep: a live request, a block about to be zeroed, or a
# prefix-cache entry that has not been evicted.

from __future__ import annotations

import gc
import importlib.util
import logging
import os

import torch

logger = logging.getLogger(__name__)

_VMM = None
_SO_PATH = os.environ.get("VLLM_DS41_VMM_SO", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "ds41_vmm.so"))


def load_vmm():
    """Load the prebuilt extension. Workers must not compile it themselves."""
    global _VMM
    if _VMM is not None:
        return _VMM
    spec = importlib.util.spec_from_file_location("ds41_vmm", _SO_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"DS41 VMM extension not found at {_SO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _VMM = module
    return module


def chunks_for_blocks(
    block_ids,
    page_size: int,
    granularity: int,
    nbytes: int,
    num_blocks: int,
) -> set[int]:
    """Chunks that cover ``block_ids`` plus chunk 0 (the null block)."""
    if granularity <= 0 or nbytes <= 0 or page_size <= 0:
        raise ValueError("invalid demand-KV geometry")
    nchunks = (nbytes + granularity - 1) // granularity
    need = {0}
    for block_id in block_ids:
        block_id = int(block_id)
        if block_id < 0 or block_id >= num_blocks:
            continue
        start = block_id * page_size
        if start >= nbytes:
            continue
        end = min(start + page_size, nbytes)
        first = start // granularity
        last = (end - 1) // granularity
        for chunk in range(first, last + 1):
            if chunk < nchunks:
                need.add(chunk)
    return need


def update_mapped_blocks(mapped, keep, release) -> set[int]:
    """Drop released ids unless this step still needs them, then add ``keep``.

    A block evicted and reused in the same step is in both lists. The keep
    set wins: the worker maps it and the zeroer clears it.
    """
    out = set(mapped)
    kept = {0}
    kept.update(int(block_id) for block_id in keep)
    if release:
        out.difference_update(
            int(block_id) for block_id in release if int(block_id) not in kept
        )
    out |= kept
    return out


def chunk_span(
    block_id: int,
    page_size: int,
    granularity: int,
    nbytes: int,
    num_blocks: int,
) -> range:
    """Chunks covered by one block. Empty when the id is outside the tensor."""
    if (
        granularity <= 0
        or page_size <= 0
        or nbytes <= 0
        or block_id < 0
        or block_id >= num_blocks
    ):
        return range(0)
    start = block_id * page_size
    if start >= nbytes:
        return range(0)
    end = min(start + page_size, nbytes)
    nchunks = (nbytes + granularity - 1) // granularity
    first = start // granularity
    last = min((end - 1) // granularity, nchunks - 1)
    if first > last:
        return range(0)
    return range(first, last + 1)


class ChunkUsers:
    """How many accounted block ids cover each chunk. Chunk 0 starts at 1."""

    def __init__(self, nchunks: int) -> None:
        self.users = [0] * nchunks
        if nchunks > 0:
            self.users[0] = 1

    def add(self, chunks: range) -> list[int]:
        fresh: list[int] = []
        for chunk in chunks:
            if chunk < 0 or chunk >= len(self.users):
                continue
            self.users[chunk] += 1
            if self.users[chunk] == 1:
                fresh.append(chunk)
        return fresh

    def remove(self, chunks: range) -> list[int]:
        drop: list[int] = []
        for chunk in chunks:
            # Chunk 0 is the null block and stays mapped for the life of the pool.
            if chunk <= 0 or chunk >= len(self.users) or self.users[chunk] <= 0:
                continue
            self.users[chunk] -= 1
            if self.users[chunk] == 0:
                drop.append(chunk)
        return drop


def flatten_block_ids(block_ids, into: set[int]) -> None:
    if block_ids is None:
        return
    if isinstance(block_ids, (list, tuple)):
        for item in block_ids:
            flatten_block_ids(item, into)
        return
    try:
        iterator = iter(block_ids)
    except TypeError:
        into.add(int(block_ids))
        return
    for item in iterator:
        flatten_block_ids(item, into)


class DemandKVSlot:
    """One KV tensor: stable virtual address, physical chunks on demand."""

    def __init__(
        self, nbytes: int, page_size: int, num_blocks: int, device_index: int
    ) -> None:
        if page_size <= 0 or nbytes != page_size * num_blocks:
            raise ValueError(
                f"KV tensor {nbytes} bytes is not {num_blocks} pages of {page_size}"
            )
        vmm = load_vmm()
        self.pool = vmm.Pool(int(nbytes), int(device_index))
        self.tensor = self.pool.view()
        self.page_size = page_size
        self.num_blocks = num_blocks
        self.granularity = int(self.pool.granularity())
        self._mapped_chunks: set[int] = set()
        self._users = ChunkUsers(int(self.pool.nchunks()))
        # Block 0 is the null/pad index. Warmup zeros it before any request.
        self._map_new({0})

    def mapped_bytes(self) -> int:
        return int(self.pool.mapped_bytes())

    def _stream(self) -> int:
        return int(torch.cuda.current_stream().cuda_stream)

    def _map_new(self, chunks: set[int]) -> None:
        pending = [chunk for chunk in sorted(chunks) if chunk not in self._mapped_chunks]
        if not pending:
            return
        try:
            self._map_pending(pending)
        except RuntimeError as exc:
            if "CUDA_ERROR_OUT_OF_MEMORY" not in str(exc):
                raise
            # cuMemCreate allocates outside PyTorch's pool. A long prefill
            # leaves the peak workspace reserved, and the driver then refuses
            # a 2 MiB chunk even though that reserve is idle. Hand it back
            # once. A second failure is a real shortage.
            free, _total = torch.cuda.mem_get_info()
            reserved = torch.cuda.memory_reserved()
            allocated = torch.cuda.memory_allocated()
            gc.collect()
            torch.cuda.empty_cache()
            free_after, _total = torch.cuda.mem_get_info()
            logger.warning(
                "DS41 demand KV: cuMemCreate OOM on %d new chunk(s); "
                "empty_cache returned %.0f MiB to the driver "
                "(free %.0f -> %.0f MiB, allocated %.0f MiB, reserved %.0f MiB).",
                len(pending),
                (free_after - free) / (1024 * 1024),
                free / (1024 * 1024),
                free_after / (1024 * 1024),
                allocated / (1024 * 1024),
                reserved / (1024 * 1024),
            )
            self._map_pending(pending)

    def _map_pending(self, pending: list[int]) -> None:
        stream = self._stream()
        for chunk in pending:
            if chunk in self._mapped_chunks:
                continue
            self.pool.map_chunk(int(chunk), stream)
            self._mapped_chunks.add(chunk)

    def _span(self, block_id: int) -> range:
        return chunk_span(
            int(block_id),
            self.page_size,
            self.granularity,
            int(self.tensor.numel()),
            self.num_blocks,
        )

    def apply_delta(self, added, removed, *, unmap: bool) -> None:
        """Map chunks for newly kept ids. Unmap only chunks whose last id left.

        The caller passes the ids that changed on this step, not the prefix
        cache. A steady decode whose blocks already sit in a mapped chunk
        makes no driver call and does not synchronize.
        """
        fresh: set[int] = set()
        for block_id in added:
            fresh.update(self._users.add(self._span(block_id)))
        drop: set[int] = set()
        if unmap:
            for block_id in removed:
                drop.update(self._users.remove(self._span(block_id)))
        if fresh:
            self._map_new(fresh)
        if not drop:
            return
        # The previous step's kernels must finish before the VA hole appears.
        # Replay of a CUDA graph cannot fault, so this runs outside capture.
        torch.cuda.current_stream().synchronize()
        for chunk in drop:
            self.pool.unmap_chunk(int(chunk))
            self._mapped_chunks.discard(chunk)
