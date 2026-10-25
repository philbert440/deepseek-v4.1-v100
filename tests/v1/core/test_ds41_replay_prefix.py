# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 decoder replay: prefix-cache hit validity (ds41_replay.py).

Groups mirror the SM70 port at block_size 256: compressed MLA (256 tokens per
block), SWA window (32 tokens per block, window 128) and compressor state (8
tokens per block, window 2), all tagged ``model_version="deepseek_v41"``.
lcm = 256, so hits are 256-aligned.

Most tests prefill in one step, so the earlier request still holds the SWA window
at every boundary of its prompt (branching off a running request). Under chunked
prefill the windows the SWA has moved past are reused first, which leaves only
the last chunk's boundaries (see ``test_chunked_prefill_hits``).
"""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import ds41_replay
from vllm.v1.core.block_pool import BlockHashToBlockMap
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

MLA_BLOCK = 256
SWA_BLOCK = 32
STATE_BLOCK = 8
HASH_BLOCK = STATE_BLOCK
K = 3072
MARGIN = ds41_replay.REPLAY_EXACT_MARGIN
CHUNK = 1024  # max_num_batched_tokens on the 8x V100 launch profile


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _config(model_version: str | None = "deepseek_v41") -> KVCacheConfig:
    def mla(name):
        return KVCacheGroupSpec(
            [name],
            MLAAttentionSpec(
                block_size=MLA_BLOCK,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                model_version=model_version,
            ),
        )

    def swa(name, block_size, window):
        return KVCacheGroupSpec(
            [name],
            SlidingWindowMLASpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=window,
                model_version=model_version,
            ),
        )

    return KVCacheConfig(
        num_blocks=40000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            mla("layers.2.attn"),
            swa("layers.21.swa_cache", SWA_BLOCK, 128),
            swa("layers.39.swa_cache", SWA_BLOCK, 128),
            swa("layers.2.state_cache", STATE_BLOCK, 2),
        ],
    )


def _manager(
    monkeypatch, rows: int | None, model_version="deepseek_v41"
) -> KVCacheManager:
    if rows is None:
        monkeypatch.delenv(ds41_replay.REPLAY_ROWS_ENV, raising=False)
    else:
        monkeypatch.setenv(ds41_replay.REPLAY_ROWS_ENV, str(rows))
    return KVCacheManager(
        kv_cache_config=_config(model_version),
        max_model_len=1 << 20,
        enable_caching=True,
        hash_block_size=HASH_BLOCK,
    )


def _request(request_id: str, token_ids: list[int]) -> Request:
    sampling_params = SamplingParams(max_tokens=16)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
    )


def _tokens(n: int, offset: int = 0) -> list[int]:
    return [offset + i for i in range(n)]


def _prefill(manager: KVCacheManager, req: Request, chunk: int | None = None) -> int:
    """Prefix lookup + (chunked) prefill like the scheduler; returns the hit length."""
    chunk = chunk or req.num_tokens
    computed, hit = manager.get_computed_blocks(req)
    lengths = {
        len(blocks) * manager.coordinator.single_type_managers[i].block_size
        for i, blocks in enumerate(computed.blocks)
    }
    # Every group's hit covers exactly the same aligned prefix.
    assert lengths == {hit}
    assert hit % MLA_BLOCK == 0
    n = min(chunk, req.num_tokens - hit)
    assert manager.allocate_slots(req, n, hit, computed) is not None
    req.num_computed_tokens = hit + n
    while req.num_computed_tokens < req.num_tokens:
        n = min(chunk, req.num_tokens - req.num_computed_tokens)
        assert manager.allocate_slots(req, n) is not None
        req.num_computed_tokens += n
    return hit


def _deep_flags(manager: KVCacheManager, req: Request, group_id: int):
    """(block_start, replay_deep) of the request's hashed blocks in one group."""
    block_size = manager.coordinator.single_type_managers[group_id].block_size
    blocks = manager.coordinator.get_blocks(req.request_id)[group_id]
    return [
        (i * block_size, blk.replay_deep)
        for i, blk in enumerate(blocks)
        if blk.block_hash is not None
    ]


def test_exact_from_rule():
    f = ds41_replay.replay_exact_from
    assert MARGIN == 2413
    assert f(10000, 0, 0) == 0  # off
    assert f(10000, 0, K) == 10000 - K + MARGIN  # fresh long prompt: P - 659
    assert f(10000, 4096, K) == 10000 - K + MARGIN  # hit below N - K
    assert f(10000, 9984, K) == 9984  # validated hit above N - K
    assert f(3000, 0, K) == 0  # nothing trimmed
    assert f(K, 0, K) == 0
    assert f(10000, 0, 128) == 10000  # approximate K: only decode blocks deep


def test_off_is_noop(monkeypatch):
    for rows in (None, 0):
        manager = _manager(monkeypatch, rows)
        assert manager.replay_rows == 0
        a = _request(f"a{rows}", _tokens(10000))
        _prefill(manager, a)
        assert a.replay_exact_from == 0
        for gid in range(4):
            assert all(deep for _, deep in _deep_flags(manager, a, gid))
        # Hit in the middle of an earlier prompt is unchanged.
        b = _request(f"b{rows}", _tokens(6000) + _tokens(500, 1 << 20))
        _, hit = manager.get_computed_blocks(b)
        assert hit == 6000 // MLA_BLOCK * MLA_BLOCK
        assert manager.num_replay_hit_shrinks == 0


def test_other_models_ignore_the_knob(monkeypatch):
    manager = _manager(monkeypatch, K, model_version=None)
    assert manager.replay_rows == 0


def test_marks_deep_from_exact_start(monkeypatch):
    manager = _manager(monkeypatch, K)
    assert manager.replay_rows == K
    a = _request("a", _tokens(10000))
    assert _prefill(manager, a) == 0
    exact_from = 10000 - K + MARGIN
    assert a.replay_exact_from == exact_from
    for gid in range(4):
        flags = _deep_flags(manager, a, gid)
        assert flags
        for start, deep in flags:
            assert deep == (start >= exact_from), (gid, start)


def test_hit_inside_earlier_prompt_shrinks(monkeypatch):
    manager = _manager(monkeypatch, K)
    _prefill(manager, _request("a", _tokens(10000)))
    # Branch at 6000 inside A's 10000-token prompt: the window before 5888
    # was never computed by layers 21-39 for A.
    n_b = 6500
    b = _request("b", _tokens(6000) + _tokens(n_b - 6000, 1 << 20))
    hit = _prefill(manager, b)
    assert hit == (n_b - K) // MLA_BLOCK * MLA_BLOCK == 3328
    assert manager.num_replay_hit_shrinks == 1
    assert b.replay_exact_from == min(n_b - K + MARGIN, n_b)


def test_hit_below_trim_start_is_kept(monkeypatch):
    manager = _manager(monkeypatch, K)
    _prefill(manager, _request("a", _tokens(10000)))
    n = 11000
    d = _request("d", _tokens(6000) + _tokens(n - 6000, 1 << 20))
    _, hit = manager.get_computed_blocks(d)
    assert hit == 5888 <= n - K
    assert manager.num_replay_hit_shrinks == 0
    assert d.replay_exact_from == n - K + MARGIN


def test_agent_continuation_hits_deep_tail(monkeypatch):
    manager = _manager(monkeypatch, K)
    _prefill(manager, _request("a", _tokens(10000)))
    c = _request("c", _tokens(10000) + _tokens(500, 1 << 20))
    hit = _prefill(manager, c)
    assert hit == 10000 // MLA_BLOCK * MLA_BLOCK == 9984
    assert manager.num_replay_hit_shrinks == 0
    assert c.replay_exact_from == hit
    # Everything C computed is exact: its own new blocks are all deep.
    for gid in range(4):
        assert all(deep for start, deep in _deep_flags(manager, c, gid) if start >= hit)


def test_short_request_cannot_use_shallow_window(monkeypatch):
    # N <= K: nothing is trimmed for E, but a hit would still read A's window.
    manager = _manager(monkeypatch, K)
    _prefill(manager, _request("a", _tokens(10000)))
    e = _request("e", _tokens(3000) + _tokens(10, 1 << 20))
    _, hit = manager.get_computed_blocks(e)
    assert hit == 0
    assert manager.num_replay_hit_shrinks == 1
    assert e.replay_exact_from == 0


def test_deep_duplicate_is_preferred(monkeypatch):
    manager = _manager(monkeypatch, K)
    _prefill(manager, _request("a", _tokens(10000)))
    # B branches at 9000: shrunk to 5888 and recomputes [5888, 9200), which
    # writes deep duplicates of A's blocks from B's exact start on.
    b = _request("b", _tokens(9000) + _tokens(200, 1 << 20))
    assert _prefill(manager, b) == 5888
    assert b.replay_exact_from == 9200 - K + MARGIN == 8541
    # Q branches at 9000 too; A's copy of the window [8833, 8960) is shallow,
    # B's copy is deep.
    q = _request("q", _tokens(9000) + _tokens(300, 2 << 20))
    _, hit = manager.get_computed_blocks(q)
    assert hit == 8960
    assert manager.num_replay_hit_shrinks == 1


def test_get_one_block_prefers_deep_copy():
    cache = BlockHashToBlockMap()
    key = make_block_hash_with_group_id(b"h", 0)
    shallow, deep = KVCacheBlock(1), KVCacheBlock(2)
    shallow.replay_deep = False
    cache.insert(key, shallow)
    assert cache.get_one_block(key) is shallow
    cache.insert(key, deep)
    assert cache.get_one_block(key) is deep
    deep.replay_deep = False
    assert cache.get_one_block(key) is shallow


def test_flag_survives_free_and_resets_on_eviction(monkeypatch):
    manager = _manager(monkeypatch, K)
    a = _request("a", _tokens(10000))
    _prefill(manager, a)
    swa_blocks = list(manager.coordinator.get_blocks("a")[1])
    manager.free(a)
    shallow = [b for b in swa_blocks if b.block_hash is not None and not b.replay_deep]
    deep = [b for b in swa_blocks if b.block_hash is not None and b.replay_deep]
    assert shallow and deep
    # Freed (LRU) blocks keep hash and flag, so a later lookup still sees them.
    assert all(b.ref_cnt == 0 for b in shallow + deep)
    blk = shallow[-1]
    assert manager.block_pool._maybe_evict_cached_block(blk)
    assert blk.block_hash is None and blk.replay_deep
    # A shallow copy still in the cache serves a hit at or below N - K.
    d = _request("d", _tokens(6000) + _tokens(5000, 1 << 20))
    _, hit = manager.get_computed_blocks(d)
    assert hit == 5888
    assert manager.num_replay_hit_shrinks == 0


def test_reset_prefix_cache_clears_flags(monkeypatch):
    manager = _manager(monkeypatch, K)
    a = _request("a", _tokens(10000))
    _prefill(manager, a)
    assert any(not b.replay_deep for b in manager.block_pool.blocks)
    manager.free(a)
    assert manager.reset_prefix_cache()
    assert all(b.replay_deep for b in manager.block_pool.blocks)


def _decode(manager: KVCacheManager, req: Request, num_tokens: int) -> None:
    req.append_output_token_ids(_tokens(num_tokens, 3 << 20))
    assert manager.allocate_slots(req, num_tokens) is not None
    req.num_computed_tokens += num_tokens


def test_decode_blocks_are_deep_and_preemption_recompute(monkeypatch):
    manager = _manager(monkeypatch, K)
    a = _request("a", _tokens(10000))
    _prefill(manager, a)
    _decode(manager, a, 600)
    for gid in range(4):
        assert all(
            deep for start, deep in _deep_flags(manager, a, gid) if start >= 10000
        )
    # Preempt and recompute: N = prompt + output; the hit's window lies in
    # decode-produced blocks.
    manager.free(a)
    a.num_computed_tokens = 0
    hit = _prefill(manager, a)
    assert hit == 10599 // MLA_BLOCK * MLA_BLOCK == 10496
    assert manager.num_replay_hit_shrinks == 0
    assert a.replay_exact_from == hit


def test_approximate_k_keeps_only_decode_blocks_deep(monkeypatch):
    manager = _manager(monkeypatch, 128)
    a = _request("a", _tokens(10000))
    _prefill(manager, a)
    assert a.replay_exact_from == 10000
    # Resend of A's prompt + one new turn: hit above N - 128 with a shallow
    # window -> recompute from floor_lcm(N - 128).
    r = _request("r", _tokens(10000) + _tokens(100, 1 << 20))
    _, hit = manager.get_computed_blocks(r)
    assert hit == (10100 - 128) // MLA_BLOCK * MLA_BLOCK == 9728
    assert manager.num_replay_hit_shrinks == 1
    _decode(manager, a, 600)
    for gid in range(4):
        for start, deep in _deep_flags(manager, a, gid):
            assert deep == (start >= 10000), (gid, start)


def test_chunked_prefill_hits(monkeypatch):
    manager = _manager(monkeypatch, K)
    a = _request("a", _tokens(10000))
    _prefill(manager, a, CHUNK)
    # The prompt's last boundary (the held window) starts >= P - 384, always
    # deep for K >= 2413 + 384: a resend + new turn hits it in full.
    c = _request("c", _tokens(10000) + _tokens(500, 1 << 20))
    _, hit = manager.get_computed_blocks(c)
    assert hit == 9984
    assert manager.num_replay_hit_shrinks == 0
    # A branch at 9300 finds the leftover window at 9216 (last chunk), which is
    # shallow (deep from 9341). The windows at or below N - K were reused
    # during A's prefill, so the shrink falls back to a full recompute.
    b = _request("b", _tokens(9300) + _tokens(500, 1 << 20))
    _, hit = manager.get_computed_blocks(b)
    assert hit == 0
    assert manager.num_replay_hit_shrinks == 1
    assert b.replay_exact_from == 9800 - K + MARGIN
    # A branch at 9500 finds the deep window at 9472.
    q = _request("q", _tokens(9500) + _tokens(500, 2 << 20))
    _, hit = manager.get_computed_blocks(q)
    assert hit == 9472
    assert manager.num_replay_hit_shrinks == 1
    assert q.replay_exact_from == 9472


def test_skip_reading_prefix_cache_sets_exact_from(monkeypatch):
    manager = _manager(monkeypatch, K)
    a = _request("a", _tokens(10000))
    a.skip_reading_prefix_cache = True
    _, hit = manager.get_computed_blocks(a)
    assert hit == 0
    assert a.replay_exact_from == 10000 - K + MARGIN


def test_window_tail_is_deep_bounds():
    def blk(deep, null=False):
        b = KVCacheBlock(7)
        b.replay_deep = deep
        b.is_null = null
        return b

    groups = [(0, 32, 128)]
    # H = 256: window [129, 256) -> blocks 4..7.
    blocks = [blk(False)] * 4 + [blk(True)] * 4
    assert ds41_replay.window_tail_is_deep((blocks,), groups, 256)
    blocks = [blk(False)] * 4 + [blk(False)] + [blk(True)] * 3
    assert not ds41_replay.window_tail_is_deep((blocks,), groups, 256)
    blocks = [blk(True, null=True)] * 4 + [blk(True, null=True)] + [blk(True)] * 3
    assert not ds41_replay.window_tail_is_deep((blocks,), groups, 256)
    assert ds41_replay.window_tail_is_deep(([],), groups, 256)
