# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the prefix-anchored sliding-window attention mask.

Semantics under test: keys inside the prompt/prefix (``kv < anchor_len``)
stay globally visible; generated keys are additionally restricted to a
sliding window (``q_abs - kv < window``); everything is AND-ed with the
causal mask.

The CPU tests validate the mask formula (the same one the masked kernels
implement) against plain causal attention with a torch reference.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn_v100 import (
    FlashAttnV100Impl,
    FlashAttnV100MetadataBuilder,
)


def anchored_swa_mask(
    seq_len: int, anchor_len: int, window: int, device: str = "cpu"
) -> torch.Tensor:
    """Boolean keep-mask [seq_len, seq_len]; True = attend.

    Mirrors the kernel formula:
    ``causal AND (kv < anchor_len OR q_abs - kv < window)``.
    """
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    causal = kv <= q
    in_prefix = kv < anchor_len
    in_window = (q - kv) < window
    return causal & (in_prefix | in_window)


def causal_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    return kv <= q


def ref_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Plain fp32 reference attention. q/k/v: [heads, seq, dim]."""
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("hqd,hkd->hqk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,hkd->hqd", probs, v.float())


def _flash_impl(window: int | None) -> FlashAttnV100Impl:
    impl = object.__new__(FlashAttnV100Impl)
    object.__setattr__(impl, "prefix_anchored_decode_window", window)
    object.__setattr__(impl, "attn_type", AttentionType.DECODER)
    object.__setattr__(impl, "kv_cache_dtype", "auto")
    return impl


@pytest.mark.cpu_test
def test_backend_default_off_does_not_inspect_metadata():
    class NoMetadataAccess:
        def __getattr__(self, name: str):
            raise AssertionError(f"default-off path inspected {name}")

    metadata = cast(Any, NoMetadataAccess())
    assert _flash_impl(None)._anchored_swa_params(metadata) == (None, 0)


@pytest.mark.cpu_test
def test_backend_enabled_fails_closed_without_matching_metadata():
    with pytest.raises(RuntimeError, match="metadata does not match"):
        _flash_impl(128)._anchored_swa_params(cast(Any, object()))


@pytest.mark.cpu_test
def test_metadata_builder_attaches_prefix_lengths_for_eager_path():
    builder = object.__new__(FlashAttnV100MetadataBuilder)
    object.__setattr__(builder, "decode_sliding_window", 128)
    object.__setattr__(builder, "device", torch.device("cpu"))
    object.__setattr__(
        builder,
        "persistent_prefix_anchor_lens",
        torch.empty(4, dtype=torch.int32),
    )
    metadata = SimpleNamespace()
    common = SimpleNamespace(
        prefix_anchor_lens=torch.tensor([7, 11], dtype=torch.int64),
        num_reqs=2,
    )

    builder._attach_prefix_anchored_metadata(
        cast(Any, metadata),
        cast(Any, common),
    )

    assert metadata.decode_sliding_window == 128
    assert metadata.prefix_anchor_lens.dtype == torch.int32
    assert metadata.prefix_anchor_lens.tolist() == [7, 11]


@pytest.mark.cpu_test
def test_metadata_builder_fails_closed_without_prefix_lengths():
    builder = object.__new__(FlashAttnV100MetadataBuilder)
    object.__setattr__(builder, "decode_sliding_window", 128)
    object.__setattr__(builder, "device", torch.device("cpu"))
    object.__setattr__(
        builder,
        "persistent_prefix_anchor_lens",
        torch.empty(4, dtype=torch.int32),
    )

    with pytest.raises(RuntimeError, match="requires per-request prefix lengths"):
        builder._attach_prefix_anchored_metadata(
            cast(Any, SimpleNamespace()),
            cast(Any, SimpleNamespace(prefix_anchor_lens=None, num_reqs=2)),
        )


@pytest.mark.cpu_test
def test_mask_equals_causal_while_within_window():
    # While the number of generated tokens is <= window, no key falls in the
    # gap, so the mask must be exactly the causal mask.
    anchor, window = 6, 8
    for seq_len in range(1, anchor + window + 1):
        assert torch.equal(
            anchored_swa_mask(seq_len, anchor, window),
            causal_mask(seq_len),
        ), f"seq_len={seq_len}"


@pytest.mark.cpu_test
def test_mask_prunes_gap_beyond_window():
    anchor, window = 6, 8
    seq_len = anchor + window + 3  # 3 generated keys fall out of the window
    mask = anchored_swa_mask(seq_len, anchor, window)
    causal = causal_mask(seq_len)
    assert not torch.equal(mask, causal)
    # The last query keeps: the full prefix and the window of recent keys...
    last = mask[-1]
    assert last[:anchor].all(), "prefix must stay globally visible"
    assert last[seq_len - window :].all(), "recent window must be visible"
    # ...and drops exactly the gap keys in between.
    gap = last[anchor : seq_len - window]
    assert not gap.any(), "gap keys must be masked"
    # Prefix queries (inside the prompt) are never affected.
    assert torch.equal(mask[: anchor + window], causal[: anchor + window])


@pytest.mark.cpu_test
def test_attention_output_equivalence_and_divergence():
    torch.manual_seed(0)
    heads, dim = 2, 32
    anchor, window = 6, 8

    # Within the window: identical outputs.
    seq_len = anchor + window
    q, k, v = (torch.randn(heads, seq_len, dim) for _ in range(3))
    out_anchored = ref_attention(q, k, v, anchored_swa_mask(seq_len, anchor, window))
    out_causal = ref_attention(q, k, v, causal_mask(seq_len))
    torch.testing.assert_close(out_anchored, out_causal)

    # Beyond the window: the late queries must diverge from full attention
    # (negative control proving the mask actually removes information).
    seq_len = anchor + 2 * window
    q, k, v = (torch.randn(heads, seq_len, dim) for _ in range(3))
    out_anchored = ref_attention(q, k, v, anchored_swa_mask(seq_len, anchor, window))
    out_causal = ref_attention(q, k, v, causal_mask(seq_len))
    assert not torch.allclose(out_anchored[:, -1], out_causal[:, -1])
    # Queries that see no gap still match exactly.
    torch.testing.assert_close(
        out_anchored[:, : anchor + window], out_causal[:, : anchor + window]
    )
