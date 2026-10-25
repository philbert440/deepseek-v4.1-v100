# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TypeAlias

import torch

from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DraftProbTokenIds: TypeAlias = Sequence[Sequence[int]] | torch.Tensor


def _assert_cached_tokens_match(
    cached_tokens: torch.Tensor,
    expected_tokens: torch.Tensor,
    req_id: str,
) -> None:
    if cached_tokens.shape != expected_tokens.shape:
        raise RuntimeError(
            "Cached draft probability token ids do not match verifier "
            f"draft token shape for request {req_id}: "
            f"cached={tuple(cached_tokens.shape)}, "
            f"expected={tuple(expected_tokens.shape)}."
        )
    matches = (cached_tokens == expected_tokens).all()
    if cached_tokens.device.type == "cuda":
        # Avoid a CPU synchronization in the decode hot path. A mismatch here
        # is an invariant violation that would corrupt rejection sampling.
        torch._assert_async(matches)
    elif not bool(matches.item()):
        raise RuntimeError(
            "Cached draft probability token ids do not match verifier "
            f"draft tokens for request {req_id}; refusing to use "
            "misaligned draft probabilities."
        )


def clone_draft_prob_token_ids(
    draft_token_ids: Sequence[Sequence[int]] | torch.Tensor,
) -> DraftProbTokenIds:
    if isinstance(draft_token_ids, torch.Tensor):
        return draft_token_ids.detach().clone()
    return [list(token_ids) for token_ids in draft_token_ids]


def get_aligned_draft_probs(
    *,
    req_ids: Sequence[str],
    draft_probs: torch.Tensor | None,
    draft_prob_req_ids: Sequence[str] | None,
    draft_prob_token_ids: DraftProbTokenIds | None,
    spec_decode_metadata: SpecDecodeMetadata,
) -> torch.Tensor | None:
    if draft_probs is None or draft_prob_req_ids is None:
        return None

    if draft_prob_token_ids is None:
        raise RuntimeError(
            "Cached draft probabilities are missing their draft token ids; "
            "cannot verify draft token/probability alignment."
        )
    if draft_probs.ndim != 3:
        raise RuntimeError(
            "Cached draft probabilities must have shape "
            "[num_requests, max_draft_tokens, vocab_size], got "
            f"{tuple(draft_probs.shape)}."
        )
    if draft_probs.shape[0] != len(draft_prob_req_ids):
        raise RuntimeError(
            "Cached draft probability row count does not match request ids: "
            f"rows={draft_probs.shape[0]}, req_ids={len(draft_prob_req_ids)}."
        )
    if isinstance(draft_prob_token_ids, torch.Tensor):
        if draft_prob_token_ids.ndim != 2:
            raise RuntimeError(
                "Cached draft probability token ids must have shape "
                "[num_requests, max_draft_tokens], got "
                f"{tuple(draft_prob_token_ids.shape)}."
            )
        if draft_prob_token_ids.shape[0] != len(draft_prob_req_ids):
            raise RuntimeError(
                "Cached draft probability token rows do not match request ids: "
                f"rows={draft_prob_token_ids.shape[0]}, "
                f"req_ids={len(draft_prob_req_ids)}."
            )
    elif len(draft_prob_token_ids) != len(draft_prob_req_ids):
        raise RuntimeError(
            "Cached draft probability token rows do not match request ids: "
            f"rows={len(draft_prob_token_ids)}, "
            f"req_ids={len(draft_prob_req_ids)}."
        )

    if len(req_ids) != len(spec_decode_metadata.num_draft_tokens):
        raise RuntimeError(
            "Spec decode draft metadata batch size does not match current "
            f"requests: metadata={len(spec_decode_metadata.num_draft_tokens)}, "
            f"req_ids={len(req_ids)}."
        )
    expected_num_draft_tokens = sum(spec_decode_metadata.num_draft_tokens)
    if spec_decode_metadata.draft_token_ids.numel() != expected_num_draft_tokens:
        raise RuntimeError(
            "Spec decode draft token count does not match per-request counts: "
            f"draft_token_ids={spec_decode_metadata.draft_token_ids.numel()}, "
            f"num_draft_tokens={expected_num_draft_tokens}."
        )

    row_by_req_id = {req_id: idx for idx, req_id in enumerate(draft_prob_req_ids)}
    draft_probs_rows: list[torch.Tensor] = []
    draft_token_offset = 0
    for req_id, num_draft in zip(req_ids, spec_decode_metadata.num_draft_tokens):
        if num_draft == 0:
            continue
        row_idx = row_by_req_id.get(req_id)
        if row_idx is None:
            raise RuntimeError(
                f"Missing cached draft probabilities for request {req_id}; "
                "cannot verify draft token/probability alignment."
            )
        if draft_probs.shape[1] < num_draft:
            raise RuntimeError(
                "Cached draft probabilities do not have enough draft "
                f"positions for request {req_id}: "
                f"available={draft_probs.shape[1]}, needed={num_draft}."
            )

        expected_tokens = spec_decode_metadata.draft_token_ids[
            draft_token_offset : draft_token_offset + num_draft
        ]
        if isinstance(draft_prob_token_ids, torch.Tensor):
            if draft_prob_token_ids.shape[1] < num_draft:
                raise RuntimeError(
                    "Cached draft probability token ids do not have enough "
                    f"positions for request {req_id}: "
                    f"available={draft_prob_token_ids.shape[1]}, "
                    f"needed={num_draft}."
                )
            cached_tokens = draft_prob_token_ids[row_idx, :num_draft].to(
                device=expected_tokens.device,
                dtype=expected_tokens.dtype,
            )
        else:
            cached_token_row = draft_prob_token_ids[row_idx]
            if len(cached_token_row) < num_draft:
                raise RuntimeError(
                    "Cached draft probability token ids do not have enough "
                    f"positions for request {req_id}: "
                    f"available={len(cached_token_row)}, needed={num_draft}."
                )
            cached_tokens = torch.tensor(
                cached_token_row[:num_draft],
                device=expected_tokens.device,
                dtype=expected_tokens.dtype,
            )
        _assert_cached_tokens_match(cached_tokens, expected_tokens, req_id)

        draft_probs_rows.append(draft_probs[row_idx, :num_draft])
        draft_token_offset += num_draft

    if not draft_probs_rows:
        return None
    return torch.cat(draft_probs_rows, dim=0).contiguous()


def get_aligned_draft_scalar_values(
    *,
    req_ids: Sequence[str],
    values: torch.Tensor | None,
    value_req_ids: Sequence[str] | None,
    value_token_ids: DraftProbTokenIds | None,
    spec_decode_metadata: SpecDecodeMetadata,
) -> torch.Tensor | None:
    """Align a cached per-draft scalar tensor to verifier request order."""
    if values is None or value_req_ids is None:
        return None
    if values.ndim != 2:
        raise RuntimeError(
            "Cached draft scalar values must have shape "
            f"[num_requests, max_draft_tokens], got {tuple(values.shape)}."
        )
    if values.shape[0] != len(value_req_ids):
        raise RuntimeError(
            "Cached draft scalar row count does not match request ids: "
            f"rows={values.shape[0]}, req_ids={len(value_req_ids)}."
        )
    if value_token_ids is None:
        raise RuntimeError(
            "Cached draft scalar values are missing their draft token ids; "
            "cannot verify alignment."
        )
    if isinstance(value_token_ids, torch.Tensor):
        if value_token_ids.ndim != 2 or value_token_ids.shape[0] != len(value_req_ids):
            raise RuntimeError(
                "Cached draft scalar token ids must have one 2-D row per "
                f"request, got shape={tuple(value_token_ids.shape)}, "
                f"req_ids={len(value_req_ids)}."
            )
    elif len(value_token_ids) != len(value_req_ids):
        raise RuntimeError(
            "Cached draft scalar token row count does not match request ids: "
            f"rows={len(value_token_ids)}, req_ids={len(value_req_ids)}."
        )
    if len(req_ids) != len(spec_decode_metadata.num_draft_tokens):
        raise RuntimeError(
            "Spec decode draft metadata batch size does not match current "
            f"requests: metadata={len(spec_decode_metadata.num_draft_tokens)}, "
            f"req_ids={len(req_ids)}."
        )
    expected_num_draft_tokens = sum(spec_decode_metadata.num_draft_tokens)
    if spec_decode_metadata.draft_token_ids.numel() != expected_num_draft_tokens:
        raise RuntimeError(
            "Spec decode draft token count does not match per-request counts: "
            f"draft_token_ids={spec_decode_metadata.draft_token_ids.numel()}, "
            f"num_draft_tokens={expected_num_draft_tokens}."
        )

    row_by_req_id = {req_id: idx for idx, req_id in enumerate(value_req_ids)}
    aligned_rows: list[torch.Tensor] = []
    draft_token_offset = 0
    for req_id, num_draft in zip(req_ids, spec_decode_metadata.num_draft_tokens):
        if num_draft == 0:
            continue
        row_idx = row_by_req_id.get(req_id)
        if row_idx is None:
            raise RuntimeError(
                f"Missing cached draft scalar values for request {req_id}; "
                "cannot verify alignment."
            )
        if values.shape[1] < num_draft:
            raise RuntimeError(
                "Cached draft scalar values do not have enough positions for "
                f"request {req_id}: available={values.shape[1]}, needed={num_draft}."
            )
        expected_tokens = spec_decode_metadata.draft_token_ids[
            draft_token_offset : draft_token_offset + num_draft
        ]
        if isinstance(value_token_ids, torch.Tensor):
            if value_token_ids.shape[1] < num_draft:
                raise RuntimeError(
                    "Cached draft scalar token ids do not have enough positions "
                    f"for request {req_id}: available={value_token_ids.shape[1]}, "
                    f"needed={num_draft}."
                )
            cached_tokens = value_token_ids[row_idx, :num_draft].to(
                device=expected_tokens.device,
                dtype=expected_tokens.dtype,
            )
        else:
            cached_token_row = value_token_ids[row_idx]
            if len(cached_token_row) < num_draft:
                raise RuntimeError(
                    "Cached draft scalar token ids do not have enough positions "
                    f"for request {req_id}: available={len(cached_token_row)}, "
                    f"needed={num_draft}."
                )
            cached_tokens = torch.tensor(
                cached_token_row[:num_draft],
                device=expected_tokens.device,
                dtype=expected_tokens.dtype,
            )
        _assert_cached_tokens_match(cached_tokens, expected_tokens, req_id)
        aligned_rows.append(values[row_idx, :num_draft])
        draft_token_offset += num_draft

    if not aligned_rows:
        return None
    return torch.cat(aligned_rows, dim=0).contiguous()
