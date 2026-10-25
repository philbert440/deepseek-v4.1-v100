# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark proposer built on the existing non-causal DFlash execution path."""

import os
import torch
from typing_extensions import override

from vllm import envs
from vllm.config import VllmConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode import ds41_dspark_trace


from vllm.logger import init_logger
logger = init_logger(__name__)

PMIN_KNOB = "VLLM_DS41_DSPARK_PMIN"
MAX_ROWS_KNOB = "VLLM_DS41_DSPARK_MAX_ROWS"
PMIN_MAX_BATCH_KNOB = "VLLM_DS41_DSPARK_PMIN_MAX_BATCH"


def ds41_dspark_pmin_max_batch() -> int:
    """``VLLM_DS41_DSPARK_PMIN_MAX_BATCH`` (int >= 0, default 0 = no limit): apply the p_min stop only while at most
    N requests are DECODING on this worker (live requests whose prompt is fully computed, scheduled this step or
    not); with more, every request verifies the full k drafts. Not the step's own request count: under the PP2
    batch queue two decoders ride alternate batches with R=1 each (decode bench v3 D2b, 2026-10-24), so a per-step
    gate never saw B=2. With two requests truncating independently the verify row count lands anywhere in 2..12,
    outside the precaptured verify shapes (2, 7, 12) (decode bench v2: B=2 lost ~30% at p_min 0.5)."""
    from vllm.models.deepseek_v41 import knobs

    return knobs.env_int(PMIN_MAX_BATCH_KNOB, 0, minimum=0)


def ds41_dspark_adaptive_knobs() -> tuple[float, int]:
    """(p_min, max_rows) of the adaptive DSpark draft length; (0, 0) = off = fixed k (the default).

    ``VLLM_DS41_DSPARK_PMIN`` (float in [0, 1]): truncate each request's draft at the first position whose drafter top-1
    probability (softmax of the drafter's own logits incl. the Markov bias, temperature 1) is below p_min
    (ik_llama.cpp #2718 rule; 0.75 there). ``VLLM_DS41_DSPARK_MAX_ROWS`` (int >= 0): verify-row ceiling per step,
    draft <= max_rows // B - 1 for a B-request batch (exllamav3 85cae3e7). Verification stays exact: the target accepts
    exactly the drafts it would have produced, so greedy output does not change; only the rows per verify step do.
    Requires synchronous scheduling (the per-request lengths reach the scheduler through take_draft_token_ids)."""
    from vllm.models.deepseek_v41 import knobs

    pmin = knobs.env_float(PMIN_KNOB, 0.0, minimum=0.0, maximum=1.0)
    max_rows = knobs.env_int(MAX_ROWS_KNOB, 0, minimum=0)
    return pmin, max_rows


def ds41_dspark_adaptive_enabled() -> bool:
    pmin, max_rows = ds41_dspark_adaptive_knobs()
    return pmin > 0.0 or max_rows > 0


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)
        self._ds41_runner = runner        # the base proposer does not keep it; the SM70 capture context needs its block table
        if self.use_local_argmax_reduction:
            raise ValueError(
                "DSpark cannot use local argmax reduction because its replicated "
                "Markov bias must be added to full-vocabulary logits."
            )
        self._anchor_indices = (
            torch.arange(self.max_batch_size, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._confidence_threshold = self.speculative_config.dspark_confidence_threshold
        self._confidence_temperatures: torch.Tensor | None = None
        if self._confidence_threshold > 0.0:
            temperatures = self.speculative_config.dspark_confidence_temperatures
            if temperatures is None:
                temperatures = [1.0] * self.num_speculative_tokens
            self._confidence_temperatures = torch.tensor(
                temperatures,
                dtype=torch.float32,
                device=device,
            ).view(1, -1)
        configured_cap = self.speculative_config.dspark_max_verification_tokens
        self._max_verification_tokens = (
            self.num_speculative_tokens if configured_cap is None else configured_cap
        )
        self.confidence_scheduling_enabled = (
            self._confidence_threshold > 0.0
            or self._max_verification_tokens < self.num_speculative_tokens
        )
        # A fixed cap needs verification lengths but not a confidence
        # projection. Collect logits only when the calibrated threshold or the
        # explicit alignment diagnostic consumes them.
        self.collect_confidence_logits = (
            self._confidence_threshold > 0.0 or envs.VLLM_SPEC_DUMP_ALIGNMENT
        )
        self._last_confidence_logits: torch.Tensor | None = None
        self._last_verification_lengths: torch.Tensor | None = None
        self._ds41_pmin, self._ds41_max_rows = ds41_dspark_adaptive_knobs()
        self._ds41_pmin_max_batch = ds41_dspark_pmin_max_batch()
        # VLLM_DS41_DSPARK_TRACE (debug, default off): keep the pos0 top-5 for the runner-side per-step trace
        self._ds41_trace_on = ds41_dspark_trace.trace_path() is not None
        if self._ds41_pmin > 0.0 or self._ds41_max_rows > 0:
            logger.info("DSpark adaptive draft length: %s=%g %s=%d %s=%d (k=%d)", PMIN_KNOB, self._ds41_pmin,
                        MAX_ROWS_KNOB, self._ds41_max_rows, PMIN_MAX_BATCH_KNOB, self._ds41_pmin_max_batch,
                        self.num_speculative_tokens)

    def take_last_confidence_logits(self) -> torch.Tensor | None:
        confidence_logits = self._last_confidence_logits
        self._last_confidence_logits = None
        return confidence_logits

    def take_last_verification_lengths(self) -> torch.Tensor | None:
        verification_lengths = self._last_verification_lengths
        self._last_verification_lengths = None
        return verification_lengths

    def _select_verification_lengths(
        self,
        confidence_logits: torch.Tensor,
    ) -> torch.Tensor:
        temperatures = getattr(self, "_confidence_temperatures", None)
        if temperatures is None:
            temperatures = torch.ones(
                (1, confidence_logits.shape[1]),
                dtype=torch.float32,
                device=confidence_logits.device,
            )
        threshold = float(getattr(self, "_confidence_threshold", 0.0))
        cap = int(
            getattr(
                self,
                "_max_verification_tokens",
                confidence_logits.shape[1],
            )
        )
        if threshold <= 0.0:
            return torch.full(
                (confidence_logits.shape[0],),
                cap,
                dtype=torch.int32,
                device=confidence_logits.device,
            )

        conditional_probs = torch.sigmoid(confidence_logits.float() / temperatures)
        confident_prefix = (
            (conditional_probs >= threshold).to(torch.int32).cumprod(dim=1)
        )
        return confident_prefix.sum(dim=1).clamp_max_(cap).to(torch.int32)

    def _ds41_adaptive_lengths(
        self,
        confident_steps: list[torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Per-request draft lengths [batch_size] int32 from the p_min stop and the row ceiling; None when both are off.
        Pure GPU ops on tensors the step already produced: no host sync (the runner copies the lengths to the host with
        the draft ids, on the same copy stream and event)."""
        pmin = float(getattr(self, "_ds41_pmin", 0.0))
        max_rows = int(getattr(self, "_ds41_max_rows", 0))
        if pmin <= 0.0 and max_rows <= 0:
            return None
        k = int(self.num_speculative_tokens)
        if pmin > 0.0 and confident_steps:
            lengths = torch.stack(confident_steps, dim=1).to(torch.int32).cumprod(dim=1).sum(dim=1, dtype=torch.int32)
        else:
            lengths = torch.full((batch_size,), k, dtype=torch.int32, device=device)
        if max_rows > 0:
            num_reqs = self._ds41_live_reqs(batch_size)
            lengths.clamp_(max=max(0, min(k, max_rows // max(num_reqs, 1) - 1)))
        return lengths

    def _ds41_live_reqs(self, batch_size: int) -> int:
        """Requests in this drafter batch (= the next verify step's R): the runner's live count, since under drafter
        CUDA graphs the sampled rows can be padded to more blocks than there are requests."""
        input_batch = getattr(getattr(self, "_ds41_runner", None), "input_batch", None)
        return int(getattr(input_batch, "num_reqs", 0) or 0) or batch_size

    def _ds41_decoding_reqs(self, batch_size: int) -> int:
        """Requests decoding on this worker: every cached request (the runner's ``requests``, finished ones are
        dropped when the scheduler reports them) whose prompt is fully computed, whether or not it is in this step;
        at least this step's own request count."""
        live = self._ds41_live_reqs(batch_size)
        requests = getattr(getattr(self, "_ds41_runner", None), "requests", None)
        if not requests:
            return live
        decoding = 0
        for state in requests.values():
            prompt = getattr(state, "num_prompt_tokens", None)
            if prompt is None or int(getattr(state, "num_computed_tokens", 0)) >= int(prompt):
                decoding += 1
        return max(live, decoding)

    def _ds41_pmin_active(self, batch_size: int) -> float:
        """p_min for this step: the knob, or 0 (full k drafts) while more than VLLM_DS41_DSPARK_PMIN_MAX_BATCH requests
        are decoding."""
        pmin = float(getattr(self, "_ds41_pmin", 0.0))
        max_batch = int(getattr(self, "_ds41_pmin_max_batch", 0))
        if pmin > 0.0 and max_batch > 0 and self._ds41_decoding_reqs(batch_size) > max_batch:
            return 0.0
        return pmin

    @override
    def dummy_run(self, num_tokens: int, use_cudagraphs: bool = True, is_graph_capturing: bool = False, **kw) -> None:
        if True:   # always (also for the eager force-static test); harmless when not capturing
            # the SM70 drafter attention captures its real path from static inputs (see sm70/dspark.py)
            from vllm.models.deepseek_v41.sm70 import dspark as sm70_dspark
            # layer names per KV-cache group from the runner's attention groups (present once the attention backends
            # exist; during the memory-profiling pass they may not be -> no context, the drafter captures its zero path
            # there, and those profiling graphs are discarded anyway)
            ctx = None
            profiling = bool(getattr(self._ds41_runner, "_ds41_cg_mem_profiling", False))
            try:
                if profiling:            # memory-profiling pass: a stub KV cache; capture the zero path (discarded graphs)
                    raise AttributeError("profiling pass")
                groups = self._ds41_runner.attn_groups
                ctx = {}
                for gid in self._draft_kv_cache_group_ids():
                    bt = self._ds41_runner.input_batch.block_table[gid].block_table.gpu
                    bs = int(self._dflash_block_size_by_gid.get(gid, self.block_size))
                    for g in groups[gid]:
                        for name in g.layer_names:
                            ctx[name] = (bt, bs)
            except (AttributeError, IndexError) as e:
                logger.info("DSpark capture context unavailable (%s): zero-path capture (profiling?)", e)
                ctx = None
            if os.environ.get("DS41_DRAFT_NO_STATIC") == "1":
                ctx = None            # diagnostic: capture the drafter attention's ZERO path (bisect graph-only faults)
            sm70_dspark.STATIC_CAPTURE_CTX = ctx
            if profiling and use_cudagraphs and os.environ.get("DS41_DRAFT_PROFILE_EAGER", "1") == "1":
                # The memory-profiling pass captured the drafter against the stub KV cache and REPLAYED it; at 2+
                # sequences that replay died with an illegal address (dspark-graph-b2h, 2026-10-20) although the same
                # rows run eagerly. Those graphs are discarded after profiling, so run the drafter eagerly there and let
                # only the real capture (real caches, static context) build graphs. The memory estimate loses the
                # drafter's small graph-pool share.
                use_cudagraphs = False
        return super().dummy_run(num_tokens, use_cudagraphs=use_cudagraphs, is_graph_capturing=is_graph_capturing, **kw)

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logits: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del spec_step_idx

        num_rows = hidden_states.shape[0]
        batch_size, remainder = divmod(num_rows, self.num_speculative_tokens)
        if remainder:
            raise ValueError(
                "DSpark sample rows must be divisible by "
                f"num_speculative_tokens={self.num_speculative_tokens}, got {num_rows}."
            )
        if logits is None:
            logits = self.model.compute_logits(hidden_states)
        base_logits = logits.view(batch_size, self.num_speculative_tokens, -1)

        # Read anchors from the persistent expanded-query buffer. The external
        # next_token_ids tensor is not guaranteed to retain a stable address or
        # contents across asynchronous scheduling and CUDA Graph replay.
        prev = self.input_ids[self._anchor_indices[:batch_size]].to(torch.long)
        import os as _os
        if _os.environ.get("DS41_DRAFT_CAPTURE_LOG") and getattr(self, "_ptr_logged", 0) < 3:
            self._ptr_logged = getattr(self, "_ptr_logged", 0) + 1
            gid = self._first_draft_kv_cache_group_id(); cad = (self._dflash_common_attn_metadata_by_gid or {}).get(gid)
            from vllm.logger import init_logger
            init_logger(__name__).info("DSpark proposer step: input_ids=%#x positions=%#x bt(meta)=%s bt(static)=%#x anchors=%s pos[:10]=%s",
                                       self.input_ids.data_ptr(), self.positions.data_ptr(),
                                       f"{cad.block_table_tensor.data_ptr():#x} {tuple(cad.block_table_tensor.shape)}" if cad is not None else None,
                                       0,
                                       prev.tolist(), self.positions[:10].tolist())
        draft_tokens: list[torch.Tensor] = []
        self._last_confidence_logits = None
        self._last_verification_lengths = None
        collect_confidence_logits = bool(
            getattr(self, "collect_confidence_logits", False)
        )
        markov_embeds: list[torch.Tensor] | None = (
            [] if collect_confidence_logits else None
        )
        draft_probs: list[torch.Tensor] | None = None
        pmin = self._ds41_pmin_active(batch_size)
        confident_steps: list[torch.Tensor] | None = [] if pmin > 0.0 else None
        from vllm.models.deepseek_v41.sm70.dspark import dump_on, dump_put
        dumping = dump_on()
        if dumping:
            dump_put("prop_prev0", prev); dump_put("prop_base_logits_top5", torch.topk(base_logits.float(), 5, dim=-1).indices)
            dump_put("prop_static_vocab", self._static_draft_vocab is not None)
        for step in range(self.num_speculative_tokens):
            markov_embed = self.model.markov_embed(prev)
            if markov_embeds is not None:
                markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            step_logits = base_logits[:, step] + bias
            if step == 0 and getattr(self, "_ds41_trace_on", False):
                ds41_dspark_trace.stash_pos0_top5(self, step_logits)
            if confident_steps is not None:
                # drafter top-1 probability of this position (ik_llama p_min stop): max softmax = exp(max - logsumexp)
                lf = step_logits.float()
                top1 = (lf.amax(dim=-1) - torch.logsumexp(lf, dim=-1)).exp_()
                confident_steps.append(top1 >= pmin)
            sampled, step_probs = self._sample_from_logits(
                step_logits,
                sampling_metadata,
            )
            if dumping:
                dump_put(f"prop_step{step}_prev", prev); dump_put(f"prop_step{step}_embed", markov_embed)
                dump_put(f"prop_step{step}_bias_top5", torch.topk(bias.float(), 5, dim=-1).indices); dump_put(f"prop_step{step}_bias_absmax", bias.float().abs().max())
                dump_put(f"prop_step{step}_logits_top5", torch.topk(step_logits.float(), 5, dim=-1).indices); dump_put(f"prop_step{step}_sampled", sampled)
            prev = self.model.map_draft_to_target(sampled)
            draft_tokens.append(prev)
            if step_probs is not None:
                if draft_probs is None:
                    draft_probs = []
                draft_probs.append(step_probs)

        if markov_embeds is not None:
            block_hidden_states = hidden_states.view(
                batch_size, self.num_speculative_tokens, -1
            )
            self._last_confidence_logits = self.model.confidence_logits(
                block_hidden_states,
                torch.stack(markov_embeds, dim=1),
            )
        if bool(getattr(self, "confidence_scheduling_enabled", False)):
            if self._last_confidence_logits is None:
                self._last_verification_lengths = torch.full(
                    (batch_size,),
                    int(self._max_verification_tokens),
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
            else:
                self._last_verification_lengths = self._select_verification_lengths(
                    self._last_confidence_logits
                )
        adaptive = self._ds41_adaptive_lengths(confident_steps, batch_size, hidden_states.device)
        if adaptive is not None:
            prev_lengths = self._last_verification_lengths
            self._last_verification_lengths = (
                adaptive if prev_lengths is None else torch.minimum(prev_lengths.to(torch.int32), adaptive)
            )
        flat_tokens = torch.stack(draft_tokens, dim=1).reshape(-1)
        if dumping:                                  # graph mode: the drafter's in-graph dumps never run; flush the proposer's
            from vllm.models.deepseek_v41.sm70.dspark import dump_flush
            dump_flush(self.model)
        if draft_probs is None:
            return flat_tokens, None
        if len(draft_probs) != self.num_speculative_tokens:
            raise RuntimeError("DSpark produced incomplete draft probabilities.")
        flat_probs = torch.stack(draft_probs, dim=1).flatten(0, 1).contiguous()
        return flat_tokens, flat_probs
