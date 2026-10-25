#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 EXL3 franken-v100 on 8x V100-SXM2-32GB: PP2 x TP4 (one stage per NVLink quad, cut at
# layer 20), EXL3 routed experts in VRAM, packed FP4 compressed KV, native 1,048,576-token context, DSpark k=5,
# RAM-resident Engram. This is the live configuration the published numbers were measured on.
#
# Paths come from the environment (see ds41.env.example):
#   DS41_MODEL       pack directory (hf download philbert440/DeepSeek-V4.1-Flash-1017-EXL3-franken-v100)
#   DS41_ENGRAM_DIR  directory with the two official Engram shards (default: $DS41_MODEL)
#   DS41_PYTHON      python of the venv with the release wheel (default: python3 on PATH)
#   DS41_TREE        source tree instead of the wheel (optional)
#   DS41_LOG_DIR     server logs (default ./logs)
#   DS41_GPUS_A / DS41_GPUS_C   GPU indices or UUIDs of the two NVLink quads (default "0 1 2 3" / "4 5 6 7")
#   DS41_PORT        default 8000
# Options: --dry-run [--strict] runs every pre-flight check and prints the command without launching; --foreground;
# -- <extra vllm args>. Every VLLM_DS41_* knob below can be overridden from the environment.
here="$(dirname "$(readlink -f "$0")")"
source "$here/ds41_common.sh"

DS41_TOKENIZER=$DS41_MODEL
DS41_MEM_WEIGHTS[pp2]="${DS41_MEM_WEIGHTS_OVERRIDE:-27.77 25.29}"   # plan bytes per GPU + 0.87 GiB of buffers vLLM measured on the 1M run
DS41_MEM_RUNTIME[pp2]="1.40 1.40"     # measured: util budget - weights - available KV on the 1M run (1.33 GiB)
DS41_MEM_POOL256K_MIB[pp2]="${DS41_MEM_POOL256K_MIB_OVERRIDE:-100 100}"   # packed-KV pools are small; util is pinned at the cap and vLLM sizes KV itself
DS41_UTIL_CAP=0.945

# Engine features. Each one is behind its own knob and was gated against the fp32 oracle before it went live.
export VLLM_DS41_MOE_EXL3_PLAN=$DS41_MODEL/exl3_moe_plan.json
export VLLM_DS41_MOE_EXL3_BALANCE="${VLLM_DS41_MOE_EXL3_BALANCE:-$here/balance/v1-ab24}"   # per-layer routing counts: experts per rank balanced by load
export VLLM_DS41_HC_FUSED_SINKHORN="${VLLM_DS41_HC_FUSED_SINKHORN:-1}"
export VLLM_DS41_HC_FUSED="${VLLM_DS41_HC_FUSED:-1}"                      # fused hc_pre/hc_post (bit-exact) + rmsnorm/mixes (~1 ulp)
export VLLM_DS41_ATTN_FUSED_DECODE="${VLLM_DS41_ATTN_FUSED_DECODE:-1}"    # split-K fused decode attention over the packed KV
export VLLM_DS41_MOE_EXL3_KSPLIT="${VLLM_DS41_MOE_EXL3_KSPLIT:-4}"
export VLLM_DS41_MOE_EXL3_GROUP_MIN_TOKENS="${VLLM_DS41_MOE_EXL3_GROUP_MIN_TOKENS:-2}"   # grouped kernel (expert decoded once per <= 8 slots) from 2 rows
export VLLM_DS41_BATCH_INVARIANT_GEMM="${VLLM_DS41_BATCH_INVARIANT_GEMM:-1}"   # fixed-M GEMM buckets: lone prefills bit-exact, mixed-step KL 0.12 -> 0.013
export VLLM_DS41_INDEXER_CANDIDATE_SCORING="${VLLM_DS41_INDEXER_CANDIDATE_SCORING:-1}"   # reindex layers score candidate keys only
export VLLM_DS41_DEMAND_KV="${VLLM_DS41_DEMAND_KV:-1}"                  # physical KV pages mapped for live and prefix-cached blocks only
export VLLM_DS41_ATTN_GRAPH="${VLLM_DS41_ATTN_GRAPH:-1}"                  # per-layer decode graphs + static-buffer builders
export VLLM_DS41_HC_FUSED_MIXES="${VLLM_DS41_HC_FUSED_MIXES:-0}"          # 0 = exact mixes
export VLLM_DS41_HC_NORM_EXACT="${VLLM_DS41_HC_NORM_EXACT:-1}"
export VLLM_DS41_ATTN_OUTER_GRAPH="${VLLM_DS41_ATTN_OUTER_GRAPH:-0}"      # speed-neutral; off = simpler
export VLLM_DS41_BUILDER_GRAPHS="${VLLM_DS41_BUILDER_GRAPHS:-1}"          # builders replay their kernels from CUDA graphs (exact)
# Prefill and decode work after #1243:
export VLLM_DS41_REPLAY_ROWS="${VLLM_DS41_REPLAY_ROWS:-3072}"             # decoder replay: layers 21-39 run each request's last 3,072 rows (exact)
export VLLM_DS41_ATTN_GATHER_ONCE="${VLLM_DS41_ATTN_GATHER_ONCE:-1}"
export VLLM_DS41_ATTN_FUSED_PREFILL="${VLLM_DS41_ATTN_FUSED_PREFILL:-1}"  # fused WMMA sparse prefill attention (ds41_sparse_prefill.so)
export VLLM_DS41_MOE_EXL3_PREFILL_KERNEL="${VLLM_DS41_MOE_EXL3_PREFILL_KERNEL:-1}"   # tiled EXL3 prefill MoE kernel
export VLLM_DS41_MOE_EXL3_PREFILL_DENSE_MIN="${VLLM_DS41_MOE_EXL3_PREFILL_DENSE_MIN:-160}"
export VLLM_DS41_MOE_EXL3_TILED_LAYOUT="${VLLM_DS41_MOE_EXL3_TILED_LAYOUT:-1}"
export VLLM_DS41_MOE_EXL3_PREFILL_KSPLIT="${VLLM_DS41_MOE_EXL3_PREFILL_KSPLIT:-1}"
export VLLM_DS41_PP_PREFILL_PIPELINE="${VLLM_DS41_PP_PREFILL_PIPELINE:-1}"
export VLLM_DS41_ATTN_PRECAPTURE="${VLLM_DS41_ATTN_PRECAPTURE:-1}"
export VLLM_DS41_ATTN_NO_LAZY_CAPTURE="${VLLM_DS41_ATTN_NO_LAZY_CAPTURE:-1}"
export VLLM_DS41_ATTN_PRECAPTURE_RESERVE_MIB="${VLLM_DS41_ATTN_PRECAPTURE_RESERVE_MIB:-160}"
export VLLM_DS41_ATTN_CAND_DECODE_ROWS="${VLLM_DS41_ATTN_CAND_DECODE_ROWS:-2}"
export VLLM_DS41_DSPARK_PMIN="${VLLM_DS41_DSPARK_PMIN:-0}"
export VLLM_DS41_ENGRAM_RESIDENT="${VLLM_DS41_ENGRAM_RESIDENT:-1}"        # Engram tables in hugepage RAM (needs tables + 32 GiB available)
export VLLM_DS41_ENGRAM_IO_THREADS="${VLLM_DS41_ENGRAM_IO_THREADS:-4}"
export VLLM_DS41_MIXED_PREFILL_PP="${VLLM_DS41_MIXED_PREFILL_PP:-0}"
export VLLM_DS41_ENGRAM_BUILD_DIR="${VLLM_DS41_ENGRAM_BUILD_DIR:-$DS41_LOG_DIR/engram-build}"   # JIT build dir of the Engram host extension
export VLLM_DS41_SLICED_PREFILL="${VLLM_DS41_SLICED_PREFILL:-1}"          # layer-sliced prefill between decode steps
export VLLM_DS41_FOLD_DRAFTS="${VLLM_DS41_FOLD_DRAFTS:-1}"
export VLLM_DS41_DECODE_LANE="${VLLM_DS41_DECODE_LANE:-0}"
export VLLM_DS41_SLICE_POLICY="${VLLM_DS41_SLICE_POLICY:-adaptive}"
export VLLM_DS41_SLICE_DECODE_KEEP="${VLLM_DS41_SLICE_DECODE_KEEP:-0.85}"
export VLLM_DS41_SLICE_MARGIN_MS="${VLLM_DS41_SLICE_MARGIN_MS:-3}"
export VLLM_DS41_SLICE_S1_ATTN_PARTS="${VLLM_DS41_SLICE_S1_ATTN_PARTS:-auto}"
export VLLM_DS41_INDEXER_CAND_PREFILL="${VLLM_DS41_INDEXER_CAND_PREFILL:-auto}"
export VLLM_DS41_INDEXER_FUSED="${VLLM_DS41_INDEXER_FUSED:-1}"            # fused indexer kernels (ds41_indexer_fused.so)
export VLLM_DS41_DSPARK_TAP_SHIFT="${VLLM_DS41_DSPARK_TAP_SHIFT:-6}"      # DSpark tap captured as (mean * 2^-6) in fp16: no inf -> NaN drafts
export VLLM_DS41_INDEXER_FUSED_DECODE="${VLLM_DS41_INDEXER_FUSED_DECODE:-1}"   # fused indexer kernels on the decode path too

# Optional host tuning (needs passwordless sudo; skipped otherwise): CPU governor `performance` and SM clocks locked at
# 1530 MHz. V100 auto-boost dips between latency-bound decode bursts. DS41_HOST_TUNING=0 skips both.
if [[ "${DS41_HOST_TUNING:-1}" == 1 ]]; then
  if [ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" != "performance" ]; then
    for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo -n tee "$c" >/dev/null 2>&1; done
    echo "cpu governor : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  fi
  if nvidia-smi --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits | grep -qv "^1530$"; then
    sudo -n nvidia-smi -lgc 1530 >/dev/null 2>&1 && sudo -n nvidia-smi -ac 877,1530 >/dev/null 2>&1
    echo "gpu clocks   : $(nvidia-smi --query-gpu=clocks.applications.graphics --format=csv,noheader -i 0)"
  fi
fi

DS41_VARIANT=pp2tp4
DS41_STAGES="A C"
DS41_PARTITION="${DS41_PARTITION_OVERRIDE:-20,20}"
DS41_MAX_MODEL_LEN="${DS41_MAX_MODEL_LEN:-1048576}"
DS41_MAX_BATCHED_TOKENS="${DS41_MAX_BATCHED_TOKENS:-1024}"   # prefill chunk; 4096 does not fit 1M
DS41_GPU_UTIL="${DS41_GPU_UTIL_OVERRIDE:-0.943}"
DS41_SPILL=0
DS41_PORT="${DS41_PORT:-8000}"
DS41_EXTRA_ARGS=(
  --numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,16,24]}'
  --override-generation-config '{"temperature": 1.0, "top_p": 1.0}'
)
# DSpark speculative decoding, k=5, proposer in CUDA graphs. DS41_DSPARK=0 -> plain decode (async scheduling, 4 seqs).
DS41_DSPARK="${DS41_DSPARK:-1}"
if [[ "$DS41_DSPARK" == 1 ]]; then
  DS41_EXTRA_ARGS+=(--hf-overrides '{"dspark_block_size": 5, "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256, "dspark_n_routed_experts": 128, "dspark_n_activated_experts": 3, "n_mtp_layers": 3}'
                  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}')
  DS41_MAX_NUM_SEQS="${DS41_MAX_NUM_SEQS_OVERRIDE:-2}"
else
  DS41_MAX_NUM_SEQS="${DS41_MAX_NUM_SEQS_OVERRIDE:-4}"
  DS41_EXTRA_ARGS+=(--async-scheduling)
  export VLLM_DS41_ENGRAM_DEFERRED="${VLLM_DS41_ENGRAM_DEFERRED:-1}"
fi
ds41_main "$@"
# CUDA context init drops the cards back to the default application clock after the pre-launch lock; re-apply once
# /health is up.
if [[ "${DS41_HOST_TUNING:-1}" == 1 ]] && sudo -n nvidia-smi -lgc 1530 >/dev/null 2>&1 && sudo -n nvidia-smi -ac 877,1530 >/dev/null 2>&1; then
  echo "gpu clocks   : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0) SM (re-locked after ready)"
fi
