# shellcheck shell=bash
# DeepSeek-V4.1-Flash on 8x V100-SXM2 (adapted from the #1031 launcher; the board map comes from the environment) --
# shared launch logic for serve_ds41.sh.
# Sourced, not executed. Every check fails loud; --dry-run reports what WOULD refuse and prints the full command.
#
# Each variant script sets, before calling ds41_main "$@":
#   DS41_VARIANT       short name (log/tmux/port defaults)
#   DS41_STAGES        space-separated board letters in pipeline order, e.g. "A C" or "A C B"
#   DS41_PARTITION     VLLM_PP_LAYER_PARTITION, e.g. "20,20" or "14,14,12"
#   DS41_MAX_MODEL_LEN, DS41_MAX_NUM_SEQS, DS41_MAX_BATCHED_TOKENS, DS41_GPU_UTIL, DS41_PORT
#   DS41_SPILL         routed experts spilled to pinned host RAM per layer (0 = none)
#   DS41_EXTRA_ARGS    array of extra vllm arguments (may be empty)

set -o pipefail

DS41_TREE="${DS41_TREE:-}"                                            # source tree (editable install); empty = installed wheel
DS41_PYTHON="${DS41_PYTHON:-}"                                        # default: $DS41_TREE/.venv/bin/python, else python3 on PATH
DS41_MODEL="${DS41_MODEL:?set DS41_MODEL to the pack directory}"
DS41_TOKENIZER="${DS41_TOKENIZER:-$DS41_MODEL}"
DS41_ENGRAM_DIR="${DS41_ENGRAM_DIR:-$DS41_MODEL}"                   # where the two official Engram shards live
DS41_VERIFIED_DIR="${DS41_VERIFIED_DIR:-$DS41_ENGRAM_DIR}"
DS41_LOG_DIR="${DS41_LOG_DIR:-$PWD/logs}"
DS41_SERVED_NAME="${DS41_SERVED_NAME:-deepseek-v4.1-flash}"
DS41_HOST="${DS41_HOST:-0.0.0.0}"
DS41_MIN_ROOT_FREE_GB="${DS41_MIN_ROOT_FREE_GB:-30}"
DS41_BUSY_MIB="${DS41_BUSY_MIB:-500}"                                # a card using more than this is "busy"

# Boards: one 4-card NVLink quad per pipeline stage. A = stage 0, C = stage 1 (letters kept from the #1031 launcher).
# Give GPU indices or UUIDs; indices are resolved to UUIDs so the pin survives CUDA_DEVICE_ORDER changes.
declare -A DS41_BOARD=(
  [A]="${DS41_GPUS_A:-0 1 2 3}"
  [C]="${DS41_GPUS_C:-4 5 6 7}"
)

DS41_DRY_RUN=0
DS41_STRICT=0
DS41_DETERMINISTIC=0
DS41_FOREGROUND=0
DS41_PROBLEMS=()

ds41_usage() {
  cat <<EOF
usage: $(basename "$0") [--dry-run [--strict]] [--deterministic] [--port N] [--spill N] [--foreground]
                        [--tree DIR] [--model DIR] [--log-dir DIR] [-- extra vllm args]
  --dry-run        run every pre-flight check, report failures as WOULD-REFUSE, print the full command, exit 0
  --strict         with --dry-run: exit 2 if any check would refuse
  --deterministic  --seed 0, temperature 0 by default (override-generation-config), CUBLAS_WORKSPACE_CONFIG=:4096:8
  --spill N        routed experts per layer in pinned host RAM (VLLM_DS41_CORE_SPILL_EXPERTS_PER_LAYER)
  --foreground     run in this terminal instead of a detached tmux session ds41-serve-<variant>
Variant $DS41_VARIANT: boards ${DS41_STAGES}, partition ${DS41_PARTITION}, max-model-len ${DS41_MAX_MODEL_LEN}
EOF
}

ds41_problem() {  # record a refusal reason (fatal unless dry-run)
  DS41_PROBLEMS+=("$1")
  if (( DS41_DRY_RUN )); then
    echo "WOULD-REFUSE: $1" >&2
  else
    echo "REFUSE: $1" >&2
    exit 2
  fi
}

ds41_parse_args() {
  DS41_PASSTHRU=()
  while (( $# )); do
    case "$1" in
      --dry-run) DS41_DRY_RUN=1 ;;
      --strict) DS41_STRICT=1 ;;
      --deterministic) DS41_DETERMINISTIC=1 ;;
      --foreground) DS41_FOREGROUND=1 ;;
      --port) DS41_PORT="$2"; shift ;;
      --spill) DS41_SPILL="$2"; shift ;;
      --tree) DS41_TREE="$2"; shift ;;
      --model) DS41_MODEL="$2"; DS41_TOKENIZER="$2"; shift ;;
      --log-dir) DS41_LOG_DIR="$2"; shift ;;
      -h|--help) ds41_usage; exit 0 ;;
      --) shift; DS41_PASSTHRU=("$@"); break ;;
      *) echo "unknown option: $1" >&2; ds41_usage >&2; exit 64 ;;
    esac
    shift
  done
  [[ "$DS41_SPILL" =~ ^[0-9]+$ ]] || { echo "--spill must be a non-negative integer" >&2; exit 64; }
  [[ "$DS41_PORT" =~ ^[0-9]+$ ]] || { echo "--port must be an integer" >&2; exit 64; }
}

ds41_gpu_list() {  # UUIDs in pipeline order (stage-major, 4 per stage)
  local board uuids=()
  for board in $DS41_STAGES; do
    [[ -n "${DS41_BOARD[$board]:-}" ]] || { echo "unknown board $board" >&2; exit 64; }
    # shellcheck disable=SC2206
    uuids+=(${DS41_BOARD[$board]})
  done
  local i
  for i in "${!uuids[@]}"; do
    if [[ "${uuids[$i]}" =~ ^[0-9]+$ ]]; then
      uuids[$i]=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${uuids[$i]}" 2>/dev/null | tr -d ' ')
      [[ -n "${uuids[$i]}" ]] || { echo "GPU index not found: ${DS41_BOARD[*]}" >&2; exit 64; }
    fi
  done
  echo "${uuids[*]}"
}

ds41_check_gpus() {
  local uuids=("$@") listing apps topo
  if ! listing=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>&1); then
    ds41_problem "nvidia-smi failed: $listing"; return
  fi
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
  topo=$(nvidia-smi topo -m 2>/dev/null)
  declare -A idx_of=()
  local line idx uuid used
  while IFS=, read -r idx uuid used; do
    idx=${idx// /}; uuid=${uuid// /}; used=${used// /}
    idx_of[$uuid]=$idx
    for u in "${uuids[@]}"; do
      if [[ "$u" == "$uuid" ]] && (( used > DS41_BUSY_MIB )); then
        ds41_problem "GPU $uuid (index $idx) is busy: ${used} MiB used"
      fi
    done
  done <<< "$listing"
  for u in "${uuids[@]}"; do
    [[ -n "${idx_of[$u]:-}" ]] || ds41_problem "GPU $u is not present (nvidia-smi -L)"
    if grep -q "$u" <<< "$apps"; then
      ds41_problem "GPU $u has running compute processes: $(grep "$u" <<< "$apps" | tr '\n' ' ')"
    fi
  done
  # every stage's 4 GPUs must form one NVLink board (all pairs NV#) -- TP4 inside a board only
  local s i j a b row cell n=${#uuids[@]}
  for (( s = 0; s < n; s += 4 )); do
    for (( i = s; i < s + 4; i++ )); do
      for (( j = s; j < s + 4; j++ )); do
        (( i == j )) && continue
        a=${idx_of[${uuids[$i]}]:-}; b=${idx_of[${uuids[$j]}]:-}
        [[ -n "$a" && -n "$b" ]] || continue
        row=$(grep -E "^GPU${a}[[:space:]]" <<< "$topo" | sed 's/\x1b\[[0-9;]*m//g')
        cell=$(awk -v col=$((b + 2)) '{print $col}' <<< "$row")
        [[ "$cell" == NV* ]] || ds41_problem "stage $((s / 4)): GPU$a and GPU$b are not NVLink peers (topo: '$cell')"
      done
    done
  done
}

ds41_check_root_disk() {
  local avail_gb
  avail_gb=$(( $(df --output=avail -B1 / | tail -1) / 1000000000 ))
  (( avail_gb >= DS41_MIN_ROOT_FREE_GB )) || \
    ds41_problem "only ${avail_gb} GB free on / (need >= ${DS41_MIN_ROOT_FREE_GB} GB)"
}

ds41_check_shards() {
  local i name missing=() absent=()
  # Packed layout (HF pack: own shards + the two official Engram shards symlinked in): every file the index names must be
  # present and non-empty, and the Engram shards must carry their verification sidecars. The 48-shard check below is for
  # the official checkpoint layout.
  if [[ -f "$DS41_MODEL/model.safetensors.index.json" ]] && ! [[ -s "$DS41_MODEL/model-00001-of-00048.safetensors" ]]; then
    local f
    while read -r f; do
      [[ -s "$DS41_MODEL/$f" ]] || absent+=("$f")
    done < <(python3 -c 'import json,sys; print("\n".join(sorted(set(json.load(open(sys.argv[1]))["weight_map"].values()))))' "$DS41_MODEL/model.safetensors.index.json")
    (( ${#absent[@]} == 0 )) || ds41_problem "${#absent[@]} index shard files missing/empty in $DS41_MODEL: ${absent[*]:0:6}"
    for name in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
      [[ -f "$DS41_ENGRAM_DIR/$name.sha256-ok" ]] || missing+=("$name")
    done
    (( ${#missing[@]} == 0 )) || ds41_problem "Engram shards lack a .sha256-ok sidecar in $DS41_ENGRAM_DIR: ${missing[*]}"
    for f in config.json tokenizer.json tokenizer_config.json; do
      [[ -f "$DS41_MODEL/$f" || -f "$DS41_TOKENIZER/$f" ]] || ds41_problem "$f missing in $DS41_MODEL"
    done
    return
  fi
  for i in $(seq 1 48); do
    name=$(printf 'model-%05d-of-00048.safetensors' "$i")
    # the FILE must be in the model dir (symlinks followed; 47/48 are D4 symlinks to $DS41_ENGRAM_DIR) and non-empty
    [[ -s "$DS41_MODEL/$name" ]] || absent+=("$name")
    if (( i >= 47 )); then
      # D5: the Engram shards count as verified only through ranged_dl.py's .sha256-ok sidecar
      [[ -f "$DS41_ENGRAM_DIR/$name.sha256-ok" && -f "$DS41_VERIFIED_DIR/$name.ok" ]] || missing+=("$name")
    else
      [[ -f "$DS41_VERIFIED_DIR/$name.ok" ]] || missing+=("$name")
    fi
  done
  (( ${#absent[@]} == 0 )) || \
    ds41_problem "${#absent[@]} of 48 shard files missing/empty in $DS41_MODEL (symlinks followed): ${absent[*]}"
  (( ${#missing[@]} == 0 )) || \
    ds41_problem "${#missing[@]} of 48 shards lack a verification marker (verify_shard.sh): ${missing[*]}"
  local f
  for f in config.json tokenizer.json tokenizer_config.json; do
    [[ -f "$DS41_MODEL/$f" || -f "$DS41_TOKENIZER/$f" ]] || ds41_problem "$f missing in $DS41_MODEL"
  done
}

ds41_check_env() {  # MC-CORE F6: debug knobs must not leak in from the calling shell
  if [[ -n "${VLLM_DS41_CORE_LAYER_SUBSET:-}" ]]; then
    ds41_problem "VLLM_DS41_CORE_LAYER_SUBSET=$VLLM_DS41_CORE_LAYER_SUBSET is set in the environment: a server would run a partial model (unset it)"
  fi
}

ds41_check_port() {
  if ss -ltnH "sport = :$DS41_PORT" 2>/dev/null | grep -q .; then
    ds41_problem "port $DS41_PORT is already in use"
  fi
}

ds41_check_tree() {
  [[ -x "$DS41_PYTHON" ]] || ds41_problem "no python at $DS41_PYTHON (set DS41_PYTHON)"
  if [[ -n "$DS41_TREE" ]]; then
    [[ -f "$DS41_TREE/vllm/models/deepseek_v41/sm70/model.py" ]] || \
      ds41_problem "$DS41_TREE has no DeepSeek-V4.1 port (vllm/models/deepseek_v41/sm70/model.py)"
  else
    (cd / && CUDA_VISIBLE_DEVICES="" "$DS41_PYTHON" -c "import importlib.util as u, sys; sys.exit(u.find_spec('vllm.models.deepseek_v41.sm70.model') is None)") 2>/dev/null || \
      ds41_problem "$DS41_PYTHON has no vllm with the DeepSeek-V4.1 SM70 port (install the release wheel or set DS41_TREE)"
  fi
}

# ---- memory sizing (PORT_DESIGN §6.1/§6.3 + §9 AM-7) ----
# Per stage, GiB/GPU: weights with all experts resident (FP16 dense), non-KV runtime (§6.3 runtime at 8K minus its
# KV), and the vLLM single block-pool KV size measured by L-ATTN at 256K / 4096-token chunks (AM-7, MiB/GPU).
declare -A DS41_MEM_WEIGHTS=( [pp2]="36.05 35.89" [pp3]="25.29 24.99 21.66" )
declare -A DS41_MEM_RUNTIME=( [pp2]="3.33 3.03" [pp3]="3.16 3.16 3.00" )
declare -A DS41_MEM_POOL256K_MIB=( [pp2]="764 844" [pp3]="509 764 626" )
DS41_SPILL_GIB_PER_EXPERT=0.0876     # one routed expert of 20 layers per GPU at TP4: 20 x 4,700,160 B
DS41_UTIL_CAP="${DS41_UTIL_CAP:-0.92}"   # keeps >= 2.5 GiB outside vLLM for Triton autotune / fragmentation (§6.3)

ds41_size_memory() {  # sets DS41_GPU_UTIL from the per-stage need; refuses if it exceeds the cap
  # vLLM's fraction is of the torch-reported total (31.73 GiB on a V100-SXM2-32GB, PORT_DESIGN §6), not nvidia-smi's
  local pp=$1 total_mib="${DS41_TORCH_TOTAL_MIB:-32491}" key="pp$1"
  local out
  out=$(awk -v w="${DS41_MEM_WEIGHTS[$key]}" -v r="${DS41_MEM_RUNTIME[$key]}" -v p="${DS41_MEM_POOL256K_MIB[$key]}" \
            -v spill="$DS41_SPILL" -v per="$DS41_SPILL_GIB_PER_EXPERT" -v len="$DS41_MAX_MODEL_LEN" \
            -v total_mib="$total_mib" -v floor="$DS41_GPU_UTIL" -v cap="$DS41_UTIL_CAP" 'BEGIN {
      n = split(w, W, " "); split(r, R, " "); split(p, P, " "); total = total_mib / 1024; need_max = 0
      for (i = 1; i <= n; i++) {
        kv = P[i] / 1024 * len / 262144
        need = W[i] - spill * per + R[i] + kv
        printf "stage %d: weights %.2f + runtime %.2f + KV pool %.2f = %.2f GiB\n", i, W[i] - spill * per, R[i], kv, need
        if (need > need_max) need_max = need
      }
      util = int((need_max + 0.10) / total * 1000 + 0.999) / 1000      # 0.1 GiB slack, rounded up to 0.001
      if (util < floor) util = floor
      printf "UTIL %.3f NEED %.2f TOTAL %.2f\n", util, need_max, total
    }')
  echo "$out" | grep -v '^UTIL' | sed 's/^/memory       : /'
  local util need total
  read -r _ util _ need _ total <<< "$(grep '^UTIL' <<< "$out")"
  DS41_GPU_UTIL=$util
  echo "memory       : --gpu-memory-utilization $util (max stage need $need of $total GiB; cap $DS41_UTIL_CAP)"
  if awk -v u="$util" -v c="$DS41_UTIL_CAP" 'BEGIN { exit !(u > c) }'; then
    ds41_problem "needs --gpu-memory-utilization $util > cap $DS41_UTIL_CAP (spill more experts or shorten --max-model-len)"
  fi
}

ds41_check_cli_args() {  # parse the server arguments with vLLM's own parser (no launch)
  local py="$DS41_PYTHON" out
  [[ -x "$py" ]] || return 0
  if out=$(cd "${DS41_TREE:-/}" && PYTHONPATH="$DS41_TREE" CUDA_VISIBLE_DEVICES="" "$py" -W ignore - "$@" 2>&1 <<'PY'
import sys
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
ns = make_arg_parser(FlexibleArgumentParser()).parse_args(sys.argv[1:])
print(f"parsed: tp={ns.tensor_parallel_size} pp={ns.pipeline_parallel_size} async_scheduling={ns.async_scheduling} "
      f"tokenizer_mode={ns.tokenizer_mode} reasoning_parser={ns.reasoning_parser} tool_call_parser={ns.tool_call_parser}")
PY
  ); then
    echo "cli-args     : $(tail -1 <<< "$out")"
  else
    ds41_problem "vLLM rejects the arguments: $(tail -3 <<< "$out" | tr '\n' ' ')"
  fi
}

ds41_main() {
  ds41_parse_args "$@"
  if [[ -z "$DS41_PYTHON" ]]; then
    if [[ -n "$DS41_TREE" ]]; then DS41_PYTHON="$DS41_TREE/.venv/bin/python"; else DS41_PYTHON=$(command -v python3); fi
  fi
  local run_dir="${DS41_TREE:-$DS41_LOG_DIR}"
  local uuids ts log session
  read -r -a uuids <<< "$(ds41_gpu_list)"
  local pp=$(( ${#uuids[@]} / 4 ))
  [[ $(tr ',' '\n' <<< "$DS41_PARTITION" | wc -l) -eq $pp ]] || { echo "partition $DS41_PARTITION != $pp stages" >&2; exit 64; }

  ds41_check_tree
  ds41_check_gpus "${uuids[@]}"
  ds41_check_root_disk
  ds41_check_shards
  ds41_check_port
  ds41_check_env
  ds41_size_memory "$pp"

  ts=$(date -u +%Y%m%dT%H%M%SZ)
  log="$DS41_LOG_DIR/${DS41_VARIANT}_${ts}.log"
  session="ds41-serve-${DS41_VARIANT}"

  local envs=(
    "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    "CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${uuids[*]}")"
    "VLLM_PP_LAYER_PARTITION=$DS41_PARTITION"
    "VLLM_DS41_CORE_SPILL_EXPERTS_PER_LAYER=$DS41_SPILL"
    "VLLM_DS41_CORE_ENGRAM_DIR=$DS41_ENGRAM_DIR"
    "VLLM_DS41_ENGRAM_IO_THREADS=${VLLM_DS41_ENGRAM_IO_THREADS:-$(( pp == 2 ? 2 : 1 ))}"     # AM-12: Engram pread threads per rank (CPU budget); caller may override (engram-io bench 2026-10-24: 4 best)
    "VLLM_USE_V2_MODEL_RUNNER=0"
    "NCCL_BUFFSIZE=4194304"
    "NCCL_MAX_NCHANNELS=4"
  )
  [[ -n "$DS41_TREE" ]] && envs+=("PYTHONPATH=$DS41_TREE")
  # forward caller-set port knobs (route histogram, spill ranking, ...) the list above doesn't set
  local v
  for v in $(compgen -e | grep '^VLLM_DS41_' | sort); do
    [[ " ${envs[*]} " == *" $v="* ]] || envs+=("$v=${!v}")
  done
  local args=(
    "$DS41_PYTHON" -m vllm.entrypoints.openai.api_server
    --model "$DS41_MODEL" --tokenizer "$DS41_TOKENIZER" --served-model-name "$DS41_SERVED_NAME"
    --host "$DS41_HOST" --port "$DS41_PORT"
    --dtype half
    --tensor-parallel-size 4 --pipeline-parallel-size "$pp"
    --max-model-len "$DS41_MAX_MODEL_LEN" --max-num-seqs "$DS41_MAX_NUM_SEQS"
    --max-num-batched-tokens "$DS41_MAX_BATCHED_TOKENS" --enable-chunked-prefill
    --gpu-memory-utilization "$DS41_GPU_UTIL"
    --no-async-scheduling
    --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41
    --enable-auto-tool-choice --tool-call-parser deepseek_v41
  )
  if (( DS41_DETERMINISTIC )); then
    envs+=("CUBLAS_WORKSPACE_CONFIG=:4096:8")
    args+=(--seed 0 --override-generation-config '{"temperature": 0.0}')
  fi
  args+=("${DS41_EXTRA_ARGS[@]}" "${DS41_PASSTHRU[@]}")

  local cmd
  cmd="env $(printf '%q ' "${envs[@]}")$(printf '%q ' "${args[@]}")"
  echo "variant      : $DS41_VARIANT (boards ${DS41_STAGES// / -> }, PP${pp}xTP4, partition $DS41_PARTITION, spill $DS41_SPILL)"
  echo "log          : $log"
  echo "health       : curl -sf http://127.0.0.1:${DS41_PORT}/health && curl -s http://127.0.0.1:${DS41_PORT}/v1/models"
  echo "command      : $cmd"
  if (( DS41_DRY_RUN )); then
    ds41_check_cli_args "${args[@]:3}"
    echo "dry-run      : ${#DS41_PROBLEMS[@]} pre-flight problem(s); nothing launched"
    (( DS41_STRICT && ${#DS41_PROBLEMS[@]} )) && exit 2
    exit 0
  fi
  mkdir -p "$DS41_LOG_DIR"
  if (( DS41_FOREGROUND )); then
    cd "$run_dir" || exit 1
    exec bash -c "$cmd 2>&1 | tee -a $(printf '%q' "$log")"
  fi
  tmux has-session -t "$session" 2>/dev/null && { echo "REFUSE: tmux session $session already exists" >&2; exit 2; }
  # run from inside the serving tree: `python -m` puts the CURRENT directory first on sys.path (ahead of PYTHONPATH), so a
  # launcher started from another vllm tree would import that tree (2026-10-19: served the dev tree's stale _C)
  tmux new-session -d -s "$session" -c "$run_dir" "cd $(printf '%q' "$run_dir") && $cmd >> $(printf '%q' "$log") 2>&1; echo EXIT=\$? >> $(printf '%q' "$log")"
  echo "launched     : tmux session $session (attach: tmux attach -t $session); waiting for /health ..."
  local t
  for t in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:${DS41_PORT}/health" >/dev/null 2>&1; then
      echo "ready after $(( t * 10 )) s:"; curl -s "http://127.0.0.1:${DS41_PORT}/v1/models"; echo; return 0
    fi
    tmux has-session -t "$session" 2>/dev/null || { echo "server exited; see $log" >&2; tail -20 "$log" >&2; exit 1; }
    sleep 10
  done
  echo "not healthy after 3600 s; still running in tmux $session, log $log" >&2
  exit 1
}
