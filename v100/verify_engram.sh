#!/bin/bash
# Check the two official Engram shards against their published sha256 and write the <shard>.sha256-ok sidecars the
# launcher's preflight requires. Reads DS41_ENGRAM_DIR (default: $DS41_MODEL). Takes a few minutes per 101.5 GB shard.
set -euo pipefail
dir=${DS41_ENGRAM_DIR:-${DS41_MODEL:?set DS41_ENGRAM_DIR or DS41_MODEL}}
# deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170 (LFS sha256)
declare -A want=(
  [model-00047-of-00048.safetensors]=76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5
  [model-00048-of-00048.safetensors]=9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd
)
for s in "${!want[@]}"; do
  f="$dir/$s"
  [ -s "$f" ] || { echo "missing: $f" >&2; exit 1; }
  got=$(sha256sum "$(readlink -f "$f")" | cut -d' ' -f1)
  if [ "$got" = "${want[$s]}" ]; then
    echo "$got  $s" > "$f.sha256-ok"; echo "ok: $s"
  else
    echo "MISMATCH: $s ($got)" >&2; exit 1
  fi
done
