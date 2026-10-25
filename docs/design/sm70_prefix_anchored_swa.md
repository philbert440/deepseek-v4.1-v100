# SM70 Prefix-Anchored Sliding-Window Attention

Prefix-anchored sliding-window attention is an explicit, model-agnostic
inference-engine option for bounded decode KV usage on V100/SM70. The request
prefix remains globally visible, while generated tokens attend only to the
most recent configured window:

```text
visible(q, kv) = kv <= q and (kv < prefix_len or q - kv < window)
```

Enable it through `AttentionConfig`; checkpoint metadata and model architecture
names do not activate or gate the route:

```bash
vllm serve <model> \
  --attention-config '{"prefix_anchored_decode_window": 4096}'
```

The option is off by default. It changes attention semantics after generation
exceeds the configured window, so it is not an exact-output optimization for a
full-attention workload.

## Current Engine Contract

The engine admits the route only when all of these operator/runtime contracts
hold:

- NVIDIA SM70 and the `FLASH_ATTN_V100` backend;
- causal, standard full decoder attention with fp16 KV;
- equal key and value head sizes, with no attention sinks or per-layer window;
- decode and prefill context-parallel sizes both equal to one;
- no speculative decoding, KV connector, or KV-cache offloading.

Prefix caching is disabled, and full CUDA graphs are reduced to piecewise
graphs. Unsupported combinations fail during configuration or layer
construction. Once enabled, missing or mismatched per-request prefix metadata
is a hard runtime error; the engine never evicts gap blocks and then silently
runs an unmasked kernel.

## Validation Status

Validation on the current `main` source includes:

- 32 CPU tests covering the mask, generic engine admission, fail-closed
  metadata, cache-spec registration, and middle-gap block reclamation;
- a clean SM70 source build of the Flash-V100 extension;
- 19 V100 operator tests for paged decode and paged prefill, including GQA,
  multiple head sizes, multi-partition decode, full-prompt and continuation
  shapes, a negative control, and the default-off path.
- same-toolchain comparison against the current `origin/main` build: all 55
  default-off scalar decode kernel instances and all 64 default-off paged
  prefill kernel instances are SASS-identical.

No model end-to-end or throughput claim is attached to this route yet. Treat
performance and workload-level quality as separate acceptance gates before
enabling it in a production configuration.
