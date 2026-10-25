# SM70 DFlash2 target-verifier graph: 20 ms control log

## Scope and frozen baseline

- Date: 2026-08-23.
- Integration base: `c62c1dcd833458054daa12e134695aa19a4ac609`
  (`codex/v100-dflash2-gdn-metadata-20260822-122723`, Draft PR #257).
- Worktree:
  `/home/ymzx/桌面/1cat-vllm/worktrees/v100-dflash2-target-graph-20ms-20260823-084427`.
- Branch: `codex/v100-dflash2-target-graph-20ms-20260823-084427`.
- Task cache: `/data/minimax-h3/task-cache/v100-dflash2-target-graph-20ms`.
- Frozen workload: Qwen3.8-27B-FP8 target, official DFlash2 draft, TP4,
  batch one, block eight (seven draft tokens), Flash-V100, target E5M2 KV,
  draft FP16/auto KV, and CUDA Graph.
- Baseline node trace:
  `/data/minimax-h3/task-cache/v100-dflash2-pr257-gdn-metadata/profiles/latest-e2e/dflash2-fused-viewcache-b1-nodes-o128-v5.sqlite`.

The 30-round, four-rank steady-state baseline (one edge round removed from
each side) is:

| Phase | Critical-path wall | Nodes or launches per rank |
|---|---:|---:|
| Draft graph | 4.046 ms | - |
| Draft to target | 4.934 ms | 153 kernels |
| Target graph | **24.740 ms** | **2612 nodes** |
| Target to draft | 2.838 ms | - |
| Complete round | 36.300 ms | - |

The target-graph objective is at most 20.000 ms under this same diagnostic.
Performance acceptance requires an improvement in the complete round. The
final quality contract uses probabilistic task scores, paired acceptance, and
PPL against target-only; greedy token identity is diagnostic evidence rather
than a production gate. The superseding measurements are recorded at the end
of this document.

## Resource-use decomposition

The target graph contains 23.365 ms of rank-average GPU kernel service inside
a 24.740 ms critical span, or 94.7% activity-envelope coverage. This rules out
a large idle bubble as the primary cause, but it does not imply efficient SM
or memory-pipeline utilization within each small kernel. Nsight Compute metrics
must remain separate from graph-span and Nsight Systems service time.

| Kernel category | Rank-average service | Launches per rank |
|---|---:|---:|
| TurboMind FP8 dense GEMM | 10.831 ms | 256 |
| Copy/cast elementwise | 3.387 ms | 941 |
| Other kernels | 2.869 ms | 240 |
| TP all-reduce/communication | 2.316 ms | 128 |
| Other Torch/Triton elementwise | 2.494 ms | 739 |
| RMSNorm/residual | 0.616 ms | 128 |
| LM-head/sample/TP gather | 0.416 ms | 48 |
| Dense GEMV/GEMM/compressor | 0.257 ms | 65 |
| Fill/mask | 0.179 ms | 67 |

The graph is therefore busy but fragmented: 1680 copy/cast/elementwise nodes
consume 5.881 ms, while the average node is only about 8.9 microseconds. The
20 ms target requires recovering at least 4.740 ms (19.2%); launch reduction
and work fusion are first-order requirements, not optional cleanup.

## Ordered experiments and stop gates

1. A/B the repaired, default-off DFlash2 packed GDN verifier with all other
   flags frozen. It preserves the current gating values, recurrent arithmetic
   order, and FP32-state contract while removing packed-QKV rearrangement and
   the final output copy. Record route-hit logs, graph wall, graph nodes,
   complete-round wall, exact output tokens, and acceptance trajectory.
2. If the packed route wins but remains above 20 ms, use its new node trace to
   isolate residual GDN state/copy traffic. Compare with SGLang-V100's fused
   recurrent verifier and all-layer state commit without importing unrelated
   scheduler or Eagle/MTP changes.
3. Profile the exact TP4 all-reduce shape. The current 2.316 ms/rank service is
   a second independent target; any replacement must preserve collective
   ordering and exact target hidden states.
4. Only after the verifier contract is stable, assess a DFlash2-safe variant of
   v100-skinny's small-M QPN8 projection work. It is not accepted on the basis
   of its non-speculative result.

Every failed experiment must be reverted or kept behind a default-off gate.
Do not report profiler-instrumented latency as unprofiled throughput. Do not
use target-only runs, and do not occupy a partial TP4 group or terminate an
unrelated process.

## External references

- SGLang-V100 fixed source and trace are compared locally at
  `/data/minimax-h3/task-cache/v100-dflash2-pr257-gdn-metadata/sources/sglang-v100`
  and
  `/data/models/v100-dflash2-20260820/sglang-audit/perf-rootcause/sglang-dflash2-single1-step20-v2.nsys-rep`.
- v100-skinny is pinned at `5b589c0dc81223e0ba65bcb3e755874723f8b515`;
  its 219.1 tok/s result is a Qwen3.8 mixed-NVFP4/FP8 MTP result, not a DFlash2
  target-verifier baseline.

## Results

### Accepted target-graph reduction

The matched TP4 node trace at source `500130882ba68fb7bac3a0b9e4eb5872647fb045`
uses the same model, draft, request, sampling, KV dtypes, and graph policy as the
frozen baseline:

- SQLite:
  `/data/minimax-h3/task-cache/v100-dflash2-target-graph-20ms/profiles/dflash2-fused-gdn-norm-split-gemma-dynamic-b1-nodes-o128-gpu0123-v3.sqlite`.
- Result JSON:
  `/data/minimax-h3/task-cache/v100-dflash2-target-graph-20ms/results/dflash2-fused-gdn-norm-split-gemma-dynamic-b1-nodes-o128-gpu0123-v3.json`.
- Four-rank synchronized target graph, with the first and last rounds removed:
  **19.317 ms mean, 19.322 ms p50, 19.444 ms p90, 19.477 ms p99**, and
  **1,257 nodes/rank**.
- Rank-average kernel service is 18.552 ms, or 96.04% of the critical span.
- Relative to the frozen 24.740 ms / 2,612-node baseline this removes
  **5.423 ms (21.92%)** and **1,355 nodes (51.88%)**. The p99 is also below
  the 20 ms objective.
- The benchmark endpoint inside this profiled run reports 125.060 steady
  decode token/s; this is trajectory evidence, not an unprofiled throughput
  claim. All 128 output
  token IDs and hash `fe0300...` match the baseline exactly. Probabilistic
  acceptance changed slightly from 4.129 (31 draft rounds) to 4.000 (32 draft
  rounds), so the Gemma suffix fusion remains a Type-B candidate pending a
  distribution-quality gate rather than being accepted from one trajectory.

The reduction is cumulative: the one-pass GDN output norm first reached
22.646 ms / 2,084 nodes; fused GDN split materialization reached 21.625 ms /
1,892 nodes; the dynamic-shape-correct Gemma residual RMS fusion reached the
19.317 ms / 1,257-node result above. A prior `v2` launch omitted the installed
`flash_attn_v100_cuda` extension directory from `PYTHONPATH` and failed during
graph capture; it is not a performance result.

### Remaining graph-boundary cost

Across 31 steady boundaries per rank, the draft-graph end to target-graph
start interval is 5.776 ms mean (5.582 ms p50). Only 1.304 ms is covered by
non-graph GPU kernels, leaving about 4.47 ms of launch, synchronization, and
CPU queueing bubbles. Every boundary submits 153 kernels:

- one TP `cross_device_reduce_1stage`: 0.785 ms;
- 20 `DeviceScanKernel`, 20 `DeviceScanInitKernel`, 20 `compute_cuda_kernel`,
  and 21 `indexSelectSmallIndex` calls: 0.276 ms service but much larger
  serialized submission cost;
- remaining input, slot, block-table, GDN metadata, and elementwise kernels:
  about 0.243 ms service.

The scan/select work arrives in five serial groups with scalar D2H ordering
points. Therefore the next boundary objective is not a faster individual
3-microsecond kernel. It is to construct persistent target metadata once and
capture or batch-submit the five repeated groups while preserving every
buffer/update dependency.

The same parser applied to the retained SGLang-V100 trace gives a useful
matched structural target. Its draft-to-target interval is 3.398 ms mean with
30 non-graph kernels and 0.115 ms GPU service, versus this branch's 5.776 ms,
153 kernels, and 1.304 ms service. SGLang still pays a 2.715 ms host
`cudaStreamSynchronize`, so it is not a zero-overhead endpoint. The immediate
vLLM gap is nevertheless concrete: remove 123 repeated launches and move the
roughly 0.785 ms TP reduction adjacent to target input/embedding preparation
into the target replay dependency chain. This comparison comes from
`/data/models/v100-dflash2-20260820/sglang-audit/perf-rootcause/sglang-dflash2-single1-step20-v2.sqlite`.

### Preliminary probabilistic quality pair

A fixed 16-question sequential GSM8K pair used graph TP4, target E5M2 KV,
draft FP16 KV, block eight, and official `temperature=1.0/top_p=.95/top_k=20`
sampling. The control artifact is
`quality/gsm8k-16-control-gemma-off.json`; the candidate artifact is
`quality/gsm8k-16-candidate-gemma-on.json` under the task cache.

- Control: 68.75% accuracy, aggregate acceptance length 4.470.
- Candidate: 75.00% accuracy, aggregate acceptance length 4.693.
- Eight of 16 full output-token trajectories match. Eight diverge under
  probabilistic sampling, as expected from the accepted one-FP16-ULP numeric
  bound; this small sample cannot establish a quality improvement.

The Gemma fusion stays default-off until paired prompt perplexity and broader
dataset gates show no regression. The GDN norm and split stages retain their
Type-A classification and can be defaulted independently of that decision.

The first deterministic distribution probe scores 1,850 prompt tokens from 32
fixed GSM8K questions in eager target prefill (Graph is irrelevant to prompt
logprobs). Weighted perplexity is 4.20119 for the decomposed control and
4.20172 for the Gemma fusion, a +0.00053 / +0.013% change. All 32 next-token
argmaxes match. Mean absolute prompt-logprob difference is 0.00173 and the
worst token is 0.02524. This is small enough to continue dataset testing but
confirms that the kernel is Type B, so it remains default-off.

### Fused Flash-V100 small-query metadata candidate

The five repeated scan groups are generated by
`FlashAttnV100MetadataBuilder._update_smallq_decode_metadata`: each of five
target KV groups performs four `repeat_interleave` scans, then materializes and
copies block-table and sequence-length temporaries. The
`VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA` candidate writes the persistent
block-table, sequence-length, and query-boundary buffers directly with one
Triton launch per group. It is limited to a DFlash2 target on SM70; draft,
DDTree, Eagle, MTP, and CPU metadata retain the existing path.

- Exact V100 tests pass for B1 and mixed B3/B4 layouts, negative block IDs,
  zero-length padded requests, and graph-token padding.
- The unchanged CPU persistent-buffer, padding, and overflow tests pass.
- A realistic `q=8`, three block columns, five-group microbenchmark reports
  1.278 ms legacy versus 0.114 ms fused wall time and 1.235 versus 0.100 ms GPU
  service: **1.164 ms saved / 11.2x** for this isolated constructor.

The matched TP4 end-to-end trace accepts the candidate as a Type-A default-on
optimization (explicit environment value `0` remains an opt-out):

- Synchronized draft-to-target falls from 5.720 ms to 1.911 ms (-66.6%).
- Per-rank non-graph work falls from 153 nodes / 1.304 ms GPU service to 23
  nodes / 0.195 ms. All 20 `DeviceScan`, 20 `DeviceScanInit`, and 20 generic
  scan-compute launches disappear; `indexSelectSmallIndex` falls from 21 to 1.
- The five replacement launches total 0.016 ms of GPU service.
- The full speculative round falls from 31.791 ms to 27.951 ms (-12.1%), while
  the target graph remains effectively unchanged at 19.261 ms / 1,257 nodes.
- Profiled single-request steady decode rises from 125.06 to 142.37 token/s
  (+13.8%). The 128-token output hash, aggregate acceptance length 4.0, and
  per-position accepted counts `[28, 23, 18, 14, 8, 3, 2]` match exactly.

The accepted trace is
`profiles/dflash2-fused-gdn-norm-split-gemma-smallqmeta-b1-nodes-o128-gpu0123-v1.sqlite`
under the task cache. Its 1.911 ms synchronized boundary is also below the
3.398 ms boundary in the retained SGLang-V100 audit trace, though that external
trace is a structural reference rather than a fully matched throughput run.

### Next boundary: target-to-draft

The accepted trace leaves a stable 2.738 ms target-to-next-draft interval on
rank 0 (33 steady intervals). A representative `M=8` interval and the aggregate
kernel service separate it as follows:

| Stage | Representative wall / service |
| --- | ---: |
| Dense target LM head, local FP16 shard | 0.947 ms |
| TP full-vocabulary all-gather | 0.139 ms |
| Full-vocabulary top-k/top-p | 0.606 ms |
| Rejection statistics, rejection, and resample | 0.225 ms |
| Target-hidden-to-draft KV precompute | about 0.60 ms |
| Post-update and next-draft metadata | about 0.22 ms |

The next candidate must therefore attack the first four rows as one semantic
unit. For the fixed no-penalty `top_k=20, top_p=0.95` contract, each TP rank can
compute its exact local top 20, exchange only 20 `(score, token-id)` pairs per
row, merge the global top 20, and perform rejection/residual sampling on that
sparse support. The repository already contains the SM70 TurboMind FP16
LM-head top-20 epilogue, but the `_C` binary linked into this worktree predates
that op. The first TP4 microbenchmark was therefore rejected at route checking
without recording a timing; an incremental SM70 build is required before this
candidate can pass the microbenchmark gate.

### Compact target sampling microbenchmarks and gated integration

The production-shaped TP4 communication probe uses eight verifier rows,
vocabulary 248,320, target top-K 20, and four V100s. It compares the existing
full-vocabulary all-gather plus global top-K with local top-K followed by two
compact all-gathers and a global merge:

- Exact top-K token IDs and values match.
- The full path moves 993,280 bytes per rank; the two-gather compact path moves
  1,920 bytes, a 517x reduction.
- Despite that byte reduction, p50 is 0.3318 ms for full gather plus top-K and
  0.3292 ms for the compact path. Local top-K (0.1208 ms), two collectives
  (0.0911 ms), and compact merge (0.0896 ms) consume the saved transport time.
- Packing values and IDs into one collective is slower at 0.3538 ms.

Compact communication alone is therefore rejected as a performance feature.
The retained artifact is
`results/tp4-compact-verifier-logits-m8-v248320-k20.json`.

The second probe fuses compact target top-p, DFlash2 acceptance, log-domain
`relu(p-q)` recovery, and token-keyed Gumbel resampling into one Triton program
per request. At the real B1/block-eight shape it is token/count exact against
the dense rejection path for block 4/8 and `top_p` 1.0/0.95. Its p50 is:

| Isolated stage | p50 |
| --- | ---: |
| Dense top-K/top-p | 0.8366 ms |
| Dense rejection only | 0.2202 ms |
| Dense top-K/top-p plus rejection | 0.8663 ms |
| Compact top-p plus sparse rejection | **0.0901 ms** |

Combining this result with the measured TP4 compact-candidate transport gives
an expected real saving of roughly 0.5-0.6 ms per complete round. The artifact
is `results/dflash2-sparse-rejection-b8-v248320-k20-q16-v1.json`.

The candidate is integrated behind default-off
`VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION=1`. Its route is limited to SM70,
MRV2 DFlash2, one active decode request, probabilistic proposals, target
`top_k=20`, positive temperature, `0 < top_p <= 1`, and no grammar, min-p,
penalty, logit bias, bad words, NaN reporting, or logprob request. Every other
configuration computes the unchanged dense logits and uses the shared
rejection sampler. DFlash2 now keeps the 16 proposal IDs and FP32 realized
scores in persistent request-slot order in addition to the dense fallback
cache; a V100 poisoned/reordered-slot test verifies the mapping.

Validation completed before the full-model run:

- DFlash2 CPU suite: 46 passed, 11 CUDA-only skipped.
- Strict route and fallback tests: 13 passed.
- Sparse dense-equivalence plus selector/cache CUDA tests: 6 passed.
- Ruff, formatting, compileall, and `git diff --check`: passed.

The matched single-request Graph/Nsight run remains pending because both TP4
GPU groups became occupied by unrelated work after these tests. No external
process was interrupted.

### Residual target-graph service after the 27.951 ms round

The accepted 1,257-node target graph has 18.357 ms of rank-0 GPU service inside
its 19.261 ms critical span. Its largest exact buckets per replay are:

| Target graph bucket | Nodes | GPU service |
| --- | ---: | ---: |
| TurboMind FP8 GEMM | 256 | 10.730 ms |
| vLLM one-stage TP4 all-reduce | 128 | 2.313 ms |
| recurrent GDN verifier | 48 | 1.213 ms |
| Flash-V100 partition/reduce | 32 | 0.802 ms |
| fused Gemma residual/RMS suffix | 127 | 0.643 ms |
| generic elementwise | 144 | 0.541 ms |

The FP8 GEMMs split into three stable launch shapes: 64 calls at 4.394 ms,
128 calls at 3.924 ms, and 64 calls at 2.411 ms. They already match the retained
SGLang target GEMM total closely, so replacing them is a separate QPN8 quality
and throughput project rather than the next low-risk leaf.

The TP4 collective remains the clearest next `>1 ms` gap. vLLM launches ten
512-thread CTAs and averages 18.07 microseconds per call. The matched
SGLang-V100 trace uses its JIT one-shot push kernel with eighty 128-thread CTAs
and averages 12.36 microseconds. Across 128 calls this structural difference is
about 0.73 ms of service; the broader matched audit found a 0.7-1.2 ms
per-graph TP reduction gap depending on critical-rank accounting. The next
microbenchmark should compare the exact `[8,5120]` FP16 target shape under one
captured 128-call chain, first sweeping the existing vLLM CTA/thread policy,
then porting SGLang's two-epoch push buffer only if the policy sweep cannot
close the gap. Bitwise FP32 accumulation order, CUDA-Graph registration, and
all 128 collective dependencies are hard gates.

### Sparse rejection end-to-end acceptance

The matched Graph/Nsight run completed after the GPU group became available:

- Result:
  `results/dflash2-sparse-target-rejection-b1-nodes-o128-gpu0123-v1.json`.
- SQLite:
  `profiles/dflash2-sparse-target-rejection-b1-nodes-o128-gpu0123-v1.sqlite`.
- Full synchronized round: **27.166 ms**, down from 27.951 ms by 0.785 ms.
- Target-to-draft: **1.994 ms**, down from 2.738 ms by 0.744 ms.
- Draft, draft-to-target, and target graph remain 4.004, 1.977, and
  19.191 ms respectively.
- All 128 output IDs and hash `fe0300...` match the accepted baseline. The
  run records acceptance length 4.129 and per-position counts
  `[27, 22, 18, 14, 9, 4, 3]`.

This is a Type-A result for the narrow sparse route contract. The environment
gate remains explicit until the default policy and paired dataset result are
landed together.

### TP4 push all-reduce audit

The existing vLLM CTA/thread-policy sweep did not improve the 128-collective
chain. A pinned SGLang-V100 checkout at `845b9fdf7a7e` measured its two-epoch
one-shot push collective at 0.846-0.886 ms for the same 128-call
`[8,5120]` FP16 chain, versus 1.854 ms for vLLM's pull path. A first vLLM port
measured 0.873 ms and matched the FP32 rank-order reference numerically across
eager, graph, and 128-call graph replay.

Two issues prevented accepting that first port:

1. Its temporary `_C` reused objects from the independent QPN8 worktree. Both
   push-on and push-off runs first diverged from the accepted output at token
   24, proving that those runs were not a clean source comparison.
2. SGLang's positive-zero empty-slot sentinel converts an all-positive-zero
   reduction to negative zero. A dedicated signed-zero probe found 40,960
   bit mismatches despite zero numerical error. This can perturb later finite
   arithmetic and is below the required bitwise gate.

A release/acquire CTA-ready protocol fixed every payload bit for exact,
random, model-like, positive-zero, and signed-zero inputs over a 128-call graph
chain, but regressed the chain to 1.884-1.901 ms. It is rejected. The accepted
microbenchmark candidate instead preinitializes the two data epochs with a
reserved FP16 NaN payload and retains SGLang's fine-grained polling. Finite
inputs, including both signed zeros, are never rewritten.

The NaN-sentinel candidate passed the strict TP4 microbenchmark gate:

- Artifact:
  `results/vllm-push-ar-nan-sentinel-correctness-timing-m8-h5120-v1.json`.
- Exact-integer, rank-marker, random-small, model-like, positive-zero, and
  signed-zero inputs are bitwise equal to the fixed rank-order FP32 reference
  in eager, one-call Graph, and 128-call Graph-chain modes on every rank.
- The 128-call chain is **0.850-0.856 ms**, or about 1.00 ms faster than the
  retained 1.854 ms vLLM pull chain. One call averages 6.64-6.68 microseconds
  inside the chain.
- The task-owned `_C` contains no QPN8 operator and has SHA256
  `275594c0b38a358c1683efa1d6351e1c572d6da992789d3b5210a74ecbfce1e8`.

The first unprofiled full-model run with that clean extension completed at
176.56 steady decode token/s. Its 128 token IDs, hash `0cc12...`, acceptance
length 4.467, and per-position accepted counts
`[28, 23, 14, 13, 11, 8, 7]` exactly match the retained push-off control built
with the current CUDA toolchain. They differ from the older `fe030...` release
extension trajectory at token 24; both push-on and push-off current-toolchain
builds make that same transition, so it is not attributed to the collective.
The result is
`results/dflash2-sparse-push-ar-nan-pure-b1-o128-gpu0123-v1.json`.

A same-binary push-off rerun was invalidated when another task claimed the TP4
group during model initialization, before KV-cache allocation; it produced no
generation or performance result. The push route therefore remains
default-off until that paired rerun and a clean Nsight complete-round trace are
recorded. The microbenchmark establishes an expected roughly 1 ms graph saving,
but the 176.56 token/s endpoint is not yet used as the paired throughput claim.

### DFlash2 target QPN8 backport

The accepted FP8 QPN8 operator and production route were cherry-picked from
`db978179de` and `e0547f1b4a`. The audited integration admits only explicit
QPN8 opt-in, SM70/TP4, block-FP8 layout, exact projection shapes, and bounded
concurrency. It does not inspect a model, architecture, or checkpoint identity;
unsupported tensor or runtime contracts retain TurboMind.

The target LM head is not a QPN8 shape and remains on the dense FP16 path. The
only admitted target projections are fused gate/up, down, and GDN/full-attention
output. GDN input and full-attention QKV also remain TurboMind. This preserves
the DFlash2 selector and target-sampling contracts while attacking the three
measured FP8 verifier buckets.

- The imported target-only evidence remains documented in
  `sm70_qwen38_qpn8_decode.md`: WikiText perplexity and all fixed GSM8K, MMLU,
  and C-Eval metrics pass, and the operator race passed 960 rows.
- The DFlash2-specific runtime-isolation tests pass for target-only, DFlash2,
  DFlash1, and a generic non-DFlash speculative configuration. Together with
  environment and warmup coverage, 70 focused CPU tests pass.
- A task-owned combined extension contains both the bitwise-gated push
  collective and all six QPN8 operators. Its SHA256 is
  `512b44cd49e99f26e0f091c68c11bf8c17e5575ffe2c42ba9af1de69938a29e9`.

No DFlash2 throughput claim is made from the target-only result. The required
next gate is a production-shape M=8 operator smoke followed by paired
single-request DFlash2 generation and a clean Nsight phase trace. The working
hypothesis is that QPN8 plus push can remove enough of the 19.2 ms target graph
to approach a 23 ms complete round, but only the new trace may accept or reject
that hypothesis.

### Quality-safe 23.678 ms checkpoint

The next accepted stack keeps the sharded target-hidden projection disabled
and collapses the five full-attention metadata refreshes into one pointer-table
kernel.  The paired single-request node traces measure 24.424136 ms for the
control and **23.678170 ms** for the candidate.  Draft-to-target falls from
1.857847 to 1.213916 ms.  All token IDs, token hash, acceptance length, and
per-position acceptance counts are exact. This grouped metadata route proved
the measured workload, but remains default-off in the audited source so the
kernel contract can be rolled out independently of a model identity.

The retained target graph has 1,254 nodes and about 16.055 ms of rank-critical
GPU service.  Its largest buckets are 4.235 ms gated-pair QPN8, 3.186 ms QPN8
`(16,1)`, 2.059 ms QPN8 `(16,2)`, 1.277 ms TP4 push all-reduce, 1.205 ms
recurrent GDN, and 0.734 ms Flash-V100 partition attention.  This decomposition
is the active short-context optimization budget; the 20 ms complete-round goal
requires at least another 3.679 ms from the measured checkpoint, not merely a
lower isolated kernel number.

Caching the prepared GDN metadata tensor tuple removes repeated empty-tensor
allocation and tuple assembly.  A ten-group/four-layer microbenchmark falls
from 50.519 to 2.663 microseconds and the targeted metadata test passes.  The
post-convolution Q/K/V packer is bitwise exact in seven V100 tests and improves
the production-shape graph from 7.475 to 4.116 microseconds, projecting about
0.161 ms over 48 GDN layers.  It stays default-off because the packed-GDN route
already subsumes the same rearrangement.

The graph-internal text-embedding experiment is rejected and fully removed.
Full target graph capture failed in the AOT Qwen signature with
`AttributeError: 'NoneType' object has no attribute 'size'` after omitting the
previously compiled `inputs_embeds` tensor.  It generated no tokens and has no
performance result; the failed run must never be counted toward the 20 ms
goal.

### QPN8 plus TP4 push fusion rejection

An exact fused epilogue was implemented for the 128 target out/down projection
calls.  It rounded the local FP32 QPN8 accumulator to FP16, pushed all four
ranks' packets, accumulated ranks 0 through 3 in FP32, and rounded the result
to FP16.  A TP4 128-call CUDA-Graph chain passed every per-rank bitwise check
against both the separate QPN8-plus-push path and the fixed rank-order
reference.

The performance gate failed: the separate chain was 2.696 ms median, while the
fused chain was 3.249 ms, or 21.06 versus 25.38 microseconds per call.  Keeping
the 512-thread QPN CTA resident while one warp polls peer packets increases SM
and NVLink contention instead of hiding communication.  The entire production
route, environment switch, bindings, and tests were removed.  Only the
standalone benchmark artifact is retained at
`results/qpn8-fused-push-tp4-m8-n5120-v1.json`; this negative 0.552 ms must not
appear in an endpoint speed claim.

### 2026-08-24 external implementation audit

The one permitted upstream-head refresh fetched SGLang-V100 at
`0083b9fd83a601b1fcd9a691f7240be4e6be111e`.  Its new native long-context work
adds grouped q=1 exact split-KV decode, grouped multi-token verification, and
exact E5M2 byte-to-FP16 expansion.  The implementation assigns a Qwen3.8 TP4
verification layer to two GQA subgroups and at most 40 context splits, instead
of scanning the same cache independently for all eight query rows.

The same redundant-read diagnosis was independently merged in
[vLLM-Metal PR #534](https://github.com/vllm-project/vllm-metal/pull/534).
That implementation shares each KV block across two verification-window rows
while retaining each row's original accumulation order and reports bitwise
parity.  This is stronger evidence than copying a benchmark number: it
identifies redundant KV traffic as a kernel-shape problem specific to
multi-token verification and supplies a quality-preserving scheduling pattern.

`dnv2003/v100-skinny` remains a useful QPN8 geometry inventory.  Its graph
sweep reports `(8,2)` around 12.27 microseconds for M=8/K=1536/N=5120, while
the active target trace uses `(16,1)` around 24.9 microseconds.  The kernels and
decoder contracts are not assumed identical.  A task-owned sweep of the exact
current operator is therefore required before changing the table; this is now
the highest-value short-context microbenchmark after rejecting QPN/AR fusion.

### Long-context verification microbenchmark

The pinned SGLang TileLang 0.1.8 compiler required its matching
`apache-tvm-ffi==0.1.8.post2` and the CUDA 12.8 toolkit still mounted at
`/mnt/nvme3n1p2/usr/local/cuda-12.8`.  These are off-tree measurement tools,
not new runtime dependencies.  The production implementation will be native
Flash-V100 code.

On the same V100 and shape `B=1, Q=8, Hq=6, Hkv=1, D=256, page=16, E5M2`, the
measured per-layer medians are:

| Prefix | Flash-V100 scalar rows | Flash-V100 XQA rows | SGLang grouped verify |
| ---: | ---: | ---: | ---: |
| 1K | 0.238 ms | 0.315 ms | **0.122 ms** |
| 32K | 1.390 ms | 1.151 ms | **0.527 ms** |
| 128K | 5.119 ms | 3.561 ms | **1.957 ms** |
| 256K | 9.064 ms | 6.481 ms | **3.803 ms** |

The actual current policy selects XQA for E5M2 at and above 16K.  Relative to
that production long-context family, grouped verification is 2.18x, 1.82x,
and 1.70x faster at 32K, 128K, and 256K.  Across the target's 16 full-attention
layers, the isolated projection is roughly 9.99, 25.67, and 42.84 ms removed
per complete verification round at those lengths.  Short-context dispatch must
still use the existing scalar path when the captured workspace is small: the
accepted 8K-capacity trace measures only about 45.5 microseconds per layer,
below the grouped kernel's fixed cost.

At 1K, both grouped and XQA outputs are within 1.526e-5 maximum and about
1.6e-6 mean absolute error of the FP32 reference.  This is currently
`B-pending`, not a default-enablement result: the port must align block tables,
causal boundaries, E5M2 scale semantics, partition order, and shuffled physical
pages; then pass non-growing numeric bounds, greedy trajectory, acceptance,
PPL, and the fixed GSM8K/MATH-500/HumanEval/MBPP/MT-Bench quality suites.

The first packed-GDN node trace on GPUs 4-7 produced no result.  A separate
task claimed the group during initialization, the reservation guard reported
`TP4_HANDOFF_COLLISION_ABORT`, and the run exited before model generation.
No external process was stopped and no timing from that attempt is retained.

### 2026-08-24 verifier follow-up: rejected seams and two bounded candidates

The complete-round acceptance gate remains **strictly below 20 ms**.  The
latest retained quality-safe result is still 23.678170 ms, so the unresolved
measured gap is at least 3.679 ms.  Isolated microbenchmarks below are not
endpoint evidence and must not be presented as completion of that gate.

Two short-path seam experiments were rejected before integration:

- The existing fused pull-all-reduce plus Gemma RMS operator measured
  3.005-3.013 ms for a TP4 128-pair graph chain, versus 1.463-1.469 ms for the
  current push-all-reduce plus RMS sequence.  It was also not bitwise equal on
  two ranks.  No source route was enabled.
- Reusing the existing paged-prefill operator for the Q=8 verifier measured
  0.436/14.872/57.935/115.838 ms per layer at 1K/32K/128K/256K.  Six head CTAs
  still serially scan the whole prefix, so merely sharing the eight query rows
  does not solve context parallelism.  This route is rejected.

The Flash-V100 split-KV framework was then extended narrowly to E5M2.  The
implementation keeps the existing FP16 routes unchanged, dispatches the
generic paged loader with `KV_DTYPE=fp8_e5m2`, preserves separate K/V scales,
and uses the existing split merge order.  The source compiles successfully;
the task build has SHA256 `e457f06f...9b0f23`.  The production-shape graph
microbenchmark, non-unit scales, shuffled physical pages, and endpoint route
remain pending because all eight GPUs are occupied by external TP4 jobs.

For the short-context target graph, the local
`Qwen3.8-27B-NVFP4` checkpoint was audited rather than assuming a foreign
checkpoint contract.  Its config hash is `1b3c7186...3a5c`; layers 0-55 use
native NVFP4 only for the MLP trunk, while attention, GDN, layers 56-63, and
the full-vocabulary LM head remain FP8.  This makes it a useful quality-gated
candidate: the DFlash2-sensitive head is not reduced to four bits, while most
of the verifier's MLP weight traffic is halved.  The already accepted no-MTP
route for this exact checkpoint measures 14.017 ms/token, but that number is
not a DFlash2 verification-round result.

The existing current-tree TurboMind NVFP4 implementation is the control.  A
prior real-weight M=1 QPN4 experiment saved a projected 1.090 ms across the 56
gate/up and 56 down calls, but it is M=1-only and therefore does not establish
an M=8 verifier gain.  A new bounded benchmark,
`benchmarks/kernels/benchmark_sm70_nvfp4_qpn2.py`, now loads the real TP-local
layer-55 shards and compares M=8 CUDA-graph replay against the production
TurboMind route.  The pinned v100-skinny QPN2 source compiled into a task-only
extension with SHA256 `09d96053...633a4`; no production dispatch has been
changed.  The next free-GPU action is this M=8 race, followed by a full DFlash2
boot only if the measured aggregate saving and numerical gate are positive.

The mixed checkpoint may become a default performance profile only after
paired FP8-target comparisons pass WikiText PPL, fixed GSM8K, MATH-500,
HumanEval, MBPP, and MT-Bench gates, plus acceptance-length and per-position
acceptance checks.  A low-bit microbenchmark alone cannot authorize that
change.

### 2026-08-24 QPN2 production admission and split-KV rejection

The generic Flash-V100 E5M2 split-KV follow-up is now rejected rather than
pending.  With production shape `B=1,Q=8,Hq=6,Hkv=1,D=256`, non-unit scales,
and shuffled physical pages, its per-layer medians were 0.124/1.936/7.791/
15.769 ms at 1K/32K/128K/256K.  The 1K result is numerically healthy (maximum
absolute error `1.526e-5` versus FP32), but the long-context results are worse
than the retained XQA path.  The generic split kernel still rereads one KV
stream per query head; it does not implement SGLang's grouped-GQA sharing and
therefore cannot be enabled.  The raw artifact is
`results/flash-v100-splitkv-verify-e5m2-b1-q8-g6-d256-v1.json`; the task-only
binary SHA256 is `e457f06f...9b0f23`.

The real-weight QPN2 operator race is positive at exact verifier width M=8,
but its control is the mixed NVFP4 checkpoint's existing TurboMind route, not
the FP8+QPN8 target used by the accepted 23.678170 ms complete-round baseline.
Across layer-0/rank-3 gate/up and down projections, replacing TurboMind with
QPN2 projects **3.814523 ms removed per complete mixed-target round**:
gate/up falls from 96.020 to 40.681 microseconds and down from 31.687 to
18.909 microseconds.  Relative L2 error versus the FP32 checkpoint oracle is
`0.000569` for QPN2 gate/up versus `0.000598` for TurboMind, and `0.000300`
for QPN2 down versus `0.000291` for TurboMind.  The retained artifacts are
`results/nvfp4-qpn2-vs-turbomind-real-layer0-rank3-m8-oracle-v1.json` and
`results/nvfp4-qpn2-vs-turbomind-real-layer55-rank0-m8-oracle-v1.json`.

That 3.814523 ms value must not be subtracted directly from 23.678170 ms.
Against the actual FP8+QPN8 baseline buckets, the current evidence projects
only about 1.8 ms, with a plausible 1.8-2.5 ms range pending the production
fused-gate measurement.  QPN2 alone therefore does not prove the sub-20 ms
gate; a new complete single-request DFlash2 trace must measure the movement,
and any remaining 1-2 ms gap must be removed from that trace's actual
critical-rank buckets.

A production QPN2 implementation is linked behind
`VLLM_SM70_NVFP4_QPN2=1`. Admission uses TP4 and the checkpoint-native gate/up
or down tensor layout; it does not inspect model or speculator identity. M<=8
uses QPN2, while larger dynamic M dispatches inside the opaque C++ operator to
the existing TurboMind layout. The switch remains default-off.

The exact production source compiles for SM70, its four routing tests pass,
and a full `_C` candidate contains QPN8, graph-only push, and all four QPN2
operators.  Candidate path:
`build/push-qpn8-qpn2-graphonly-v3/_C.abi3.so`, SHA256
`c133500e...3f068b5`.  The earlier task-only standalone hash is obsolete after
the final production-source edits and is intentionally not used as endpoint
evidence.

The production candidate passed the real-weight M=8 CUDA-Graph oracle on a
V100.  Gate/up is 96.759 versus 36.801 microseconds and down is 31.830 versus
19.794 microseconds against TurboMind, for a 4.031684 ms mixed-target
projection.  QPN2 relative L2 versus the FP32 checkpoint oracle is `0.000569`
for the fused gate/up output and `0.000300` for down.  The result is
`results/nvfp4-qpn2-production-layer0-rank3-m8-v1.json`, SHA256
`cfc0634d...1a5a6c`.  This passes the operator gate but does not alter the
correct-baseline warning above: only a complete DFlash2 trace can prove the
net movement from 23.678170 ms.

The first full-service launch exposed a checkpoint contract mismatch rather
than a QPN2 failure.  The mixed checkpoint declares calibrated E4M3 KV scales,
which upstream vLLM correctly refuses to reinterpret as E5M2 scales.  This is
the behavior retained by upstream PR #45040.  A task-cache-only model overlay
therefore keeps every original weight/tokenizer file by symlink but removes
the incompatible `kv_cache_scheme` from its copied config, making the explicit
E5M2 run use the same unit-scale storage contract as the accepted FP8 target.
The original checkpoint is unchanged.  Overlay config SHA256 is
`5b72ecd7...40f6e52`; the benchmark now records both that hash and the resolved
weight path.

The overlay launch passed the KV contract, but an unrelated TP4 job acquired
GPUs 4-7 between the availability check and worker initialization.  Free
memory fell to 24.41 GiB, below the matched 0.8 utilization contract, so vLLM
refused startup before model loading.  No endpoint or timing result is claimed
from either failed launch.  The next valid action remains the matched complete
single-request trace on a genuinely free TP4 group; the hard acceptance gate
is strictly below 20.000 ms.

The next owned launch reached DFlash2 warmup and exposed a second checkpoint
contract rather than a timing result: the mixed checkpoint quantizes
`lm_head`, while DFlash2 candidate selection intentionally requires an
unquantized target LM head.  The final task-cache overlay therefore maps only
`lm_head.weight` to the accepted FP8 target's BF16 `[248320,5120]` tensor and
removes `lm_head` from both the mixed quantization target list and its FP8 scale
entry.  All other target weights still resolve to the original mixed
checkpoint.  The combined overlay config and index SHA256 values are
`1b418636...01e8a` and `dacc00bf...836e`; the BF16 sidecar SHA256 is
`ddff1d66...b698ff`.  The original FP8 and mixed model directories remain
unchanged.

The launch audit also found that the production checkpoint selects
`CompressedTensorsW4A4Fp4` because its config contains dynamic NVFP4 input
activation metadata.  The first QPN2 integration had been attached to the
W4A16 scheme, so its real-weight microbenchmark remained valid but the full
model did not actually dispatch QPN2.  The exact admission and fused
gate/up/down dispatch now live in the W4A4 scheme, and the W4A16 file is back to
its original content.  The routing, FP8-prefill, and DFlash2 targeted suites
pass **78 tests**.  No endpoint speed claim is made until the newly queued
single-request node trace contains the QPN2 kernels and measures a complete
round strictly below the 20.000 ms gate.

Because the default safetensors iterator visits every tensor in each referenced
shard, deleting only the index entry was insufficient: the original monolithic
body still exposed `lm_head.weight_scale` during loading.  A streaming
task-cache rewrite now removes exactly `lm_head.weight` and
`lm_head.weight_scale` from the body without decoding or re-encoding any other
tensor.  The slim body is 21,296,297,056 bytes with SHA256
`9f49b437...389267`; all 1,951 retained tensor dtype/shape records match and
the first, middle, and last bytes of every retained tensor match the source.
The final index SHA256 is `94f500bf...c7ef2c`, and the benchmark records that
index plus every actually referenced weight realpath.  This replaces neither
the original checkpoint nor the DFlash2 unquantized-LM-head guard.

The first valid mixed-target node trace is a rejection, not a speed result to
retain.  Its complete round is **40.775579 ms**: draft 3.941496 ms,
draft-to-target 1.228315 ms, target 33.567780 ms, and target-to-draft
2.037988 ms.  The target graph has 1,462 nodes.  QPN2 itself matches its
microbenchmark, consuming 2.521294 ms for 56 gate/up calls and 1.518048 ms for
56 down calls.  The regression is instead 144 residual channel-FP8 projections
falling through to SM70 Marlin for **21.110017 ms**.  The single-request output
has acceptance length 3.823529; neither its 91.451 tok/s steady decode nor that
one-prompt acceptance value is a quality or performance acceptance claim.

The missing dependency was already implemented and DCO-signed in local commit
`5123a6a7b2` (`[Kernel] Recover Qwen3.8 NVFP4 SM70 decode`): its narrow
compressed-tensors W8A16 channel-FP8 adapter reuses the accepted TurboMind and
QPN8 operators.  Only that adapter, its fused-SiLU delegation seam, and focused
tests were ported here; unrelated attention and tuning changes from the commit
were not duplicated.  This also fixes the QPN2 production seam: without
`CompressedTensorsLinearMethod.apply_fused_silu_and_mul`, the full model ran a
separate Triton activation even though the scheme exposed the fused operator.
The combined compressed-FP8, QPN2, FP8-runtime, and DFlash2 suites pass **83
tests**.  A new complete trace must prove removal of the 21.110 ms Marlin bucket
and still meet the strict sub-20.000 ms gate.

### 2026-08-24 mixed-target recovery and remaining 1.807 ms

The channel-FP8 adapter and QPN2 route are now present in a clean combined
extension.  The retained `v8` complete single-request trace uses the mixed
NVFP4 target, QPN2 MLP, QPN8 residual projections, graph-only TP4 push,
sharded target-hidden projection, E5M2 target KV, FP16 draft KV, and an
unquantized target LM head.  Across 140 steady rank-cycles it measures:

| Phase | Mean wall |
| --- | ---: |
| Draft graph | 3.914734 ms |
| Draft to target | 1.097309 ms |
| Target graph | 15.367610 ms |
| Target to draft | 1.729242 ms |
| Complete round | **22.108896 ms** |

The target graph has 1,254 nodes.  Its largest service buckets are QPN8
projections (3.327 ms), QPN2 gate/up (2.511 ms), QPN2 down (1.568 ms), push
all-reduce (1.294 ms), recurrent GDN (1.200 ms), Flash-V100 attention
(0.738 ms), and the 127 Gemma suffix kernels (0.676 ms).  Both the draft
selector and target rejection still execute an approximately 0.98 ms dense
FP16 LM head.

The draft-to-target interval contained six four-byte device-to-host copies and
stream synchronizations.  Five were accidental: the five Flash-V100 groups
recomputed `seq_lens_cpu` even though `InputBatch.seq_lens_cpu_upper_bound`
already held the required contract.  Passing that tensor through
`MambaHybridModelState.prepare_attn()` removes exactly those five syncs.  The
new focused unit test passes.  The corrected `v10` trace, with the sharded
target-hidden projection explicitly restored after reboot, measures:

| Phase | Mean wall |
| --- | ---: |
| Draft graph | 3.914056 ms |
| Draft to target | **0.790517 ms** |
| Target graph | 15.357529 ms |
| Target to draft | 1.745331 ms |
| Complete round | **21.807433 ms** |

The complete median/min/max are 21.823373/21.557190/22.653044 ms.  This is the
current accepted local best, but it **does not pass** the strict sub-20.000 ms
gate: the measured deficit remains 1.807433 ms.  The one remaining D2H fence
is the batch-level GDN metadata ordering assertion.  A retained historical
GPU-assert experiment measured 24.093540 ms versus 23.679850 ms for its
same-generation synchronized control, so it is rejected; the final ordering
edge will not be removed on the basis of a theoretical sync saving.

At the exact production LM-head shape `M=8,K=5120,N_local=62080`, a new V100
probe measures `torch.mm` at 1.018880 ms p50 and the existing SM70 dense
full-logits operator at 0.812032 ms p50.  The 0.206848 ms per-call gain projects
to about 0.414 ms for the two heads in one complete round.  The probe's top-16
support and values are exact, but full-logit non-bitwise behavior still makes
this a quality-gated candidate rather than accepted endpoint evidence.  Raw
artifact: `results/lmhead-dense-sm70-vs-torch-m8-n62080-k5120-v11.json`.

The next bounded candidate preserves the accepted push transport and fuses
its following FP32 residual/Gemma RMSNorm suffix.  It targets the 128 push
collectives plus 127 RMS kernels as a 128-node chain.  It must first beat the
current production-shaped TP4 chain microbenchmark, keep the reduced residual
bitwise exact, and satisfy the FP16 normalized-output bound.  Only then may it
be combined with the LM-head candidate in a full Graph/Nsight run; neither
microbenchmark may be subtracted from 21.807433 ms to claim completion.

That push-suffix candidate is now rejected and fully removed from the
production source.  The first centralized-row implementation measured
2.544-2.550 ms for the 128-pair TP4 graph chain versus 1.475-1.480 ms for the
current push-plus-Triton sequence.  A second implementation fused each CTA's
partial square sum and distributed normalization across all five row CTAs; it
reduced the candidate to 1.638 ms but the matched control remained
1.470-1.472 ms.  The best fused form is therefore still about 11.3% slower.
Both versions kept residual output bitwise exact; the second version was
normalized-output bitwise exact on three ranks and differed in two FP16 values
on the fourth, with maximum absolute error 0.000244.  Artifacts are
`results/tp4-push-vs-push-fused-gemma-rms-m8-h5120-v2.json` and `v3.json`.
No endpoint run or speed claim is permitted from this failed branch.

### 2026-08-24 synchronized-fence removal and 20.746 ms endpoint

The final batch-level GDN metadata assertion was audited against both the
metadata builders and `ModelCudaGraphManager.run_fullgraph()`.  All metadata
kernels, copies, and graph replay are enqueued on the same current CUDA stream;
the device `.item()` checked a debug invariant but supplied no ordering edge.
The assertion is separately gated by `VLLM_SM70_DFLASH2_GDN_SYNC_ASSERT`.
After the exact token/acceptance trajectory passed, the production default was
changed to off; it remains an explicit opt-in metadata audit.  A structural CPU
invariant remains unconditional.

The `v17` single-request Graph/Nsight trace disables only that debug fence and
enables the already quality-gated dense FP16 LM-head route.  Across 140 steady
rank-cycles it measures:

| Phase | Mean wall |
| --- | ---: |
| Draft graph | 3.756561 ms |
| Draft to target | 0.156985 ms |
| Target graph | 15.288068 ms |
| Target to draft | 1.543990 ms |
| Complete round | **20.745603 ms** |

The complete median/min/max are 20.747498/20.523095/20.939874 ms.  This is a
1.061830 ms improvement over `v10`, but it still misses the strict sub-20 ms
gate by 0.745603 ms and is therefore not an accepted endpoint.  The output
token SHA256 remains
`22db7b2e81df17d3afdc6982fc1425769335c6cc7931da10e64bd6bac3459f28`;
the answer is correct and the complete acceptance contract is unchanged at
3.685714 with per-position counts `[28,22,17,12,8,5,2]`.  Steady decode is
174.016 tokens/s.  Result/report/sqlite SHA256 values are respectively
`1ee462b4...78a2e`, `7f686d51...f3ba`, and `1f82d151...3302`.

The remaining two local LM-head passes are now the bounded next target.  With
the real rank-0 FP16 shard `[62080,5120]`, channel-QPN8 local top-64 contains
every dense top-16 token in 512/512 random rows.  Re-evaluating those 64 rows
with the original FP16 weight via `index_select` plus batched dot products
reduces the complete local selector from 0.904192 to 0.518144 ms p50.  The
reranked top-1 matches 511/512 rows; maximum/mean absolute logit differences
are 0.00390625 and 6.68e-6.  Artifact SHA256 is
`39fde704...718a7`.  These random-hidden results authorize only a production
candidate, not a quality or endpoint claim.

The implementation is isolated behind
`VLLM_SM70_DFLASH2_QPN8_RERANK=0`.  It admits only SM70, TP4, the exact local
LM-head shape, FP16 weights, zero vocabulary padding, top-k 16/20, and at most
eight rows.  QPN8 supplies only a coarse top-64 support; the returned logits
are always recomputed from the original FP16 rows.  The same shared target LM
head serves both the draft selector and sparse target rejection.  An eager-only
shadow gate executes the candidate, logs real-hidden support/set/top-1
agreement, and returns the dense result so the audit trajectory cannot change.
The path remains default-off until four-rank real-weight coverage, real-hidden
shadow coverage, dataset quality/acceptance, and a new complete Graph trace all
pass.  Eighteen focused CPU routing tests and static checks currently pass.

The stricter four-shard `v18` probe uses local top-20, candidate width 64,
1,024 random rows per TP rank, and the same real checkpoint weights.  All
4,096 dense top-20 sets are fully contained in the QPN8 support (minimum
recall 1.0 on every rank).  Fixed split-8/non-fast QPN8 is 0.425-0.427 ms;
complete QPN8 plus original-FP16 rerank is 0.499-0.531 ms versus
0.893-0.923 ms for the dense selector.  The reranked top-1 matches 4,081/4,096
rows, while exact local top-20 set agreement is 3,971/4,096.  The maximum
returned-logit difference is one FP16 step (0.00390625).  Thus the coarse
support gate passes, but reduction-order differences still require real-hidden
and model-quality gates; `v18` alone does not authorize endpoint use.  The four
result SHA256 prefixes are `b07428ec`, `85e94de8`, `e56ed9b0`, and `fb4c1016`.
The complete CPU DFlash2 targeted file now passes 52 tests with 12 GPU tests
skipped under an explicitly hidden CUDA environment.

The `v19` eager shadow then exercised the same candidate on real hidden states
while always returning the dense local result.  Across four TP ranks it
observed 8,564 local rows (draft top-16 and target top-20), with **zero** dense
tokens missing from the QPN8 top-64 support.  Original-FP16 rerank set agreement
is 8,406/8,564 (98.1551%) and local top-1 agreement is 8,547/8,564 (99.8015%);
the largest ordered logit difference is 0.015625.  This passes the coarse
support gate and quantifies the remaining reduction/tie-order effect.  Because
shadow is eager-only and deliberately returns dense outputs, neither its
throughput nor its four truncated GSM8K answers are candidate quality evidence.
Log/result SHA256 values are `6dad2b23...482e` and `29c8ff88...753f`.  The next
required artifact is the active single-request Graph/Nsight trace; the hard
gate remains a measured complete mean strictly below 20.000 ms.

### 2026-08-24 exact-rerank endpoint and sub-20 ms replication

The first active Graph launch, `v20`, is rejected before timing.  Startup,
compilation, and QPN8 preparation succeeded, but the draft selector's TP
all-gather rejected a non-contiguous top-16 tensor.  The implementation had
sliced the first 16 columns from one persistent `[8,20]` output, retaining a
row stride of 20.  It now owns separate graph-stable `[8,16]` and `[8,20]`
value, position, and ID buffers.  Row-only views preserve contiguity without a
runtime copy.  Six layout cases (top-16/top-20 by 1/7/8 rows), the target
candidate route, Ruff, and `git diff --check` pass.  `v20` produced no timing
or quality result.

The corrected `v21` single-request probabilistic CUDA Graph/Nsight trace
crosses the strict complete-mean gate for the first time.  Across 140 steady
rank-cycles on physical GPUs 4--7 it measures:

| Phase | Mean wall |
| --- | ---: |
| Draft graph | 3.332724 ms |
| Draft to target | 0.158264 ms |
| Target graph | 15.281917 ms |
| Target to draft | 1.163312 ms |
| Complete round | **19.936216 ms** |

The complete median/min/max are 19.956518/19.751641/20.129946 ms.  Steady
single-request decode is 180.686 tokens/s.  The 128-token SHA256 is exactly
`22db7b2e81df17d3afdc6982fc1425769335c6cc7931da10e64bd6bac3459f28`,
the answer is correct, and acceptance is exactly unchanged at 3.685714 with
per-position counts `[28,22,17,12,8,5,2]`.  Result/report/sqlite SHA256 values
are `3d22d740...e1248`, `27175b2f...09c7`, and `5fe2773c...60e0`.

An unchanged `v22` replication then ran on physical GPUs 0--3.  Its 140
steady rank-cycles measure 3.339601 ms draft, 0.155429 ms draft-to-target,
15.330372 ms target, 1.165597 ms target-to-draft, and **19.990999 ms** complete
mean.  The median/min/max are 20.011181/19.785692/20.212695 ms, and steady
decode is 181.177 tokens/s.  Token SHA256, correct answer, acceptance length,
and every per-position acceptance count again match the dense control.  The
result/report/sqlite SHA256 values are `06fa010e...19b`, `af23b773...031`, and
`c34083f8...f62e`.

Thus the complete mean is below 20.000 ms in two independent runs and on both
physical TP4 groups; this is direct Graph/Nsight evidence rather than a
microbenchmark projection.  The margin is still only 0.009--0.064 ms and the
upper tail crosses 20 ms, so the candidate remains default-off pending a
larger stability margin and the multi-dataset/PPL quality gate.  The next
bounded microbenchmark checks whether the QPN8 support top-64 can skip ordering
before exact FP16 reranking; candidate order is not consumed by the rerank.

The `v23` sorted/unsorted support experiment rejects that optimization.  On the
real rank-0 shard, removing QPN8 top-64 ordering changes complete rerank p50
from 0.513024 to 0.501760 ms.  In the production fixed split-8 core the saving
is only about 4.1 microseconds per LM-head call, or about 8 microseconds per
complete DFlash2 round.  This is below run-to-run noise and changes tie order,
so production retains sorted support.  The sorted/unsorted artifact SHA256
values are `5eb1ad6f...31de` and `da3499cc...fb6`.

The accepted next micro-optimization removes the 5.2 MiB FP16
`index_select` temporary and its batched matrix multiplication.  A fixed-grid
Triton kernel directly computes each selected original-FP16 vocabulary-row dot
product.  For the production `[8,62080,5120]`, top-64 to top-20 shape, the
complete selector p50 falls from 0.517120 to 0.464896 ms.  All 512/512 top-20
sets and top-1 IDs match the gathered-bmm reference; maximum ordered-logit
difference is 0.001953125.  The selected configuration is `BLOCK_K=1024`, four
warps, with a graph-stable 512-program grid masking inactive rows.  The
artifact SHA256 is `b898c410...0631`.  Dedicated V100 tests for one, seven, and
eight rows pass, as do the focused CPU routing/layout tests and static checks.

Two unchanged full Graph/Nsight runs then validate the direct indexed-dot
endpoint on both TP4 groups:

| Run / physical GPUs | Draft | D-to-T | Target | T-to-D | Complete mean | Median | Max | Steady decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v25` / 4--7 | 3.324374 | 0.157883 | 15.272063 | 1.139965 | **19.894284 ms** | 19.909407 | 20.087468 | 182.516 tok/s |
| `v26` / 0--3 | 3.332363 | 0.157819 | 15.313486 | 1.145328 | **19.948997 ms** | 19.960269 | 20.148380 | 181.216 tok/s |

Each result contains 140 steady rank-cycles.  Both target and draft are FULL
CUDA Graphs and the indexed-dot path is captured without a graph break.  Both
runs reproduce the baseline 128-token SHA256
`22db7b2e81df17d3afdc6982fc1425769335c6cc7931da10e64bd6bac3459f28`,
the correct answer, acceptance length 3.685714, and per-position counts
`[28,22,17,12,8,5,2]` exactly.  The `v25` result/report/sqlite SHA256 values are
`c82ef281...0952`, `f9c26647...6c81`, and `7c7645ef...006a`; the corresponding
`v26` values are `04498a79...0cbc`, `0ebd0b86...4244`, and
`3de27587...9d1`.  Parsed round artifacts have SHA256
`a49fe2cd...be3b` and `bb9d09dc...5e2d`.

The strict complete-mean and median target is therefore independently
replicated below 20 ms after the final kernel change.  It is not yet an
all-sample tail claim: the observed maxima remain 20.087--20.148 ms.  The
rerank and GDN debug-fence defaults remain conservative until paired
multi-dataset generation and prompt-logprob/PPL gates pass; a further
0.2--0.4 ms safety-margin reduction remains the immediate performance target.

### 2026-08-24 paired quality rejection and dense-order correction

The first paired 16-question GSM8K quality run (`v27`) rejects the active
QPN8 reranker despite its exact single-prompt trace.  Both variants are
sequential single-request FULL Graph runs with probabilistic draft sampling,
temperature 1.0, top-p 0.95, top-k 20, identical shuffled rows and request
seed.  Dense control scores 11/16 with aggregate acceptance 4.490716; the
candidate scores 9/16 with acceptance 4.412224.  The acceptance delta is
-0.078493, below the fixed -0.05 gate, and only 5/16 token streams match.
Artifacts `gsm8k16-qpn8-indexeddot-v27-{control,candidate}.json` have SHA256
`ced400d1...aee` and `e4c89772...4eeb`; the failed comparison has SHA256
`9650f531...045c`.  Consequently the candidate remains default-off and its
sub-20 traces are not accepted as a default quality-safe endpoint.

The follow-up `v28` microbenchmark corrects the reference used during rerank
development.  It compares against the production TurboMind
`sm70_f16_gemm_out`, not `index_select+bmm`.  Across 1,024 random rows on the
four real TP shards, QPN top-64 contains every production dense top-20 token.
Depending on reduction configuration, 99.969--99.994% of candidate logits are
bitwise equal, yet candidate-order top-20 set equality is only 250--253/256
rows per shard.  This isolates a second source of drift: `torch.topk` tie
selection over a shuffled 64-element support does not reproduce its selection
over the original 62,080-element vocabulary order.

`v29` restores that dense order by scattering the 64 re-evaluated FP16 logits
into a persistent full-shard buffer initialized to negative infinity before
local top-k.  All four shards then match production dense output exactly for
all 1,024 rows: ordered token IDs, top-20 sets, top-1 IDs, and returned FP16
values.  Complete local selector p50 is 0.5007--0.5038 ms, versus
0.4608--0.4628 ms without dense-order restoration and 0.894--0.930 ms for the
production dense selector.  The four result SHA256 values are
`93ebf045...15ec`, `6ec9f526...c064`, `9e4e9868...2a`, and
`904790b6...bb3`.  Focused CPU tests (12) and V100 tests (4) pass.  A new
same-source paired GSM8K run is required before this correction can re-enter a
complete Graph/Nsight trace.

The same-source `v30` paired run rejects the corrected active path at the
acceptance gate.  Both variants are sequential single-request FULL Graph runs
over the same 16 shuffled GSM8K questions, probabilistic draft sampling,
temperature 1.0, top-p 0.95, top-k 20, and identical request seeds; the only
environment difference is `VLLM_SM70_DFLASH2_QPN8_RERANK=0/1`.  Control and
candidate both score 11/16 with zero invalid answers, but pooled acceptance
changes from 4.581867 to 4.498670 (delta -0.083197), below the fixed -0.05
gate.  Seven of 16 token streams are identical.  Candidate mean steady decode
is 249.052 token/s versus 244.506 token/s for control, but that speed is not
accepted because the paired acceptance gate fails.  Control, candidate, and
comparison SHA256 values are respectively `a3289717...db1`,
`d446ec34...9776`, and `6b32738b...71f`.  The reranker remains default-off and
must not be included in a quality-safe sub-20 ms claim.

This result also narrows the remaining numerical mechanism.  The earlier
real-hidden shadow observed zero dense top-k tokens missing from QPN8 top-64,
so the failure is not coarse-support recall.  The candidate recomputes only
the selected rows with a different reduction schedule from the production
TurboMind dense GEMM; real-hidden ordered logits can differ by up to 0.015625.
Restoring dense vocabulary tie order fixes exact random-row tests but cannot
restore those production reduction values.  The next selector candidate must
either execute the selected rows with the same Tensor-Core reduction contract
as the dense kernel or pass a larger paired distribution/acceptance gate; it
cannot be promoted from random-hidden equality alone.

### 2026-08-24 exact context-projection overlap rejection

The default-off `v31` experiment moves the position-independent DFlash2
target-hidden combine, TP4 sharded context projection, and fused five-layer K/V
projection onto an auxiliary CUDA stream before target sampling.  The main
stream waits on one event only when RoPE and cache slots are available.  It is
numerically exact in the full single-request probabilistic run: the 128-token
SHA256 remains `22db7b2e...9f28`, the answer remains correct, and acceptance is
exactly 3.685714 with per-position counts `[28,22,17,12,8,5,2]`.  Both target
and draft remain FULL CUDA Graphs and there is no deadlock or graph break.

Performance rejects the overlap.  Across 140 steady rank-cycles, the trace
measures 3.802137 ms draft, 0.254715 ms draft-to-target, 16.041994 ms target,
1.748872 ms target-to-draft, and **21.847718 ms complete**.  Complete
median/min/max are 21.858843/21.612554/22.046909 ms, and observed steady decode
is 165.686 token/s.  This is about 1.10 ms slower than the 20.745603 ms exact
`v17` baseline, so the gate remains default-off.

The dual-stream timeline explains why subtracting the isolated projection cost
was invalid.  The 0.824 ms dense LM head occupies the SMs first; the 0.118 ms
context projection cannot start until its tail.  It then overlaps the many
small target top-k/rejection kernels and stretches the main sampling tail by
about 0.25 ms.  The auxiliary all-gather and K/V projection finish before the
event wait, but the contention and cross-stream launch schedule cost more than
the hidden work.  Result, qdstrm, report, sqlite, and parsed-round SHA256 values
are `7a75a159...181`, `0cfa07d8...17d`, `83a05726...99e`,
`1891c169...6c1`, and `6b87003b...869`.  No speed from this experiment is
accepted; a future overlap attempt needs a fused single-stream schedule or an
explicitly prioritized kernel, not another generic auxiliary stream.

### 2026-08-24 native grouped verifier and long-context result

The retained verifier optimization does not reuse DDTree indices or skip target
attention. It implements the MRV2 block-eight semantics directly in
Flash-V100: eight query rows share each E5M2 K/V page, accumulate all six GQA
heads, and write the eight verifier outputs in one pass. Admission requires
SM70, FP16 Q, E5M2 KV, B1, Q=8, Hq=6, Hkv=1, D=256, causal decode, and the
original single-request metadata. All other shapes keep the previous path.
The dedicated CUDA-Graph test poisons and replays the sequence length and cache
pages; six V100 cases pass against the FP32 oracle.

The paired same-process CUDA-Graph microbenchmark at page size 3,296 measures:

| Context | Native one-pass | Native two-pass | Previous XQA |
| ---: | ---: | ---: | ---: |
| 1K | 0.068608 ms | 0.097280 ms | 0.230400 ms |
| 32K | 0.356352 ms | 0.544768 ms | 1.024000 ms |
| 128K | 1.214464 ms | 1.864704 ms | 3.021312 ms |
| 256K | 2.396160 ms | 3.690496 ms | 6.011392 ms |

The microbenchmark is
`results/flash-v100-grouped-paired-page3296-v72.json`, SHA256
`aeccb306...657d9`. It establishes the isolated attention saving; endpoint
claims use the full-model traces below.

Matched 32K and warmed 128K single-request FULL-Graph runs preserve the output
token stream and complete acceptance counters within each on/off pair:

| Input | Grouped | Complete round | Target phase | Steady decode |
| ---: | :---: | ---: | ---: | ---: |
| 32K | off | 36.310865 ms | 30.248113 ms | 77.537 tok/s |
| 32K | on | **26.361447 ms** | **20.451644 ms** | **88.392 tok/s** |
| 128K | off | 69.854780 ms | 63.708214 ms | 106.428 tok/s |
| 128K | on | **40.778072 ms** | **34.646866 ms** | **182.294 tok/s** |

At 32K the complete round falls 27.40% and the target phase 32.39%; at 128K
they fall 41.62% and 45.62%. The 128K pure-decode result rises 71.28%. The
repetitive fixed long-context prompt has acceptance 6.095 at 32K and 7.647 at
128K in both variants; these values are paired route checks, not corpus
acceptance estimates. The four phase artifacts and SHA256 values are
`v77`/`2d705aa9...0c0e`, `v76`/`de6d922d...4235`,
`v79`/`e61e858f...cc1c`, and `v80`/`76164c8c...8deb`.

The short fixed 128-token trace is neutral rather than a claimed win: grouped
is 21.117097 ms versus 21.060181 ms off. The long-context admission threshold
therefore remains `max_model_len >= 32768`; the feature can be disabled with
`VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY=0`. Dataset generation below is the
deciding endpoint gate, where grouped improves aggregate throughput despite
mixed prompt and output lengths.

### 2026-08-24 replicated sub-20 ms short-round bundle

Before the grouped-attention addition, the complete candidate bundle was
replicated on both physical TP4 groups under FULL target and draft CUDA Graphs:

| Run | Draft | D-to-T | Target | T-to-D | Complete mean / median / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v52-r2` | 3.374848 | 0.156565 | 15.292271 | 1.073660 | **19.897343 / 19.895756 / 20.098654 ms** |
| `v53` | 3.385334 | 0.157777 | 15.320752 | 1.079131 | **19.942993 / 19.948144 / 20.145704 ms** |

Thus mean and median are below 20 ms twice; this is not an all-sample-tail
claim because both maxima remain slightly above 20 ms. Phase artifacts have
SHA256 `eb8feded...5482` and `412ad1a0...7599`. These traces authorize the
performance candidate, while the larger score/PPL run below authorizes its
sampling and model-quality behavior.

### 2026-08-24 probabilistic 2K score and PPL gate

Per the final acceptance decision, greedy token-for-token equivalence is no
longer a production gate. Quality is evaluated with probabilistic DFlash2,
temperature 1.0, top-p 0.95, target top-k 20, reasoning effort `xhigh`, fixed
seed, `max_tokens=2048`, and natural EOS. Every run is sequential B1 and uses
the same target checkpoint, TP4, E5M2 target KV, FP16 draft KV, Flash-V100, and
FULL Graph. The target is the local mixed NVFP4/FP8 checkpoint with a BF16
LM-head; it must not be described as the official unquantized checkpoint.

The first 128-row dataset contains 32 GSM8K, 32 MATH-500, 32 HumanEval, and 32
raw-MBPP prompts. The raw MBPP prompts copied the speed benchmark and omitted
the function name and tests, so their 0--2/32 results are invalid as task
scores and excluded. The remaining 96 scored cases are:

| Variant | GSM8K | MATH-500 | HumanEval | Included total |
| --- | ---: | ---: | ---: | ---: |
| Target-only | 29/32 | 26/32 | 28/32 | **83/96 (86.458%)** |
| Dense verifier DFlash2 | 30/32 | 24/32 | 28/32 | 82/96 (85.417%) |
| Native grouped DFlash2 | 30/32 | 25/32 | 28/32 | **83/96 (86.458%)** |

Target-only versus grouped has three wins on each side and McNemar exact
two-sided p=1.0. Dense versus grouped has two dense-only and three grouped-only
wins. Grouped therefore matches target-only aggregate score and improves one
case over the dense-verifier DFlash2 sample; no task-score regression is
observed.

The corrected MBPP prompt includes the requested function name and all three
held-out tests, following the Google MBPP evaluation contract. Target-only and
grouped both score **25/32 (78.125%)**. They have two exclusive wins each, 23
shared passes, five shared failures, and McNemar exact p=1.0. Target has eight
length finishes and grouped six. The grouped MBPP run overlapped an unrelated
process that appeared after reservation, so its throughput is discarded;
pass/fail scoring completed without OOM or engine error.

Across the valid 96-case mixed run, performance is:

| Variant | Aggregate output | Mean request steady decode | Mean completion tokens / verify step |
| --- | ---: | ---: | ---: |
| Target-only | 67.128 tok/s | 68.008 tok/s | n/a |
| Dense verifier DFlash2 | 176.341 tok/s | 218.436 tok/s | 4.2243 |
| Native grouped DFlash2 | **193.484 tok/s** | **231.126 tok/s** | **4.2384** |

Grouped is +9.72% in aggregate output and +5.81% in mean steady decode versus
the dense verifier. Its pooled acceptance is 3.749104 versus 3.749153 dense;
the difference is negligible, while the request-mean completion length is
+0.0141. The summary artifact is
`quality/dflash2-mixed32-quality-2k-v83-summary.json`, SHA256
`9ca37f6a...dab7`. Corrected MBPP score artifacts have SHA256
`92908b64...08f3` and `326e559b...339e`.

The supplementary WikiText-2 gate scores 16,376 prompt tokens in eight equal
2,048-token windows. Token-weighted/geometric PPL is 5.497872 target-only and
5.497376 grouped, a relative **-0.0090%** change. Mean absolute prompt-logprob
difference is 0.007519 and the maximum is 1.101103. The generic comparison tool
labels this as failure only because its historical bounds require every
logprob and every per-window PPL to agree within `1e-6`; that exactness contract
is intentionally superseded here by the user-approved score/PPL contract.
Aggregate PPL does not regress. Artifact SHA256 is `d2c54784...e0c7`.

The official DFlash2 model card publishes lossless-distribution semantics plus
acceptance/throughput, but no GSM8K, MATH-500, HumanEval, or MBPP task accuracy.
Its published acceptance lengths are 5.46/5.28/4.39/4.79 respectively under
H200, SGLang, FlashAttention-3, block eight, maximum 4,096 tokens, and the full
official datasets. They are useful reference points, not a direct comparison
to this 32-row-per-suite V100/mixed-checkpoint/max-2K gate. See
<https://huggingface.co/incoai/Qwen3.8-27B-DFlash2>.

### Audit disposition and isolation

The score, PPL, paired acceptance, microbenchmark, short-trace, and long-context
results remain evidence for the measured workload. They do not install a model
profile. Every new leaf is default-off and independently selected by explicit
environment opt-in plus hardware, algorithm configuration, dtype, tensor
layout, cache/graph mode, sampling, and bounded-concurrency contracts.

The earlier clean 8-token TP4 auto-profile smoke remains historical route
evidence only; automatic profile injection was removed during merge audit.
Unsupported requests execute the previous dense/general path rather than
changing routing semantics.
