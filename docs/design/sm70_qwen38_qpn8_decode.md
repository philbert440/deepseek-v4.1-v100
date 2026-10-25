# SM70 Qwen3.8-27B-FP8 QPN8 Decode Acceptance

Date: 2026-08-23

## Decision

The memory-neutral QPN8 dense route is available as an opt-in for the
Qwen3.8-27B-FP8 TP4 no-MTP model and projection shapes when configured
`max_num_seqs` is at most eight. Set `VLLM_SM70_FP8_QPN8=1` to request it.
The model-quality gate passes, but pure-FP8 default promotion still requires a
same-source speed rerun with the same number of steady decode intervals. An
explicit request with missing QPN8 operators fails closed.

The gate checks the Qwen3.8 text architecture, 5120 hidden size, 17408
intermediate size, 64 layers, full-attention interval 4, head dimension 256,
and TP4. Qwen3.6 and both 35B acceptance routes remain unchanged.
Configurations admitting more than eight live sequences retain TurboMind at
load time until a measured M>8 QPN8 route exists. MTP configurations also
retain TurboMind until their verification widths pass speed and quality gates.
These guards prevent the functional large-M dequantization fallback from
becoming a concurrency or speculative-decode regression.

This acceptance is a no-MTP, M=1 decode speed result with numerical tests up
to M=8 and model-quality batches up to eight. M>8 remains functional through
one bounded 85 MiB FP16 weight workspace. Large-M fused gate/up also has a
transient `M x 8704` FP16 GEMM output, so that fallback is not an accepted
high-concurrency memory or speed result.

## Contract

- Model: `Qwen3.8-27B-FP8`, TP4 on physical V100-SXM2-32GB GPUs 4-7.
- Runtime: Python 3.12, Torch 2.10.0+cu128, CUDA 12.8, source checkout and
  source-built SM70 extension; no wheel changes are used.
- Production speed case: no MTP, E5M2 KV, `FLASH_ATTN_V100`, CUDA graphs,
  `max_model_len=262144`, input 1024, output 256, and one live request.
- Official sampling: `temperature=1.0`, `top_p=0.95`, `top_k=20`, request seed
  20260815. Pure decode excludes TTFT/prefill and uses the 255 steady token
  intervals.
- Quality case: identical baseline/candidate datasets and seeds, 4096-token
  rolling-PPL windows, and generation/log-likelihood batches 1, 2, and 8.

## Source Route

Checkpoint-native block-FP8 weights are repacked once at model load into one
QPN8 `[K, N]` code tensor plus grouped FP16 scales. The old TurboMind tensor
is replaced rather than retained, avoiding the approximately 6 GiB/rank
duplicate-weight experiment. The matched candidate reported 19.89 GiB
available KV memory versus 19.80 GiB for the control.

The opt-in model and shape gate admits only:

| Projection | TP-local K | TP-local N | Decode schedule |
|---|---:|---:|---|
| MLP gate/up with fused SiLU×up | 5120 | 8704 | split-K 8, two accumulator chains, prefetch |
| MLP down | 4352 | 5120 | split-K 16, one accumulator chain |
| GDN/full-attention output | 1536 | 5120 | split-K 16, one accumulator chain |

GDN input and full-attention QKV remain on TurboMind. The M decision is inside
the opaque C++ operator so AOTInductor cannot incorrectly specialize a Python
M=1 branch for a later prefill. M=1-8 calls QPN8 directly; larger M
materializes one layer at a time in the shared weight workspace and calls FP16
GEMM. The large-M gate/up temporary described above is bounded by the engine's
`max_num_batched_tokens`, not by the 85 MiB weight workspace.

## Pure Decode Result

| Route | Pure TPOT | Steady decode | Change |
|---|---:|---:|---:|
| Matched TurboMind control | 16.985327 ms | 58.874 tok/s | baseline |
| QPN8 accepted subset | 15.808000 ms | 63.259 tok/s | +7.448%, -1.177327 ms/token |
| Complete-source TurboMind control | 16.961295 ms | 58.958 tok/s | source A/B baseline |
| Complete-source QPN8 candidate | 16.347973 ms | 61.170 tok/s | +3.752%, -0.613322 ms/token |

The matched operator-integration pair generated all 256 tokens and finished by
length. Their sampled token streams first differ at output token 91; sampled
token identity is not the quality gate because both trajectories are coherent
and the acceptance criterion is PPL plus fixed reasoning, knowledge, and
Chinese datasets.

The complete-source control generated 256 tokens and finished by length. The
complete-source QPN8 request used the same prompt, sampling contract,
and source extension, but encountered the model's normal stop token 248044 at
token 233. Its TPOT therefore covers 232 steady intervals rather than the
control's 255. The earlier 63.259 tok/s matched operator-integration result is
retained as evidence of headroom. Because 61.170 tok/s uses 232 intervals while
the control uses 255, another same-source equal-interval run is required before
pure-FP8 default promotion.

## Nsight Systems Per-Token Decomposition

Node tracing adds profiler overhead, so these rows explain composition rather
than replace the unprofiled TPOT above.

| Steady replay metric | Control | QPN8 | Delta |
|---|---:|---:|---:|
| Replay interval | 17.418 ms | 16.437 ms | -0.981 ms |
| GPU activity union | 16.870 ms | 15.894 ms | -0.976 ms |
| Total idle gaps | 0.548 ms | 0.543 ms | -0.005 ms |
| Kernel launches/rank/token | 1085.5 | 1085.9 | unchanged |

The trace gain is therefore GPU kernel service, not a hidden CPU wait or
launch-gap reduction. Dense service changes from 10.335 ms/rank/token to:

| Candidate dense component | Time | Launches/rank/token |
|---|---:|---:|
| Fused QPN8 gate/up | 4.018 ms | 64 |
| QPN8 down and output projections | 3.040 ms | 128 |
| Remaining TurboMind GDN/full-QKV input | 2.332 ms | 64 |
| Total dense | 9.391 ms | 256 |

This accounts for about 0.944 ms/token of the trace improvement. TP
communication is 1.680 to 1.637 ms and LM-head/sample/gather remains about
1.33 ms, so neither explains the main win.

## Resource Release and Peak-Band Residency

Whole-request NVML averages are not used as achieved FLOPS or bandwidth.
Nsight Systems establishes how long grids can occupy the machine, and Nsight
Compute supplies achieved per-kernel counters.

The grid-limited occupancy ceiling moves from 93.56% of service below 25%
occupancy in the control to 48.56% below 25%, 32.11% at 25-50%, and 19.26% at
50-75% with QPN8. This is a duration distribution, not an average-utilization
claim.

For the accepted QPN8 kernels, NCU reports:

| Shape | Registers/thread | Achieved occupancy | SM throughput | DRAM throughput | Tensor-pipe elapsed |
|---|---:|---:|---:|---:|---:|
| Down | 50 | 48.90% | 27.20% | 82.74% | 9.50% |
| Output | 50 | 48.17% | 27.69% | 71.32% | 8.23% |
| Fused gate/up | 64 | 42.33% | 29.98% | 87.74% | 10.08% |

Counting the useful M=1 dense multiply-add work, throughput is 1.177 to
1.295 TFLOP/s per GPU, or 4.71 to 5.18 TFLOP/s across TP4.
Dense-service-weighted Tensor Core pipe activity is 7.97% to 9.15%,
equivalent to about 9.99 to 11.47 TFLOP/s per GPU, or 39.96 to 45.88 TFLOP/s
across TP4, relative to the V100's nominal FP16 Tensor Core peak. These answer
different questions: the first is model-useful arithmetic, while the second
includes tensor instructions spent inside the split-K implementation. Neither
is a whole-request average.

Across the 9.391 ms candidate dense service, 7.059 ms resides in the 25-50%
SM-throughput band and 2.332 ms remains below 25%; no dense kernel reaches
50% SM throughput. For DRAM, 6.172 ms resides at 75-90% and 3.219 ms at
50-75%; no dense kernel reaches 90%. Dense-service-weighted achieved
occupancy rises from 9.96% to 36.25%, but the duration bands above are the
primary evidence.

The result is not “the V100 cannot reach peak compute.” QPN8 makes a much
larger fraction of dense time sustain high memory duty and occupancy, while
the unchanged approximately 0.54 ms idle budget shows that host launch work
is not the present first-order limit.

## SGLang-V100 Comparison Boundary

The current [sglang-V100 README](https://github.com/haohervchb/sglang-V100/blob/main/README.md)
now reports a Qwen3.8-27B-FP8 target-only, E5M2 KV, TP4 row at 58.2 tok/s for
1K decode and 50.8 tok/s for 25K decode. Its table uses one cold request and
256 greedy output tokens. The linked
[target-only audit](https://github.com/haohervchb/sglang-V100/blob/main/benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md)
records 17.180 ms TPOT, or 58.21 tok/s, at 1,024 input tokens. The 63.259
tok/s QPN8 result here is 8.67% above that published number, so the public
evidence points to this route being faster. It is not a strict head-to-head
because this acceptance uses fixed random sampling and a different
prompt/client harness. A formal claim still requires running SGLang's
target-only command with the same prompt, sampling, graph, and pure-decode
accounting used here.

The published 136.6 tok/s DFlash2-8 row is speculative and is not a no-MTP
comparison.

## Quality Gate

The gate compares the prior TurboMind route and QPN8 with identical prompts,
few-shot examples, seeds, and batching. Lower is better for the three
WikiText metrics; higher is better for the remaining accuracy metrics.

| Dataset and metric | Samples | Control | QPN8 | Result |
|---|---:|---:|---:|---|
| WikiText word perplexity | 62 documents | 8.790013 | 8.789904 | pass, lower |
| WikiText byte perplexity | 62 documents | 1.501519 | 1.501515 | pass, lower |
| WikiText bits/byte | 62 documents | 0.586422 | 0.586419 | pass, lower |
| GSM8K 5-shot strict exact match | 128 | 69.531% | 70.312% | pass, +1 item |
| GSM8K 5-shot flexible extraction | 128 | 73.438% | 73.438% | pass, equal |
| MMLU abstract algebra 5-shot | 100 | 67.0% | 67.0% | pass, equal |
| MMLU high-school computer science 5-shot | 100 | 89.0% | 89.0% | pass, equal |
| MMLU professional law 5-shot | 100 | 63.0% | 66.0% | pass, +3 items |
| MMLU three-subject macro | 300 | 73.0% | 74.0% | pass |
| C-Eval computer network 5-shot | 19 | 73.684% | 73.684% | pass, equal |
| C-Eval high-school Chinese 5-shot | 19 | 73.684% | 73.684% | pass, equal |
| C-Eval advanced mathematics 5-shot | 19 | 57.895% | 57.895% | pass, equal |

No evaluated metric regresses. This closes the quality side of the opt-in
route; it does not replace the missing equal-interval performance rerun.

## Build and Tests

- The complete source `_C` target built 47/47 objects for CUDA 12.8 and SM70;
  the resulting 45 MiB extension registers all six prepare, dequantize,
  prefill, dispatch, GEMM, and fused-gate QPN8 operators.
- The operator race passed 960 rows. Maximum relative L2 was `2.851e-4`,
  minimum cosine was `0.99999988`, and maximum absolute error was `9.77e-4`;
  CUDA Graph replay deltas were zero.
- Seventeen focused Python tests pass, covering opt-in/explicit-off policy,
  model and shape gates, workspace reuse, M=1 and M>8 opaque dispatch, fused
  gate/up dispatch, and warmup registration.
- Ruff lint/format, changed-line clang-format, Python byte compilation, shell
  syntax checks, and `git diff --check` pass.

## Next Optimization Order

1. Extend QPN8 with a single-launch two-dimensional row-tile grid and race
   M=9, 11, and 16 against TurboMind. Do not compose an 8-row launch with a
   separate tail launch, and do not assume the batch-one winner stays best.
2. Race M=17, 32, and 64 and keep a measured route map. The external
   [v100-skinny QPN study](https://github.com/dnv2003/v100-skinny/blob/main/docs/qpn_race_notes.md)
   also finds a crossover to WMMA/TurboMind above the small-batch band.
3. Before high-concurrency acceptance, replace or chunk the large-M fused
   gate/up temporary and record peak memory under the supported batched-token
   limit.
4. Add a bounded large-M route for GDN input/full-QKV, then quality-gate it.
   Those 64 remaining TurboMind launches consume 2.332 ms/token.
5. Continue shape-specific work on fused gate/up (4.018 ms) and down/output
   (3.040 ms), measuring time above 90% DRAM and 50% SM rather than optimizing
   a whole-run average.
6. Revisit the 1.637 ms communication and 1.33 ms LM-head only after the dense
   residue falls. The current trace does not justify prioritizing Python launch
   overhead.
7. Repeat the decomposition at 25K and 70K. SGLang's target-only 70K trace
   reports 10.60 ms/token in FP8 projections and 10.35 ms/token in attention,
   so the 1K optimization order must not be hard-coded for long context.

The ordering is also consistent with external evidence, without treating a
newer-architecture implementation as a V100 drop-in. The
[v100-skinny M sweep](https://github.com/dnv2003/v100-skinny/blob/main/docs/REPRODUCE.md)
uses a measured SIMT/QPN/WMMA dispatch map rather than one kernel for every M.
The [CUTLASS grouped scheduler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/grouped_scheduler.md)
shows how a bounded persistent CTA work queue can retain residency across many
independent small tiles; it becomes relevant only if independent problems can
be grouped without crossing transformer dependencies. The
[FlashInfer paper](https://arxiv.org/abs/2501.01005) similarly makes dynamic,
load-balanced scheduling compatible with a static CUDA Graph. For this route,
those are design constraints to test under concurrency, not reasons to replace
the measured SM70 kernels speculatively.
