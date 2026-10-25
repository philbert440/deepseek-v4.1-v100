# SM70 DeepSeek V4 DSpark Verifier and Long-Context Ledger

This ledger tracks the 2026-08-24 TP8/V100 campaign for
`/data/models/DeepSeek-V4-Flash-0731`. It complements the historical N7 ledger
and prevents short route-hit smokes from being reused as verifier, long-context,
or quality evidence.

## Acceptance Contract

- Eight V100 GPUs, TP8, one request, DeepSeek V4 Flash DSpark with seven draft
  tokens, temperature 1.0, top-p 1.0, and the production strict rejection
  sampler.
- Primary speed gate: at least 100 pure decode token/s on the pinned 120-token
  Chinese HTML prompt with 3,500 generated tokens. TTFT and prefill are reported
  separately.
- Primary latency gate: complete verifier at most 20 ms per speculative round.
  Complete verifier means target forward, target logits, and strict
  rejection/sampling. Draft work, accepted-state commit, CPU scheduling, and
  total round wall are reported separately rather than hidden in that number.
- Long-context gate: report raw contexts 1K, 4K, 16K, 64K, 128K, and 252K with
  identical output length and sampling. Report verifier/round cost, acceptance
  by position, emitted tokens/round, and pure decode separately.
- Quality gate: compare no-speculation and DSpark on deterministic GSM8K,
  HumanEval, a multilingual/multitask LongBench subset, and 128K/252K needle
  retrieval. A speed result cannot promote a route that loses the paired quality
  or output-health gates.
- Shared-machine rule: launch only after an event-driven free-GPU gate, never
  stop unrelated owners, avoid frequent polling, and terminate every task-owned
  worker after the artifact is complete.

## Reproduced Baselines

Current latest-main source plus the admitted SM70 route repairs reproduces the
historical endpoint target. Three 3,500-token runs measured 118.481, 115.975,
and 116.288 token/s; median throughput is 116.288 token/s. The third run
accepted 2,816 of 4,781 proposed draft tokens (0.589), emitted 5.123 tokens per
round, and had per-position unconditional acceptance
`0.8873/0.7818/0.6911/0.6047/0.5066/0.3880/0.2635`.

The matched no-speculation route measured 66.014 token/s and 15.1483 ms/token
at 1,024 input plus 256 output tokens. A synchronized DSpark profile measured:

| Component | Steady median per round |
|---|---:|
| Target forward, M=8 | 34.295 ms |
| Target logits | 0.326 ms |
| Strict rejection/sample | 0.499 ms |
| Complete verifier | 35.120 ms |
| Draft GPU work | 4.419 ms |
| Total round wall | 43.505 ms |

The 20 ms goal therefore requires target-forward work, not sampler-only tuning.

## Admitted Source Gates

The following changes have passed focused current-source gates but still require
their stated end-to-end promotion gate:

1. FP32 mHC staging now covers verifier rows M=1..8. M=8 falls from 0.05383 to
   0.01257 ms per call and projects 3.507 ms saved across 85 calls. Four V100
   numerical cases were bitwise exact.
2. The M=8 graph can use a bounded 2,048-token short-context bucket instead of
   reserving the whole maximum context. Speculative buckets require an explicit
   `VLLM_SM70_DSV4_DECODE_CONTEXT_BUCKETS`; CPU dispatcher tests pass, while the
   synchronized TP8 transfer remains required.
3. A default-off `VLLM_SM70_DSV4_PRIVATE_COMPRESSOR_STATE=1` route stores
   single-request SM70 compressor C4/C128 intermediates in bounded private
   device rings when prefix caching, pipeline parallelism, KV transfer, and
   parallel drafting are absent. Seven V100 ring/workspace tests and eleven CPU
   route tests pass. It cannot become the default before the long-context
   quality and capacity gates finish.
4. A default-off `VLLM_SM70_INDEXER_DECODE_CUBLAS=1` route gathers long-context
   paged FP8 keys once and uses FP32 cuBLAS scores for bounded row shapes. It
   retains per-head ReLU and FP8 scale semantics. The captured-graph test,
   shorter dynamic replay, graph-tail mask, and workspace-width gate pass on
   V100; the incomplete long-context quality gate keeps it opt-in.

5. A default-off `VLLM_SM70_MXFP4_MOE_GROUPED_VERIFIER=1` route extends the
   single-launch, one-row-per-slot MXFP4 contract from M=8 to M=2..M=8. This is
   required before a shorter or confidence-scheduled DSpark block can reduce
   verifier latency; the previous M=2..M=7 path launched one operation per
   compact expert and erased the benefit of verifying fewer rows.

The focused M=5 and M=6 CUDA Graph gates covered mixed overlap, all-distinct
slots, and six hot experts. Slot grouping and real-expert grouping were bitwise
at every pipeline stage and after a changed-route replay in all six cases. The
mixed-route results are:

| Verifier rows | Active-expert loop | Slot grouped | Real-expert grouped |
|---:|---:|---:|---:|
| M=5, 22 unique of 30 slots | 1.98837 ms | 0.18552 ms | 0.18000 ms |
| M=6, 26 unique of 36 slots | 1.32993 ms | 0.34895 ms | 0.27260 ms |

All-distinct expert grouping is slightly slower than slot grouping
(`+0.121 ms` projected over 43 layers at M=5 and `+0.210 ms` at M=6), so the
full-model route must report the observed production overlap. Evidence is
retained under `/data/models/dsv4-verifier-20ms-variable-grouped-r1`; the
single V100 was released after the gate.

The 0731 checkpoint declares an official DSpark block size of five and stores
`mtp.2.confidence_head.proj.weight`. The runtime instantiates and loads that
optional head only for explicit confidence-threshold scheduling or alignment
diagnostics and fails cleanly if a requested projection is absent. Ordinary
serving performs no confidence projection. DeepSpec computes conditional step
probabilities with a sigmoid,
uses their cumulative product for prefix reliability reporting, and truncates
at the first conditional probability below a threshold. Before production
confidence scheduling, this runtime must first collect calibration data on the
pinned datasets and profile M=2..M=8 service; a raw uncalibrated threshold is
not an acceptance or latency proof. Static N=4/N=5 trials use the same strict
rejection sampler and therefore preserve the target distribution while the
hardware width crossover is measured.

The indexer crossover gate uses FP32 scores, signed head weights, top-k 512,
M=8, H=64, D=128, and CUDA Graph replay:

| Compressed keys | Fused paged Triton | Gather + FP32 cuBLAS | 21-layer projection |
|---:|---:|---:|---:|
| 1,024 | 0.06854 ms | 0.02096 ms | -0.999 ms |
| 2,048 | 0.13455 ms | 0.02519 ms | -2.297 ms |
| 4,096 | 0.23433 ms | 0.04738 ms | -3.926 ms |
| 8,192 | 0.45660 ms | 0.07680 ms | -7.976 ms |
| 16,384 | 0.85016 ms | 0.14493 ms | -14.810 ms |

All 40 rows retained the exact top-k set; maximum logit absolute difference was
`1.221e-4`. The opt-in crossover is 1,024 compressed keys. Generic batches,
noncontiguous inputs, more than eight rows, and non-ReLU semantics retain the
existing fused paged kernel.

The production block-size candidate verifies M=5 rows. Its matched crossover
gate retained the exact top-k set in all 30 rows and measured:

| Compressed keys | Fused paged Triton | Gather + FP32 cuBLAS | 21-layer projection |
|---:|---:|---:|---:|
| 1,024 | 0.03548 ms | 0.01905 ms | -0.345 ms |
| 2,048 | 0.06881 ms | 0.02196 ms | -0.984 ms |
| 4,096 | 0.13532 ms | 0.03410 ms | -2.126 ms |
| 8,192 | 0.26614 ms | 0.05806 ms | -4.370 ms |
| 16,384 | 0.58173 ms | 0.10066 ms | -10.103 ms |
| 65,536 | 1.92835 ms | 0.32707 ms | -33.627 ms |

The maximum logit absolute difference was `1.221e-4`. The microbenchmark owner
released its V100 immediately after writing the artifact.

The speculative SM70 graph context candidate uses explicit raw widths 2K, 4K,
16K, 64K, and 128K when the maximum is 256K. Longer requests retain the generic
full-context graph. `VLLM_SM70_DSV4_DECODE_CONTEXT_BUCKETS` is required to
enable the candidate; an explicit MTP bucket override, including an empty
value, remains authoritative. Automatic non-speculative selection uses only
the compressed-index operator configuration and never architecture or
checkpoint identity.

## Long-Context Diagnosis

DeepSeek V4 does not use the repository's classic Flash-V100 dense-attention
decode route. Its production log identifies packed-FP8 SM70 sparse MLA with C4,
C128, SWA-128, and top-k 512. Sparse attention after selection is nearly fixed
work; the C4 indexer scan grows with compressed history and was the expected
linear decay source. The opt-in private compressor rings address capacity and
memory growth, while opt-in gather-once FP32 cuBLAS addresses the repeated
index scan.

`benchmarks/benchmark_dsv4_dspark_long_context.py` records each completed
context incrementally. It uses server-reported prompt usage, separates TTFT
from pure decode, snapshots speculative counters around each request, and gates
suffix-marker retrieval, replacement characters, token diversity, and repeated
token runs. When given the TP0 server log it also records interior synchronized
profiler windows for target forward, target logits, strict rejection/sampling,
and their summed complete-verifier cost. The verifier row count and acceptance
positions follow the configured speculative width instead of assuming N=7. A
zero speculative width records the matched no-speculation long-context speed
and output-health curve without fabricating verifier counters.

## Quality Gate

`benchmarks/benchmark_dsv4_quality_api.py` provides deterministic HumanEval and
LongBench-subset API gates. HumanEval code executes with Landlock filesystem
isolation, seccomp network/process restrictions, a private temporary directory,
and CPU, address-space, output-file, and process limits. The sandbox has proven
that normal Python works while `/home`, `/data`, global `/tmp`, networking,
forking, and signalling other processes are denied.

The first paired LongBench subset is `hotpotqa`, `multifieldqa_zh`,
`gov_report`, and `lcc`. Official LongBench metrics are loaded from the pinned
local checkout; `jieba==0.42.1`, `rouge==1.0.1`, and `fuzzywuzzy==0.18.0` are
isolated under `/data/models/dsv4-quality-deps` rather than installed into the
shared runtime environment. Every quality artifact includes SHA-256 hashes for
the HumanEval tasks, LongBench prompt/max-length configs, and each selected
dataset. LongBench records retain their source-row indices so the matched
no-speculation and DSpark scores cannot silently use different samples.
`benchmarks/compare_dsv4_quality_results.py` then rejects any reference-pass to
candidate-fail transition in HumanEval, GSM8K, an individual LongBench row, or
either final/anywhere needle retrieval. Aggregate scores alone cannot hide a
per-sample regression.

## Rejected or Deferred Paths

- A TP8 M=8 64 KiB hierarchical all-reduce was slower than NCCL in isolation
  and saved only 0.028 ms across 87 joined calls, within noise. It was fully
  reverted and must not be repeated without a different communication design.
- The expert-row grouped MXFP4 candidate is tracked separately. Its bitwise
  microbenchmarks justify a full TP8 trial but not promotion by themselves.
- Disabling per-head ReLU, changing index-cache layout, reducing index top-k,
  or weakening strict rejection are outside the quality contract.

## Artifact Roots

- 100+ endpoint: `/data/models/v100-dsv4-0731-tp8-dspark-prob-n7-latest-c4-20260824-r6`
- no-speculation: `/data/models/v100-dsv4-0731-tp8-nospec-latest-c4-20260824-r1`
- synchronized breakdown: `/data/models/v100-dsv4-0731-tp8-dspark-prob-n7-profile-latest-c4-20260824-r1`
- mHC source gate: `/data/models/dsv4-verifier-20ms-mhc-source-gate-r2-cuda128`
- private rings: `/data/models/dsv4-verifier-20ms-private-ring-source-gate-r1`
- indexer crossover: `/data/models/dsv4-verifier-20ms-indexer-crossover-r3`
- indexer source gate: `/data/models/dsv4-verifier-20ms-indexer-source-gate-r1`
- M=5 indexer crossover: `/data/models/dsv4-verifier-20ms-indexer-m5-r1`
- M=5/M=6 grouped verifier: `/data/models/dsv4-verifier-20ms-variable-grouped-r1`
- rejected TP8 all-reduce: `/data/models/dsv4-verifier-20ms-tp8-ar-screen-r1`
