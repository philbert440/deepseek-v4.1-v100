# SM70 DFlash2 long-context verification decay

## Scope

This task measures, attributes, and reduces the context-length growth of one
MRV2 DFlash2 verification round on V100. The historical study started on
private DFlash2 verifier head `e8daa778f55cf16df311f8ce8dd032e2281f1680`;
the public change set contains only the five retained verifier commits rebased
onto current `main`. It does not replace or retune DFlash1, `dflash_ddtree`,
Eagle, or MTP.

DFlash2 names the measured workload, not a runtime activation identity. The
kernel entry admits only the explicit SM70, E5M2 paged-KV, Q1-Q8, Hq6/Hkv1,
D256, page-layout, causal-mask, and workspace contracts. It does not inspect a
model name, checkpoint, `model_type`, or architecture identity.

Quality promotion remains score based. Greedy token identity is diagnostic;
the production gates are probabilistic task scores, PPL, acceptance length,
and output health against the same target-only checkpoint.

## Frozen workload

- Target: local Qwen3.8-27B mixed NVFP4/FP8 checkpoint, TP4.
- Draft: `incoai/Qwen3.8-27B-DFlash2`, FP16 draft KV.
- Target KV: `fp8_e5m2`.
- Attention: `FLASH_ATTN_V100` target and draft routes.
- Execution: FULL CUDA Graph, sequential batch one.
- Sampling: temperature 1.0, top-p 0.95, target top-k 20, probabilistic
  proposals, fixed seed.
- Selector top-K: checkpoint contract 16; speculative tokens: 7.
- Context sweep: 1K, 32K, 128K, and 256K boundary.

All comparisons must keep model, TP/GPU set, graph mode, KV dtypes, sampling,
and route gates identical. Prefill/TTFT are reported separately from steady
decode and verification-round wall time.

## Existing evidence to reproduce

The parent task recorded the following unprofiled paired measurements:

| Context | Prior complete round | Grouped complete round | Prior target phase | Grouped target phase |
| --- | ---: | ---: | ---: | ---: |
| 32K | 36.311 ms | 26.361 ms | 30.248 ms | 20.452 ms |
| 128K | 69.855 ms | 40.778 ms | 63.708 ms | 34.647 ms |

From 32K to 128K, the grouped complete round grows by 14.417 ms and its target
phase grows by 14.195 ms. This strongly localizes the observed growth to target
verification, but it does not yet distinguish paged attention service,
dependency gaps, launch geometry, communication placement, or trace overhead.

The retained grouped-attention microbenchmark reported 0.0686/0.3564/1.2145/
2.3962 ms at 1K/32K/128K/256K. These values must be re-run and reconciled with
the full-model critical-path wall before using them as a closed decomposition.

## Closed trace attribution

The retained 32K and 128K Nsight Systems traces use the same grouped one-pass
kernel and target CUDA Graph route. Summing the partial and combine kernels,
then dividing by the number of target graph replays across all four TP ranks,
gives the following service time per target verification graph:

| Context | Partial average | Partial calls | Combine average | Attention per target graph |
| --- | ---: | ---: | ---: | ---: |
| 32K | 0.313606 ms | 1,344 | 0.008282 ms | 5.150214 ms |
| 128K | 1.198762 ms | 1,088 | 0.008295 ms | 19.312912 ms |

There are sixteen full-attention layers per target graph. The attention service
therefore grows by 14.162698 ms between 32K and 128K, while the measured target
phase grows by 14.270586 ms. Grouped attention explains 99.24% of target-phase
growth and 95.83% of complete-round growth in this pair. Draft, selector,
sampling, TP exchange, and host residuals are not material causes of this
context-length slope.

Both traces launch the partial kernel with grid `(2, 40, 1)`, block size 256,
128 registers per thread, and 39,424 bytes of static shared memory. The fixed
80-CTA grid maps one CTA to each V100 SM even as the KV length quadruples. The
kernel's `__launch_bounds__(256, 2)` and resource arithmetic permit two resident
CTAs per SM (65,536 registers and 78,848 shared-memory bytes), so the first
bounded experiment is 40 versus 80 context splits. This is a hypothesis about
exposed parallelism, not yet a performance claim: an exact-shape paired
microbenchmark must distinguish occupancy benefit from memory-bandwidth
saturation and extra split-reduction cost.

The experiment follows the same split-KV principle used by the official
[FlashInfer scheduler](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh),
which derives its maximum grid from CUDA occupancy, and by
[Flash-Decoding](https://princeton-nlp.github.io/flash-decoding/). Recent
[PersistentKV](https://arxiv.org/abs/2606.26666) results likewise motivate
adaptive sequence splitting and page-aware scheduling for long-context decode.
These sources guide the search space; only same-machine measurements can
promote a V100-specific default.

## Independent upstream audit

`sglang-V100` is derived from this project's V100 work. It remains useful for
route and result cross-checks, but it is not independent evidence for an
optimization decision. The bounded search therefore uses primary upstream
implementations and papers, pinned at the audited revisions:

- Official FlashAttention revision
  [`0251105a`](https://github.com/Dao-AILab/flash-attention/tree/0251105a2fb19d2957484b7f023cd8c115286ced)
  packs `seqlen_q * qheads_per_kv_head` into the M dimension in
  [`pack_gqa.h`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/hopper/pack_gqa.h),
  then computes dynamic Split-KV occupancy from the packed M work in
  [`flash_prepare_scheduler.cu`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/hopper/flash_prepare_scheduler.cu).
  This is the direct algorithmic source for combining the six Q heads that
  share the Qwen3.8 verifier KV head. Its SM80/SM90 implementation is not
  copied; only the exact PackGQA scheduling principle is ported to Volta WMMA.
- FlashInfer revision
  [`c18a974a`](https://github.com/flashinfer-ai/flashinfer/commit/c18a974a88b8c7da88be0338a7478636f1dffa17)
  explicitly keeps head-dimension-256 prefill within the 64-KiB SM70/SM75
  shared-memory limit. Its release JIT successfully compiled on this V100,
  but the grouped Q8/H6/D256 paged launch still ended in an unspecified launch
  failure during CUDA Graph setup. The direct backend route is therefore
  rejected; its packing and scheduler layout remain useful source material.
- TensorRT-LLM revision
  [`958d651b`](https://github.com/NVIDIA/TensorRT-LLM/tree/958d651b18800da1da0677160e0277f701c0ff84)
  supports multi-block and multi-query-token XQA, but
  `supportConfigHMMA` rejects `SM < 80`. ONNX Runtime XQA likewise explicitly
  requires compute capability 8.0 or newer. Neither is a drop-in V100 kernel;
  both independently confirm head-group packing plus context splitting as the
  intended speculative-verification topology. TensorRT-LLM's speculative
  multi-block scheduler further caps the packed grid at one SM wave and avoids
  scaling short histories, independently matching the bounded 80-CTA policy
  tested below.
- [PersistentKV](https://arxiv.org/abs/2606.26666) maps native paged GQA work
  by `(request, KV head, sequence split)`, explicitly reusing each K/V tile
  across the grouped query heads. Its compact workqueue is valuable for ragged
  concurrent batches; the frozen single-request Q8 verifier has no empty work
  items to compact. The head-group mapping is directly applicable, while the
  workqueue is retained as a later concurrency experiment rather than being
  conflated with the B1 result.
- xFormers Split-K combines split states with LSE and supports small causal
  query counts plus experimental paged attention. Its current Triton route
  requires SM80, so it is algorithmic cross-check material, not a new V100
  dependency.
- [FlashAttention-3](https://papers.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf),
  [LeanAttention](https://arxiv.org/abs/2405.10480), and
  [FlashDecoding++](https://arxiv.org/abs/2311.01282) provide the exact
  split-softmax reduction, Stream-K work distribution, and GEMM-oriented
  decode alternatives. They motivate bounded follow-ups, not unmeasured
  performance claims on Volta.

TinyFA revision `bc755512` was also inspected and excluded: its fast path
requires SM75-era asynchronous-copy machinery, lacks E5M2 paged KV support,
and its paged decode mapping does not pack all six query heads into one KV
load. This audit leaves native SM70 PackGQA as the smallest dependency-closed
candidate.

## Split-count microbenchmark

The first candidate doubles the maximum context splits from 40 to 80. With
two GQA head groups this changes the fixed partial launch from 80 to 160 CTAs,
matching the kernel's two-block-per-SM launch bound. No attention arithmetic,
mask, FP8 decode, or MRV2 scheduling code changes.

An A/B/A CUDA-Graph run on physical GPU 1 used the production Qwen3.8 target
shape (Q8/H6/KV1/D256, E5M2 KV, page size 3,296), 40 warmups and 200 measured
rounds per point:

| Context | Split 40 A | Split 80 | Split 40 A2 | Split 80 versus A/A2 |
| --- | ---: | ---: | ---: | ---: |
| 1K | 0.058368 ms | 0.067584 ms | 0.058368 ms | +15.79% / +15.79% |
| 32K | 0.319488 ms | 0.261120 ms | 0.322560 ms | -18.27% / -19.05% |
| 128K | 1.204736 ms | 0.955392 ms | 1.206272 ms | -20.70% / -20.80% |
| 256K | 2.377728 ms | 1.864704 ms | 2.378752 ms | -21.58% / -21.61% |

At sixteen full-attention layers, the operator delta projects to -0.934 ms at
32K, -3.989 ms at 128K, and -8.208 ms at 256K per verification graph. The 1K
operator regression projects to +0.147 ms, or about +0.74% of the retained
approximately 20 ms complete short-context round. This remains a projection
until the TP4 full-model A/B is complete.

The split-80 extension passes all six targeted grouped-verifier tests: five
random physical-page cases against the FP32 reference, and CUDA Graph replay
while mutating runtime sequence length. These are operator correctness gates;
acceptance length, PPL, and task scores remain required before promotion.

## Full-model 32K and 128K confirmation

A TP4 FULL-Graph Nsight run used the same 128K prompt, 128 generated tokens,
probabilistic sampling seed, model/KV dtypes, and route gates as retained
split-40 run `v80`. The only implementation change was the split-80 Flash-V100
extension:

| Phase | Split 40 | Split 80 | Delta |
| --- | ---: | ---: | ---: |
| Complete verification round | 40.778072 ms | 36.683712 ms | -4.094360 ms (-10.04%) |
| Target | 34.646866 ms | 30.557332 ms | -4.089534 ms (-11.80%) |
| Draft | 4.604373 ms | 4.627743 ms | +0.023370 ms |
| Draft to target | 0.230538 ms | 0.225575 ms | -0.004963 ms |
| Target to draft | 1.296296 ms | 1.273062 ms | -0.023233 ms |

The partial kernel changed from grid `(2, 40, 1)` to `(2, 80, 1)`. Its mean
service time fell from 1.198762 ms to 0.944041 ms per full-attention layer
(-21.25%); combine rose from 0.008295 ms to 0.010430 ms. Across sixteen layers,
the net attention prediction is -4.041 ms, explaining 98.8% of the measured
target-phase improvement. The earlier operator projection was -3.989 ms, only
about 0.10 ms from the full target result.

The fixed diagnostic request retained identical accepted-token counts
`[17, 17, 17, 16, 16, 15, 15]`, mean acceptance length 7.647059, and all 128
output token IDs. This is trajectory evidence, not a substitute for the
required score/PPL gates.

The matching 32K TP4 FULL-Graph trace independently confirms the shorter
context point:

| Phase | Split 40 | Split 80 | Delta |
| --- | ---: | ---: | ---: |
| Complete verification round | 26.361447 ms | 25.238442 ms | -1.123005 ms (-4.26%) |
| Target | 20.451644 ms | 19.468326 ms | -0.983318 ms (-4.81%) |
| Draft | 4.365014 ms | 4.257568 ms | -0.107446 ms |
| Draft to target | 0.260419 ms | 0.235765 ms | -0.024654 ms |
| Target to draft | 1.284369 ms | 1.276783 ms | -0.007586 ms |

The split-80 partial kernel averages 0.248999 ms at 32K, versus 0.313606 ms
for split 40. The measured target reduction is within 0.05 ms of the operator
projection. Across the paired full-model endpoints, the 32K-to-128K complete
round growth falls from 14.416625 ms to 11.445270 ms (-20.61%), while target
growth falls from 14.195222 ms to 11.089005 ms (-21.88%). The remaining slope
is still serious: target accounts for 96.89% of the optimized complete-round
growth, so subsequent work remains scoped to grouped paged attention.

## Physical-page and tile-balanced follow-up

Two address-equivalent scheduling changes pass the same six operator tests:

- Specialize the actual FP8 kernel-page sizes (1,648 and 3,296) at compile
  time, replacing runtime integer division with constant arithmetic. The
  Qwen3.8 hybrid cache reports a 1,648-token logical attention block, while the
  grouped verifier receives the 3,296-element physical FP8 view; the final
  trace proves the `<page=3296>` specialization is selected.
- Divide the sequence by complete 32-token WMMA tiles, distributing the
  remainder across splits. This removes per-split padded tail tiles without
  changing visibility, loaded KV bytes, softmax, or WMMA arithmetic.

Repeated CUDA-Graph A/B/A runs show no 32K regression and a further
approximately 1.7% operator reduction at 128K versus plain split 80. The final
TP4 FULL-Graph endpoints are:

| Phase | Split 80 at 32K | Final at 32K | Split 80 at 128K | Final at 128K |
| --- | ---: | ---: | ---: | ---: |
| Complete verification round | 25.238442 ms | 25.237987 ms | 36.683712 ms | 36.482220 ms |
| Target | 19.468326 ms | 19.455442 ms | 30.557332 ms | 30.354906 ms |
| Draft | 4.257568 ms | 4.269400 ms | 4.627743 ms | 4.606652 ms |

At 128K, the partial kernel falls from 0.944041 ms to 0.926568 ms per
full-attention layer; sixteen layers predict -0.280 ms and the measured target
delta is -0.202 ms. At 32K, the partial kernel is 0.245912 ms and complete wall
is unchanged within 0.001 ms. The final 32K-to-128K complete-round slope is
11.244233 ms, 22.01% below split 40 and 1.76% below plain split 80. The final
target slope is 10.899464 ms, 23.22% below split 40. Target attention still
accounts for 96.93% of the residual complete-round slope.

A logical-work roofline proxy narrows the remaining operator bottleneck. The
two padded 32-row GQA groups perform about 2.15/8.59 GFLOP per layer at
32K/128K; the measured partial times correspond to only 8.73/9.27 logical
TFLOP/s. Their requested E5M2 K/V payload corresponds to about 136/145 GB/s
before accounting for duplicate-group L2 reuse. Both rates stay nearly flat as
context grows and are far below V100 peak figures. This is not an NCU hardware
counter claim, but it rules out treating the residual as unavoidable saturated
HBM traffic and prioritizes flat-QK/PV utilization, softmax synchronization,
and GQA packing over further page-address arithmetic.

The final 128K diagnostic request preserves the split-80 accepted-token counts
and 7.647059 mean acceptance length. At 32K, the single-request diagnostic
changes from 6.142857 to 6.095238 (-0.047619), inside the 0.05 diagnostic gate;
dataset-level acceptance and model-quality gates remain authoritative.

An attempted compact mapping for the two-head GQA tail passed the six operator
correctness tests but was 6.3-7.0% slower than split 80 across the micro sweep.
The run also observed a late external-worker ownership race, so its absolute
timings are not promotion evidence; the uniformly negative direction was
sufficient to revert the experiment completely. Nsight Compute hardware
counters are unavailable to the current user (`ERR_NVGPUCTRPERM`), so no NCU
counter claim is made.

A second exact-address experiment replaced sixteen redundant page divisions,
modulos, and block-table loads per D=256 row with one half-warp calculation and
broadcast. It preserved register count and passed all six operator correctness
tests, but regressed the one-pass kernel by 3.51%/2.36%/2.47%/2.63% at
1K/32K/128K/256K in a production-page-size A/B/A run. On this Volta path, the
independent address arithmetic is hidden better than the added shuffle
dependency. The candidate was reverted completely.

## Short-context floor and rejected split-K QK

The missing 1K TP4 FULL-Graph endpoint now uses the same final binary and
sampling/configuration contract as the 32K and 128K traces:

| Phase | 1K mean per verification round |
| --- | ---: |
| Complete | 21.634854 ms |
| Target | 16.348011 ms |
| Draft | 3.760873 ms |
| Draft to target | 0.242986 ms |
| Target to draft | 1.282984 ms |

The grouped partial and combine kernels average 0.057043 and 0.005948 ms per
full-attention layer. Across sixteen layers they contribute about 1.007854 ms
to the target graph. Subtracting this measured service leaves approximately
15.340 ms of context-independent target work and 20.627 ms of complete-round
work. Thus long-attention work explains the decay, but eliminating it entirely
would not by itself satisfy a sub-20-ms short-context goal; the fixed target,
draft, and return path remain a separate optimization problem.

An exact split-K experiment divided the D=256 QK accumulation between two
warps and joined the FP32 partials before the unchanged softmax. Limiting
unrolling removed all local stack spills (128/126 registers for the one-/two-
pass kernels), and all six grouped-verifier correctness tests passed. A/B/A
medians nevertheless changed by -0.40% at 32K, +0.11% at 128K, and +0.28% at
256K versus the second baseline. There is no stable long-context gain, so the
experiment was reverted and is not part of the performance result.

## Pack-GQA48 follow-up

The accepted grouped baseline launches two 256-thread CTAs per context split:
one four-head group and one two-head tail. Both CTAs load and decode the same
single-KV-head page. Following the official FlashAttention PackGQA mapping,
the follow-up packs all six heads and eight verifier tokens into one 48-row,
512-thread CTA. QK and PV remain Volta WMMA, online softmax remains FP32, and
the split outputs retain the same FP16-partial/FP32-LSE reduction contract.
The production partial kernel uses 128 registers, no local stack, and 50,944
bytes of opt-in dynamic shared memory.

The first packed prototype retained 128 tokens per split. It improved the
32K/128K/256K operator by approximately 20-21%, but regressed 1K by 19-21%
because only eight large CTAs were launched. Reducing the packed split quantum
to 64 tokens recovered the 1K parallelism, but the single-query utility shape
then crossed its strict FP32-reference mean-error gate. The retained dispatch
therefore keeps 128-token splits for Q1, forces one split for Q2-Q8 at total KV
length at most 128, and uses 64-token splits otherwise. This is a numerical
dispatch boundary, not a sampling or acceptance heuristic.

The final production-page A/B/A CUDA-Graph medians are:

| Context | Accepted grouped A/A2 | Pack-GQA48 | Delta versus A/A2 |
| --- | ---: | ---: | ---: |
| 1K | 0.057344 / 0.057344 ms | 0.046080 ms | -19.64% / -19.64% |
| 32K | 0.257024 / 0.257024 ms | 0.205824 ms | -19.92% / -19.92% |
| 128K | 0.939008 / 0.937984 ms | 0.745472 ms | -20.61% / -20.52% |
| 256K | 1.838080 / 1.838080 ms | 1.448960 ms | -21.17% / -21.17% |

At the ultra-short 128/256-token fixed shapes the packed route costs about
0.002-0.003 ms more than the accepted grouped kernel; at 512 tokens it is
already 10.9-12.5% faster. These sub-millisecond boundary points do not affect
the frozen 1K-or-longer promotion gate, but remain covered by correctness
tests.

The expanded operator suite covers Q1/Q3/Q4/Q5/Q6/Q8, 128/256/512/1K and
longer boundaries, random physical pages, one-/two-pass equivalence, and CUDA
Graph replay. All thirteen collected cases pass. The maximum absolute error is
unchanged from the accepted grouped baseline at every audited shape; the largest
positive mean-error delta is `1.60e-7`, below the per-shape `5e-7` guard.

The TP4 FULL-Graph endpoints confirm that the operator gain lands on the full
target critical path without a short-context regression:

| Context/phase | Accepted grouped | Pack-GQA48 | Delta |
| --- | ---: | ---: | ---: |
| 1K complete | 21.634854 ms | 21.303621 ms | -0.331233 ms (-1.53%) |
| 1K target | 16.348011 ms | 16.043125 ms | -0.304886 ms (-1.86%) |
| 32K complete | 25.237987 ms | 24.390177 ms | -0.847810 ms (-3.36%) |
| 32K target | 19.455442 ms | 18.633571 ms | -0.821871 ms (-4.22%) |
| 128K complete | 36.482220 ms | 33.334558 ms | -3.147662 ms (-8.63%) |
| 128K target | 30.354906 ms | 27.208276 ms | -3.146630 ms (-10.37%) |

The traced packed partial averages 0.733521 ms per full-attention layer and its
combine averages 0.010906 ms. The diagnostic request keeps the exact 128 output
token IDs, accepted counts `[17, 17, 17, 16, 16, 15, 15]`, and mean acceptance
length 7.647059. Seventeen verification steps produce 128 output tokens, so
the conservative steady-round projection is about 225.9 token/s at 128K,
versus 206.4 token/s for the accepted grouped endpoint.

The 32K-to-128K complete-round slope falls from 11.244233 to 8.944381 ms
(-20.45%), and the target slope falls from 10.899464 to 8.574705 ms (-21.33%).
The 32K/128K geometric-mean complete round falls from 30.343662 to 28.513782
ms (-6.03%). Diagnostic mean acceptance changes by +0.030303 at 1K, +0.047619
at 32K, and exactly zero at 128K, all inside the frozen 0.05 boundary.

WikiText PPL is exactly equal to the accepted grouped candidate across all
16,376 prompt-logprob positions. Against target only, the maximum PPL delta is
0.003952 and the mean absolute prompt-logprob delta is 0.007519, exactly the
same quality envelope recorded before Pack-GQA. Mixed task-score gates remain
required before promotion.

## Score and PPL promotion evidence

The final split-80/page-specialized/tile-balanced binary was evaluated with
probabilistic proposals, target sampling at temperature 1.0/top-p 0.95/top-k
20, fixed seed, a 2,048-token cap, and natural EOS. The mixed dataset supplies
32 GSM8K, 32 HumanEval, and 32 MATH500 cases. Its embedded MBPP prompts have an
incompatible function-name contract and are excluded; the independent
`mbpp-tests-32` dataset supplies the valid MBPP result.
The tested extension SHA256 is
`4a925265cfed934ebd7772be2a1eccaa64163d5324f87d31ba8ee99d10c3a044`.

| Suite | Target only | Prior grouped | Final candidate |
| --- | ---: | ---: | ---: |
| GSM8K | 29/32 | 30/32 | 30/32 |
| HumanEval | 28/32 | 28/32 | 27/32 |
| MATH500 | 26/32 | 25/32 | 26/32 |
| MBPP, independent tests | 25/32 | 25/32 | 26/32 |
| Valid total | 108/128 | 108/128 | 109/128 |

The suite-level movement is one HumanEval case down and a net two MATH/MBPP
cases up; the valid aggregate is therefore one case above both controls, not a
quality-regression claim. WikiText prompt-logprob/PPL output is exactly equal
to the prior grouped candidate across all eight samples (zero maximum and mean
difference). Against target only, the maximum PPL difference is 0.003952 and
the mean absolute prompt-logprob difference is 0.007519.

Acceptance also remains inside the frozen 0.05 gate. On the 128-request mixed
run, request-mean completion tokens per verification step change from 4.238430
to 4.218840 (-0.019589), and pooled acceptance length changes from 3.749104 to
3.714972 (-0.034132). On independent MBPP they change from 3.527945 to
3.509266 (-0.018679), while pooled acceptance changes from 3.421634 to
3.431016 (+0.009382). The final runs produce 101,312 and 28,070 output tokens,
respectively; 103/128 and 24/32 requests terminate naturally before the cap.

## Measurement order

1. Reproduce low-overhead full-model B1 baselines at 1K/32K/128K/256K.
2. Record synchronized target, draft, selector/sampling, communication, and
   host residual timing from the same requests.
3. Capture short Nsight Systems graph-node traces at 32K and 128K, aligned by
   replay ordinal across all four TP workers.
4. Classify target graph nodes into attention, quantized GEMM, GDN/norm,
   LM-head/selector, collectives, and dependency gaps.
5. Build the smallest exact-shape microbenchmark for the confirmed growing
   kernel family. Run Nsight Compute only after the runtime route is proved.

## Promotion gates

- Reduce geometric-mean 32K/128K complete verification-round wall without a
  regression greater than 1% at either length.
- Do not increase 1K complete-round wall by more than 1%.
- Preserve request-mean acceptance length within 0.05 and pooled acceptance
  within sampling noise.
- Preserve the existing task-score and WikiText PPL gates before enabling a
  numerical or scheduling change by default.
- Validate the 256K boundary for capacity, output health, and latency after a
  candidate passes at 128K.

## Current status

The 32K/128K trace attribution is closed: target grouped attention is the
long-context verification bottleneck. Pack-GQA48 is the retained candidate;
its 1K/32K/128K full-graph endpoints, 1K-256K operator sweep, task scores, PPL,
and dataset-level acceptance gates are recorded above. At the initial public
merge, the route remained explicitly opt-in through
`VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY`. Two bounded follow-ups (tail
compaction and page-address broadcast) are rejected and must not be resurrected
without new evidence.

The final source audit rebased all five patches without semantic change onto
`main@acc0f6fb92`, after the generic prefix-anchored and E4M3 batch-XQA merges.
The complete changed-file static suite passes and the combined SM70 extension
builds and links. Removing a redundant translation-unit-wide dynamic-shared
alignment attribute preserves the packed verifier layout while restoring all
121 pre-existing XQA kernel SASS instruction streams exactly; none are changed
or missing. All thirteen grouped-verifier cases pass on a V100 against this
current-main build. No new full-model or 256K production-configuration claim
is made by the current-main source audit.

## 2026-08-26 current-main long-context follow-up

This follow-up starts from `main@b06f54cee3` and keeps the practical production
contract fixed: Qwen3.8-27B NVFP4 target, E5M2 target KV, FP16 draft KV, TP4,
seven probabilistic draft tokens, prefix cache enabled, 256K maximum context,
4,096 maximum batched tokens, and FULL CUDA Graph. The frozen unprofiled
complete-round baseline is 18.477/21.583/30.641/42.300 ms at
1K/32K/128K/256K. The corresponding Nsight diagnostic values are
21.323/24.430/33.400/45.307 ms; the approximately 2.8-3.0 ms tracing overhead
is kept separate from service-wall results.

The trace attribution remains consistent with the earlier audit. Sixteen target
full-attention layers dominate the context-dependent slope, while five draft
sliding-window attention layers contribute a smaller avoidable tail. The target
grouped partial kernel is launched once per full-attention layer with grid
`(1, 80)`, 512 threads, 128 registers per thread, and 50,944 bytes of dynamic
shared memory. Its retained single-layer baseline is approximately
0.048/0.206/0.746/1.451 ms at 1K/32K/128K/256K.

### Draft sliding-window history skip

The non-anchored draft paged-prefill kernel previously started from key tile
zero even though a 2,048-token sliding window masked all older tiles. Clamping
the first visited tile to `min_key_pos / BLOCK_N` removes that masked history;
the anchored/prefix path is deliberately unchanged. Exact DFlash2 shape
microbenchmarks report the following five-layer totals:

| Context | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| 1K | 0.737280 ms | 0.757760 ms | +0.020480 ms |
| 32K | 1.367040 ms | 1.264640 ms | -0.102400 ms |
| 128K | 1.730560 ms | 1.218560 ms | -0.512000 ms |
| 256K | 2.114560 ms | 1.233920 ms | -0.880640 ms |

Baseline and candidate tensors are bitwise equal at all four lengths. The
targeted non-causal DFlash2 SWA and anchored-SWA suites pass all 21 cases. The
small 1K operator movement is 0.02048 ms per complete round and remains subject
to the full-model one-percent promotion gate.

### Volta shared-row bank-conflict removal

The packed grouped verifier used power-of-two shared row strides for Q/K/V
(`256` halves), scores (`32` floats), and probabilities (`32` halves). Those
strides repeatedly present the same bank pattern to Volta WMMA shared loads.
The initial candidate padded Q/K/V rows by eight halves and both score and
probability rows by eight elements. This changes only shared-memory addressing:

- Q and K/V row stride: `256 -> 264`;
- score and probability row stride: `32 -> 40`;
- softmax arithmetic, FP32 reductions, visibility masks, split count, launch
  geometry, output workspace, and sampling remain unchanged.

The candidate uses 54,528 bytes of opt-in dynamic shared memory and retains one
CTA per SM and 128 registers per thread. Q-only and K/V-only ablations improved
the 32K/128K/256K one-pass kernel to
0.168/0.592/1.147 ms and 0.184/0.618/1.200 ms, respectively. Padding both sides
reached 0.136/0.469/0.900 ms; padding the score/probability rows as well produced
the retained result below.

The production-page-size CUDA-Graph A/B/A medians use 40 warmups and 200 timed
rounds on the same V100:

| Context | Baseline A1 | Padded candidate | Baseline A2 | Delta vs. A mean |
| --- | ---: | ---: | ---: | ---: |
| 1K | 0.051200 ms | 0.031744 ms | 0.044032 ms | -33.33% |
| 32K | 0.206848 ms | 0.124928 ms | 0.204800 ms | -39.30% |
| 128K | 0.745472 ms | 0.419840 ms | 0.746496 ms | -43.72% |
| 256K | 1.451008 ms | 0.807936 ms | 1.451008 ms | -44.32% |

Candidate outputs are bitwise identical to the retained baseline at all four
lengths, and all thirteen grouped-verifier GPU cases pass. The raw A/B/A records
are `grouped-sp40-aba-a1.json`, `grouped-sp40-aba-b.json`, and
`grouped-sp40-aba-a2.json` under the task microbenchmark directory.

Two bounded alternatives were rejected and fully reverted. A 192-thread,
160-split layout intended to admit two CTAs per SM slowed the four points to
0.063/0.279/0.889/1.703 ms. An in-CTA QK-next/PV-current software pipeline used
97,280 bytes of shared memory and slowed them to
0.058/0.264/0.972/1.900 ms. Both were numerically correct, but neither improved
the measured critical path; they must not be revived without new hardware
evidence.

The combined candidate was then run with the frozen TP4 production contract.
The four-request unprofiled diagnostic improves steady decode from
147.5/259.0/176.6 to 170.3/317.7/240.0 token/s at 32K/128K/256K. The 1K request
is excluded from this throughput comparison because both runs include first-use
JIT. These rates remain acceptance-sensitive and are not used as the verifier
latency proof.

Independent Nsight Systems traces provide that proof. Each result drops the
first graph round and averages all four ranks; tracing overhead is therefore
kept separate from unprofiled service wall. Values are milliseconds per
complete verification cycle:

| Context | Phase | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| 1K | Draft | 3.774 | 3.933 | +0.159 |
| 1K | Draft to target | 0.251 | 0.297 | +0.046 |
| 1K | Target verify | 16.003 | 15.881 | -0.122 |
| 1K | Target to draft | 1.295 | 1.304 | +0.009 |
| 1K | Complete | 21.323 | 21.415 | +0.092 (+0.43%) |
| 32K | Draft | 4.248 | 4.215 | -0.033 |
| 32K | Draft to target | 0.255 | 0.285 | +0.030 |
| 32K | Target verify | 18.623 | 17.403 | -1.220 |
| 32K | Target to draft | 1.303 | 1.307 | +0.003 |
| 32K | Complete | 24.430 | 23.210 | -1.219 (-4.99%) |
| 128K | Draft | 4.598 | 4.342 | -0.256 |
| 128K | Draft to target | 0.256 | 0.332 | +0.076 |
| 128K | Target verify | 27.239 | 22.179 | -5.060 |
| 128K | Target to draft | 1.307 | 1.318 | +0.012 |
| 128K | Complete | 33.400 | 28.171 | -5.229 (-15.65%) |
| 256K | Draft | 5.091 | 4.443 | -0.648 |
| 256K | Draft to target | 0.254 | 0.256 | +0.002 |
| 256K | Target verify | 38.651 | 28.377 | -10.274 |
| 256K | Target to draft | 1.311 | 1.308 | -0.004 |
| 256K | Complete | 45.307 | 34.384 | -10.923 (-24.11%) |

The complete-round medians are 21.216/23.125/27.910/34.090 ms, versus
21.320/24.435/33.399/45.219 ms for the baseline. The mean 32K-to-256K growth
falls from 20.877 to 11.173 ms (-46.48%), while the 1K point stays within the
one-percent non-regression gate. This closes the speed and long-context-decay
gates for the kernel change.

The unprofiled four-request run has exactly the same aggregate mean acceptance
length (`4.932692`), accepted count (`409`), and drafted count (`728`) as the
frozen baseline. One 1K probabilistic trace differs by -0.058824 in mean
acceptance, narrowly outside the single-sample 0.05 boundary; the source
operator outputs are nevertheless bitwise equal at all four audited lengths.
Consequently no quality-promotion claim is made from the synthetic traces. The
frozen dataset-level acceptance, task-score, and WikiText PPL gates remain
required before default promotion.

### PR 285 fixed-layout follow-up

The operator reference identified for this follow-up is
[PR 285](https://github.com/1CatAI/1Cat-vLLM/pull/285), not PR 206. PR 285
improves the independent batch-one E4M3 XQA route from 40.561 to 61.834 token/s
at 128K and from 27.456 to 50.376 token/s at 256K. Its relevant retained
mechanisms are a compile-time interleaved physical stride and split-local page
ID staging. The DFlash2 verifier already combines QK, online softmax, and WMMA
P@V in one 80-CTA kernel, so PR 285's merged long-wave launch is already the
local topology. Its scalar shared-V `half2` P@V loop is not copied because that
would replace the verifier's tensor-core P@V arithmetic rather than merely
specialize addressing.

The new dispatch admits a fixed layout only for the exact production unbound
views from `[blocks, 2, page, 1, 256]`, page size 1,648 or 3,296, and matching
physical strides. All other layouts retain the generic stride path. The fixed
kernel removes runtime 64-bit block-stride products, and stages the few page
IDs used by each context split into shared memory. The page IDs share the
existing Q-publication barrier, so this adds no CTA barrier. Both changes have
independent default-on rollback flags:

- `VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED=0` disables fixed addressing;
- `VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS=0` disables page-ID staging.

The same audit found and fixed a generic paired-load defect: its physical
interleaved block stride was hard-coded to page 800. It is now
`2 * page_size * head_dim`. This protects page-1,568 E4M3 XQA while admitting
the new page-1,648/3,296 DFlash2 layouts. The existing page-1,568 E4M3 scalar
and wave-partition CUDA tests pass all fourteen parametrized cases after the
change.

A same-process CUDA-Graph comparison on the production page-3,296 layout keeps
the initial score/probability-padding binary fixed and toggles only fixed
addressing plus page staging:

| Context | Generic | Fixed plus staged | Delta |
| --- | ---: | ---: | ---: |
| 1K | 0.035840 ms | 0.035840 ms | 0.00% |
| 32K | 0.142336 ms | 0.140288 ms | -1.44% |
| 128K | 0.447488 ms | 0.440320 ms | -1.60% |
| 256K | 0.801792 ms | 0.789504 ms | -1.53% |

Every output is bitwise equal. Separate ablations attribute approximately
0.7-1.3% to the fixed physical stride and a further 0.4-0.7% to page staging.
This is deliberately a small address-path improvement, not a claim that the
full PR 285 throughput ratio transfers to the different DFlash2 WMMA kernel.

An A/B/A shared-row follow-up then removed score padding while retaining
probability padding. On the same physical GPU, the selected
Q/K/V-264, score-32, probability-40 layout compares with the earlier
Q/K/V-264, score-40, probability-40 layout as follows:

| Context | Selected score-32 | Earlier score-40 | Delta |
| --- | ---: | ---: | ---: |
| 1K | 0.030720 ms | 0.030720 ms | 0.00% |
| 32K | 0.120832 ms | 0.121856 ms | -0.84% |
| 128K | 0.407552 ms | 0.410624 ms | -0.75% |
| 256K | 0.784384 ms | 0.789504 ms | -0.65% |

The output SHA256 is identical across the two separately built extensions at
every length. The final extension SHA256 is
`b6952509ffb117c16ba43720c1affd5024f350ccd2ca4324d89022953692bc96`.
Its production fixed/staged one-pass specialization uses 127 registers per
thread, zero stack, and 52,992 bytes of dynamic shared memory; launch geometry
and one-CTA-per-SM occupancy are unchanged. All fifteen grouped-verifier GPU
cases, all fourteen page-1,568 E4M3 regression cases, and all 56 environment
tests pass.

The bounded probabilistic mixed-quality A/B uses 32 shuffled requests, a
2,048-token cap, natural EOS, and the production sampling contract. The first
control and candidate both score 19/32, with identical per-suite pass counts:
GSM8K 7/9, HumanEval 7/9, MATH500 5/7, and the known-invalid embedded MBPP
subset 0/7. The embedded MBPP result is useful only as a relative regression
check; the valid independent MBPP quality gate is recorded earlier. Candidate
natural termination improves from 25/32 to 26/32. Request-mean acceptance
changes from 4.809806 to 4.841060 (+0.031254), while length-weighted pooled
acceptance changes from 4.226907 to 4.154068 (-0.072839).

An identical-control replay proves that separate probabilistic engine starts
are not token-stream reproducible: only 4/32 streams match the first control,
despite no implementation or configuration change. Its task score is 18/32,
pooled acceptance is 4.190773, and request-mean acceptance is 4.834261. Against
this adjacent replay, the fixed/staged candidate is one task higher, changes
pooled acceptance by -0.036705, changes request-mean acceptance by +0.006799,
and stays inside the frozen 0.05 acceptance gate. This evidence rejects a
quality-regression attribution; it does not claim that address specialization
improves model quality. Operator outputs remain bitwise equal and the earlier
valid 128-request, independent-MBPP, and WikiText gates continue to define the
absolute production-quality envelope.

### Default promotion

The private current-main audit promotes the grouped verifier to default-on for
its existing exact runtime contract. Admission remains shape- and route-based:
SM70, DFlash selector verification, Q8/H6/Hkv1/D256, E5M2 KV, page 1,648 or
3,296, one request, causal full attention, and a model length of at least 32K.
It does not inspect a model or checkpoint identity. The retained operator is
bitwise equal to the prior route, the inherited dataset/PPL gates remain valid,
and the 1K complete-round movement stays inside the one-percent guard. Set
`VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY=0` to restore the independent-row
fallback.
