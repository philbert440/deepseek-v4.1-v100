# Persistent EXL3 MoE prefill kernel for SM70 — design note (build after the stall profile)

**Problem.** Stage-0 prefill of a 1016-token chunk spends 130 ms in `moe.experts` (6.5 ms/layer/rank, profile
20261025T021427Z), 40% of the 330 ms chunk. The row-tiled kernel (`exl3_tiled_kwarp_kernel`, layout 1) already decodes
each expert's trellis ~once (1.03x: 257/384 experts have < 16 rows at T=1016, 11 have 64–159, 1.4 take the cuBLAS dense
path), so "decode once" is done. Per rank and layer: 96 experts x 35.4M weights x ~3.1 bpw = 1.32 GB -> **HBM floor
1.47 ms**; **trellis-decode ALU floor 1.2–1.75 ms** at the GEMV's 2–3e12 weights/s; **HMMA floor 1.1 ms** (108 GFLOP at
100 TFLOPS; ~1.8 ms at the 10/16 average row fill of a 16-row tile). Measured 6.5 ms is 3.7–4.4x over every floor.
Hypothesis: **decode-throughput-bound** — each warp decodes a 32-column strip serially along K=5120 from scattered
16-bit words (4–6 words per 16 weights), the accumulator budget caps residency at 8–16 warps/SM (127/164/168 registers
for the 16/32/64-row tiles), and with ~1 tile per expert there is no reuse of a decoded tile across tiles.

**Go / no-go (scripts/moe_prefill_profile.sh).** Build the kernel if ncu shows, for `exl3_tiled_kwarp_kernel`:
achieved occupancy <= 25% (<= 16 warps/SM), DRAM throughput < 40% of peak, SM (tensor) throughput < 30%, and the warp
stall breakdown dominated by long-scoreboard / LG throttle (memory latency) or by ALU dependency (`short scoreboard` /
`math pipe throttle` on the integer decode). If instead DRAM throughput is > 70%, the kernel is already near the HBM
floor and the lever is weight bytes, not the kernel (stop). If issue slots are > 80% busy with the integer pipes, the
decode arithmetic itself is the floor and the gain is bounded by the decode-ALU floor (~2x, still worth it).

**Design (persistent, grouped).**
1. **Work list** from the existing plan kernel: (expert, row tile) items sorted by expert; the kernel is launched with
   `min(#SM x blocks_per_SM, items)` blocks that pull items from a global counter (persistent), so the tail of a few
   large experts does not idle the grid and the host launches once per projection (3 launches per layer: gate+up, down,
   plus the dense list).
2. **Cooperative strip loads.** A block (8 warps) owns one item and walks its expert's K in 64-wide k-tiles; all 8 warps
   issue coalesced 128-bit loads of the *whole* block's trellis words for the k-tile (strip-major words of 32-column
   strips are contiguous along k) into a double-buffered smem ring (2 x 64 x N_cols x bits/16 words ~ 2 x 18 KB at
   N=4608, K=3), so the global loads are wide and in flight while the previous tile decodes.
3. **Staged decode.** Each warp decodes its strips from smem into fp16 B fragments in registers (the same `mul1`
   codebook math as today, no change in numerics) and multiplies against the row tile's A fragments held in registers
   (16 rows -> 1 m16n16k16 WMMA tile per 16 columns; 32/64 rows -> 2/4). FP32 accumulate, same per-element products as
   today; the K-summation order within a strip stays sequential as today (k-tile order), so the result is the same class
   as today's tiled kernel (not bit-identical to the 8-slot kernel either; gate with oracle KL as every MoE change).
4. **Occupancy.** 8 warps x 2 blocks/SM with <= 128 registers (only one row tile of accumulators per warp: 16 rows x
   16 columns x 2 fragments) = 16 warps/SM, plus the smem ring hides load latency, which is what the current kernel lacks.
5. **Dense experts (>= 160 rows)** keep the reconstruct + cuBLAS path (4.7% of rows); the persistent kernel takes
   `rows < dense_min`.
6. **Knob** `VLLM_DS41_MOE_EXL3_PREFILL_PERSISTENT=1` (default off); same plan / pre / mid / post glue; the no-sync
   (capture) path is unchanged.

**Expected.** Decode at the GEMV's measured 2.5e12 weights/s plus coalesced loads: ~2.5–3 ms/layer/rank (from 6.5)
-> **−70 to −80 ms per 1016-token chunk** on stage 0 (−50 to −60 ms if the decode ALU floor binds). Effort: ~1–2 weeks
(CUDA, the trellis decode path, ncu-driven tuning), KL-gated. Not worth building first if the profile shows the kernel
is already HBM-bound.

**Related small items (not built):** the per-layer host sync of the plan (`info.cpu()`, ~0.1–0.3 ms launch bubble x 20)
can be removed with the no-sync tier bounds, but that path disables the dense experts (dense_min = 0) — measure the
bubble in the nsys table first. `fixed_m` pad-once (audit #7) is moot on live: `VLLM_DS41_BATCH_INVARIANT_GEMM` is off,
so `fixed_m.mm` is a plain `torch.mm`.
