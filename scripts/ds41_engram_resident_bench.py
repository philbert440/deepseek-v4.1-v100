#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram RAM-resident microbench (VLLM_DS41_ENGRAM_RESIDENT): load time + per-chunk gather vs the cold pread path.
CPU only (no GPU, no server). Uses the same code the service uses (ResidentRowTables, csrc/engram_resident.cpp,
the C++ RowReader).

  1. LOAD: one TP rank's sub-table rows of layers 1 + 14 (~48 GB at TP4) -- or, by default, only the first
     --slice-gib of them (dry run) -- with the resident loader (threads / O_DIRECT / pread size / NUMA interleave as
     the knobs), timed; arcstats size before/after; a pread spot-check of the loaded bytes.
  2. GATHER: the real row set of ONE --tokens chunk of a prompt (random ids from --seed, or --text) for this rank:
       pread arm    the step gather the service does today (RowReader, --pread-threads, WILLNEED as --willneed),
                    rows never read before (fresh --seed) -> cold; arcstats deltas printed.
       resident arm the same unique rows from RAM (gather_resident), first call + min/median of --reps.
     With a slice load the resident arm cannot hold the real rows, so it gathers the same NUMBER of rows remapped
     into the slice (smaller footprint than 48 GB -> fewer TLB/DRAM-page misses: a lower bound); --slice-gib 0
     (full rank, needs --full-ok) gathers the real rows. Both arms' bytes are compared.

Dry run (fits the ~2 GB test budget): --slice-gib 1. Full rank: --slice-gib 0 --full-ok (box quiet; ~50 GB RAM).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--model", default=os.environ.get("DS41_MODEL"))
ap.add_argument("--engram-dir", default=os.environ.get("DS41_ENGRAM_DIR"))
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--tp", type=int, default=4)
ap.add_argument("--slice-gib", type=float, default=1.0, help="load only this much of the rank's rows (0 = all)")
ap.add_argument("--slice-offset-gib", type=float, default=0.0,
                help="skip this much of the rank's rows before the slice (a fresh region per run)")
ap.add_argument("--full-ok", action="store_true", help="required with --slice-gib 0 (allocates ~48 GB)")
ap.add_argument("--threads", type=int, default=32, help="load threads")
ap.add_argument("--o-direct", type=int, default=1)
ap.add_argument("--chunk-mib", type=int, default=2)
ap.add_argument("--nodes", default="all", help="VLLM_DS41_ENGRAM_RESIDENT_NODES syntax")
ap.add_argument("--margin-gib", type=int, default=32)
ap.add_argument("--verify-rows", type=int, default=4096)
ap.add_argument("--tokens", type=int, default=1024)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--text", default="", help="tokenize this file instead of random token ids")
ap.add_argument("--pread-threads", type=int, default=4)
ap.add_argument("--willneed", type=int, default=1)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--json", default="")
a = ap.parse_args()
if a.slice_gib <= 0 and not a.full_ok:
    sys.exit("--slice-gib 0 loads the whole rank (~48 GB): pass --full-ok once the box is scheduled for it")
sys.path.insert(0, a.src)

from vllm.models.deepseek_v41.common.engram import (  # noqa: E402
    ROW_SCALE_BYTES, ROW_VALUE_BYTES, EngramHasher, EngramLayout, load_compressed_token_map)
from vllm.models.deepseek_v41.common.engram_host import (  # noqa: E402
    ResidentRowTables, check_resident_memory, load_engram_resident_extension, load_engram_rows_extension,
    locate_engram_tables, memory_report, parse_resident_nodes, resident_ranges, zfs_arc_bytes)

ARC = "/proc/spl/kstat/zfs/arcstats"
ARC_KEYS = ("demand_data_hits", "demand_data_iohits", "demand_data_misses", "prefetch_data_hits",
            "prefetch_data_misses")


def arc() -> dict[str, int]:
    out = {}
    try:
        for line in open(ARC).read().splitlines()[2:]:
            k, _, v = line.split()
            if k in ARC_KEYS:
                out[k] = int(v)
    except OSError:
        pass
    return out


def arc_delta(a0, a1):
    return {k: a1.get(k, 0) - a0.get(k, 0) for k in ARC_KEYS}


cfg = types.SimpleNamespace(**json.load(open(f"{a.model}/config.json")))
layout = EngramLayout.from_hf_config(cfg)
layers = tuple(layout.layer_ids)
subtables = layout.subtables_for_rank(a.rank, a.tp)
tables = locate_engram_tables(a.engram_dir, layout, layers)
for loc in tables:     # DECISIONS D5: same sidecar rule as the service
    for p in (loc.weight_path, loc.scale_path):
        if not os.path.isfile(p + ".sha256-ok"):
            sys.exit(f"{p} has no .sha256-ok sidecar (DECISIONS D5)")
L, S = len(layers), len(subtables)
lis = [layout.layer_index(lid) for lid in layers]
sub_off = np.array([[layout.offsets[li][s] for s in subtables] for li in lis], np.int64)
sub_rows = np.array([[layout.primes[li][s] for s in subtables] for li in lis], np.int64)
base = np.concatenate([np.zeros((L, 1), np.int64), np.cumsum(sub_rows, axis=1)[:, :-1]], axis=1)
woff = np.array([t.weight_offset for t in tables], np.int64)
full = [resident_ranges(int(woff[l]), sub_off[l], sub_rows[l], base[l]) for l in range(L)]
rank_bytes = sum(int(r[:, 2].sum()) for r in full)

# ---- truncate to the slice (whole rows, in layer / sub-table order)
budget = rank_bytes if a.slice_gib <= 0 else int(a.slice_gib * 2**30) // ROW_VALUE_BYTES * ROW_VALUE_BYTES
skip = 0 if a.slice_gib <= 0 else int(a.slice_offset_gib * 2**30) // ROW_VALUE_BYTES * ROW_VALUE_BYTES
ranges, loaded_rows = [], []
for l in range(L):
    keep, dpos = [], 0
    for fo, d, n in full[l].tolist():
        drop = min(n, skip)
        skip -= drop
        take = min(n - drop, budget)
        if take > 0:   # packed from 0 in the slice table (slice mode only checks bytes against the file)
            keep.append((fo + drop, dpos, take))
            dpos += take
            budget -= take
    ranges.append(np.array(keep, np.int64).reshape(-1, 3))
    loaded_rows.append(int(ranges[-1][:, 2].sum()) // ROW_VALUE_BYTES)
use = [l for l in range(L) if loaded_rows[l] > 0]
load_bytes = sum(loaded_rows) * ROW_VALUE_BYTES
is_full = load_bytes == rank_bytes
print(f"rank {a.rank}/{a.tp} sub-tables {subtables}: rank rows {rank_bytes / 2**30:.2f} GiB; loading "
      f"{load_bytes / 2**30:.2f} GiB ({'FULL' if is_full else 'slice'}) rows/layer {loaded_rows}")
check_resident_memory(load_bytes, a.margin_gib << 30, f"bench rank {a.rank}")
nodes = parse_resident_nodes(a.nodes, a.rank)
ext = load_engram_resident_extension()

# ---- 1. load
print("before:", memory_report())
arc_sz0, a0 = zfs_arc_bytes(), arc()
res = ResidentRowTables(ext, [tables[l].weight_path for l in use], [ranges[l] for l in use], nodes, a.threads,
                        bool(a.o_direct), a.chunk_mib << 20)
arc_sz1, a1 = zfs_arc_bytes(), arc()
gbps = res.nbytes / res.load_seconds / 1e9
print(f"LOAD {res.nbytes / 2**30:.2f} GiB in {res.load_seconds:.2f} s = {gbps:.2f} GB/s ({a.threads} threads, "
      f"{'O_DIRECT' if a.o_direct else 'buffered'}, {a.chunk_mib} MiB preads, nodes {nodes}); placement "
      f"{res.placement()}")
if arc_sz0 and arc_sz1:
    print(f"  ARC size {arc_sz0[0] / 2**30:.2f} -> {arc_sz1[0] / 2**30:.2f} GiB; arcstats delta {arc_delta(a0, a1)}")
print(f"  extrapolated at this rate: one rank {rank_bytes / 1e9 / gbps:.0f} s, 4 ranks serialized "
      f"{4 * rank_bytes / 1e9 / gbps:.0f} s")
t0 = time.perf_counter()
print(f"  verify: {res.verify(a.verify_rows, seed=a.seed)} rows re-read by pread == resident "
      f"({time.perf_counter() - t0:.2f} s)")
print("after:", memory_report())

# scales of the loaded rows (same index space), small
scale_tabs = []
for l in use:
    sr = [(tables[l].scale_offset + (fo - int(woff[l])) // ROW_VALUE_BYTES * ROW_SCALE_BYTES,
           d // ROW_VALUE_BYTES * ROW_SCALE_BYTES, n // ROW_VALUE_BYTES * ROW_SCALE_BYTES)
          for fo, d, n in ranges[l].tolist()]
    t = ext.alloc_resident(loaded_rows[l] * ROW_SCALE_BYTES, [], False)
    ext.fill_resident(t, tables[l].scale_path, torch.tensor(sr, dtype=torch.int64), 4, False, 1 << 20)
    scale_tabs.append(t)

# ---- 2. the chunk's real rows for this rank
tmap = load_compressed_token_map(f"{a.model}/tokenizer.json", layout.compressed_vocab_size)
if a.text:
    from tokenizers import Tokenizer

    ids = np.array(Tokenizer.from_file(f"{a.model}/tokenizer.json").encode(open(a.text).read()).ids[: a.tokens])
else:
    ids = np.random.default_rng(a.seed).integers(0, tmap.shape[0], a.tokens)
h = EngramHasher(layout, tmap, layers, subtables)
rows = h.hash_positions(h.compress(ids), np.arange(ids.shape[0]))      # [T, L, S] global row ids
T = rows.shape[0]
l_flat = np.broadcast_to(np.arange(L, dtype=np.int64)[None, :, None], rows.shape).reshape(-1)
j_flat = np.broadcast_to(np.arange(S, dtype=np.int64)[None, None, :], rows.shape).reshape(-1)
r_flat = rows.reshape(-1)
keys, first = np.unique((l_flat << 40) | r_flat, return_index=True)
l_u, j_u, r_u = l_flat[first], j_flat[first], r_flat[first]
sidx = base[l_u, j_u] + (r_u - sub_off[l_u, j_u])
n = keys.shape[0]
paths = sorted({t.weight_path for t in tables})
file_of_layer = np.array([paths.index(t.weight_path) for t in tables], np.int32)
offsets = woff[l_u] + r_u * ROW_VALUE_BYTES
print(f"CHUNK {T} tokens x {L} layers x {S} sub-tables = {T * L * S} lookups, {n} unique rows")

rd = load_engram_rows_extension().RowReader(paths, a.pread_threads, False)
rd.set_gather_willneed(bool(a.willneed))
dst_p = torch.zeros(n * 264, dtype=torch.uint8)
a0 = arc()
t0 = time.perf_counter()
st = rd.wait(rd.submit(torch.from_numpy(file_of_layer[l_u]), torch.from_numpy(offsets),
                       torch.full((n,), -1, dtype=torch.int32), torch.zeros(n, dtype=torch.int64), dst_p, 264, 256))
t_pread = time.perf_counter() - t0
d_arc = arc_delta(a0, arc())
print(f"PREAD (today's step gather, cold rows, {a.pread_threads} threads, willneed={a.willneed}): "
      f"{t_pread * 1e3:.1f} ms for {n} rows; arcstats delta {d_arc}")

# resident arm
if is_full:    # table ids index the loaded tables (= all layers)
    tab_t = torch.from_numpy(np.array([use.index(int(x)) for x in l_u], np.int32))
    sidx_r = sidx.astype(np.int64)
    mode = "real rows"
else:          # the same number of rows, uniformly random inside the loaded slice (layer use[0])
    lo = use[0]
    tab_t = torch.zeros(n, dtype=torch.int32)
    sidx_r = np.sort(np.random.default_rng(a.seed + 1).integers(0, loaded_rows[lo], n)).astype(np.int64)
    mode = f"{n} random rows inside the {loaded_rows[lo] * 256 / 2**30:.2f} GiB slice"
rows_list = list(res.tables)
dst_r = torch.zeros(n * 264, dtype=torch.uint8)
sidx_t = torch.from_numpy(sidx_r)
times = []
for _ in range(max(1, a.reps)):
    t0 = time.perf_counter()
    ext.gather_resident(rows_list, scale_tabs, tab_t, sidx_t, dst_r, 264, 256, 8)
    times.append(time.perf_counter() - t0)
print(f"RESIDENT gather ({mode}): first {times[0] * 1e3:.2f} ms, min {min(times) * 1e3:.2f} ms, median "
      f"{float(np.median(times)) * 1e3:.2f} ms for {n} rows (incl. 8 B scales, single thread)")

# byte check
got = dst_r.numpy().reshape(n, 264)
if is_full:
    same = np.array_equal(got[:, :256], dst_p.numpy().reshape(n, 264)[:, :256])
    what = "resident == pread on the real rows (256 B values)"
else:
    lo = use[0]
    rg = ranges[lo]
    # resident row index -> file offset (inside the loaded ranges of layer lo)
    starts = rg[:, 1] // 256
    k = np.searchsorted(starts, sidx_r, side="right") - 1
    off_r = rg[k, 0] + (sidx_r - starts[k]) * 256
    dst_c = torch.zeros(n * 264, dtype=torch.uint8)
    rd.wait(rd.submit(torch.from_numpy(np.full(n, file_of_layer[lo], np.int32)), torch.from_numpy(off_r.astype(np.int64)),
                      torch.full((n,), -1, dtype=torch.int32), torch.zeros(n, dtype=torch.int64), dst_c, 264, 256))
    same = np.array_equal(got[:, :256], dst_c.numpy().reshape(n, 264)[:, :256])
    what = "resident == pread on the remapped rows (256 B values)"
print(f"BYTES {what}: {'IDENTICAL' if same else 'MISMATCH'}")
rd.shutdown()
speedup = t_pread / max(min(times), 1e-9)
print(f"SUMMARY rank {a.rank} load {res.nbytes / 2**30:.2f} GiB {res.load_seconds:.1f} s ({gbps:.2f} GB/s) | chunk {n} rows: "
      f"pread cold {t_pread * 1e3:.1f} ms vs resident {min(times) * 1e3:.2f} ms ({speedup:.0f}x)")
if a.json:
    json.dump(dict(args=vars(a), load_s=res.load_seconds, load_bytes=res.nbytes, load_gbps=gbps, rows=int(n),
                   pread_ms=t_pread * 1e3, resident_ms=[x * 1e3 for x in times], arc=d_arc, identical=bool(same),
                   full=is_full), open(a.json, "w"))
if not same:
    sys.exit(1)
