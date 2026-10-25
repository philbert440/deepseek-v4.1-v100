# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngramHostService: host-side Engram row store for one TP rank of one PP stage (lane L-ENGRAM, PORT_DESIGN §3.6).

Storage (DECISIONS D4/D5 -- no repacked row file): the 256-byte FP8 rows are read IN PLACE from the official
safetensors shards (``layers.{L}.engram.embed.weight``; tensor offsets from the safetensors header; 1 row in 16
straddles a 4 KiB page because the tensors start at byte 664 / 672) through the page cache, which is the warm tier.
The rank's UE8M0 scales (``embed.scale``, 8 B per row, only its own sub-tables: ~0.77 GB per layer at TP4) are loaded
once into page-locked RAM and appended to every row on the CPU, so each staged row is the 264-byte
``[256 x E4M3 | 8 x UE8M0]`` record the module decodes.

Per step (hooks called by L-CORE, §3.6):
  begin_step(plan)   worker thread, before the PP receive: hash the scheduled tokens (CPU, numpy), dedup the
                     (layer, row) keys, submit ONE high-priority gather to the C++ reader (csrc/engram_rows.cpp)
                     into pinned slot k; new/resumed requests also queue a low-priority page-cache warm-up of the
                     rest of their prompt (admission prefetch). Never waits for I/O.
  bind_batch(layout) runner thread, after _prepare_inputs: resolves async-PP sampled tokens, writes the runner-order
                     index map (padding -> the all-zero row 0) and, when the gather has already finished, enqueues the
                     H2D copy on the service's copy stream. Never waits for I/O.
  wait_rows(layer)   inside the module's eager break: finishes the gather and the H2D if still pending, makes the
                     current stream wait for it, expands unique rows -> the static dense buffer [T_pad, n_sub, 264].
  dummy_rows(...)    AM-2: profiling / CUDA-graph-capture forwards (ForwardContext.is_dummy_run) get zero rows.
Two pinned/device slots alternate so step k+1's gather and copy overlap step k's compute; the copy into a device
slot waits for the event recorded after that slot's previous expansion.

Lookahead (VLLM_DS41_ENGRAM_LOOKAHEAD_TOKENS = W > 0; default 0 = off). The first PP stage has no PP receive to hide
the step gather behind, so in chunked prefill every chunk's ~12K cold rows per rank were read on the critical path
(the CPU blocks in wait_rows before layer 1). With W > 0, begin_step(k) also submits, at mid priority (after any
step gather, before the WILLNEED warm-up), a real gather of the next W prompt positions of every request still in
its prompt into a host-side lookahead buffer (sorted unique keys -> rows). begin_step(k+1) serves its keys from that
buffer (a host copy in _issue_h2d once the lookahead job is done) and reads only the misses. Rows are a pure function
of the key, so the staged bytes are identical. Two lookahead buffers alternate (fill one during step k, consume the
other). With lookahead on, the whole-prompt admission WILLNEED defaults to off (VLLM_DS41_ENGRAM_PREFETCH=1 still
forces it): on ZFS it is async-read class I/O (3 in flight per vdev) issued in table-offset order, so it warms every
chunk a little instead of the next chunk fully, and it fills the ARC with 16 KiB records of the whole prompt.
VLLM_DS41_ENGRAM_PREFETCH_ORDER=position issues that warm-up in prompt order instead of offset order.

RAM-resident rows (VLLM_DS41_ENGRAM_RESIDENT=1; default 0 = the pread path above, unchanged). At init each rank
loads the 256-byte rows of ITS OWN sub-tables (6 of 24 at TP4, ~48 GB per rank for layers 1 + 14) into one anonymous
mapping per layer (MADV_HUGEPAGE, mbind-interleaved over NUMA nodes: _RESIDENT_NODES; NOT page-locked), with large
multi-threaded sequential preads from the same sha256-verified shards, laid out like the rank's scale table (sub-tables concatenated in rank order, so the scale index
of a row is also its resident row index). Ranks load one at a time under a host-wide flock and each refuses to start
unless MemAvailable >= its tables + _RESIDENT_MARGIN_GIB. A sample of rows is then re-read with pread and compared.
The step gather becomes a memcpy per unique row (csrc/engram_resident.cpp) into the same pinned staging, done inline
in begin_step: no reader threads, no WILLNEED, no lookahead (forced off), same 264-byte records.
Measured on 8x V100-SXM2 (ZFS 2.4, recordsize 16K, one 990): the load runs at ~1 GB/s (32-64 threads, 1-2 MiB preads; dd
alone does ~0.6), i.e. ~50 s per rank / ~3.5 min for 4 ranks, and O_DIRECT does NOT keep the blocks out of the ARC
there (it grows ~0.8 GiB per GiB loaded until zfs_arc_max) -- lower zfs_arc_max for resident serving.

Slice lane (VLLM_DS41_SLICED_PREFILL=1; default 0 = not allocated; hol-design.md §2a). Layer-sliced prefill runs one
STAGED prompt chunk a few layers per engine step, after the step's decode forward, so the chunk reaches layer 1 and
layer 14 in different engine steps while the decode batch keeps using the step slots above. The lane has its own
pinned/device staging, index map, dense buffer and events (no scratch shared with the step slots); one staged chunk
per stage at a time:
  begin_slice(req_id, start_pos, num_tokens, prompt_token_ids)  when the chunk is STAGED: hash the chunk (prompt
                     positions only) and submit its gather (mid priority: behind the step gathers). RAM-resident: a
                     memcpy done inline. Never waits for I/O. Returns a handle.
  rows_ready(handle) cheap, non-blocking: has the gather finished (always True when resident)?
  slice_lane(handle) context manager around a slice forward: while active, rows_buffer / wait_rows serve the slice
                     (the Engram module is unchanged). wait_rows_slice(handle, layer) is the same thing called directly.
  end_slice(handle)  after layer 14 consumed its rows, or on abort / preempt (drains a pending gather).
The slice forward must run on the same CUDA stream as the step forwards (stream order protects the dense buffer and
the next slice's H2D waits for the event recorded after the last expansion of this one).
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import mmap
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import (
    ENGRAM_HEAD_DIM,
    ENGRAM_ROW_BYTES,
    EngramBatchLayout,
    EngramStepPlan,
)
from vllm.models.deepseek_v41.common.engram import (
    ROW_SCALE_BYTES,
    ROW_VALUE_BYTES,
    EngramHasher,
    EngramLayout,
    load_compressed_token_map,
    row_bias_for_exponents,
)

logger = init_logger(__name__)

O_DIRECT_KNOB = "VLLM_DS41_ENGRAM_O_DIRECT"                # bypass the page cache (cold-path measurements)
PREFETCH_KNOB = "VLLM_DS41_ENGRAM_PREFETCH"                # admission prefetch of whole prompts
REQUIRE_VERIFIED_KNOB = "VLLM_DS41_ENGRAM_REQUIRE_VERIFIED"  # shard needs its <file>.sha256-ok sidecar
BUILD_DIR_KNOB = "VLLM_DS41_ENGRAM_BUILD_DIR"              # torch extension build directory
IO_THREADS_KNOB = "VLLM_DS41_ENGRAM_IO_THREADS"            # overrides the ctor's io_threads (CPU budget per rank)
LOOKAHEAD_KNOB = "VLLM_DS41_ENGRAM_LOOKAHEAD_TOKENS"       # 0 = off; W > 0: gather the next W prompt positions early
PREFETCH_ORDER_KNOB = "VLLM_DS41_ENGRAM_PREFETCH_ORDER"    # "offset" (default) | "position" (admission warm-up order)
GATHER_WILLNEED_KNOB = "VLLM_DS41_ENGRAM_GATHER_WILLNEED"  # 1 (default): WILLNEED each miss before the blocking
                                                           # preads; 0: plain preads (ZFS sync-read class)
RESIDENT_KNOB = "VLLM_DS41_ENGRAM_RESIDENT"                # 1: the rank's row tables live in RAM (see docstring)
RESIDENT_NODES_KNOB = "VLLM_DS41_ENGRAM_RESIDENT_NODES"    # "all" (default: interleave over Mems_allowed) | "none"
                                                           # (task policy) | "0,1" | per TP rank "0,1;0,1;1;1"
RESIDENT_MARGIN_KNOB = "VLLM_DS41_ENGRAM_RESIDENT_MARGIN_GIB"   # RAM that must stay available after the load (32)
RESIDENT_THREADS_KNOB = "VLLM_DS41_ENGRAM_RESIDENT_LOAD_THREADS"  # load threads per rank (32)
RESIDENT_O_DIRECT_KNOB = "VLLM_DS41_ENGRAM_RESIDENT_O_DIRECT"   # 1 (default): load with O_DIRECT (see docstring)
RESIDENT_CHUNK_KNOB = "VLLM_DS41_ENGRAM_RESIDENT_CHUNK_MIB"     # pread size of the load (2 MiB)
RESIDENT_LOCK_KNOB = "VLLM_DS41_ENGRAM_RESIDENT_LOCK"           # host-wide lock file serialising the loads
RESIDENT_VERIFY_KNOB = "VLLM_DS41_ENGRAM_RESIDENT_VERIFY_ROWS"  # rows re-read with pread and compared (4096)
SLICE_KNOB = "VLLM_DS41_SLICED_PREFILL"                    # 1: allocate the slice lane (see docstring)
N_SLOTS = 2
N_AHEAD = 2
_LATENCY_WINDOW = 4096


# ------------------------------------------------------------------------------------------------ extension
@functools.lru_cache(maxsize=1)
def load_engram_rows_extension() -> Any:
    """Build/load csrc/engram_rows.cpp as a runtime torch extension (no CMakeLists entry at P2)."""
    return _load_extension("ds41_engram_rows", "engram_rows.cpp", "")


@functools.lru_cache(maxsize=1)
def load_engram_resident_extension() -> Any:
    """Build/load csrc/engram_resident.cpp (VLLM_DS41_ENGRAM_RESIDENT=1 only; own build sub-directory)."""
    return _load_extension("ds41_engram_resident", "engram_resident.cpp", "resident")


def _load_extension(name: str, source: str, subdir: str) -> Any:
    src = Path(__file__).resolve().parents[4] / "csrc" / source
    if not src.is_file():  # installed wheel: sources ship inside the package
        src = Path(__file__).resolve().parents[3] / "_ds41_csrc" / source
    if not src.is_file():
        raise FileNotFoundError(f"Engram row reader source not found at {src} (needs a source checkout of 1Cat-vLLM)")
    default_dir = Path.home() / ".cache" / "vllm" / "ds41_engram_rows" / torch.__version__.replace("+", "_")
    build_dir = Path(knobs.env_str(BUILD_DIR_KNOB, str(default_dir)))
    if subdir:
        build_dir = build_dir / subdir
    build_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ninja") is None:
        venv_ninja = Path(sys.executable).parent / "ninja"
        if not venv_ninja.is_file():
            raise RuntimeError("building the Engram row reader needs `ninja` on PATH (or next to the interpreter)")
        os.environ["PATH"] = f"{venv_ninja.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    from torch.utils.cpp_extension import load

    return load(name=name, sources=[str(src)], build_directory=str(build_dir),
                extra_cflags=["-O3", "-std=c++17"], verbose=False)


# ------------------------------------------------------------------------------------------------ shards
@dataclass(frozen=True)
class EngramTableLocation:
    layer_id: int
    weight_path: str      # realpath of the shard holding embed.weight
    weight_offset: int    # absolute byte offset of row 0
    scale_path: str
    scale_offset: int
    num_rows: int


def read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """(header dict, absolute offset of the data section)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (short header length)")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 << 20:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        header = json.loads(f.read(n))
    return header, 8 + n


def locate_engram_tables(row_dir: str, layout: EngramLayout, layers: tuple[int, ...]) -> list[EngramTableLocation]:
    """Find ``layers.{L}.engram.embed.{weight,scale}`` under ``row_dir`` (a checkpoint dir with an index json, or
    a directory of shards) and return absolute byte offsets, validating dtype and shape against the layout."""
    index = os.path.join(row_dir, "model.safetensors.index.json")
    wanted = {f"layers.{lid}.engram.embed.{part}" for lid in layers for part in ("weight", "scale")}
    files: dict[str, str] = {}
    if os.path.isfile(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        for name in wanted:
            if name not in weight_map:
                raise KeyError(f"{index} has no entry for {name}")
            files[name] = os.path.join(row_dir, weight_map[name])
    else:
        for p in sorted(Path(row_dir).glob("*.safetensors")):
            header, _ = read_safetensors_header(str(p))
            for name in wanted & header.keys():
                files[name] = str(p)
        missing = wanted - files.keys()
        if missing:
            raise FileNotFoundError(f"Engram tables {sorted(missing)} not found in any *.safetensors under {row_dir}")
    out = []
    for lid in layers:
        rows = layout.num_embeddings[layout.layer_index(lid)]
        locs = {}
        for part, dtype, width in (("weight", "F8_E4M3", ROW_VALUE_BYTES), ("scale", "F8_E8M0", ROW_SCALE_BYTES)):
            name = f"layers.{lid}.engram.embed.{part}"
            path = os.path.realpath(files[name])
            header, data_start = read_safetensors_header(path)
            meta = header[name]
            if meta["dtype"] != dtype or list(meta["shape"]) != [rows, width]:
                raise ValueError(f"{path}: {name} is {meta['dtype']} {meta['shape']}, expected {dtype} {[rows, width]}")
            a, b = meta["data_offsets"]
            if b - a != rows * width:
                raise ValueError(f"{path}: {name} spans {b - a} bytes, expected {rows * width}")
            if data_start + b > os.path.getsize(path):
                raise ValueError(f"{path}: {name} ends at {data_start + b}, beyond the file ({os.path.getsize(path)})")
            locs[part] = (path, data_start + a)
        out.append(EngramTableLocation(lid, locs["weight"][0], locs["weight"][1], locs["scale"][0],
                                       locs["scale"][1], rows))
    return out


def _check_verified(path: str) -> None:
    sidecar = path + ".sha256-ok"
    if not os.path.isfile(sidecar):
        raise RuntimeError(f"Engram shard {path} has no {os.path.basename(sidecar)} sidecar: refusing to serve rows "
                           f"from an unverified shard (DECISIONS D5; set {REQUIRE_VERIFIED_KNOB}=0 only for tests)")


# ------------------------------------------------------------------------------------------------ host memory
class _PinnedHostBuffer:
    """Exact-size anonymous mapping, THP-advised, page-locked with cudaHostRegister (RLIMIT_MEMLOCK does not cap it;
    torch's pinned allocator would round 0.77 GB up to 1 GiB)."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self._map = mmap.mmap(-1, max(nbytes, 1), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        if hasattr(mmap, "MADV_HUGEPAGE"):
            self._map.madvise(mmap.MADV_HUGEPAGE)
        self.array = np.frombuffer(self._map, dtype=np.uint8, count=nbytes)
        self.tensor = torch.from_numpy(self.array)
        self._registered = False

    def register(self) -> None:
        rc = torch.cuda.cudart().cudaHostRegister(self.tensor.data_ptr(), self.nbytes, 0)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister of {self.nbytes} B Engram scale table failed: {rc}")
        self._registered = True

    def release(self) -> None:
        if self._registered:
            rc = torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr())
            self._registered = False
            if int(rc) != 0:
                raise RuntimeError(f"cudaHostUnregister of the Engram scale table failed: {rc}")


def _pread_into(fd: int, dst: np.ndarray, offset: int, path: str) -> None:
    view = memoryview(dst)
    done = 0
    chunk = 64 << 20
    while done < dst.nbytes:
        n = os.preadv(fd, [view[done:done + chunk]], offset + done)
        if n <= 0:
            raise OSError(f"short read of Engram scales from {path} at {offset + done}: {n}")
        done += n


# ------------------------------------------------------------------------------------------------ resident tables
def _parse_node_list(text: str) -> list[int]:
    """'0-1,3' -> [0, 1, 3]."""
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b) if b else int(a)
        if lo < 0 or hi < lo:
            raise ValueError(f"bad NUMA node range {part!r}")
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError(f"empty NUMA node list {text!r}")
    return sorted(out)


def mems_allowed() -> list[int]:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Mems_allowed_list:"):
                return _parse_node_list(line.split(":", 1)[1])
    return [0]


def parse_resident_nodes(spec: str, tp_rank: int) -> list[int] | None:
    """VLLM_DS41_ENGRAM_RESIDENT_NODES for one rank: None = no mbind (the task's policy, e.g. numactl --membind),
    else the interleave set. 'all' = every node in Mems_allowed; 'a;b;c;d' = one spec per TP rank."""
    spec = spec.strip()
    if ";" in spec:
        parts = [p.strip() for p in spec.split(";")]
        if tp_rank >= len(parts):
            raise ValueError(f"{RESIDENT_NODES_KNOB}={spec!r} has {len(parts)} per-rank entries, TP rank {tp_rank}")
        spec = parts[tp_rank]
    if spec == "none":
        return None
    if spec == "all":
        return mems_allowed()
    try:
        return _parse_node_list(spec)
    except ValueError as exc:
        raise ValueError(f"{RESIDENT_NODES_KNOB}={spec!r}: {exc} (expected all | none | 0,1 | 0,1;0,1;1;1)") from exc


def meminfo_bytes(path: str = "/proc/meminfo") -> dict[str, int]:
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            # /proc/meminfo: "MemAvailable: 123 kB"; node meminfo: "Node 0 MemFree: 123 kB"
            if len(parts) >= 3 and parts[-1] == "kB":
                out[parts[-3].rstrip(":")] = int(parts[-2]) * 1024
    return out


def node_free_bytes() -> dict[int, int]:
    out = {}
    for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        try:
            out[int(p.name[4:])] = meminfo_bytes(str(p / "meminfo")).get("MemFree", 0)
        except OSError:
            pass
    return out


def zfs_arc_bytes() -> tuple[int, int] | None:
    """(ARC size, c_min) or None without ZFS. The ARC is NOT in MemAvailable."""
    try:
        vals = {}
        with open("/proc/spl/kstat/zfs/arcstats") as f:
            for line in f.read().splitlines()[2:]:
                k, _, v = line.split()
                if k in ("size", "c_min"):
                    vals[k] = int(v)
        return vals["size"], vals["c_min"]
    except (OSError, KeyError, ValueError):
        return None


def memory_report() -> str:
    mi = meminfo_bytes()
    arc = zfs_arc_bytes()
    nodes = ", ".join(f"node {n} free {b / 2**30:.1f}" for n, b in node_free_bytes().items())
    arc_s = f"ZFS ARC {arc[0] / 2**30:.1f} GiB (c_min {arc[1] / 2**30:.1f})" if arc else "no ZFS ARC"
    return f"MemAvailable {mi.get('MemAvailable', 0) / 2**30:.1f} GiB, {nodes} GiB, {arc_s}"


def check_resident_memory(need_bytes: int, margin_bytes: int, what: str) -> None:
    """Refuse (clear error) instead of OOMing: MemAvailable must cover the tables plus the margin."""
    avail = meminfo_bytes().get("MemAvailable", 0)
    if avail < need_bytes + margin_bytes:
        raise RuntimeError(
            f"Engram resident: {what} needs {need_bytes / 2**30:.1f} GiB + {margin_bytes / 2**30:.0f} GiB margin "
            f"({RESIDENT_MARGIN_KNOB}) but only {avail / 2**30:.1f} GiB is available ({memory_report()}). The ARC "
            f"is not counted: lower zfs_arc_max (the resident tables replace it for Engram), free RAM, or unset "
            f"{RESIDENT_KNOB} (the pread path).")


@contextlib.contextmanager
def host_lock(path: str):
    """Host-wide exclusive flock (ranks load one at a time, so each memory check sees the previous loads)."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def resident_ranges(weight_offset: int, sub_offset: np.ndarray, sub_rows: np.ndarray,
                    base: np.ndarray) -> np.ndarray:
    """[S, 3] int64 (file offset, resident byte offset, nbytes) of one layer's rank-local sub-tables: resident row
    ``base[j] + local`` = global row ``sub_offset[j] + local`` (the scale table's layout)."""
    return np.stack([weight_offset + sub_offset.astype(np.int64) * ROW_VALUE_BYTES,
                     base.astype(np.int64) * ROW_VALUE_BYTES,
                     sub_rows.astype(np.int64) * ROW_VALUE_BYTES], axis=1).astype(np.int64)


class ResidentRowTables:
    """One rank's resident row tables (one anonymous mapping per layer), filled from the shards."""

    def __init__(self, ext: Any, paths: list[str], ranges: list[np.ndarray], nodes: list[int] | None,
                 threads: int, o_direct: bool, chunk_bytes: int) -> None:
        self.paths, self.ranges = list(paths), [np.ascontiguousarray(r, np.int64) for r in ranges]
        self.nodes = nodes
        self.tables: list[torch.Tensor] = []
        t0 = time.perf_counter()
        for path, rg in zip(self.paths, self.ranges):
            nbytes = int((rg[:, 1] + rg[:, 2]).max())
            t = ext.alloc_resident(nbytes, list(nodes or []), True)
            got, _ns = ext.fill_resident(t, path, torch.from_numpy(rg), int(threads), bool(o_direct), int(chunk_bytes))
            if got != int(rg[:, 2].sum()):
                raise RuntimeError(f"Engram resident: loaded {got} B of {int(rg[:, 2].sum())} from {path}")
            self.tables.append(t)
        self.load_seconds = time.perf_counter() - t0
        self.nbytes = sum(t.numel() for t in self.tables)

    def verify(self, n_rows: int, seed: int) -> int:
        """Re-read ~n_rows rows (plus the first and last row of every range) with pread and compare the bytes."""
        if n_rows <= 0:
            return 0
        rng = np.random.default_rng(seed)
        per = max(1, n_rows // max(1, sum(r.shape[0] for r in self.ranges)))
        checked = 0
        for path, rg, t in zip(self.paths, self.ranges, self.tables):
            arr = t.numpy()
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                for fo, d, n in rg.tolist():
                    nr = n // ROW_VALUE_BYTES
                    for p in np.unique(np.concatenate([[0, nr - 1], rng.integers(0, nr, per)])).tolist():
                        want = os.pread(fd, ROW_VALUE_BYTES, fo + p * ROW_VALUE_BYTES)
                        if arr[d + p * ROW_VALUE_BYTES:d + (p + 1) * ROW_VALUE_BYTES].tobytes() != want:
                            raise RuntimeError(f"Engram resident: row at {path}:{fo + p * ROW_VALUE_BYTES} differs "
                                               "from the loaded copy (load bug or a shard changed under us)")
                        checked += 1
            finally:
                os.close(fd)
        return checked

    def placement(self) -> str:
        """Per-node GiB of the mappings holding the tables (/proc/self/maps + numa_maps; adjacent tables may share
        one VMA, which is then reported once)."""
        try:
            vmas = []
            with open("/proc/self/maps") as f:
                for line in f:
                    a, b = line.split()[0].split("-")
                    vmas.append((int(a, 16), int(b, 16)))
            mine = {a for a, b in vmas for t in self.tables if a <= t.data_ptr() < b}
            nodes: dict[int, float] = {}
            with open("/proc/self/numa_maps") as f:
                for line in f:
                    addr, *fields = line.split()
                    if int(addr, 16) not in mine:
                        continue
                    kb = next((int(x.split("=")[1]) for x in fields if x.startswith("kernelpagesize_kB=")), 4)
                    for x in fields:
                        if x[:1] == "N" and "=" in x and x[1:].split("=")[0].isdigit():
                            n = int(x[1:].split("=")[0])
                            nodes[n] = nodes.get(n, 0.0) + int(x.split("=")[1]) * kb / 2**20
        except (OSError, ValueError):
            return "n/a"
        return ", ".join(f"N{n} {g:.2f} GiB" for n, g in sorted(nodes.items())) or "n/a"

    def release(self) -> None:
        self.tables = []


class _ResidentReader:
    """Drop-in for the C++ RowReader (same submit/wait/done/... surface) over resident tables: submit gathers at once
    (rows[xtab[i]][xidx[i]] || scales[xtab[i]][xidx[i]]); the file offsets are not used. Tickets are born done."""

    def __init__(self, ext: Any, tables: ResidentRowTables) -> None:
        self._ext = ext
        self._tables = tables
        self._rows = list(tables.tables)
        self._extras: list[torch.Tensor] = []
        self._extra_bytes = 0
        self._jobs: dict[int, list[int]] = {}
        self._next = 1
        self._willneed = False

    def add_table(self, t: torch.Tensor, entry_bytes: int) -> int:
        i = len(self._extras)
        if i >= len(self._rows) or t.numel() != self._rows[i].numel() // ROW_VALUE_BYTES * entry_bytes:
            raise ValueError(f"Engram resident: extra table {i} ({t.numel()} B) does not match resident table {i}")
        self._extras.append(t)
        self._extra_bytes = int(entry_bytes)
        return i

    def submit(self, file_idx, offsets, xtab, xidx, dst, dst_stride: int, row_bytes: int, priority: int = 0) -> int:
        if row_bytes != ROW_VALUE_BYTES:
            raise ValueError(f"Engram resident: row_bytes {row_bytes} != {ROW_VALUE_BYTES}")
        n = int(xidx.numel())
        ns = self._ext.gather_resident(self._rows, self._extras, xtab, xidx, dst, int(dst_stride), int(row_bytes),
                                       self._extra_bytes)
        ticket, self._next = self._next, self._next + 1
        self._jobs[ticket] = [n, n, 0, 0, int(ns)]
        return ticket

    def done(self, ticket: int) -> bool:
        if ticket not in self._jobs:
            raise RuntimeError(f"Engram resident: unknown or already-waited ticket {ticket}")
        return True

    def wait(self, ticket: int) -> list[int]:
        if ticket not in self._jobs:
            raise RuntimeError(f"Engram resident: unknown or already-waited ticket {ticket}")
        return self._jobs.pop(ticket)

    def prefetch(self, file_idx, offsets, row_bytes: int) -> int:
        return 0                                  # nothing to warm: rows are in RAM

    def pending_prefetch_rows(self) -> int:
        return 0

    def drop_cache(self, file: int) -> None:
        pass

    def set_gather_willneed(self, on: bool) -> None:
        self._willneed = bool(on)

    def gather_willneed(self) -> bool:
        return self._willneed

    def shutdown(self) -> None:
        self._rows, self._extras = [], []
        self._jobs.clear()
        self._tables.release()


# ------------------------------------------------------------------------------------------------ request state
class _History:
    """Compressed token ids of one request by absolute position.

    Positions < prompt_len (the prompt, or every committed token of a resumed request) are never dropped; a write
    beyond them ends the known history at its last position, so tokens of rejected speculative drafts can never be
    mistaken for known ones (spec-decode rollback)."""

    def __init__(self, ids: np.ndarray) -> None:
        self.buf = np.empty(max(16, int(ids.shape[0] * 1.25) + 16), dtype=np.int64)
        self.buf[: ids.shape[0]] = ids
        self.length = int(ids.shape[0])
        self.prompt_len = self.length

    def write(self, start: int, ids: np.ndarray) -> None:
        if start > self.length:
            raise ValueError(f"history gap: write at {start}, only {self.length} positions known")
        end = start + ids.shape[0]
        if end > self.buf.shape[0]:
            grown = np.empty(max(end + 16, self.buf.shape[0] * 2), dtype=np.int64)
            grown[: self.length] = self.buf[: self.length]
            self.buf = grown
        self.buf[start:end] = ids
        self.length = max(end, self.prompt_len)

    def view(self) -> np.ndarray:
        return self.buf[: self.length]


@dataclass
class _ReqSlot:
    off: int          # first token of this request in the step's scheduler-order token space
    n: int
    start_pos: int
    resolved: bool


@dataclass
class _Step:
    step_id: int
    slot: int
    reqs: dict[str, _ReqSlot]
    total_tokens: int
    inv: np.ndarray                     # [total_tokens, L, S] int32 unique-row index (+1; 0 = zero row)
    n_unique: int                       # unique rows written into the slot (row 0 excluded)
    tickets: list[int] = field(default_factory=list)
    t_begin: float = 0.0
    layout: EngramBatchLayout | None = None
    h2d_issued: bool = False
    expanded: bool = False
    t_pad: int = 0
    # lookahead hits: (buffer, source rows in the buffer, first destination row in the slot); copied in _issue_h2d
    copies: list[tuple[_AheadBuffer, np.ndarray, int]] = field(default_factory=list)


@dataclass
class _Slice:
    handle: int
    req_id: str
    start_pos: int
    n: int
    t_pad: int
    n_unique: int                       # unique rows in the slice slot (row 0 = zero row excluded)
    tickets: list[int] = field(default_factory=list)
    t_begin: float = 0.0
    h2d_issued: bool = False
    expanded: set[int] = field(default_factory=set)    # layer indices already expanded into the dense buffer


class _AheadBuffer:
    """Rows of one lookahead gather: ``keys`` (sorted, unique, (layer << 40) | row) -> ``rows[i]`` [264 B], written by
    one reader job (``ticket``). Not usable until the job is waited for (``_ahead_ready``)."""

    def __init__(self, cap_rows: int) -> None:
        self.cap = int(cap_rows)
        self.rows = torch.empty((max(self.cap, 1), ENGRAM_ROW_BYTES), dtype=torch.uint8, device="cpu")
        self.keys = np.empty(0, np.int64)
        self.ticket: int | None = None

    def clear(self) -> None:
        self.keys = np.empty(0, np.int64)


def split_lookahead_hits(keys: np.ndarray, ahead_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """keys, ahead_keys: sorted unique int64. Returns (hit mask over keys, index into ahead_keys of every hit)."""
    if ahead_keys.size == 0 or keys.size == 0:
        return np.zeros(keys.shape[0], bool), np.empty(0, np.int64)
    pos = np.searchsorted(ahead_keys, keys)
    pos_c = np.minimum(pos, ahead_keys.shape[0] - 1)
    hit = ahead_keys[pos_c] == keys
    return hit, pos_c[hit].astype(np.int64)


def misses_first(inv: np.ndarray, hit: np.ndarray) -> tuple[np.ndarray, int]:
    """Renumber the unique keys so the misses (in their sorted order) come first, then the hits (sorted order).
    ``inv`` indexes the unique keys; returns (renumbered inv, number of misses)."""
    order = np.concatenate([np.flatnonzero(~hit), np.flatnonzero(hit)])
    new_pos = np.empty(hit.shape[0], np.int64)
    new_pos[order] = np.arange(hit.shape[0])
    return new_pos[inv], int(hit.shape[0] - int(hit.sum()))


class EngramHostService:
    """See the module docstring. One instance per TP rank of a stage that owns Engram layers."""

    def __init__(self, hf_config, layers: tuple[int, ...], tp_rank: int, tp_size: int, row_dir: str,
                 tokenizer_path: str, max_num_batched_tokens: int, device: torch.device,
                 io_threads: int = 4) -> None:
        try:
            self._init(hf_config, layers, tp_rank, tp_size, row_dir, tokenizer_path, max_num_batched_tokens,
                       device, io_threads)
        except BaseException:
            # a failing ctor must not leave scale tables cudaHostRegister'ed (or reader threads running): the
            # mapping is freed by GC while still registered and a later registration of a recycled address fails
            # (cudaErrorHostMemoryAlreadyRegistered). Re-raise unchanged.
            self.shutdown()
            raise

    def _init(self, hf_config, layers: tuple[int, ...], tp_rank: int, tp_size: int, row_dir: str,
              tokenizer_path: str, max_num_batched_tokens: int, device: torch.device, io_threads: int) -> None:
        t0 = time.perf_counter()
        self.layout = EngramLayout.from_hf_config(hf_config)
        if layout_bad := [lid for lid in layers if lid not in self.layout.layer_ids]:
            raise ValueError(f"layers {layout_bad} are not Engram layers {self.layout.layer_ids}")
        if not layers:
            raise ValueError("EngramHostService needs at least one Engram layer")
        if self.layout.head_dim != ENGRAM_HEAD_DIM:
            raise ValueError(f"Engram head_dim {self.layout.head_dim} != contract {ENGRAM_HEAD_DIM}")
        self.layers = tuple(layers)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.subtables = self.layout.subtables_for_rank(tp_rank, tp_size)
        self.n_sub = len(self.subtables)
        self.device = torch.device(device)
        self.max_tokens = int(max_num_batched_tokens)
        self.o_direct = knobs.env_bool(O_DIRECT_KNOB, False)
        self.lookahead_tokens = knobs.env_int(LOOKAHEAD_KNOB, 0, minimum=0, maximum=self.max_tokens)
        self.resident = knobs.env_bool(RESIDENT_KNOB, False)
        self._resident: ResidentRowTables | None = None
        if self.resident and self.lookahead_tokens:
            logger.info("Engram resident: %s=%d ignored (rows are in RAM)", LOOKAHEAD_KNOB, self.lookahead_tokens)
            self.lookahead_tokens = 0
        # lookahead replaces the whole-prompt warm-up unless PREFETCH is set explicitly
        self.prefetch_enabled = (knobs.env_bool(PREFETCH_KNOB, self.lookahead_tokens == 0) and not self.o_direct
                                 and not self.resident)
        self.prefetch_order = knobs.env_str(PREFETCH_ORDER_KNOB, "offset", choices=("offset", "position"))
        token_map = load_compressed_token_map(tokenizer_path, self.layout.compressed_vocab_size)
        self.hasher = EngramHasher(self.layout, token_map, self.layers, self.subtables)

        self.tables = locate_engram_tables(row_dir, self.layout, self.layers)
        if knobs.env_bool(REQUIRE_VERIFIED_KNOB, True):
            for loc in self.tables:
                _check_verified(loc.weight_path)
                _check_verified(loc.scale_path)

        # -- per layer: weight file index, absolute row offsets, rank-local scale tables (pinned, exact size)
        L, S = len(self.layers), self.n_sub
        lis = [self.layout.layer_index(lid) for lid in self.layers]
        self._sub_offset = np.array([[self.layout.offsets[li][s] for s in self.subtables] for li in lis], np.int64)
        sub_rows = np.array([[self.layout.primes[li][s] for s in self.subtables] for li in lis], np.int64)
        self._sub_rows = sub_rows
        self._scale_base = np.concatenate([np.zeros((L, 1), np.int64), np.cumsum(sub_rows, axis=1)[:, :-1]], axis=1)
        self._weight_offset = np.array([loc.weight_offset for loc in self.tables], np.int64)
        self._scales: list[_PinnedHostBuffer] = []
        self._row_bias: dict[int, int] = {}
        self.scale_exponent_range: dict[int, tuple[int, int]] = {}
        for l_idx, loc in enumerate(self.tables):
            buf = _PinnedHostBuffer(int(sub_rows[l_idx].sum()) * ROW_SCALE_BYTES)
            fd = os.open(loc.scale_path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                for j, s in enumerate(self.subtables):
                    off = loc.scale_offset + self._sub_offset[l_idx, j] * ROW_SCALE_BYTES
                    n = int(sub_rows[l_idx, j]) * ROW_SCALE_BYTES
                    dst_a = int(self._scale_base[l_idx, j]) * ROW_SCALE_BYTES
                    _pread_into(fd, buf.array[dst_a:dst_a + n], int(off), loc.scale_path)
                    os.posix_fadvise(fd, int(off), n, os.POSIX_FADV_DONTNEED)  # the pinned copy is the only one kept
            finally:
                os.close(fd)
            e_min, e_max = int(buf.array.min()), int(buf.array.max())
            if e_max == 255:
                raise ValueError(f"layer {loc.layer_id}: Engram scales hold UE8M0 NaN (0xFF)")
            self.scale_exponent_range[loc.layer_id] = (e_min - 127, e_max - 127)
            self._row_bias[loc.layer_id] = row_bias_for_exponents(e_min, e_max)
            buf.register()
            self._scales.append(buf)

        io_threads = knobs.env_int(IO_THREADS_KNOB, int(io_threads), minimum=1, maximum=64)
        paths = [loc.weight_path for loc in self.tables]
        uniq_paths = sorted(set(paths))
        self._file_of_layer = np.array([uniq_paths.index(p) for p in paths], np.int32)
        if self.resident:
            self._resident = self._load_resident()
            self._reader = _ResidentReader(load_engram_resident_extension(), self._resident)
        else:
            ext = load_engram_rows_extension()
            self._reader = ext.RowReader(uniq_paths, int(io_threads), self.o_direct)
        self.gather_willneed = knobs.env_bool(GATHER_WILLNEED_KNOB, True)
        self._reader.set_gather_willneed(self.gather_willneed)
        self._table_ids = [self._reader.add_table(b.tensor, ROW_SCALE_BYTES) for b in self._scales]
        self._table_ids_arr = np.array(self._table_ids, np.int32)
        self.io_threads = int(io_threads)

        # -- staging: 2 slots of unique rows (row 0 = all zeros = padding) + runner-order index maps
        self.max_unique = self.max_tokens * L * S + 1
        # explicit CPU: vLLM constructs the model (and this service) under a `with torch.device("cuda")` context
        self._host_rows = torch.zeros((N_SLOTS, self.max_unique, ENGRAM_ROW_BYTES), dtype=torch.uint8,
                                      device="cpu").pin_memory()
        self._host_idx = torch.zeros((N_SLOTS, L * self.max_tokens * S), dtype=torch.int32, device="cpu").pin_memory()
        self._dev_rows = torch.zeros((N_SLOTS, self.max_unique, ENGRAM_ROW_BYTES), dtype=torch.uint8,
                                     device=self.device)
        self._dev_idx = torch.zeros((N_SLOTS, L * self.max_tokens * S), dtype=torch.int32, device=self.device)
        self._dense = torch.zeros((L, self.max_tokens, S, ENGRAM_ROW_BYTES), dtype=torch.uint8, device=self.device)
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._h2d_done = [torch.cuda.Event() for _ in range(N_SLOTS)]
        self._consumed = [torch.cuda.Event() for _ in range(N_SLOTS)]
        self._h2d_recorded = [False] * N_SLOTS
        self._consumed_recorded = [False] * N_SLOTS
        self._next_slot = 0
        # lookahead buffers (plain CPU memory: only the reader writes and a host copy reads them); [0] is filled by
        # the newest begin_step, [1] holds the rows gathered one step earlier (consumed by the current step)
        self._ahead: list[_AheadBuffer] = []
        if self.lookahead_tokens > 0:
            cap = self.lookahead_tokens * L * S
            self._ahead = [_AheadBuffer(cap) for _ in range(N_AHEAD)]

        # -- slice lane: one slot, allocated only with the knob (~2 x max_unique x 264 B + dense, pinned + device)
        self.slice_enabled = knobs.env_bool(SLICE_KNOB, False)
        self._slice: _Slice | None = None
        self._slice_lane: int | None = None
        self._next_handle = 1
        if self.slice_enabled:
            self._slice_host_rows = torch.zeros((self.max_unique, ENGRAM_ROW_BYTES), dtype=torch.uint8,
                                                device="cpu").pin_memory()
            self._slice_host_idx = torch.zeros(L * self.max_tokens * S, dtype=torch.int32, device="cpu").pin_memory()
            self._slice_dev_rows = torch.zeros((self.max_unique, ENGRAM_ROW_BYTES), dtype=torch.uint8,
                                               device=self.device)
            self._slice_dev_idx = torch.zeros(L * self.max_tokens * S, dtype=torch.int32, device=self.device)
            self._slice_dense = torch.zeros((L, self.max_tokens, S, ENGRAM_ROW_BYTES), dtype=torch.uint8,
                                            device=self.device)
            self._slice_h2d_done = torch.cuda.Event()
            self._slice_consumed = torch.cuda.Event()
            self._slice_h2d_recorded = False
            self._slice_consumed_recorded = False

        self._hist: dict[str, _History] = {}
        self._step: _Step | None = None
        self._stats = dict(steps=0, dummy_steps=0, lookups=0, unique_rows=0, hits=0, misses=0, pages=0, prefetch_rows=0,
                           wait_blocked_s=0.0, h2d_bytes=0, lookahead_rows=0, lookahead_hits=0, lookahead_wait_s=0.0,
                           slices=0, slice_rows=0, slice_wait_blocked_s=0.0)
        self._lat_ms: list[float] = []
        self.init_seconds = time.perf_counter() - t0
        logger.info("Engram host service: layers %s, TP rank %d/%d sub-tables %s, scales %.2f GiB page-locked, "
                    "row bias %s, scale exponents %s, %d I/O threads (%s), staging %.1f MiB/slot, lookahead %d "
                    "tokens, admission prefetch %s, init %.1f s",
                    self.layers, tp_rank, tp_size, self.subtables, self.pinned_scale_bytes / 2**30, self._row_bias,
                    self.scale_exponent_range, 0 if self.resident else io_threads,
                    "RAM-resident rows" if self.resident else "O_DIRECT" if self.o_direct else "buffered",
                    self.max_unique * ENGRAM_ROW_BYTES / 2**20, self.lookahead_tokens,
                    self.prefetch_order if self.prefetch_enabled else "off", self.init_seconds)

    def _load_resident(self) -> ResidentRowTables:
        """VLLM_DS41_ENGRAM_RESIDENT=1: load this rank's sub-table rows into RAM (one rank at a time per host)."""
        L = len(self.layers)
        ranges = [resident_ranges(int(self._weight_offset[l]), self._sub_offset[l], self._sub_rows[l],
                                  self._scale_base[l]) for l in range(L)]
        need = sum(int(r[:, 2].sum()) for r in ranges)
        nodes = parse_resident_nodes(knobs.env_str(RESIDENT_NODES_KNOB, "all"), self.tp_rank)
        margin = knobs.env_int(RESIDENT_MARGIN_KNOB, 32, minimum=0, maximum=4096) << 30
        threads = knobs.env_int(RESIDENT_THREADS_KNOB, 32, minimum=1, maximum=64)
        o_direct = knobs.env_bool(RESIDENT_O_DIRECT_KNOB, True)
        chunk = knobs.env_int(RESIDENT_CHUNK_KNOB, 2, minimum=1, maximum=1024) << 20
        n_verify = knobs.env_int(RESIDENT_VERIFY_KNOB, 4096, minimum=0)
        lock = knobs.env_str(RESIDENT_LOCK_KNOB, "/tmp/ds41-engram-resident.lock")
        ext = load_engram_resident_extension()
        t_wait = time.perf_counter()
        with host_lock(lock):
            waited = time.perf_counter() - t_wait
            what = f"TP rank {self.tp_rank} layers {self.layers} sub-tables {self.subtables}"
            check_resident_memory(need, margin, what)
            before = memory_report()
            res = ResidentRowTables(ext, [loc.weight_path for loc in self.tables], ranges, nodes, threads, o_direct,
                                    chunk)
        checked = res.verify(n_verify, seed=self.tp_rank)
        logger.info("Engram resident: %s: %.2f GiB loaded in %.1f s (%.2f GB/s, %d threads, %s, %d MiB preads, "
                    "nodes %s; waited %.1f s for the host lock), %d rows verified by pread. Placement: %s. Before: "
                    "%s. After: %s", what, res.nbytes / 2**30, res.load_seconds,
                    res.nbytes / max(res.load_seconds, 1e-9) / 1e9, threads, "O_DIRECT" if o_direct else "buffered",
                    chunk >> 20, "task policy" if nodes is None else nodes, waited, checked, res.placement(), before,
                    memory_report())
        return res

    # ---------------------------------------------------------------- properties used by the module / tests
    @property
    def pinned_scale_bytes(self) -> int:
        return sum(b.nbytes for b in self._scales)

    @property
    def pinned_staging_bytes(self) -> int:
        return self._host_rows.numel() + self._host_idx.numel() * 4

    def row_bias(self, layer_id: int) -> int:
        return self._row_bias[layer_id]

    def rows_buffer(self, layer_id: int, num_tokens_padded: int) -> torch.Tensor:
        """Static device view [T_pad, n_sub, 264] that wait_rows(layer_id) fills (stable across graph replays). Inside
        slice_lane(): the slice lane's own buffer (the chunk's T_pad must match begin_slice)."""
        if self._slice_lane is not None:
            sl = self._live_slice(self._slice_lane)
            if num_tokens_padded != sl.t_pad:
                raise ValueError(f"Engram slice {sl.handle}: forward has {num_tokens_padded} tokens, the staged chunk "
                                 f"{sl.t_pad}")
            return self._slice_dense[self.layers.index(layer_id), :num_tokens_padded]
        if num_tokens_padded > self.max_tokens:
            raise ValueError(f"Engram: {num_tokens_padded} tokens exceed max_num_batched_tokens {self.max_tokens}")
        return self._dense[self.layers.index(layer_id), :num_tokens_padded]

    # ---------------------------------------------------------------- key -> I/O addresses
    def _addresses(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                    np.ndarray]:
        """rows [n, L, S] global row ids -> (inverse [n, L, S] into the unique keys, the sorted unique keys
        (layer << 40) | row, then per unique key: file index, absolute byte offset of the 256-B row, scale table id,
        scale entry index). A global row id lies in exactly one sub-table, so any occurrence of a key gives its
        sub-table column."""
        n, L, S = rows.shape
        l_flat = np.broadcast_to(np.arange(L, dtype=np.int64)[None, :, None], (n, L, S)).reshape(-1)
        j_flat = np.broadcast_to(np.arange(S, dtype=np.int64)[None, None, :], (n, L, S)).reshape(-1)
        r_flat = rows.reshape(-1)
        keys, first, inv = np.unique((l_flat << 40) | r_flat, return_index=True, return_inverse=True)
        l_u, j_u, r_u = l_flat[first], j_flat[first], r_flat[first]
        local = r_u - self._sub_offset[l_u, j_u]
        if local.size and (int(local.min()) < 0 or bool((local >= self._sub_rows[l_u, j_u]).any())):
            raise AssertionError("Engram row id outside its sub-table (hash/layout mismatch)")
        file_idx = self._file_of_layer[l_u]
        offsets = self._weight_offset[l_u] + r_u * ROW_VALUE_BYTES
        tab = self._table_ids_arr[l_u]
        sidx = self._scale_base[l_u, j_u] + local
        return inv.reshape(n, L, S).astype(np.int32), keys, file_idx, offsets, tab, sidx

    def _submit_rows(self, rows: np.ndarray, slot: int, first_unique: int,
                     step: _Step | None = None, host_rows: torch.Tensor | None = None,
                     priority: int = 0) -> tuple[np.ndarray, int, int]:
        """Gather the unique rows of ``rows`` [n, L, S] into slot rows [first_unique, ...). Returns (inverse+offset,
        number of unique rows, ticket). With ``step`` and a filled lookahead buffer, keys found there are not read:
        they take the slot rows after the misses and ``step.copies`` records the host copy _issue_h2d performs."""
        inv, keys, file_idx, offsets, tab, sidx = self._addresses(rows)
        n_u = offsets.shape[0]
        if first_unique + n_u > self.max_unique:
            raise RuntimeError(f"Engram staging overflow: {first_unique + n_u} rows > {self.max_unique}")
        n_read = n_u
        if step is not None and self._ahead and self._ahead[1].keys.size:
            buf = self._ahead[1]
            hit, src = split_lookahead_hits(keys, buf.keys)
            if src.size:
                inv, n_read = misses_first(inv, hit)
                miss = ~hit
                file_idx, offsets, tab, sidx = file_idx[miss], offsets[miss], tab[miss], sidx[miss]
                step.copies.append((buf, src, first_unique + n_read))
                self._stats["lookahead_hits"] += int(src.size)
        rows_t = self._host_rows[slot] if host_rows is None else host_rows
        dst = rows_t[first_unique:first_unique + n_read].view(-1)
        ticket = self._reader.submit(torch.from_numpy(file_idx), torch.from_numpy(offsets),
                                     torch.from_numpy(tab), torch.from_numpy(sidx), dst, ENGRAM_ROW_BYTES,
                                     ROW_VALUE_BYTES, priority)
        self._stats["lookups"] += int(rows.size)
        self._stats["unique_rows"] += n_u
        return (inv + first_unique).astype(np.int32), n_u, ticket

    def _ahead_ready(self, buf: _AheadBuffer) -> None:
        """Block until ``buf``'s lookahead job is done (normally it finished during the previous step)."""
        if buf.ticket is None:
            return
        t0 = time.perf_counter()
        self._reader.wait(buf.ticket)
        self._stats["lookahead_wait_s"] += time.perf_counter() - t0
        buf.ticket = None

    def _rotate_lookahead(self) -> None:
        """Start of begin_step: [1] becomes the buffer gathered during the previous step (this step's hits), [0] the
        one this step refills. [0] last served the previous step, whose host copies ran in its _issue_h2d (or were
        dropped with an aborted step); its own job is drained before the buffer is reused."""
        self._ahead.reverse()
        self._ahead_ready(self._ahead[0])
        self._ahead[0].clear()

    def _submit_lookahead(self, rows: list[np.ndarray]) -> None:
        """Gather ``rows`` (each [n, L, S]) into lookahead buffer [0] at mid priority (after step gathers)."""
        buf = self._ahead[0]
        if not rows:
            return
        _, keys, file_idx, offsets, tab, sidx = self._addresses(np.concatenate(rows, axis=0))
        n = keys.shape[0]
        if n > buf.cap:
            raise RuntimeError(f"Engram lookahead overflow: {n} rows > {buf.cap}")
        buf.ticket = self._reader.submit(torch.from_numpy(file_idx), torch.from_numpy(offsets),
                                         torch.from_numpy(tab), torch.from_numpy(sidx), buf.rows[:n].view(-1),
                                         ENGRAM_ROW_BYTES, ROW_VALUE_BYTES, 1)
        buf.keys = keys
        self._stats["lookahead_rows"] += n

    def _prefetch_rows(self, rows: np.ndarray) -> None:
        n, L, S = rows.shape
        flat = ((np.arange(L, dtype=np.int64)[None, :, None] << 40) | rows).reshape(-1)
        if self.prefetch_order == "position":   # warm the prompt front to back (unique keys by first occurrence)
            _, first = np.unique(flat, return_index=True)
            keys = flat[np.sort(first)]
        else:                                   # today: table-offset order
            keys = np.unique(flat)
        l_idx = keys >> 40
        offsets = self._weight_offset[l_idx] + (keys & ((1 << 40) - 1)) * ROW_VALUE_BYTES
        self._reader.prefetch(torch.from_numpy(self._file_of_layer[l_idx].astype(np.int32)),
                              torch.from_numpy(offsets), ROW_VALUE_BYTES)
        self._stats["prefetch_rows"] += int(keys.shape[0])

    # ---------------------------------------------------------------- hooks
    def begin_step(self, plan: EngramStepPlan) -> None:
        """Worker thread, BEFORE the PP receive. Hash every request whose token ids are known and submit this step's
        gather; admission prefetch for new/resumed requests. Never blocks on I/O (a CPU wait happens only if the H2D
        that last read this pinned slot -- two steps ago -- has not finished)."""
        self._retire_step()
        for rid in plan.finished_req_ids:
            self._hist.pop(rid, None)
        if self._ahead:
            self._rotate_lookahead()
        ahead_rows: list[np.ndarray] = []
        ahead_budget = self.lookahead_tokens
        slot = self._next_slot
        self._next_slot ^= 1
        if self._h2d_recorded[slot]:
            self._h2d_done[slot].synchronize()
        reqs: dict[str, _ReqSlot] = {}
        known_rows: list[np.ndarray] = []
        known_meta: list[tuple[int, int]] = []   # (token offset, n) of each known block
        prefetch: list[np.ndarray] = []
        off = 0
        for r in plan.reqs:
            if r.req_id in reqs:
                raise ValueError(f"Engram step {plan.step_id}: request {r.req_id} scheduled twice")
            if r.num_tokens <= 0:
                raise ValueError(f"Engram step {plan.step_id}: request {r.req_id} has {r.num_tokens} tokens")
            hist = self._hist.get(r.req_id)
            if r.prompt_token_ids is not None:
                hist = _History(self.hasher.compress(np.asarray(r.prompt_token_ids)))
                self._hist[r.req_id] = hist
            elif hist is None:
                raise KeyError(f"Engram: request {r.req_id} has no token history; its first (or resumed) step must "
                               "carry prompt_token_ids")
            start, n = int(r.start_pos), int(r.num_tokens)
            if r.token_ids is not None:
                ids = np.asarray(r.token_ids)
                if ids.shape != (n,):
                    raise ValueError(f"Engram: request {r.req_id} token_ids {ids.shape} != num_tokens {n}")
                hist.write(start, self.hasher.compress(ids))
                resolved = True
            elif start + n <= hist.prompt_len:
                resolved = True                     # continuation of the known prompt (chunked prefill)
            elif n == 1 and hist.prompt_len <= start <= hist.length:
                resolved = False                    # sampled token arrives in bind_batch (no PP); positions past
                                                    # start (rejected drafts) are dropped when it is written
            else:
                raise ValueError(f"Engram: request {r.req_id} positions [{start}, {start + n}) without token ids: "
                                 f"only prompt positions (< {hist.prompt_len}) or one sampled token at <= "
                                 f"{hist.length} can be resolved")
            if resolved:
                known_rows.append(self.hasher.hash_positions(hist.view(), np.arange(start, start + n)))
                known_meta.append((off, n))
                if r.prompt_token_ids is not None and self.prefetch_enabled and start + n < hist.length:
                    prefetch.append(self.hasher.hash_positions(hist.view(), np.arange(start + n, hist.length)))
                if ahead_budget > 0 and start + n < hist.prompt_len:   # the next chunk of a prompt being prefilled
                    end = min(hist.prompt_len, start + n + ahead_budget)
                    ahead_rows.append(self.hasher.hash_positions(hist.view(), np.arange(start + n, end)))
                    ahead_budget -= end - (start + n)
            reqs[r.req_id] = _ReqSlot(off=off, n=n, start_pos=start, resolved=resolved)
            off += n
        L, S = len(self.layers), self.n_sub
        inv = np.zeros((off, L, S), np.int32)
        step = _Step(step_id=plan.step_id, slot=slot, reqs=reqs, total_tokens=off, inv=inv, n_unique=0,
                     t_begin=time.perf_counter())
        if known_rows:
            rows = np.concatenate(known_rows, axis=0)
            kinv, n_u, ticket = self._submit_rows(rows, slot, 1, step)
            step.tickets.append(ticket)
            self._step = step          # published at once: a later exception still leaves the gather owned/drained
            pos = 0
            for t_off, n in known_meta:
                inv[t_off:t_off + n] = kinv[pos:pos + n]
                pos += n
            step.n_unique = n_u
        self._step = step
        self._stats["steps"] += 1
        if self._ahead:
            self._submit_lookahead(ahead_rows)
        for rows in prefetch:
            self._prefetch_rows(rows)

    def bind_batch(self, layout: EngramBatchLayout) -> None:
        """Runner thread, after _prepare_inputs and before the forward of a REAL scheduled step. Dummy forwards
        (profiling, CUDA-graph capture) are not bound: the module asks for ``dummy_rows`` when the forward context
        says ``is_dummy_run`` (AM-2)."""
        t_pad = int(layout.num_tokens_padded)
        if t_pad > self.max_tokens:
            raise ValueError(f"Engram: T_pad {t_pad} > max_num_batched_tokens {self.max_tokens}")
        if layout.step_id < 0 or layout.num_tokens <= 0:
            raise ValueError(f"Engram: bind_batch got step {layout.step_id} with {layout.num_tokens} tokens; dummy "
                             "forwards are marked by ForwardContext.is_dummy_run and are never bound (AM-2)")
        step = self._step
        if step is None or step.step_id != layout.step_id:
            raise RuntimeError(f"Engram: bind_batch for step {layout.step_id} but begin_step saw "
                               f"{None if step is None else step.step_id}")
        if step.layout is not None:
            raise RuntimeError(f"Engram: step {layout.step_id} bound twice")
        qsl = np.asarray(layout.query_start_loc)
        if qsl.shape[0] != len(layout.req_order) + 1 or int(qsl[-1]) != layout.num_tokens:
            raise ValueError(f"Engram: query_start_loc {qsl.tolist()} inconsistent with {len(layout.req_order)} "
                             f"requests / {layout.num_tokens} tokens")
        if set(layout.req_order) != set(step.reqs) or layout.num_tokens != step.total_tokens:
            raise ValueError(f"Engram: runner batch {sorted(layout.req_order)} / {layout.num_tokens} tokens differs "
                             f"from the plan {sorted(step.reqs)} / {step.total_tokens}")
        unresolved = [(i, rid) for i, rid in enumerate(layout.req_order) if not step.reqs[rid].resolved]
        if unresolved:
            self._resolve_sampled(step, layout, unresolved)
        L, S = len(self.layers), self.n_sub
        perm = np.empty(layout.num_tokens, np.int64)
        for i, rid in enumerate(layout.req_order):
            a, b = int(qsl[i]), int(qsl[i + 1])
            rs = step.reqs[rid]
            if b - a != rs.n:
                raise ValueError(f"Engram: request {rid} has {b - a} tokens in the runner batch, {rs.n} planned")
            perm[a:b] = np.arange(rs.off, rs.off + rs.n)
        idx = np.zeros((L, t_pad, S), np.int32)
        idx[:, : layout.num_tokens] = step.inv[perm].transpose(1, 0, 2)
        self._host_idx[step.slot, : L * t_pad * S].numpy()[:] = idx.reshape(-1)
        step.layout = layout
        step.t_pad = t_pad
        if all(self._reader.done(t) for t in step.tickets) and all(
                buf.ticket is None or self._reader.done(buf.ticket) for buf, _, _ in step.copies):
            self._issue_h2d(step)

    def _resolve_sampled(self, step: _Step, layout: EngramBatchLayout, unresolved: list[tuple[int, str]]) -> None:
        if layout.sampled_fill is None:
            raise RuntimeError(f"Engram step {step.step_id}: requests {[r for _, r in unresolved]} have unknown "
                               "tokens but the runner passed no sampled_fill")
        ids_t, event = layout.sampled_fill
        event.synchronize()
        ids = ids_t.reshape(-1).numpy()
        if ids.shape[0] != len(layout.req_order):
            raise ValueError(f"Engram step {step.step_id}: sampled_fill holds {ids.shape[0]} ids, expected one per "
                             f"request in runner order ({len(layout.req_order)})")
        rows, blocks = [], []
        for i, rid in unresolved:
            rs = step.reqs[rid]
            hist = self._hist[rid]
            hist.write(rs.start_pos, self.hasher.compress(np.array([int(ids[i])])))
            rows.append(self.hasher.hash_positions(hist.view(), np.array([rs.start_pos])))
            blocks.append(rs.off)
            rs.resolved = True
        kinv, n_u, ticket = self._submit_rows(np.concatenate(rows), step.slot, 1 + step.n_unique)
        for k, t_off in enumerate(blocks):
            step.inv[t_off] = kinv[k]
        step.n_unique += n_u
        step.tickets.append(ticket)

    def _issue_h2d(self, step: _Step) -> None:
        for t in step.tickets:
            t0 = time.perf_counter()
            n, hits, misses, pages, ns = self._reader.wait(t)
            self._stats["wait_blocked_s"] += time.perf_counter() - t0
            self._stats["hits"] += hits
            self._stats["misses"] += misses
            self._stats["pages"] += pages
            self._lat_ms.append(ns / 1e6)
        step.tickets.clear()
        if len(self._lat_ms) > _LATENCY_WINDOW:
            del self._lat_ms[: len(self._lat_ms) - _LATENCY_WINDOW]
        slot = step.slot
        for buf, src, dst0 in step.copies:     # lookahead hits -> slot rows after the misses
            self._ahead_ready(buf)
            torch.index_select(buf.rows, 0, torch.from_numpy(src),
                               out=self._host_rows[slot, dst0:dst0 + src.shape[0]])
        step.copies.clear()
        n_rows = step.n_unique + 1
        n_idx = len(self.layers) * step.t_pad * self.n_sub
        with torch.cuda.stream(self._copy_stream):
            if self._consumed_recorded[slot]:
                self._copy_stream.wait_event(self._consumed[slot])
            self._dev_rows[slot, :n_rows].copy_(self._host_rows[slot, :n_rows], non_blocking=True)
            self._dev_idx[slot, :n_idx].copy_(self._host_idx[slot, :n_idx], non_blocking=True)
            self._h2d_done[slot].record(self._copy_stream)
        self._h2d_recorded[slot] = True
        self._stats["h2d_bytes"] += n_rows * ENGRAM_ROW_BYTES + n_idx * 4
        step.h2d_issued = True

    def wait_rows(self, layer_id: int) -> torch.Tensor:
        """Inside the Engram eager break: block the CPU until this step's rows are gathered and their H2D is enqueued,
        make the current stream wait for it, expand to the static device view [T_pad, n_sub, 264] u8 of layer_id."""
        if self._slice_lane is not None:
            return self.wait_rows_slice(self._slice_lane, layer_id)
        li = self.layers.index(layer_id)
        step = self._step
        if step is None or step.layout is None:
            raise RuntimeError("Engram wait_rows: a REAL step reached Engram without bound rows -- the worker must call "
                               "begin_step(plan) and the runner bind_batch(layout) before every non-dummy forward "
                               "(dummy forwards must set ForwardContext.is_dummy_run, AM-2)")
        if not step.h2d_issued:
            self._issue_h2d(step)
        if not step.expanded:
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(self._h2d_done[step.slot])
            span = step.t_pad * self.n_sub
            for l_idx in range(len(self.layers)):
                out = self._dense[l_idx, : step.t_pad].view(span, ENGRAM_ROW_BYTES)
                torch.index_select(self._dev_rows[step.slot], 0,
                                   self._dev_idx[step.slot, l_idx * span:(l_idx + 1) * span], out=out)
            self._consumed[step.slot].record(cur)
            self._consumed_recorded[step.slot] = True
            step.expanded = True
        return self._dense[li, : step.t_pad]

    def dummy_rows(self, layer_id: int, num_tokens_padded: int) -> torch.Tensor:
        """AM-2 dummy forward (profiling / CUDA-graph capture): all-zero rows in the static buffer -> the Engram
        contribution is exactly zero; no host I/O, no step state touched. Counted once per forward."""
        out = self.rows_buffer(layer_id, num_tokens_padded)
        out.zero_()
        if layer_id == self.layers[0]:
            self._stats["dummy_steps"] += 1
        return out

    def end_step(self, step_id: int) -> None:
        step = self._step
        if step is not None and step.step_id == step_id:
            self._retire_step()

    def _retire_step(self) -> None:
        step = self._step
        if step is None:
            return
        if step.tickets:          # gathered but never consumed (aborted step): drain so the slot can be reused
            for t in step.tickets:
                self._reader.wait(t)
            step.tickets.clear()
        self._step = None

    # ---------------------------------------------------------------- slice lane (hol-design §2a)
    def begin_slice(self, req_id: str, start_pos: int, num_tokens: int,
                    prompt_token_ids: np.ndarray | None = None, num_tokens_padded: int | None = None,
                    token_ids: np.ndarray | None = None) -> int:
        """Worker thread, when the chunk [start_pos, start_pos + num_tokens) of ``req_id`` is STAGED. Prompt positions
        only (the sampled tail takes the normal step path). History, as begin_step: ``prompt_token_ids`` (full prompt)
        creates it; ``token_ids`` (this chunk's ids, SliceDirective.token_ids) creates it at start_pos 0 or extends a
        history known up to start_pos, and marks the chunk as prompt (never rolled back); with neither, the chunk must
        lie inside the known prompt. Never blocks on I/O (a CPU wait happens only if the previous slice's H2D has not
        finished reading the pinned slot)."""
        if not self.slice_enabled:
            raise RuntimeError(f"Engram: begin_slice needs {SLICE_KNOB}=1 (the slice lane is not allocated)")
        if self._slice is not None:
            raise RuntimeError(f"Engram: begin_slice({req_id!r}) while slice {self._slice.handle} "
                               f"({self._slice.req_id!r}) is still staged: one staged chunk per stage")
        start, n = int(start_pos), int(num_tokens)
        t_pad = n if num_tokens_padded is None else int(num_tokens_padded)
        if n <= 0 or t_pad < n or t_pad > self.max_tokens:
            raise ValueError(f"Engram: slice of {n} tokens (T_pad {t_pad}) outside (0, {self.max_tokens}]")
        if prompt_token_ids is not None:
            self._hist[req_id] = _History(self.hasher.compress(np.asarray(prompt_token_ids)))
        if token_ids is not None:
            ids = np.asarray(token_ids)
            if ids.shape != (n,):
                raise ValueError(f"Engram: slice of request {req_id} token_ids {ids.shape} != num_tokens {n}")
            hist = self._hist.get(req_id)
            if hist is None and start == 0:
                self._hist[req_id] = _History(self.hasher.compress(ids))
            elif hist is None:
                raise KeyError(f"Engram: slice of request {req_id} at {start} without token history")
            else:
                hist.write(start, self.hasher.compress(ids))           # raises on a gap
                hist.prompt_len = max(hist.prompt_len, start + n)       # staged positions are prompt: keep them
        hist = self._hist.get(req_id)
        if hist is None:
            raise KeyError(f"Engram: slice of request {req_id} without token history; its first (or resumed) work must "
                           "carry prompt_token_ids")
        if start < 0 or start + n > hist.prompt_len:
            raise ValueError(f"Engram: slice positions [{start}, {start + n}) of request {req_id} are not prompt "
                             f"positions (< {hist.prompt_len})")
        if self._slice_h2d_recorded:
            self._slice_h2d_done.synchronize()       # the previous slice's H2D still reads the pinned rows
        rows = self.hasher.hash_positions(hist.view(), np.arange(start, start + n))
        kinv, n_u, ticket = self._submit_rows(rows, -1, 1, host_rows=self._slice_host_rows, priority=1)
        L, S = len(self.layers), self.n_sub
        idx = np.zeros((L, t_pad, S), np.int32)
        idx[:, :n] = kinv.transpose(1, 0, 2)
        self._slice_host_idx[: L * t_pad * S].numpy()[:] = idx.reshape(-1)
        sl = _Slice(handle=self._next_handle, req_id=req_id, start_pos=start, n=n, t_pad=t_pad, n_unique=n_u,
                    tickets=[ticket], t_begin=time.perf_counter())
        self._next_handle += 1
        self._slice = sl
        self._stats["slices"] += 1
        self._stats["slice_rows"] += n_u
        return sl.handle

    def _live_slice(self, handle: int) -> _Slice:
        sl = self._slice
        if sl is None or sl.handle != handle:
            raise RuntimeError(f"Engram: slice {handle} is not staged (current: {None if sl is None else sl.handle})")
        return sl

    def rows_ready(self, handle: int) -> bool:
        """Non-blocking: the slice's gather has finished (its rows can be consumed without a CPU wait)."""
        sl = self._live_slice(handle)
        return sl.h2d_issued or all(self._reader.done(t) for t in sl.tickets)

    def _slice_issue_h2d(self, sl: _Slice) -> None:
        t0 = time.perf_counter()
        for t in sl.tickets:
            self._reader.wait(t)
        sl.tickets.clear()
        self._stats["slice_wait_blocked_s"] += time.perf_counter() - t0
        n_rows = sl.n_unique + 1
        n_idx = len(self.layers) * sl.t_pad * self.n_sub
        with torch.cuda.stream(self._copy_stream):
            if self._slice_consumed_recorded:
                self._copy_stream.wait_event(self._slice_consumed)
            self._slice_dev_rows[:n_rows].copy_(self._slice_host_rows[:n_rows], non_blocking=True)
            self._slice_dev_idx[:n_idx].copy_(self._slice_host_idx[:n_idx], non_blocking=True)
            self._slice_h2d_done.record(self._copy_stream)
        self._slice_h2d_recorded = True
        self._stats["h2d_bytes"] += n_rows * ENGRAM_ROW_BYTES + n_idx * 4
        sl.h2d_issued = True

    def wait_rows_slice(self, handle: int, layer_id: int) -> torch.Tensor:
        """Inside the slice forward that reaches ``layer_id``: finish the gather / H2D if still pending, make the
        current stream wait for it and expand THIS layer's rows into the lane's dense buffer [T_pad, n_sub, 264]."""
        sl = self._live_slice(handle)
        li = self.layers.index(layer_id)
        if not sl.h2d_issued:
            self._slice_issue_h2d(sl)
        if li not in sl.expanded:
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(self._slice_h2d_done)
            span = sl.t_pad * self.n_sub
            out = self._slice_dense[li, : sl.t_pad].view(span, ENGRAM_ROW_BYTES)
            torch.index_select(self._slice_dev_rows, 0, self._slice_dev_idx[li * span:(li + 1) * span], out=out)
            self._slice_consumed.record(cur)
            self._slice_consumed_recorded = True
            sl.expanded.add(li)
        return self._slice_dense[li, : sl.t_pad]

    @contextlib.contextmanager
    def slice_lane(self, handle: int):
        """Route rows_buffer / wait_rows to the staged slice for the duration of one slice forward."""
        self._live_slice(handle)
        if self._slice_lane is not None:
            raise RuntimeError(f"Engram: slice lane {self._slice_lane} already active")
        self._slice_lane = handle
        try:
            yield
        finally:
            self._slice_lane = None

    def end_slice(self, handle: int) -> None:
        """After the chunk's last Engram layer consumed its rows, or on abort / preempt mid-slice (a pending gather
        is drained so the slot can be reused). Ending an unknown handle raises; ending twice is an error too."""
        sl = self._live_slice(handle)
        if self._slice_lane == handle:
            raise RuntimeError(f"Engram: end_slice({handle}) inside its own slice_lane")
        for t in sl.tickets:
            self._reader.wait(t)
        sl.tickets.clear()
        self._slice = None

    def stats(self) -> dict[str, float]:
        lat = np.array(self._lat_ms) if self._lat_ms else np.zeros(1)
        s = {k: float(v) for k, v in self._stats.items()}
        s.update(gather_p50_ms=float(np.percentile(lat, 50)), gather_p99_ms=float(np.percentile(lat, 99)),
                 gather_max_ms=float(lat.max()), warmup_backlog_rows=float(self._reader.pending_prefetch_rows()),
                 pinned_scale_gib=self.pinned_scale_bytes / 2**30, pinned_staging_mib=self.pinned_staging_bytes / 2**20)
        if self._resident is not None:
            s.update(resident_gib=self._resident.nbytes / 2**30, resident_load_s=self._resident.load_seconds)
        return s

    def drop_page_cache(self) -> None:
        """POSIX_FADV_DONTNEED on the row shards (cold-path measurements; no privileges needed)."""
        for i in range(len(set(loc.weight_path for loc in self.tables))):
            self._reader.drop_cache(i)

    def shutdown(self) -> None:
        """Joins the reader's threads BEFORE the staging/scale memory they write into or read from is released."""
        if getattr(self, "_reader", None) is not None:
            sl = getattr(self, "_slice", None)
            if sl is not None and sl.tickets:
                for t in sl.tickets:
                    self._reader.wait(t)
                sl.tickets.clear()
            self._reader.shutdown()
            self._reader = None
        self._resident = None          # drops the resident tables (munmap once no tensor references them)
        for b in getattr(self, "_scales", []):
            b.release()
        self._scales = []
