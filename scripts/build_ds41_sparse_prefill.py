# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the VLLM_DS41_ATTN_FUSED_PREFILL extension (csrc/ds41_sparse_prefill_sm70.cu) once, for sm_70 only.

Workers never compile it: they import the .so by path (VLLM_DS41_SPARSE_PREFILL_SO, default
build/sparse-prefill/ds41_sparse_prefill.so). Needs nvcc + ninja, no GPU. Prints the ptxas
register / spill / smem report.

    python scripts/build_ds41_sparse_prefill.py [--out build/sparse-prefill]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/sparse-prefill")
    a = ap.parse_args()
    src = Path(__file__).resolve().parents[1] / "csrc" / "ds41_sparse_prefill_sm70.cu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    if shutil.which("ninja") is None:
        os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    from torch.utils.cpp_extension import load
    load(name="ds41_sparse_prefill", sources=[str(src)], build_directory=str(out),
         extra_cflags=["-O3", "-std=c++17"],
         extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_70,code=sm_70", "-Xptxas=-v",
                            "-lineinfo"],          # no --use_fast_math: expf / RNE casts must stay IEEE
         verbose=True)
    so = out / "ds41_sparse_prefill.so"
    print(f"built {so} ({so.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
