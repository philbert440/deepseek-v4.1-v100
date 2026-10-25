# deepseek-v4.1-v100

1Cat-vLLM engine snapshot for serving
[DeepSeek-V4.1-Flash-1017 EXL3 franken-v100](https://huggingface.co/philbert440/DeepSeek-V4.1-Flash-1017-EXL3-franken-v100)
at its native 1,048,576-token context on 8× V100-SXM2-32GB, plus a prebuilt SM70 wheel.

The tree is the engine that is live and benchmarked as of 2026-10-25. It contains everything in
[1CatAI/1Cat-vLLM#1243](https://github.com/1CatAI/1Cat-vLLM/pull/1243) (the engine as of 2026-10-21) and the prefill
and decode work since then, which #1243 doesn't have: decoder replay, fused prefill attention, the tiled EXL3 prefill
MoE kernel, RAM-resident Engram, layer-sliced prefill, fused indexer kernels and the DSpark tap fix. Since tag
`ds41-vision-20261025` it also takes image input: the vision tower and aligner, which the port used to skip.

- Model page: https://philtompkins.com/models/deepseek-v4.1-flash
- Checkpoint: https://huggingface.co/philbert440/DeepSeek-V4.1-Flash-1017-EXL3-franken-v100
- Upstream engine: https://github.com/1CatAI/1Cat-vLLM

## Exact base

| | |
|---|---|
| Upstream | 1Cat-vLLM `main` @ [`3d56d991002dc132565ac585c8dabb76c4ff877b`](https://github.com/1CatAI/1Cat-vLLM/commit/3d56d991002dc132565ac585c8dabb76c4ff877b) (2026-10-07, after the v1.5.1 release) |
| Bring-up | [#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) (PLI Labs) at [`a36df7c2239a13ae8da3f257151bdf92f335cc5f`](https://github.com/1CatAI/1Cat-vLLM/commit/a36df7c2239a13ae8da3f257151bdf92f335cc5f), five commits |
| This engine | [#1243](https://github.com/1CatAI/1Cat-vLLM/pull/1243) plus the later work: engine commit `714597f12461127be033257760fa438d0ee5267f` (tag `ds41-final-20261025`), then a sliced-prefill fix (`a9c8c9d6f55b`) and image input (`71c3e6e14fef`) |
| Snapshot tags | [`ds41-vision-20261025`](https://github.com/philbert440/deepseek-v4.1-v100/tree/ds41-vision-20261025) (current), [`ds41-final-20261025`](https://github.com/philbert440/deepseek-v4.1-v100/tree/ds41-final-20261025) (text-only) |

The history starts with two commits. The first imports #1031's head tree (upstream `main` plus #1031) unmodified,
without upstream history. The second is the engine, squashed. Its tree equals engine commit `714597f124` except for
publication edits:

- comments that named the test host;
- the default paths of the three prebuilt extensions (`ds41_sparse_prefill.so`, `ds41_indexer_fused.so`,
  `ds41_vmm.so`), which now point at the `vllm/` package directory, where the release wheel ships them;
- bench-script defaults, which now come from `DS41_MODEL`, `DS41_ENGRAM_DIR` and `DS41_EXL3_PLAN`;
- the Engram row reader, which also looks for its C++ sources inside the package (`vllm/_ds41_csrc`) when there is no
  source checkout;
- internal benchmark-gate drivers, which were removed.

No kernel or model logic differs. The commit after that tag adds this README and `v100/`. The two engine commits
after it (the sliced-prefill fix and image input) are replayed as they are, and their trees match the served engine
the same way.

Every addition sits behind its own `VLLM_DS41_*` knob, and each one was gated against an fp32 CPU run of DeepSeek's
reference code before it went live.

## Hardware

- 8× V100-SXM2-32GB on NVLink, two quads of four: PP2 × TP4, cut at layer 20, one pipeline stage per quad. Stage 1
  holds 27.77 GiB of weights per GPU and stage 2 holds 25.29 GiB plus the drafter, so 16 GB cards are out.
- Host RAM for the Engram tables: they load RAM-resident (45.78 GiB per stage-0 rank), and the engine refuses to
  start unless the tables plus 32 GiB are available. The test box has 372 GB and 64 cores.
- Disk: 218.6 GB for the pack plus the two official Engram shards (101.5 GB each).
- CUDA 12.8 (CUDA 13 dropped Volta), Python 3.12, torch 2.10 cu128.

## Install (prebuilt wheel)

```bash
# download 1cat_vllm-1.5.2.dev1148+ds41vision1025-cp312-cp312-linux_x86_64.whl and SHA256SUMS from
# https://github.com/philbert440/deepseek-v4.1-v100/releases/tag/ds41-vision-20261025
sha256sum -c SHA256SUMS
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ./1cat_vllm-1.5.2.dev1148+ds41vision1025-cp312-cp312-linux_x86_64.whl ninja --torch-backend=cu128
```

The wheel is built from the tagged tree with `TORCH_CUDA_ARCH_LIST=7.0`, CUDA 12.8.1, gcc 14 and torch 2.10.0+cu128 on
Ubuntu 24.04 (glibc 2.39). It bundles the three prebuilt DS41 extensions in `vllm/`. The CUDA 12.8 toolkit, a C++
compiler and `ninja` are still needed at runtime: the Engram reader and the TileLang kernels compile on first start.

From source instead: clone the tag, then
`TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70 CC=gcc-14 CXX=g++-14 uv pip install -e . --torch-backend=cu128 --no-build-isolation`.
Build the extensions with `python scripts/build_ds41_sparse_prefill.py` and `python scripts/build_ds41_indexer_fused.py`,
and set `VLLM_DS41_SPARSE_PREFILL_SO`, `VLLM_DS41_INDEXER_FUSED_SO` and `VLLM_DS41_VMM_SO` to the built files.

## Download

```bash
hf download philbert440/DeepSeek-V4.1-Flash-1017-EXL3-franken-v100 \
  --revision 99d85a463b446e4ff07e173be6a8faf9b920b97d --local-dir ds41-franken-v100
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ds41-franken-v100
DS41_MODEL=$PWD/ds41-franken-v100 v100/verify_engram.sh   # sha256 check + the .sha256-ok sidecars preflight wants
```

The pack holds the EXL3 experts and every official non-expert tensor. The Engram tables are unmodified, so they come
from DeepSeek's repo. Engram reads 256-byte rows at random, so a small filesystem record size helps (ZFS
`recordsize=16K` on the test box).

## Launch

```bash
cp v100/ds41.env.example ds41.env   # edit DS41_MODEL, DS41_PYTHON, DS41_GPUS_A / DS41_GPUS_C
source ds41.env
v100/serve_ds41.sh --dry-run --strict   # pre-flight: GPUs free, each quad all-NVLink, shards, port, memory; no launch
v100/serve_ds41.sh                      # detached tmux session; add --foreground to run in this terminal
./v100/smoke.py http://127.0.0.1:$DS41_PORT
```

[`v100/serve_ds41.sh`](v100/serve_ds41.sh) is the live configuration: PP2 × TP4 cut at layer 20,
`--max-model-len 1048576`, `--max-num-batched-tokens 1024`, `--gpu-memory-utilization 0.943`, DSpark with
5 draft tokens, `--max-num-seqs 2`, and every `VLLM_DS41_*` knob the published numbers were measured with. Each knob can be
overridden from the environment, and `DS41_DSPARK=0` switches to plain decode. With passwordless sudo, the launcher sets the CPU
governor to `performance` and locks SM clocks at 1530 MHz (`DS41_HOST_TUNING=0` skips both). With Engram loaded into
RAM, a boot takes about 12 minutes.

Served model name: `deepseek-v4.1-flash`. Thinking is off by default. Pass
`chat_template_kwargs: {"thinking": true}` and it arrives in the `reasoning` field.

## Image input

Send images as OpenAI `image_url` content parts (data URLs work). Images inside tool results work too. Each image
becomes up to 1,024 prompt tokens: a 800×600 image is about 317. Nothing extra to configure.

The vision tower and aligner weights (926 MiB as FP16) stay in pinned host memory. TP rank 0 of the first stage streams
them through two small GPU buffers per image and broadcasts the image embeddings to its TP group. The KV pool, and so
the 1M context, are unchanged. Steps without image tokens run exactly as before, CUDA-graph decode included. Image tokens
route through the MoE gate's `bias_vl`, and Engram skips them, as in DeepSeek's reference code. Against an fp32 run of
the reference tower, this tower's output has relative L2 error 1.5e-2, closer than the reference's own bf16 path (2.7e-2).

Measured on the vision build: a text image read back exactly in 0.8 s (330-token prompt); a 2400×1600 image read in
5.3 s (991-token prompt); a screenshot returned in a tool result read correctly in 1.2 s. Text quality and speed are
the same as the text-only build: KL 0.0358, self-consistency worst mean 0.0237, KV pool 1,055,580 tokens, decode within
noise.

## Measured (this engine, 2026-10-25)

8× V100-SXM2-32GB, PP2 × TP4, franken-v100 pack, `v100/serve_ds41.sh` settings, 1M context (KV pool 1,055,580 tokens).

| | |
|---|---|
| Fidelity vs fp32 reference | KL 0.0358, top-1 95.79% (8 domains, 3,134 positions) |
| Solo prefill 8K / 32K / 128K | 2,906 / 3,246 / 3,185 tok/s |
| Single-stream decode, DSpark (code / structured / chat / prose) | 64.9 / 65.9 / 35.0 / 34.5 tok/s |
| Two streams, aggregate (code / structured / chat / prose) | 95.5 / 103.3 / 57.3 / 52.3 tok/s |
| 128K-token prompt | prefill 3,207 tok/s, then 52.6 tok/s decode |
| 32K prefill arriving during a decode | prefill 1,460 tok/s, the running decode keeps 56.0 tok/s |

Decode speed depends on the text, because it depends on how many drafts are accepted. First-draft acceptance on code is
0.90; on prose it is 0.60.

## Credits

- 1CatAI for [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), the SM70 vLLM fork that makes Volta a real target. It is
  built on [vLLM](https://github.com/vllm-project/vllm).
- PLI Labs: [#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) brought DeepSeek-V4.1 up on it. The launcher in `v100/`
  is adapted from theirs.
- 0xSero, HeNryous and the sudoingX lineage, and Mia-AiLab for the expert packs the franken pack is assembled from.
- DeepSeek for the model and for the reference inference code that the fidelity check runs. Also for DeepInfer, which
  supplied the drafter-graph shape, batch invariance and the self-consistency gate.

## License

Apache-2.0, as upstream: see [LICENSE](LICENSE). Vendored components keep their own licenses and notices.
