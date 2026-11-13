# Environment setup and image generation

[Documentation index](README.md) · [Tool reference](../tools/README.md) · [Model sources](../models/README.md)

These commands target the existing R9700 workspace foundation: Linux, Docker, `rocm-smi`, the parent workspace's `rocm10-launch.sh`, its locked image, and its shared reservation and GPU-lock directories. The launcher uses `deepinfer_workspace/artifacts/amd/rocm10/` and `minimax_h3/.standard-opt-state/locks/` under that parent workspace. This repository does not provision those shared components on a new machine.

Run commands from the project root. Every `--run-dir` must be a new directory under `runs/`. Select currently available physical GPUs; GPU3 is excluded on this host.

## Prepare the pinned sources

The application and extension checkouts stay in the ignored `.runtime/` directory. The setup tools check their commit identities.

```bash
git clone --depth 1 --branch v0.45.0 https://github.com/Comfy-Org/ComfyUI.git .runtime/engines/ComfyUI-image
git clone --depth 1 --branch v0.2.41 https://github.com/Comfy-Org/comfy-kitchen.git .runtime/sources/comfy-kitchen
git clone --depth 1 --branch v0.6.0 https://github.com/Comfy-Org/comfy-aimdo.git .runtime/sources/comfy-aimdo
```

The ComfyUI commit is `989d93d0d968e704c589ce0146bb18c77f6b50cc`. The [runtime lock](../configs/runtime-lock.json) records the foundation image, exact Torch/Triton pairing, extension commits, and previously verified binary hashes.

## Install and check the environment

Set `SETUP_GPU_ID` to the physical ID of one available card on your machine. The launcher checks device identity, KFD users, and shared reservations, then holds the shared lock for its own job.

```bash
python3 tools/run_rocm10.py --gpu "${SETUP_GPU_ID:?Set SETUP_GPU_ID to an available physical GPU ID}" --run-dir runs/setup-image-env -- python tools/prepare_environment.py --install
python3 tools/run_rocm10.py --gpu "$SETUP_GPU_ID" --run-dir runs/build-image-native -- .runtime/venv/bin/python tools/build_native.py
python3 tools/run_rocm10.py --gpu "$SETUP_GPU_ID" --run-dir runs/check-native-image-env -- .runtime/venv/bin/python tools/probe_native_runtime.py --attention-4k
```

The project venv references the foundation packages through a `.pth` file. Dependency preparation rejects a resolver plan that would replace protected Torch, Triton, TorchVision, or ROCm packages. Native extensions are rebuilt from pinned source inside the same foundation for `gfx1201`; build and JIT caches remain project-local.

The native probe checks imports, device copies, and a forced Flash Attention call at a 4K image-token shape. It is an environment check; full image generation is a separate validation.

## Download the original models

```bash
python3 tools/download_native_models.py --workers 8 --chunk-mib 64
```

This host-side downloader uses Python and `requests`. It downloads 40 selected files from two fixed revisions, totaling 122,332,347,043 bytes. It verifies sizes and upstream LFS SHA256 values and checks the structure of 25 safetensors files. Local download state and the verification report stay in `models/.download/native/`.

See [model sources and checksums](../models/README.md). Original-precision generation does not require the historical INT8 weights.

## Generate a 1K, 2K, or 4K image

Set `GPU_IDS` to four available physical GPU IDs on your machine, separated by commas. The launcher maps that order to logical devices, reserves all four cards, and samples whole-card VRAM and process-tree RAM every 0.5 seconds.

```bash
python3 tools/run_rocm10.py \
  --gpus "${GPU_IDS:?Set GPU_IDS to four comma-separated physical GPU IDs}" \
  --run-dir runs/native-4k-seed42 \
  --timeout 1800 \
  --monitor-interval 0.5 \
  --purpose native-4k-qualification \
  -- .runtime/venv/bin/python tools/run_native_image.py \
  --mode four-static \
  --config configs/native_t2i.json \
  --width 4096 --height 4096 --seed 42
```

Use `--width 1024 --height 1024` for 1K, or `--width 2048 --height 2048` for 2K. Add `--prompt "..."` to replace the configured prompt. Use a new run directory each time.

The configuration fixes six model evaluations, CFG=1, the released PDD sigma schedule, and original BF16/FP32 precision. Text encoding streams weights from the CPU in an isolated process. A hash-verified BF16 conditioning artifact feeds the main process, which holds complete DiT layers across four GPUs. The first DiT partition is released before FP32 tiled VAE decoding.

DiT sequences of at least 4096 tokens require PyTorch Flash. The measured 1K prompt falls below that threshold and uses ordinary PyTorch SDPA dispatch; the measured 2K and 4K requests require Flash. See [measured performance](../README.md#measured-performance) for the scope of the published numbers.

## Inspect timing and memory

A completed run retains its PNG, `result.json`, `monitor.jsonl`, `postflight.json`, conditioning, initial noise, and six step latents. Generate the compact report after the launcher exits:

```bash
python3 tools/summarize_native_run.py runs/native-4k-seed42
```

The resulting `summary.json` and `summary.md` report request and launcher duration, per-stage timing, sampled VRAM by physical card, PyTorch allocation peaks by logical card, process-tree RSS, summed process VmHWM, and whole-host RAM growth. The [benchmark methodology](benchmarks/R9700_NATIVE_RESOLUTIONS.md) defines each metric and its limits.

## Repeat an identical request

For A/A validation, keep the prompt, seed, dimensions, configuration, source, environment, and ordered physical GPUs identical:

```bash
python3 tools/run_rocm10.py \
  --gpus "${GPU_IDS:?Set GPU_IDS to four comma-separated physical GPU IDs}" \
  --run-dir runs/native-4k-seed42-aa \
  --timeout 1800 \
  --monitor-interval 0.5 \
  --purpose native-4k-aa \
  -- .runtime/venv/bin/python tools/run_native_image.py \
  --mode four-static \
  --config configs/native_t2i.json \
  --width 4096 --height 4096 --seed 42 \
  --reference-run runs/native-4k-seed42
```

The reference is used only for comparison; the new request recomputes text encoding and image generation. Inspect `reference_comparisons` in `result.json`: successful execution alone is not an assertion that all comparisons are equal.

This comparison work runs inside the measured request. For timing comparisons using the published benchmark definition, run both requests without `--reference-run`, then compare their retained artifacts offline. The [resolution summary tool](../tools/summarize_resolution_sweep.py) validates and exports matched pairs at all three resolutions.

## Historical INT8 baseline

The frozen [single-card INT8 release](releases/v0.1.0-image-int8-first-image.md) uses three different model files, listed in the historical INT8 section of [model sources](../models/README.md).

After those files are present, the legacy environment check and image entry point are:

```bash
python3 tools/run_rocm10.py --gpu "${SETUP_GPU_ID:?Set SETUP_GPU_ID to an available physical GPU ID}" --run-dir runs/check-int8-image-env -- .runtime/venv/bin/python tools/verify_environment.py
python3 tools/run_rocm10.py --gpu "$SETUP_GPU_ID" --run-dir runs/image-int8-2048-seed42 -- .runtime/venv/bin/python tools/run_image.py
```

The [INT8 API workflow](../configs/workflows/t2i_2048_api.json) uses 2048 × 2048, batch 1, seed 42, six steps, CFG=1, Euler/simple, and denoise=1. It disables prompt expansion, the Comfy compiler, and node-result caching; DynamicVRAM remains enabled. Its output is under the run's `output/` directory.

Both entry points retain diagnostic artifacts and stage synchronization. Neither is a persistent Web UI or a qualified concurrent-serving endpoint. Images, raw run data, weights, local plans, and research stay outside Git.
