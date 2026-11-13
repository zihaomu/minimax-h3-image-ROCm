# R9700 original-precision resolution benchmark

Measured on **November 13, 2026**. This report compares 1024 × 1024, 2048 × 2048, and 4096 × 4096 text-to-image requests using the same four AMD Radeon AI PRO R9700 GPUs and the same implementation.

| Resolution | Full request (s) | Six-step denoise (s) | Peak VRAM / GPU (GiB) | Peak CPU RAM, RSS (GiB) | Host RAM growth (GiB) |
|---|---:|---:|---:|---:|---:|
| 1K · 1024 × 1024 | **68.61** | 7.30 | 16.38 | 6.56 | 6.43 |
| 2K · 2048 × 2048 | **80.58** | 15.23 | 16.76 | 6.47 | 5.96 |
| 4K · 4096 × 4096 | **144.12** | 67.34 | 18.34 | 6.65 | 6.04 |

Each resolution has **two completed independent requests**. Timings are arithmetic means. Memory values are the maximum observed across those samples, rather than averages of the peaks. These measurements characterize an instrumented single-image pipeline; they do not establish persistent-server throughput or a minimum GPU count.

## Fixed workload and environment

- **Hardware:** four R9700s, RDNA4 / `gfx1201`, ordered physical IDs `1,5,6,7`. Physical GPU3 was excluded. The host reported 503.63 GiB of total RAM.
- **Runtime:** locked ROCm 10.1 foundation; Torch `2.14.0+rocm10.1.0`, Triton `3.8.0+git669b31ac.rocm10.1.0`, Python `3.13.13`; ComfyUI `v0.45.0` at `989d93d0d968e704c589ce0146bb18c77f6b50cc`. See the [environment lock](../../configs/runtime-lock.json).
- **Image model:** [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/5f90473023975e6271b0ed1dd84c4e88e4bbc6d3), revision `5f90473023975e6271b0ed1dd84c4e88e4bbc6d3`.
- **Text encoder:** [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317/text_encoder), revision `0919fb684102ef115126612f3fbb84d1027e7317`; Image consumes layers 0–49.
- **Precision:** original BF16 encoder and DiT weights, released FP32 modules and PDD heads, FP32 Euler updates, and FP32 VAE. Weight quantization is disabled.
- **Request:** the product-photography prompt in [`native_t2i.json`](../../configs/native_t2i.json), seed 42, batch 1, six model evaluations, CFG=1, the released sigma schedule, and no extra guidance branch.
- **Placement:** `four-static`; an isolated text-encoder process streams CPU weights to the first GPU. DiT blocks 0–10 / 11–23 / 24–36 / 37–49 stay on the four GPUs during denoising. The first partition is released before tiled VAE decoding.
- **Attention policy:** sequences of at least 4096 tokens require PyTorch Flash. The 2K and 4K DiT requests each record 300 strict Flash calls. The 1K request falls below the threshold and uses ordinary PyTorch SDPA dispatch; its selected kernel was not independently profiled.
- **Process and cache policy:** each sample starts a new container/process and recomputes conditioning. OS file caches and compiled-kernel caches may already be warm. No disk-cache flush or cache-cold storage benchmark was performed.
- **Monitoring:** 0.5-second whole-card VRAM and process-tree RAM samples, with ownership checks throughout the request and after exit.

## Timing ranges

| Resolution | Request mean (range), s | Launcher mean (range), s | Denoise mean (range), s |
|---|---:|---:|---:|
| 1K · 1024 × 1024 | 68.61 (68.60–68.63) | 77.16 (76.63–77.69) | 7.30 (7.24–7.35) |
| 2K · 2048 × 2048 | 80.58 (80.32–80.84) | 88.74 (88.43–89.04) | 15.23 (15.12–15.34) |
| 4K · 4096 × 4096 | 144.12 (144.00–144.24) | 152.37 (152.13–152.60) | 67.34 (67.31–67.37) |

**Request time** covers isolated text encoding, weight loading, all six denoising steps, tensor-artifact capture, VAE loading/decoding, PNG saving, and in-process cleanup. It excludes container/process bootstrap and foundation validation. **Launcher time** includes that startup and the launcher's sampling/polling overhead. The pipeline synchronizes phases and retains intermediate tensors for validation.

Two samples give an observed range, not a confidence interval or a latency percentile. No resolution was measured with fewer steps or reduced weight precision.

## Stage breakdown

Arithmetic mean seconds across each pair:

| Resolution | Text encoding | DiT loading | Six-step denoise | VAE loading | VAE decoding | PNG saving |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 29.34 | 28.27 | 7.30 | 1.04 | 1.68 | 0.38 |
| 2K | 29.76 | 28.62 | 15.23 | 1.04 | 3.43 | 1.88 |
| 4K | 30.14 | 28.68 | 67.34 | 1.03 | 9.89 | 6.36 |

Phase totals omit small boundary/artifact operations and cleanup, so they do not sum exactly to request time. Text encoding and DiT loading account for most of the 1K request. The six denoising steps alone should not be presented as end-to-end image generation time.

## Memory measurements

All values below are **GiB (2³⁰ bytes)**, taking the maximum over the two samples.

| Resolution | GPU1 VRAM | GPU5 VRAM | GPU6 VRAM | GPU7 VRAM | Process-tree RSS | Sum of process VmHWM | Host RAM growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | 15.46 | 16.29 | 16.28 | 16.38 | 6.56 | 6.67 | 6.43 |
| 2K | 15.94 | 16.68 | 16.68 | 16.76 | 6.47 | 6.68 | 5.96 |
| 4K | 17.75 | 18.26 | 18.26 | 18.34 | 6.65 | 6.68 | 6.04 |

- **VRAM:** total usage sampled on each physical card, including the card's small baseline allocation. The README reports the largest of the four per-card peaks. Independent card peaks need not occur simultaneously.
- **Process-tree RSS:** the largest sampled sum of resident memory for the owned container's processes, including the temporary encoder process when it exists. Shared pages may be counted more than once.
- **Sum of process VmHWM:** the highest observed sum of the currently present processes' historical resident-memory high-water marks. This is a separate metric from simultaneous RSS, not an exact process-tree high-water mark.
- **Host RAM growth:** the peak increase in `MemTotal - MemAvailable` relative to the launch baseline. It includes page-cache effects and unrelated host activity and is not attributable solely to this request.
- Sampling can miss short VRAM/RSS spikes. These RAM figures were measured on a large-memory host with potentially warm caches and do **not** establish that a small-RAM machine can run the workload. The fixed model download is 122.33 GB on disk.

## Completed samples and exclusions

| Local run ID | Request (s) | Launcher (s) | Six-step denoise (s) |
|---|---:|---:|---:|
| `readme-native-1024-seed42-r1` | 68.63 | 76.63 | 7.24 |
| `readme-native-1024-seed42-r2` | 68.60 | 77.69 | 7.35 |
| `readme-native-2048-seed42-r1` | 80.84 | 89.04 | 15.34 |
| `readme-native-2048-seed42-r2` | 80.32 | 88.43 | 15.12 |
| `readme-native-4096-seed42-r1` | 144.24 | 152.60 | 67.31 |
| `readme-native-4096-seed42-r3` | 144.00 | 152.13 | 67.37 |

The additional attempt `readme-native-4096-seed42-r2` was stopped during its first denoising step after an external process began using a selected GPU. It did not produce a completed request, and its mixed VRAM measurements are excluded from every table. The failed evidence remains local. The replacement `r3` started from scratch on the original device order after the external process had exited.

## Reproducibility and qualification

For each resolution, the two successful requests produced byte-identical conditioning, initial noise, all six step-latent files, and final PNGs. Comparisons ran after inference and therefore do not inflate the request timings.

| Resolution | PNG SHA256, identical for both samples |
|---|---|
| 1K | `339ef3032765cce98661070a6d44b42fcfb370ec37c0ea5ec0aace02b7b0c457` |
| 2K | `6f62fac137f4a5b331deb9ae524a70b3d1aeb97135cecbb93a3695d2b4824ffd` |
| 4K | `5dbf521490a347facf6e2e71ddac12bdd040c79510fba11dc06090cf5ca5308c` |

The 1K and 2K images passed manual inspection for the requested fox toy, cup, flowers, table, and green backdrop. The 4K PNG is byte-identical to the previously inspected original-precision qualification output. This resolution sweep uses one prompt and one seed; the earlier 4K functional matrix separately covered three prompts and two seeds.

All six successful samples passed completion, official workload, original-precision policy, finite-output, device ownership, resource-release, source-identity, and artifact-hash checks. Same-implementation A/A equality does not imply pixel equivalence to another engine. Concurrency and all-GPU-resident serving remain unqualified.

## Evidence and regeneration

The [machine-readable summary](r9700-native-resolutions.json) contains unrounded measurements, per-sample stage and memory data, output/tensor hashes, raw-evidence hashes, model revisions, environment identity, and hashes of the 12 runtime source files. The recorded Git base identifies the checkout before these native source files were committed; the per-file SHA256 values identify the actual code that ran.

The CPU-only [summary tool](../../tools/summarize_resolution_sweep.py) verifies the completed local evidence, identical workload/source/runtime/device order, artifact hashes, and exact A/A results before exporting. It rejects the incomplete attempt.

With the local evidence present, regenerate the summary from the project root:

```bash
python3 tools/summarize_resolution_sweep.py \
  runs/readme-native-1024-seed42-r1 \
  runs/readme-native-1024-seed42-r2 \
  runs/readme-native-2048-seed42-r1 \
  runs/readme-native-2048-seed42-r2 \
  runs/readme-native-4096-seed42-r1 \
  runs/readme-native-4096-seed42-r3
```

To collect new measurements, follow the [README run command](../../README.md#quick-start), choose four available permitted GPUs, and keep their order fixed for every resolution and repeat. Use a new run directory for every request and generate each run's `summary.json` with `tools/summarize_native_run.py`. Compare paired runs offline to retain the timing definition above.

Only this prepared report and sanitized summary are public. Model weights, generated images, raw tensors, monitors, host paths, and local planning/research files stay outside Git.
