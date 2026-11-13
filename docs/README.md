# Documentation

Start with the [project overview and performance table](../README.md), then follow the [environment and generation guide](QUICKSTART.md).

## Current original-precision pipeline

| Document | Contents |
|---|---|
| [Environment setup and image generation](QUICKSTART.md) | Pinned dependencies, model download, 1K/2K/4K generation, resource reports, and A/A comparison |
| [R9700 resolution benchmark](benchmarks/R9700_NATIVE_RESOLUTIONS.md) | Same-workload latency, per-card VRAM, CPU RAM, sample counts, and measurement definitions |
| [Benchmark JSON](benchmarks/r9700-native-resolutions.json) | Unrounded measurements, source identities, and output/evidence hashes |
| [Model sources](../models/README.md) | Original BF16/FP32 components, fixed revisions, sizes, and checksums |
| [Source and license identities](NATIVE_SOURCES.md) | Pinned ComfyUI modules, Diffusers reference files, and model license identities |
| [Tool reference](../tools/README.md) | Which entry point to use, where it runs, and what it produces |
| [Configuration reference](../configs/README.md) | Workloads, environment locks, and historical configurations |
| [Development guide](../CONTRIBUTING.md) | Local checks, numerical validation, and publication boundaries |

## Historical INT8 baseline

| Document | Scope |
|---|---|
| [First-image report](FIRST_IMAGE.md) | Original single-card INT8 ConvRot 2048 × 2048 result |
| [Frozen release report](releases/v0.1.0-image-int8-first-image.md) | Evidence associated with tag `v0.1.0-image-int8-first-image` |
| [Release manifest](../configs/releases/v0.1.0-image-int8-first-image.json) | Machine-readable identity of the frozen INT8 snapshot |

Historical reports retain their original language and results. Their precision and workload differ from the current four-card original-precision benchmark.

## Local development records

The development workspace also contains ignored `vibe/`, `research/`, `runs/`, and `.runtime/` directories. These hold plans, diagnostic sources, raw measurements, and application dependencies; they are not included in a public checkout. Public support and performance claims are documented in the reports above.
