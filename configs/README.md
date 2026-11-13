# Configuration reference

[Project overview](../README.md) · [Setup guide](../docs/QUICKSTART.md)

## Original-precision workloads

| File | Role |
|---|---|
| [`native_t2i.json`](native_t2i.json) | Product-photography prompt, seed 42, official six-step sigma schedule, CFG=1, process isolation, and original BF16/FP32 policy |
| [`o5_qualification_matrix.json`](o5_qualification_matrix.json) | Three prompt categories and two seeds used for the 4K functional qualification |

The default size in `native_t2i.json` is 2048 × 2048. The CLI overrides both dimensions for 1024 × 1024 or 4096 × 4096; it preserves the declared precision and six-step schedule. The matrix file records a qualification workload and its gates; the single-request entry point does not automatically execute the matrix.

## Environment identity

| File | Role |
|---|---|
| [`runtime-target.json`](runtime-target.json) | Foundation and application identities used during preparation and native extension builds |
| [`runtime-lock.json`](runtime-lock.json) | Verified foundation/framework pairing, application commits, extension hashes, and historical INT8 model identities |
| [`requirements-image.lock`](requirements-image.lock) | Pinned application and build dependencies layered onto the foundation |

Original model revisions and checksums are in [`models/README.md`](../models/README.md) and the fixed [downloader](../tools/download_native_models.py). Their local verification metadata is ignored.

## Historical INT8 inputs

| File | Role |
|---|---|
| [`workflows/t2i_2048_api.json`](workflows/t2i_2048_api.json) | Executable ComfyUI API graph for the single-card INT8 baseline |
| [`workflows/t2i_2048_source.json`](workflows/t2i_2048_source.json) | Saved source workflow associated with that graph |
| [`releases/v0.1.0-image-int8-first-image.json`](releases/v0.1.0-image-int8-first-image.json) | Manifest for the frozen first-image release |

Historical inputs remain associated with their original precision and validation scope. The [current benchmark](../docs/benchmarks/R9700_NATIVE_RESOLUTIONS.md) records hashes of the configuration and runtime files used for its measurements.
