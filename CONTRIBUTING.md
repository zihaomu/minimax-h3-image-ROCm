# Development guide

This repository contains the original-precision MiniMax-H3-Image backend for the pinned R9700 workspace. Start with the [README](README.md), [setup guide](docs/QUICKSTART.md), and [tool reference](tools/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/minimax_h3_image_rocm/` | Runtime modules: weight loading, encoder, DiT placement, VAE, pipeline, and measurements |
| `tools/` | Maintained setup, inference, and reporting entry points |
| `configs/` | Pinned environment identities, workloads, and historical release inputs |
| `docs/` | Public guides, source identities, prepared benchmarks, and release reports |
| `tests/` | CPU-only regression fixtures |
| `models/README.md` | Public model links, revisions, sizes, and checksums |

Local `vibe/`, `research/`, `runs/`, `.runtime/`, and model files are ignored. The shared workspace foundation and sibling video project are external dependencies; their environments and historical artifacts remain separate.

## CPU-only checks

From the project root:

```bash
python3 -m compileall -q src tools tests
python3 -m unittest discover -s tests -v
git diff --check
```

These checks require no GPU inference. The existing unit tests cover GPU-user ownership classification using fixtures; passing them does not qualify model numerics or performance.

For documentation-only changes, check relative links and preserve the recorded source/model hashes and benchmark values. Runtime entry-point paths are kept stable because retained run commands and source manifests refer to them.

## Runtime changes

Use the shared launcher and the exact foundation Torch/Triton pairing. Application dependencies and rebuilt extensions belong in the project environment. GPU work requires occupancy, shared-reservation, ownership, and physical-to-logical mapping checks; physical GPU3 is excluded.

Freeze the prompt, seed, dimensions, six-step schedule, CFG, precision, and device order before comparing implementations. Keep numerical, attention, placement, and transport changes independently attributable. An unavailable required backend must fail explicitly.

Retain the baseline, unsuccessful attempts, and new evidence under `runs/`. Validate output semantics, finite values, intermediate tensors, same-input reproducibility, resource usage, and cleanup within the affected workload. A previously measured benchmark describes its recorded source snapshot; changed runtime code needs fresh qualification before inheriting those numbers.

## Publish prepared evidence

Review a specific list of source, configuration, and documentation files before staging. Keep model weights and partial downloads, raw images/tensors/traces, environments, credentials, and local plans/research outside commits and their ancestor history. Under `models/`, only `README.md` is public.

Public benchmark summaries should identify the model revision, runtime/source snapshot, workload, sample count, timing boundaries, and memory measurement method. Mark interrupted or contaminated attempts and explain exclusions. The [current report](docs/benchmarks/R9700_NATIVE_RESOLUTIONS.md) and [summary tool](tools/summarize_resolution_sweep.py) provide the existing format.

Inspect `git diff --cached --stat` and `git diff --cached --check` before committing. Use normal commits and pushes; keep the frozen INT8 tag and its historical reports intact.
