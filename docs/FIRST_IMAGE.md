# R9700 single-GPU first-image validation

This is an English translation of the historical INT8 first-image report. Its results and validation scope describe that request. The frozen release manifest's report SHA256 identifies the [original file at the release tag](https://github.com/zihaomu/minimax-h3-image-ROCm/blob/v0.1.0-image-int8-first-image/docs/FIRST_IMAGE.md), rather than this translated copy on `main`.

On 2026-11-13, the independent ComfyUI environment completed one 2048×2048 text-to-image request with six steps and CFG=1. The native ComfyUI executor used pinned INT8 ConvRot DiT/encoder weights and the FP16 Image VAE. Custom Attention, multi-GPU code, and vLLM-Omni were not part of this request.

Environment: ComfyUI v0.45.0 at its pinned commit, ROCm 10.1, Torch `2.14.0+rocm10.1.0`, and Triton `3.8.0+git669b31ac.rocm10.1.0`. comfy-kitchen 0.2.41 and comfy-aimdo 0.6.0 were rebuilt from pinned sources. The request used a Radeon AI PRO R9700, gfx1201, recorded as physical GPU1, with native DynamicVRAM and asynchronous weight offload enabled. See the [runtime lock](../configs/runtime-lock.json) for exact identities.

## Results

Local run ID: `p2-first-image-2048-r2`. The request used seed 42, batch 1, euler/simple, and denoise=1, with a detailed prompt describing a fox plush toy, a blue cup, and yellow flowers. The output dimensions were correct, retained floating-point tensors were finite, and six sampling callbacks executed. Visual inspection found the main objects and materials consistent with the prompt, with no obvious blank or corrupted output. This inspection does not establish general image quality or numerical accuracy.

| Metric | Observed value |
|---|---:|
| Full workflow request | 17.642 s |
| Text conditioning, including stage loading | 5.274 s |
| KSampler, including loading and diagnostics | 9.294 s |
| VAE decode, including stage loading | 1.673 s |
| PNG save node | 0.497 s |
| Sampled whole-card VRAM peak | 31.618 GiB |
| Peak process RSS | 2.857 GiB |

Request timing starts at the executor call and ends after image saving and GPU synchronization; container and application startup are excluded. The request includes tensor capture, synchronization at node boundaries, dispatch counting, and CPU profiling of the first SDPA call in each relevant node. It is an instrumented first-image functional check. An earlier attempt was interrupted during encoding, and the filesystem cache was not cleared. Stage times omit some diagnostics between nodes and cannot simply be added to obtain end-to-end time.

VRAM was sampled from sysfs approximately every two seconds. The observed peak was 33,949,990,912 bytes; shorter transients may have been missed. Torch allocator statistics do not cover all DynamicVRAM weight allocations: the observed Torch allocated peak was approximately 0.601 GiB while whole-card VRAM was close to capacity. These figures do not establish feasibility for longer prompts, editing, or 4K.

The request recorded 1,200 `int8_linear` calls through comfy-kitchen HIP dispatch and 350 INT8 ConvRot weight-dequantization calls. INT8 describes the weight format and part of the computation path; some layers and Attention use floating-point computation. The sampled first SDPA call used efficient-attention in text encoding and flash-attention in sampling and VAE nodes. Only the first call in each relevant node was recorded, so this does not establish kernel coverage for all calls or shapes.

PNG SHA256: `83b075e1aa8deca8bc8e8fed4f08d3b0bc078fde91617370282ed12d5e9d9c3f`.

## Validation scope at the time of the report

P1 environment qualification passed, and the first complete P2 image passed. A/A reproducibility, at least three prompt categories × two seeds, unprofiled warm-request statistics, editing, and higher-precision controls had not been completed for this baseline. The custom compute backend, dual-GPU execution, and four-GPU execution had not yet been validated.

After the request, the container exited, the reservation was released, and no owned KFD processes remained on the selected GPU. Evidence from the initial resource-monitoring interruption was retained. The monitor now identifies new owned processes by container cgroup; CPU regression fixtures cover ownership classification and rejection of external GPU use.

See [QUICKSTART](QUICKSTART.md) for current reproduction guidance and the [frozen release report](releases/v0.1.0-image-int8-first-image.md) for the historical checkout. Raw outputs, tensors, and logs remain in local `runs/` directories.
