# Original-precision backend sources and audit identities

This document records the pinned upstream sources used to implement and audit the four-GPU original-precision Image backend. The runtime imports modules from the pinned ComfyUI checkout. Diffusers files serve as references for the released Image parameter semantics, FP32 timestep/AdaLN ordering, and six-step scheduler. File SHA256 values come from the O0 mapping audit; listing a file here does not mean its source was copied into this repository.

## ComfyUI runtime modules

- Repository: [`Comfy-Org/ComfyUI`](https://github.com/Comfy-Org/ComfyUI)
- Commit: `989d93d0d968e704c589ce0146bb18c77f6b50cc`
- Upstream license: GPL-3.0. Refer to LICENSE in the pinned checkout.

| Pinned file | SHA256 |
|---|---|
| [`comfy/ldm/minimax/model.py`](https://github.com/Comfy-Org/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy/ldm/minimax/model.py) | `68900fb081fa87bec62a7eb26631f9e154db2e2ac55073c4944dd289cf1317f8` |
| [`comfy/ldm/minimax/vae.py`](https://github.com/Comfy-Org/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy/ldm/minimax/vae.py) | `fd32db0e1fd277fac561c6232323afc53957ed6eb48cc50eca18730a0a6bb0d1` |
| [`comfy/text_encoders/llama.py`](https://github.com/Comfy-Org/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy/text_encoders/llama.py) | `6ac9fd4b9878261c0a225c9d0a2607df3d282a07fe994220f40c97fa50fe51ab` |
| [`comfy/ldm/modules/attention.py`](https://github.com/Comfy-Org/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy/ldm/modules/attention.py) | `2e688c4c96ba10915c7ca847ea692faea3be4b9c2605d74d785b7abb23ee2502` |

## Diffusers numerical references

- Repository: [`huggingface/diffusers`](https://github.com/huggingface/diffusers)
- Commit: `2c58e66a08a61ea5c58b9add560cf808e2c851de`
- Upstream license: Apache-2.0. Refer to LICENSE in the pinned checkout.

| Pinned file | SHA256 |
|---|---|
| [`transformer_minimax_h3_image.py`](https://github.com/huggingface/diffusers/blob/2c58e66a08a61ea5c58b9add560cf808e2c851de/src/diffusers/models/transformers/transformer_minimax_h3_image.py) | `0928960f80447c5e4db43d61a361ccc4c785021207325aabe65416f8f6f94e62` |
| [`scheduling_minimax_h3.py`](https://github.com/huggingface/diffusers/blob/2c58e66a08a61ea5c58b9add560cf808e2c851de/src/diffusers/schedulers/scheduling_minimax_h3.py) | `ee59acb005d7b6e38c520b9c8846f41c9e4fc7fbdbc0128ee5880002a2b2faf4` |
| [`encoders.py`](https://github.com/huggingface/diffusers/blob/2c58e66a08a61ea5c58b9add560cf808e2c851de/src/diffusers/modular_pipelines/minimax_h3_image/encoders.py) | `e4ddd0e51d9be072575f5efbb607d3d3238cb40b1274b75c487b294d0f80fe9c` |
| [`before_denoise.py`](https://github.com/huggingface/diffusers/blob/2c58e66a08a61ea5c58b9add560cf808e2c851de/src/diffusers/modular_pipelines/minimax_h3_image/before_denoise.py) | `0f30d146dec35a7cdb5e53696d666188f46070eae9e68c2c5a6f7b6a820219c4` |
| [`denoise.py`](https://github.com/huggingface/diffusers/blob/2c58e66a08a61ea5c58b9add560cf808e2c851de/src/diffusers/modular_pipelines/minimax_h3_image/denoise.py) | `f150624c4adecc430f17d205b8db415d304ba1519e50294c5fec9224454288bb` |
| [`decoders.py`](https://github.com/huggingface/diffusers/blob/2c58e66a08a61ea5c58b9add560cf808e2c851de/src/diffusers/modular_pipelines/minimax_h3_image/decoders.py) | `01612ce0ab9785812da58b9dde3bb26bec57c43c1e4a07bec9d32d1fe064ec46` |

## Model license identities

| Model | Pinned revision / LICENSE | SHA256 of the local LICENSE copy |
|---|---|---|
| MiniMax-H3-Image | [`5f90473023975e6271b0ed1dd84c4e88e4bbc6d3`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/5f90473023975e6271b0ed1dd84c4e88e4bbc6d3/LICENSE), MiniMax-H3-Image Community License (file date: 2026-11-10) | `0ec3bb5e6d9e885fa2aaf15e84689e50fd51f42b53fa24d3a64ed9d8c76d423b` |
| MiniMax-H3 text encoder | [`0919fb684102ef115126612f3fbb84d1027e7317`](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/0919fb684102ef115126612f3fbb84d1027e7317/LICENSE), MiniMax H3 Community License (file date: 2026-08-02) | `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44` |

See [`models/README.md`](../models/README.md) for weight sources, pinned download links, and weight SHA256 values. Model weights and local license copies remain ignored under the repository's publication rules. Users must read and comply with the licenses at the corresponding pinned revisions.

The sibling `minimax_h3` project served only as a read-only engineering reference. Its protected research, archive, and historical experiment trees were not copied into the public repository.
