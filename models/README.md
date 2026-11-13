# 模型来源与固定版本

本项目选用 MiniMax-H3-Image 的 ComfyUI INT8 ConvRot 组合。权重下载到本目录对应子目录；Git 只跟踪本说明文件，权重、下载缓存及校验报告均留在本地。

- 原始模型：[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
- 权重仓库：[Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)
- 固定 revision：`0bb149f51e3156ed4b90dd45d5620346bfe1df8f`

| 组件 | 本地相对路径（相对于 models/） | 大小 GB / GiB | 固定下载链接 |
|---|---|---:|---|
| 生成主模型（INT8 ConvRot） | `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` | 20.98 / 19.54 | [下载](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors) |
| 文本／参考图编码器（INT8 ConvRot） | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.14 / 25.28 | [下载](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) |
| 图片 VAE（FP16） | `vae/minimax_h3_image_vae_fp16.safetensors` | 1.17 / 1.09 | [下载](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/vae/minimax_h3_image_vae_fp16.safetensors) |

合计 **49,289,162,266 bytes = 49.29 GB = 45.90 GiB**。大小以固定 revision 的文件元数据为准；这是磁盘文件大小，不是运行时显存。

## SHA256

```text
83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622  diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors
bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af  vae/minimax_h3_image_vae_fp16.safetensors
```

## 使用范围

- 第一阶段使用以上三个组件，直接输入固定 prompt；可选 prompt enhancer 不属于此次下载。
- 编码器名称必须是 `int8_convrot`，图片 VAE 必须是 `minimax_h3_image_vae_fp16`。
- 下载文件的尺寸、SHA256 和 safetensors 结构校验，不等于已完成 R9700 推理验证。
