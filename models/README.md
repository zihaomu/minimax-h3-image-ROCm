# 模型来源与固定版本

权重下载到本目录对应子目录；Git 只跟踪本说明文件，权重、partial、下载缓存及本地校验报告均不进入仓库。当前四卡原始精度 4K 路线使用第一组模型；下方 INT8 ConvRot 组合仅保留为历史基线。

## 当前：官方原始 BF16/FP32 组件

| 组件 | 固定来源 | 本地目录 | 文件数 / bytes |
|---|---|---|---:|
| DiT、PDD、scheduler、tokenizer、FP32 VAE | [MiniMaxAI/MiniMax-H3-Image@5f904730](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/5f90473023975e6271b0ed1dd84c4e88e4bbc6d3) | `native/minimax-h3-image/` | 26 / 68,639,995,438 |
| Image 实际消费的文本编码器前 50 层 | [MiniMaxAI/MiniMax-H3@0919fb68/text_encoder](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317/text_encoder) | `native/minimax-h3/` | 14 / 53,692,351,605 |

合计 40 个文件、122,332,347,043 bytes。下载器只接受上述两个固定 revision，支持分段续传，下载后逐文件核对上游 LFS SHA256，并对 safetensors header/data offset 做结构检查：

```bash
python3 tools/download_native_models.py --workers 8 --chunk-mib 64
```

完整 40 文件的本地验证报告写到被忽略的 `models/.download/native/verification.json`。以下固定的是实际权重文件 SHA256；配置、索引、tokenizer 和 LICENSE 同样在本地报告中逐文件核验。

```text
c07093a098e3bef4d43cd29c33e239a0c45424c5ff40dfcc864df00b8d664738  native/minimax-h3-image/pdd_fused_heads.safetensors
200667b632a6004bb8442008c4a8e975d64a7dbbc3d5569ed482650c3d61ff7b  native/minimax-h3-image/transformer/model-00001-of-00012.safetensors
2227018562b2d92db391bce2fe641ea7f887585b581d6a73092a8fcfc59ef348  native/minimax-h3-image/transformer/model-00002-of-00012.safetensors
165469f7dccbb9fb6cbdc7a88e89d2a0220697af4da7ee40a96b369ccfa5abc9  native/minimax-h3-image/transformer/model-00003-of-00012.safetensors
2bf7dff80d49203cee29834ce6bd86077b09c4b73174d1c4f696f8fecab83ed3  native/minimax-h3-image/transformer/model-00004-of-00012.safetensors
dcc7c79d46fa1ca910396fa001657985057db491af1bfadeda4cf3a82da329f2  native/minimax-h3-image/transformer/model-00005-of-00012.safetensors
12c76e39604f04d11d282172719bad6ad374e98479b8892f4ea0c9d2695b3086  native/minimax-h3-image/transformer/model-00006-of-00012.safetensors
59ed575a374c0304da09beb234225c17b5ecc1e491b76f72455f0b3b7a1fbbff  native/minimax-h3-image/transformer/model-00007-of-00012.safetensors
b71af5f2f42fee2530b4154caba6ce595ce98072d831090a6b63a546420c0072  native/minimax-h3-image/transformer/model-00008-of-00012.safetensors
a2764f44f306f6eecfd3d52567bcebe64ddfafb0a2fbdaf089efc1e74cbe82f8  native/minimax-h3-image/transformer/model-00009-of-00012.safetensors
0e82a47b5fd0498674718c7a0b3830d042d349f6b5fe16cd2b35cccca5063365  native/minimax-h3-image/transformer/model-00010-of-00012.safetensors
d728554ce210b10062e70db51080d2f6d6e41c6d0fa033dc77b021db20745d2b  native/minimax-h3-image/transformer/model-00011-of-00012.safetensors
8d9e1c67066a5fa646f2e87f0868f73cb50be66cc1739e51eb12791fd9123b83  native/minimax-h3-image/transformer/model-00012-of-00012.safetensors
a7c8bb1e181865e7cf87c29d3d6bc728d944ce4862b92d0e1cbdcc36b3b180b2  native/minimax-h3-image/vae/diffusion_pytorch_model.safetensors
6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb  native/minimax-h3/text_encoder/model-00001-of-00014.safetensors
d8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa  native/minimax-h3/text_encoder/model-00002-of-00014.safetensors
54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940  native/minimax-h3/text_encoder/model-00003-of-00014.safetensors
ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f  native/minimax-h3/text_encoder/model-00004-of-00014.safetensors
fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2  native/minimax-h3/text_encoder/model-00005-of-00014.safetensors
82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974  native/minimax-h3/text_encoder/model-00006-of-00014.safetensors
fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934  native/minimax-h3/text_encoder/model-00007-of-00014.safetensors
431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f  native/minimax-h3/text_encoder/model-00008-of-00014.safetensors
3825e3f4302f4d2f7d76aa7430d2ce0864fde6b9e540a5806bf0d8e38e4d9f47  native/minimax-h3/text_encoder/model-00009-of-00014.safetensors
aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f  native/minimax-h3/text_encoder/model-00010-of-00014.safetensors
3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f  native/minimax-h3/text_encoder/model-00011-of-00014.safetensors
```

只下载到第 11 个文本编码器 shard 是有意设计：Image 路径读取第 50 层 hidden state，上述固定集合已覆盖 embedding 与实际消费的 layers 0–49；没有把未消费的后续层冒充运行依赖。

## 历史：ComfyUI INT8 ConvRot 组合

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
