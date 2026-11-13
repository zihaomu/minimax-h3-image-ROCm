# R9700 单卡首图验证

2026-11-13，独立 ComfyUI 环境完成一个 2048×2048、6 步、CFG 1 的完整文生图请求。原生 ComfyUI 执行器使用固定 INT8 ConvRot DiT/编码器和 FP16 Image VAE，未接入自有 Attention、多卡代码或 vLLM-Omni。

环境：ComfyUI v0.45.0 固定提交、ROCm 10.1、Torch `2.14.0+rocm10.1.0`、Triton `3.8.0+git669b31ac.rocm10.1.0`。comfy-kitchen 0.2.41 和 comfy-aimdo 0.6.0 从固定源码重新编译。使用物理 GPU1，Radeon AI PRO R9700，gfx1201；开启原生 DynamicVRAM 与异步权重卸载。精确身份见 [运行时锁](../configs/runtime-lock.json)。

## 结果

本地 run ID：`p2-first-image-2048-r2`。seed 42，batch 1，euler/simple，denoise 1；详细 prompt 描述狐狸毛绒玩具、蓝色杯子和黄色花朵。图片尺寸正确，保留的浮点张量均为有限值，执行了六个采样回调；目视检查显示主要物体和材质符合提示，无明显空白或损坏。此检查不证明通用图像质量或数值精度。

| 指标 | 本次观测 |
|---|---:|
| 完整工作流请求 | 17.642 秒 |
| 文本 conditioning，包含阶段加载 | 5.274 秒 |
| KSampler，包含加载和诊断记录 | 9.294 秒 |
| VAE decode，包含阶段加载 | 1.673 秒 |
| PNG 保存节点 | 0.497 秒 |
| 整卡 VRAM 采样峰值 | 31.618 GiB |
| 进程主机内存峰值 RSS | 2.857 GiB |

请求时间从执行器调用开始，到图片保存完成并同步 GPU；不含容器和应用启动。该请求带张量捕获、节点边界同步、dispatch 计数，以及每个相关节点首个 SDPA 调用的 CPU profiler 记录，属于首图功能检查，不是热请求性能基线。此前有一个编码阶段中断的尝试，文件系统缓存状态未清空。阶段时间不包含全部节点间诊断开销，不能简单相加替代端到端时间。

VRAM 从 sysfs 约每两秒采样，观测峰值为 33,949,990,912 bytes，并非连续测得的真实峰值。DynamicVRAM 的权重分配不能仅靠 Torch allocator 统计衡量：本次 Torch allocated 峰值约 0.601 GiB，但整卡显存已接近可用容量。不可据此推断长 prompt、编辑或 4K 同样可行。

记录到 1,200 次 `int8_linear` 经 comfy-kitchen HIP dispatch，另有 350 次 INT8 ConvRot 权重反量化调用。INT8 是权重格式及部分计算路径，不代表所有层和 Attention 都执行 INT8。抽查到文本编码首个 SDPA 调用使用 efficient-attention，采样和 VAE 节点首个 SDPA 调用使用 flash-attention；记录只覆盖各节点首次调用，不能外推为所有形状的完整内核覆盖。

PNG SHA256：`83b075e1aa8deca8bc8e8fed4f08d3b0bc078fde91617370282ed12d5e9d9c3f`。

## 验证范围

P1 已通过；P2 的首张完整图片已通过。A/A 重复性、至少三类 prompt × 两个 seed、无 profiler 热请求统计、编辑和高精度控制尚未完成。自有计算后端、双卡和四卡尚未验证。

任务结束后容器退出、预约释放，GPU1 无本实验残留 KFD 进程。首轮资源监控中断的记录保留；监控现按容器 cgroup 识别新进程，CPU 回归检查覆盖归属识别与外部进程阻止。

复现入口见 [QUICKSTART](QUICKSTART.md)。原始输出、张量与日志仅保存在本地 `runs/`。
