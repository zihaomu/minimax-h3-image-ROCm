# MiniMax H3-Image ROCm

面向 AMD Radeon AI PRO R9700 的 MiniMax-H3-Image 推理项目。以 ComfyUI 和 INT8 ConvRot 为基础，逐步实现自己的计算后端和 Image 多卡执行，比较四卡协同与两组双卡的延迟和吞吐。

**当前状态：已确定模型与环境目标，推理实现和 GPU 验证尚未完成。**

- [环境目标](configs/runtime-target.json)
- [模型来源与固定版本](models/README.md)

计划、阶段状态和来源审计保存在本地 `vibe/`、`research/`，不上传 GitHub。

环境使用项目独立的 ComfyUI checkout 与依赖，基础目标为 ROCm 10.1。参考 [MiniMax H3 视频项目](https://github.com/zihaomu/minimax-h3-ROCm) 的量化分片、R9700 Attention 和双卡执行；Image 前向与四卡扩展按本项目计划重新验证。

源码、配置和专门整理的公开文档同步至本仓库。`vibe/`、`research/`、模型权重、运行环境、缓存和原始实验产物存放在本地。`models/` 仅提交记录来源链接、版本及校验值的 `README.md`，不提交权重文件。官方模型与第三方源码遵循各自许可；实际代码移植时记录来源并保留相应许可证及 NOTICE。
