# 独立环境与首图复现

当前工具针对已配置共享 ROCm 10.1 foundation 的 R9700 工作区。需要父目录的 `rocm10-launch.sh`、锁定镜像、Docker、`rocm-smi`，以及现有共享预约目录和 `minimax_h3/.standard-opt-state/locks/`。镜像 ID 见 [运行时锁](../configs/runtime-lock.json)；这不是其他机器上的通用安装器。

权重按 [models/README.md](../models/README.md) 放到本项目 `models/`。三个文件不进入 Git。所有命令从项目根目录执行；`--run-dir` 必须是尚不存在的 `runs/` 子目录，每次执行使用新名称。

## 准备源码

以下三个 checkout 仅保存在 `.runtime/`。准备工具会核对提交身份。

```bash
git clone --depth 1 --branch v0.45.0 https://github.com/Comfy-Org/ComfyUI.git .runtime/engines/ComfyUI-image
git clone --depth 1 --branch v0.2.41 https://github.com/Comfy-Org/comfy-kitchen.git .runtime/sources/comfy-kitchen
git clone --depth 1 --branch v0.6.0 https://github.com/Comfy-Org/comfy-aimdo.git .runtime/sources/comfy-aimdo
```

ComfyUI 提交为 `989d93d0d968e704c589ce0146bb18c77f6b50cc`；两个扩展的提交和已验证二进制哈希记录在运行时锁中。

## 安装与检查

确认执行时 GPU1 可用。启动工具检查物理编号、BDF/UUID、KFD 使用情况和共享预约，并持有既有 GPU 锁。GPU3 被禁止；被占用时命令直接失败，不接管其他实验。

```bash
python3 tools/run_rocm10.py --gpu 1 --run-dir runs/setup-image-env -- python tools/prepare_environment.py --install
python3 tools/run_rocm10.py --gpu 1 --run-dir runs/build-image-native -- .runtime/venv/bin/python tools/build_native.py
python3 tools/run_rocm10.py --gpu 1 --run-dir runs/check-image-env -- .runtime/venv/bin/python tools/verify_environment.py
```

项目 venv 通过 `.pth` 引用基础镜像的包，应用包安装在项目 venv 中；安装前解析依赖并阻止替换基础 Torch、Triton、TorchVision 和 ROCm 包。HIP 扩展从固定源码在同一镜像内重新编译，仅构建 gfx1201。所有构建和 JIT 缓存留在本项目。

下载后先按模型说明核对完整 SHA256。环境检查脚本使用公开运行时锁中的模型清单核对 loader 路径和实际大小，不在每次启动时重新读取并哈希全部 49.29 GB。已有本地下载校验报告留在 `models/.download/`。

## 生成一张图片

```bash
python3 tools/run_rocm10.py --gpu 1 --run-dir runs/image-2048-seed42 -- .runtime/venv/bin/python tools/run_image.py
```

执行 [API 工作流](../configs/workflows/t2i_2048_api.json)：2048×2048、batch 1、seed 42、6 步、CFG 1、euler/simple、denoise 1。使用固定详细 prompt，关闭 expander、Comfy 编译器与节点结果缓存；原生 DynamicVRAM 开启。

输出位于指定运行目录的 `output/`。同目录保留 `result.json`、`console.log`、六步 latent、conditioning、噪声、sigmas、显存采样及资源释放记录。此入口为带诊断记录的首图验证工具，包括阶段同步、张量保存和少量 CPU profiler 记录，不能直接作为无插桩热请求基准。当前不启动持久的 Web UI 服务。

源码、锁文件与工作流公开；图片、权重、原始记录、`vibe/` 和 `research/` 保持本地。下一阶段须完成 A/A、多 prompt 和无 profiler 热请求测试，再建立优化前的数值策略。
