# 独立环境与原始精度四卡复现

当前工具针对已配置共享 ROCm 10.1 foundation 的 R9700 工作区。需要父目录的 `rocm10-launch.sh`、锁定镜像、Docker、`rocm-smi`，以及现有共享预约目录和 `minimax_h3/.standard-opt-state/locks/`。镜像 ID 见 [运行时锁](../configs/runtime-lock.json)；这不是其他机器上的通用安装器。

权重按 [models/README.md](../models/README.md) 放到本项目 `models/`。当前路线使用两个固定官方 revision 的原始 BF16/FP32 组件；历史 INT8 三文件也不进入 Git。所有命令从项目根目录执行；`--run-dir` 必须是尚不存在的 `runs/` 子目录，每次执行使用新名称。

## 准备源码

以下三个 checkout 仅保存在 `.runtime/`。准备工具会核对提交身份。

```bash
git clone --depth 1 --branch v0.45.0 https://github.com/Comfy-Org/ComfyUI.git .runtime/engines/ComfyUI-image
git clone --depth 1 --branch v0.2.41 https://github.com/Comfy-Org/comfy-kitchen.git .runtime/sources/comfy-kitchen
git clone --depth 1 --branch v0.6.0 https://github.com/Comfy-Org/comfy-aimdo.git .runtime/sources/comfy-aimdo
```

ComfyUI 提交为 `989d93d0d968e704c589ce0146bb18c77f6b50cc`；两个扩展的提交和已验证二进制哈希记录在运行时锁中。

## 安装与检查

确认执行时所选四张物理卡可用。启动工具检查物理编号、BDF/UUID、KFD 使用情况和共享预约，并持有既有 GPU 锁。GPU3 被禁止；被占用时命令直接失败，不接管其他实验。

```bash
python3 tools/run_rocm10.py --gpu 1 --run-dir runs/setup-image-env -- python tools/prepare_environment.py --install
python3 tools/run_rocm10.py --gpu 1 --run-dir runs/build-image-native -- .runtime/venv/bin/python tools/build_native.py
python3 tools/run_rocm10.py --gpu 1 --run-dir runs/check-native-image-env -- .runtime/venv/bin/python tools/probe_native_runtime.py --attention-4k
```

项目 venv 通过 `.pth` 引用基础镜像的包，应用包安装在项目 venv 中；安装前解析依赖并阻止替换基础 Torch、Triton、TorchVision 和 ROCm 包。HIP 扩展从固定源码在同一镜像内重新编译，仅构建 gfx1201。所有构建和 JIT 缓存留在本项目。

当前原始组件可由固定下载器准备：

```bash
python3 tools/download_native_models.py --workers 8 --chunk-mib 64
```

下载器对 40 个目标文件的固定 revision、大小和 SHA256 逐项核验，并检查 25 个 safetensors 的结构；本地结果留在 `models/.download/native/`。原始精度流程不依赖历史 INT8 权重；旧 `tools/verify_environment.py` 是 INT8 ConvRot 检查入口，需要历史三文件，仅在复现旧路线时运行。两组权重都不会进入 Git。

## 当前：四卡原始精度 1K / 2K / 4K

下例物理卡 `1,2,6,7` 只是已通过资格的同映射示例；每次运行前根据当前占用选择四张允许卡，物理 GPU3 始终不能使用。运行器会同时持锁并以 0.5 秒记录四卡整卡 VRAM、进程树 RSS/VmHWM 和整机 RAM 增量。

```bash
python3 tools/run_rocm10.py \
  --gpus 1,2,6,7 \
  --run-dir runs/native-4k-seed42 \
  --timeout 1800 \
  --monitor-interval 0.5 \
  --purpose native-4k-qualification \
  -- .runtime/venv/bin/python tools/run_native_image.py \
  --mode four-static \
  --config configs/native_t2i.json \
  --width 4096 \
  --height 4096 \
  --seed 42
```

配置固定官方六步 PDD sigma schedule、CFG=1、关闭额外 guidance，命令行不提供降步数或改精度的入口。文本编码器在一次性 GPU 子进程中逐层 offload；conditioning 以 hash-verified BF16 CPU artifact 进入干净主进程。DiT 按完整层静态分到四卡并强制大序列 PyTorch Flash Attention，最后释放第一卡 DiT 分片并运行 FP32 VAE。量化和 vLLM-Omni 均关闭。

将宽高同时设为 `1024` 可生成 1K，同设为 `2048` 可生成 2K；每次更换 `--run-dir`。1K 的短序列沿用 PyTorch SDPA dispatcher，2K/4K 的长 DiT 序列强制 Flash，其余精度、六步和 offload 策略相同。分辨率实测见根 [README](../README.md#measured-performance)。

成功目录包含 `result.json`、`monitor.jsonl`、`postflight.json`、PNG、conditioning、初始噪声及六张逐步 latent。运行结束后生成资源汇总：

```bash
python3 tools/summarize_native_run.py runs/native-4k-seed42
```

生成的 `summary.json` / `summary.md` 同时给出：

- measured request 与 launcher wall；
- 六个去噪步及各阶段耗时；
- 四张物理卡的整卡采样 VRAM 峰值与逻辑卡 PyTorch allocated/reserved 峰值；
- 进程树 RSS、VmHWM 汇总与整机 RAM 相对启动基线增量。

严格 A/A 必须保持完全相同的物理卡顺序，并从头执行第二个请求：

```bash
python3 tools/run_rocm10.py \
  --gpus 1,2,6,7 \
  --run-dir runs/native-4k-seed42-aa \
  --timeout 1800 \
  --monitor-interval 0.5 \
  --purpose native-4k-aa \
  -- .runtime/venv/bin/python tools/run_native_image.py \
  --mode four-static \
  --config configs/native_t2i.json \
  --width 4096 \
  --height 4096 \
  --seed 42 \
  --reference-run runs/native-4k-seed42
```

reference 只用于比较，不会替代本次文本编码或生成。当前资格观测是六个 batch=1 请求，不可直接当作无插桩热吞吐或并发指标。

## 历史：单卡 INT8 首图

```bash
python3 tools/run_rocm10.py --gpu 1 --run-dir runs/image-2048-seed42 -- .runtime/venv/bin/python tools/run_image.py
```

执行 [API 工作流](../configs/workflows/t2i_2048_api.json)：2048×2048、batch 1、seed 42、6 步、CFG 1、euler/simple、denoise 1。使用固定详细 prompt，关闭 expander、Comfy 编译器与节点结果缓存；原生 DynamicVRAM 开启。

输出位于指定运行目录的 `output/`。同目录保留 `result.json`、`console.log`、六步 latent、conditioning、噪声、sigmas、显存采样及资源释放记录。此入口为带诊断记录的首图验证工具，包括阶段同步、张量保存和少量 CPU profiler 记录，不能直接作为无插桩热请求基准。当前不启动持久的 Web UI 服务。

源码、锁文件与工作流公开；图片、权重、原始记录、`vibe/` 和 `research/` 保持本地。原始精度路线已经完成同映射严格 A/A 和三类 prompt × 两 seed 功能矩阵；下一阶段若进行并发/吞吐优化，必须单独冻结 workload 和验收门，不能覆盖当前数值基线。
