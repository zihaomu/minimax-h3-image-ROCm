"""End-to-end original-precision MiniMax-H3-Image execution."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .dit import NativeImageDiT
from .text_encoder import StreamingTextEncoder
from .vae import OriginalImageVAE


OFFICIAL_SIGMAS = [
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def official_schedule(declared: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    if declared != OFFICIAL_SIGMAS:
        raise RuntimeError(f"Only the official H3-Image schedule is accepted: {declared}")
    # Reproduce the released modular pipeline's float32 sigma -> timestep ->
    # sigma round trip, then the scheduler's timestep construction.
    initial_timesteps = torch.tensor([1.0 - sigma for sigma in declared[:-1]], dtype=torch.float32)
    sigmas = torch.cat([1.0 - initial_timesteps, torch.zeros(1, dtype=torch.float32)])
    timesteps = 1.0 - sigmas[:-1]
    if timesteps.numel() != 6 or sigmas.numel() != 7:
        raise RuntimeError("The official workload must have six model evaluations")
    return sigmas, timesteps


def tensor_artifact(metrics, path: Path, tensor: torch.Tensor, name: str, **fields) -> dict:
    started = time.perf_counter()
    source = tensor.device
    cpu = tensor.detach().to(device="cpu").contiguous()
    if source.type == "cuda":
        torch.cuda.synchronize(source)
    transfer_seconds = time.perf_counter() - started
    if source.type == "cuda":
        metrics.transfer(
            "device_to_host_artifact",
            cpu.numel() * cpu.element_size(),
            transfer_seconds,
            component=name,
            source_logical_device=source.index,
            **fields,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    save_file({"tensor": cpu}, str(path))
    save_seconds = time.perf_counter() - save_started
    row = {
        "path": str(path),
        "sha256": sha256(path),
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "bytes": cpu.numel() * cpu.element_size(),
        "transfer_seconds": transfer_seconds,
        "save_seconds": save_seconds,
        **fields,
    }
    return row


def compare_tensor(reference_path: Path, actual_path: Path) -> dict:
    with safe_open(reference_path, framework="pt", device="cpu") as reference_file:
        reference = reference_file.get_tensor("tensor").float()
    with safe_open(actual_path, framework="pt", device="cpu") as actual_file:
        actual = actual_file.get_tensor("tensor").float()
    if reference.shape != actual.shape:
        raise RuntimeError(f"Reference shape mismatch: {reference.shape} != {actual.shape}")
    difference = actual - reference
    reference_norm = torch.linalg.vector_norm(reference.double())
    difference_norm = torch.linalg.vector_norm(difference.double())
    denominator = max(reference_norm.item(), torch.finfo(torch.float64).tiny)
    return {
        "reference": str(reference_path),
        "actual": str(actual_path),
        "relative_l2": difference_norm.item() / denominator,
        "max_absolute": difference.abs().max().item(),
        "mean_absolute": difference.abs().mean().item(),
        "exact_equal": bool(torch.equal(reference, actual)),
    }


class NativeImagePipeline:
    def __init__(self, root: Path, run: Path, config: dict, mode: str, metrics):
        self.root = Path(root)
        self.run = Path(run)
        self.config = config
        self.mode = mode
        self.metrics = metrics
        expected_devices = 1 if mode == "single_offload" else 4
        if torch.cuda.device_count() != expected_devices:
            raise RuntimeError(
                f"{mode} requires {expected_devices} visible devices, got {torch.cuda.device_count()}"
            )
        self.devices = [torch.device("cuda", index) for index in range(expected_devices)]
        self.image_root = self.root / "models/native/minimax-h3-image"
        self.text_root = self.root / "models/native/minimax-h3/text_encoder"
        self.dit = None
        self.vae = None
        self.text_encoder_stage = None

    def _validate(self) -> None:
        width = int(self.config["width"])
        height = int(self.config["height"])
        if width != height or width not in (1024, 2048, 4096) or width % 16:
            raise ValueError("Qualification accepts square 1024, 2048 or 4096 images divisible by 16")
        if int(self.config["steps"]) != 6 or float(self.config["cfg"]) != 1.0:
            raise ValueError("Qualification freezes six NFEs and CFG=1 (no CFG branch)")
        if self.config["sigmas"] != OFFICIAL_SIGMAS:
            raise ValueError("Qualification freezes the released PDD sigma grid")
        if not self.config.get("prompt"):
            raise ValueError("A non-empty prompt is required")
        required = [
            self.image_root / "pdd_config.json",
            self.image_root / "pdd_fused_heads.safetensors",
            self.image_root / "transformer/model.safetensors.index.json",
            self.image_root / "vae/diffusion_pytorch_model.safetensors",
            self.image_root / "tokenizer/tokenizer.json",
            self.text_root / "model.safetensors.index.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Native model preparation is incomplete: {missing}")

    def _cleanup(self) -> dict:
        report = {"vae_released_bytes": 0, "dit_released_bytes": 0}
        if self.vae is not None:
            report["vae_released_bytes"] = self.vae.release()
            self.vae = None
        if self.dit is not None:
            report["dit_released_bytes"] = self.dit.release_all()
            self.dit = None
        gc.collect()
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        report["post_cleanup"] = self.metrics.device_memory()
        return report

    def _encode_context_isolated(self, prompt: str) -> tuple[torch.Tensor, dict]:
        """Run the streamed text encoder in a disposable GPU process.

        The conditioning artifact is the explicit process boundary.  No HIP
        allocator, library, or kernel-selection state from the 50-layer text
        encoder can survive into the DiT process.
        """
        stage_run = self.run / "stages/text-encoder"
        stage_run.mkdir(parents=True, exist_ok=False)
        request_path = stage_run / "request.json"
        request_path.write_text(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
        log_path = stage_run / "console.log"
        command = [
            sys.executable,
            str(self.root / "tools/encode_native_conditioning.py"),
            "--request",
            str(request_path),
            "--stage-run",
            str(stage_run),
        ]
        started = time.perf_counter()
        with log_path.open("x") as log:
            completed = subprocess.run(
                command,
                cwd=self.root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        subprocess_seconds = time.perf_counter() - started
        result_path = stage_run / "result.json"
        if not result_path.is_file():
            raise RuntimeError(
                f"Isolated text process produced no result (exit {completed.returncode}); see {log_path}"
            )
        stage_result = json.loads(result_path.read_text())
        if completed.returncode != 0 or stage_result.get("status") != "complete":
            raise RuntimeError(
                f"Isolated text process failed (exit {completed.returncode}); see {log_path}"
            )
        artifact = stage_result["conditioning"]
        artifact_path = Path(artifact["path"]).resolve()
        if artifact_path != stage_run / "conditioning.safetensors":
            raise RuntimeError(f"Unexpected isolated conditioning path: {artifact_path}")
        if artifact.get("sha256") != sha256(artifact_path):
            raise RuntimeError("Isolated conditioning artifact hash changed before consumption")
        with safe_open(artifact_path, framework="pt", device="cpu") as tensors:
            context = tensors.get_tensor("tensor")
        if context.dtype != torch.bfloat16 or context.ndim != 3 or context.shape[-1] != 5120:
            raise RuntimeError(
                f"Unexpected isolated conditioning tensor: {context.dtype} {tuple(context.shape)}"
            )
        if not bool(torch.isfinite(context).all().item()):
            raise RuntimeError("Isolated conditioning contains non-finite values")
        self.metrics.transfer(
            "device_to_host_activation",
            artifact["bytes"],
            artifact["transfer_seconds"],
            component="conditioning_process_boundary",
            source_logical_device=0,
            bit_preserving=True,
            executed_in_child_process=True,
        )
        self.text_encoder_stage = {
            "process_isolation": True,
            "run": str(stage_run),
            "subprocess_seconds": subprocess_seconds,
            "child_request_seconds": stage_result.get("request_seconds"),
            "torch_global_peaks": stage_result.get("torch_global_peaks"),
            "phases": stage_result.get("phases"),
            "transfers": stage_result.get("transfers"),
            "cleanup": stage_result.get("cleanup"),
            "conditioning": artifact,
        }
        return context.contiguous(), stage_result["text_encoder"]

    def run_request(self, reference_run: Path | None = None) -> dict:
        self._validate()
        width = int(self.config["width"])
        height = int(self.config["height"])
        seed = int(self.config["seed"])
        sigmas, timesteps = official_schedule(self.config["sigmas"])
        artifacts = []
        comparisons = []
        denoise_rows = []
        cleanup = None

        try:
            if self.config.get("text_encoder_process_isolation") is not True:
                raise RuntimeError(
                    "Qualification requires text_encoder_process_isolation=true; "
                    "the in-process CPU barrier did not pass strict 4K A/A"
                )
            with self.metrics.phase(
                "text_encode_isolated_process", prompt_characters=len(self.config["prompt"])
            ) as phase:
                context, text_report = self._encode_context_isolated(self.config["prompt"])
                phase["child"] = self.text_encoder_stage

            with self.metrics.phase(
                "conditioning_process_boundary",
                dtype=str(context.dtype),
                shape=list(context.shape),
                process_isolated=True,
            ) as phase:
                phase["bytes"] = context.numel() * context.element_size()
                phase["artifact_sha256"] = self.text_encoder_stage["conditioning"]["sha256"]
                phase["gpu0_allocated_before_dit"] = torch.cuda.memory_allocated(self.devices[0])

            with self.metrics.phase("save_conditioning_reference"):
                conditioning_path = self.run / "tensors/conditioning.safetensors"
                artifacts.append(
                    tensor_artifact(self.metrics, conditioning_path, context, "conditioning")
                )
                if reference_run is not None:
                    comparison = compare_tensor(
                        reference_run / "tensors/conditioning.safetensors", conditioning_path
                    )
                    comparison["artifact"] = "conditioning"
                    comparisons.append(comparison)

            with self.metrics.phase("dit_load", placement=self.mode):
                self.dit = NativeImageDiT(
                    self.image_root / "transformer",
                    self.image_root / "pdd_config.json",
                    self.image_root / "pdd_fused_heads.safetensors",
                    self.devices,
                    self.mode,
                    self.metrics,
                )

            generator = torch.Generator(device="cpu").manual_seed(seed)
            latent = torch.randn(
                (1, 24, 1, height // 16, width // 16),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            with self.metrics.phase("save_initial_noise"):
                initial_path = self.run / "tensors/latent-initial.safetensors"
                artifacts.append(tensor_artifact(self.metrics, initial_path, latent, "initial_noise"))
                if reference_run is not None:
                    comparison = compare_tensor(
                        reference_run / "tensors/latent-initial.safetensors", initial_path
                    )
                    comparison["artifact"] = "initial_noise"
                    comparisons.append(comparison)

            for step_index in range(6):
                with self.metrics.phase(
                    "denoise_step",
                    step=step_index,
                    sigma=float(sigmas[step_index]),
                    sigma_next=float(sigmas[step_index + 1]),
                    timestep=float(timesteps[step_index]),
                ) as phase:
                    velocity, state, row = self.dit.forward(
                        latent,
                        context,
                        timesteps[step_index],
                        step_index,
                    )
                    delta = (sigmas[step_index] - sigmas[step_index + 1]).to(
                        device=velocity.device, dtype=torch.float32
                    )
                    latent = state.to(torch.float32) + delta * velocity.to(torch.float32)
                    torch.cuda.synchronize(velocity.device)
                    if latent.dtype != torch.float32 or not bool(torch.isfinite(latent).all().item()):
                        raise RuntimeError(f"Invalid latent after denoise step {step_index}")
                    row["sigma"] = float(sigmas[step_index])
                    row["sigma_next"] = float(sigmas[step_index + 1])
                    row["euler_delta"] = float(delta)
                    denoise_rows.append(row)
                    phase["dit"] = row

                with self.metrics.phase("save_step_latent", step=step_index):
                    step_path = self.run / f"tensors/latent-step-{step_index + 1:02d}.safetensors"
                    artifacts.append(
                        tensor_artifact(
                            self.metrics,
                            step_path,
                            latent,
                            "step_latent",
                            step=step_index,
                        )
                    )
                if reference_run is not None:
                    reference_path = reference_run / "tensors" / step_path.name
                    if not reference_path.is_file():
                        raise FileNotFoundError(f"Missing reference latent: {reference_path}")
                    comparison = compare_tensor(reference_path, step_path)
                    comparison["step"] = step_index
                    comparisons.append(comparison)
                    self.metrics.report["reference_comparisons"] = comparisons
                    self.metrics.write()

            del context, velocity, state
            released_first = self.dit.release_first_device()
            self.metrics.report["dit_first_device_release_bytes"] = released_first
            self.metrics.write()

            with self.metrics.phase("vae_load", dtype="float32"):
                self.vae = OriginalImageVAE(
                    self.image_root / "vae/diffusion_pytorch_model.safetensors",
                    self.devices[0],
                    self.metrics,
                )

            with self.metrics.phase("vae_decode", width=width, height=height) as phase:
                pixels, vae_report = self.vae.decode(latent)
                expected_shape = (1, 3, 1, height, width)
                if tuple(pixels.shape) != expected_shape:
                    raise RuntimeError(
                        f"VAE returned {tuple(pixels.shape)}, expected {expected_shape}"
                    )
                phase["vae"] = vae_report

            with self.metrics.phase("save_png", quantization="uint8_truncation"):
                image_tensor = (
                    pixels[0, :, 0].permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu()
                )
                output = self.run / f"minimax-h3-image-{width}-seed{seed}.png"
                Image.fromarray(image_tensor.numpy(), mode="RGB").save(output)
                output_report = {
                    "path": str(output),
                    "sha256": sha256(output),
                    "width": width,
                    "height": height,
                    "mode": "RGB",
                    "byte_count": output.stat().st_size,
                    "pixel_min": int(image_tensor.min()),
                    "pixel_max": int(image_tensor.max()),
                    "nonconstant": bool(image_tensor.min() != image_tensor.max()),
                }
                if not output_report["nonconstant"]:
                    raise RuntimeError("Generated image is constant")

            del pixels, image_tensor, latent
            cleanup = self._cleanup()
            result = {
                "mode": self.mode,
                "width": width,
                "height": height,
                "seed": seed,
                "steps": 6,
                "cfg": 1.0,
                "guidance_branch": False,
                "declared_sigmas": self.config["sigmas"],
                "effective_sigmas_float32": sigmas.tolist(),
                "timesteps_float32": timesteps.tolist(),
                "scheduler_update": "float32 data-ward velocity Euler",
                "text_encoder": text_report,
                "text_encoder_stage": self.text_encoder_stage,
                "denoise": denoise_rows,
                "artifacts": artifacts,
                "reference_comparisons": comparisons,
                "vae": vae_report,
                "output": output_report,
                "cleanup": cleanup,
            }
            from .bootstrap import attention_policy_report

            result["attention_policy"] = attention_policy_report()
            self.metrics.report["request"] = result
            self.metrics.finish(output=output_report, cleanup=cleanup)
            return result
        except BaseException:
            if cleanup is None:
                try:
                    cleanup = self._cleanup()
                    self.metrics.report["cleanup_after_failure"] = cleanup
                    self.metrics.write()
                except BaseException as cleanup_error:
                    self.metrics.report["cleanup_error"] = repr(cleanup_error)
                    self.metrics.write()
            raise
