"""Original-FP32 MiniMax-H3-Image VAE decoder loading and execution."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

from safetensors import safe_open
import torch

from .bootstrap import materialized_module_construction
from .weights import assign_tensor, unload_module_to_meta


DECODE_PREFIXES = ("post_quant_conv.", "decoder.")
CONFIG_BUFFER_TENSORS = frozenset(("latents_mean", "latents_std"))


class OriginalImageVAE:
    """Load only the original FP32 tensors needed for text-to-image decode.

    The released file also carries the image encoder. Text-to-image never calls
    it, so keeping those weights on the meta device avoids an unrelated GPU
    copy without changing any decoder tensor or arithmetic.
    """

    def __init__(self, checkpoint: Path, device: torch.device, metrics):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.metrics = metrics
        vae_config = json.loads((self.checkpoint.parent / "config.json").read_text())

        import comfy.ops
        from comfy.ldm.minimax.vae import MiniMaxH3ImageVAE

        with materialized_module_construction(), torch.device("meta"):
            self.model = MiniMaxH3ImageVAE(operations=comfy.ops.disable_weight_init)

        expected_all = set(self.model.state_dict().keys())
        started = time.perf_counter()
        byte_count = 0
        loaded = []
        with safe_open(self.checkpoint, framework="pt", device="cpu") as tensors:
            source_keys = set(tensors.keys())
            expected_checkpoint = expected_all - CONFIG_BUFFER_TENSORS
            if source_keys != expected_checkpoint:
                missing = sorted(expected_checkpoint - source_keys)
                extra = sorted(source_keys - expected_checkpoint)
                raise RuntimeError(
                    f"VAE checkpoint mapping mismatch: missing={missing[:10]} extra={extra[:10]}"
                )
            decode_keys = sorted(
                name
                for name in source_keys
                if name.startswith(DECODE_PREFIXES)
            )
            for name in decode_keys:
                source = tensors.get_tensor(name)
                if source.dtype != torch.float32:
                    raise RuntimeError(f"Original VAE tensor is not FP32: {name} {source.dtype}")
                moved = source.to(device=self.device, non_blocking=False)
                assign_tensor(self.model, name, moved)
                byte_count += moved.numel() * moved.element_size()
                loaded.append(name)
                del source, moved

        self._materialize_config_and_nonpersistent_buffers(vae_config)
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        unresolved = self._unresolved_decode_tensors()
        if unresolved:
            raise RuntimeError(f"VAE decoder has unresolved tensors: {unresolved}")
        self.load_report = {
            "checkpoint": str(self.checkpoint),
            "loaded_tensor_count": len(loaded),
            "loaded_bytes": byte_count,
            "checkpoint_tensor_count": len(source_keys),
            "encoder_weights_loaded": False,
            "config_buffers": sorted(CONFIG_BUFFER_TENSORS),
            "dtype_policy": "released FP32 decoder tensors; FP32 execution",
            "seconds": seconds,
        }
        metrics.transfer(
            "host_to_device_weights",
            byte_count,
            seconds,
            component="vae_fp32_decode_only",
            shard=self.checkpoint.name,
            logical_devices=[self.device.index],
        )
        metrics.report["vae_load"] = self.load_report
        metrics.write()

    def _materialize_config_and_nonpersistent_buffers(self, vae_config: dict) -> None:
        from comfy.ldm.minimax.vae import (
            IMAGENET_MEAN,
            IMAGENET_STD,
            LATENTS_MEAN,
            LATENTS_STD,
        )

        for name, reference in (("latents_mean", LATENTS_MEAN), ("latents_std", LATENTS_STD)):
            configured = vae_config[name]
            if len(configured) != len(reference) or any(float(a) != float(b) for a, b in zip(configured, reference)):
                raise RuntimeError(f"VAE {name} differs between fixed config and pinned Comfy source")
            assign_tensor(
                self.model,
                name,
                torch.tensor(configured, dtype=torch.float32, device=self.device),
            )

        assign_tensor(
            self.model,
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1, 1),
        )
        assign_tensor(
            self.model,
            "pixel_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1, 1),
        )
        inv_freq = self.model.decoder.pos_embed.inv_freq
        count = inv_freq.numel()
        values = 1.0 / 100.0 ** (
            torch.arange(count, dtype=torch.float32, device=self.device) / count
        )
        assign_tensor(self.model, "decoder.pos_embed.inv_freq", values)
        for index, block in enumerate(self.model.decoder.transformer_blocks):
            scale = block.attn.qk_norm_scale
            assign_tensor(
                self.model,
                f"decoder.transformer_blocks.{index}.attn.qk_norm_scale",
                torch.ones(scale.shape, dtype=torch.float32, device=self.device),
            )

    def _unresolved_decode_tensors(self) -> list[str]:
        unresolved = []
        for name, value in self.model.named_parameters():
            if (name.startswith(DECODE_PREFIXES) or name in CONFIG_BUFFER_TENSORS) and value.is_meta:
                unresolved.append(name)
        for name, value in self.model.named_buffers():
            required = (
                name.startswith(DECODE_PREFIXES)
                or name in CONFIG_BUFFER_TENSORS
                or name in ("pixel_mean", "pixel_std")
            )
            if required and value is not None and value.is_meta:
                unresolved.append(name)
        return sorted(unresolved)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, dict]:
        started = time.perf_counter()
        source_device = latent.device
        latent = latent.to(device=self.device, dtype=torch.float32, non_blocking=False)
        torch.cuda.synchronize(self.device)
        transfer_seconds = time.perf_counter() - started
        if source_device != self.device:
            self.metrics.transfer(
                "device_to_device_activation" if source_device.type == "cuda" else "host_to_device_activation",
                latent.numel() * latent.element_size(),
                transfer_seconds,
                component="final_latent_to_vae",
                source_logical_device=source_device.index,
                source_device_type=source_device.type,
                destination_logical_device=self.device.index,
            )

        decode_started = time.perf_counter()
        pixels = self.model.decode(latent)
        torch.cuda.synchronize(self.device)
        decode_seconds = time.perf_counter() - decode_started
        if pixels.dtype != torch.float32 or not bool(torch.isfinite(pixels).all().item()):
            raise RuntimeError(f"Invalid VAE output: dtype={pixels.dtype} shape={tuple(pixels.shape)}")
        return pixels, {
            "input_shape": list(latent.shape),
            "output_shape": list(pixels.shape),
            "output_dtype": str(pixels.dtype),
            "latent_transfer_seconds": transfer_seconds,
            "decode_seconds": decode_seconds,
            "tile_size_pixels": self.model.tile_size,
            "minimum_overlap_pixels": self.model.tile_overlap_min,
        }

    def release(self) -> int:
        released = unload_module_to_meta(self.model)
        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        return released
