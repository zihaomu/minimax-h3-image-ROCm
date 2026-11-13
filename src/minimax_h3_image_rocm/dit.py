"""Original-precision MiniMax-H3-Image DiT with explicit layer placement."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import re
import time

import torch
import torch.nn.functional as functional

from .bootstrap import materialized_module_construction
from .weights import (
    SafetensorIndex,
    block_device,
    load_absolute_pdd_heads,
    meta_tensors,
    plan_dit,
    raw_head_major_qkv_to_component_major,
    unload_module_to_meta,
)


BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.")


class NativeImageDiT:
    @staticmethod
    def _raw_tensor_transform(
        source_name: str, _target_name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        if source_name.endswith(".attn.qkv_proj.weight"):
            return raw_head_major_qkv_to_component_major(tensor)
        return tensor

    def __init__(
        self,
        model_root: Path,
        pdd_config: Path,
        pdd_heads: Path,
        devices: list[torch.device],
        mode: str,
        metrics,
    ):
        if mode not in ("single_offload", "four_static"):
            raise ValueError(mode)
        if mode == "single_offload" and len(devices) != 1:
            raise ValueError("single_offload requires exactly one visible device")
        if mode == "four_static" and len(devices) != 4:
            raise ValueError("four_static requires exactly four visible devices")
        self.root = Path(model_root)
        self.index = SafetensorIndex(self.root)
        self.pdd = json.loads(Path(pdd_config).read_text())
        self.declared_sigmas = [float(value) for value in self.pdd["sigmas"]]
        if self.pdd.get("num_inference_steps") != 6 or len(self.declared_sigmas) != 7:
            raise RuntimeError(f"The native backend only accepts the official six-NFE PDD schedule: {self.pdd}")
        self.devices = [torch.device(device) for device in devices]
        self.mode = mode
        self.metrics = metrics

        import comfy.ops
        from comfy.ldm.minimax.model import MiniMaxH3ImageModel

        with materialized_module_construction(), torch.device("meta"):
            self.model = MiniMaxH3ImageModel(
                pdd_sigmas=self.declared_sigmas,
                pdd_steps=6,
                dtype=torch.bfloat16,
                device=torch.device("meta"),
                operations=comfy.ops.disable_weight_init,
            )

        # The released mixed-precision path keeps the timestep MLP output in
        # FP32 through SiLU, then casts that activation to the BF16 AdaLN
        # projection.  Pinned Comfy normally casts the timestep embedding
        # before its AdalnProj-owned SiLU; disable that internal activation so
        # forward() can reproduce the released ordering explicitly.
        for block in self.model.blocks:
            block.adaln_proj.apply_silu = False
        self.model.final_layer.adaln_proj.apply_silu = False

        self.plan = plan_dit(self.index) if mode == "four_static" else {
            "ranges": [{"logical_device": 0, "start": 0, "stop": 50}],
            "mode": "one block at a time",
        }
        raw_output = {"final_layer.video_out.weight", "final_layer.video_out.bias"}
        mapping = {name: name for name in self.index.specs if name not in raw_output}

        if mode == "single_offload":
            mapping = {name: target for name, target in mapping.items() if not BLOCK_PATTERN.match(name)}

        def destination(source_name: str, _target_name: str) -> torch.device:
            match = BLOCK_PATTERN.match(source_name)
            if match:
                return self.devices[block_device(self.plan, int(match.group(1)))]
            if source_name.startswith("final_layer."):
                return self.devices[-1]
            return self.devices[0]

        load = self.index.load_mapping(
            self.model,
            mapping,
            destination,
            metrics=metrics,
            component=f"dit_{mode}",
            tensor_transform=self._raw_tensor_transform,
        )
        pdd_load = load_absolute_pdd_heads(
            self.model, Path(pdd_heads), self.devices[-1], metrics=metrics
        )
        unresolved = meta_tensors(self.model)
        if mode == "four_static":
            if unresolved:
                raise RuntimeError(f"Static DiT has unresolved tensors: {unresolved[:20]}")
        else:
            unexpected = [name for name in unresolved if not name.startswith("blocks.")]
            if unexpected:
                raise RuntimeError(f"Offload DiT has unresolved non-block tensors: {unexpected}")
        self.metrics.report["dit_placement"] = {
            "mode": mode,
            "plan": self.plan,
            "base_load": load,
            "pdd_load": pdd_load,
            "pdd_head_semantics": "six absolute official heads, selected directly by step index",
            "time_embedding_policy": "FP32 time MLP and SiLU, then BF16 AdaLN projection",
            "qkv_weight_policy": "official raw head-major rows permuted to Comfy component-major Q|K|V rows",
            "unresolved_block_tensor_count": len(unresolved) if mode == "single_offload" else 0,
        }
        self.metrics.write()

    def _block_mapping(self, index: int) -> dict[str, str]:
        prefix = f"blocks.{index}."
        expected = set(self.model.blocks[index].state_dict().keys())
        mapping = {prefix + name: name for name in expected}
        missing = sorted(set(mapping) - set(self.index.specs))
        if missing:
            raise RuntimeError(f"DiT block {index} mapping is incomplete: {missing}")
        return mapping

    def _move(self, tensor: torch.Tensor, device: torch.device, name: str, step: int) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        started = time.perf_counter()
        output = tensor.to(device=device, non_blocking=True)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        self.metrics.transfer(
            "device_to_device_activation" if tensor.is_cuda else "host_to_device_activation",
            output.numel() * output.element_size(),
            seconds,
            tensor=name,
            step=step,
            source_logical_device=tensor.device.index if tensor.is_cuda else None,
            source_device_type=tensor.device.type,
            destination_logical_device=device.index,
            transport_qualification="copy completed; physical transport path not inferred",
        )
        return output

    def _load_offload_block(self, index: int) -> dict:
        block = self.model.blocks[index]
        return self.index.load_mapping(
            block,
            self._block_mapping(index),
            lambda _source, _target: self.devices[0],
            metrics=self.metrics,
            component=f"dit_step_block_{index:02d}",
            tensor_transform=self._raw_tensor_transform,
        )

    @torch.no_grad()
    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        from comfy.ldm.minimax.model import PackedLayout, rope_rotation_table, unpatchify_video

        first = self.devices[0]
        # The released input projection is an FP32 island. Keep the scheduler's
        # FP32 latent intact until _embed_and_pack projects it, then the hidden
        # stream follows the BF16 context dtype.
        latent = self._move(latent, first, "latent", step_index).to(torch.float32)
        context = self._move(context, first, "context", step_index)
        _, _, latent_t, latent_h, latent_w = latent.shape
        layout = PackedLayout(context.shape[1], latent_t, latent_h, latent_w, 0)
        transformer_options = {}
        text_length = context.shape[1]
        mod_segments = [(0, text_length, 1), (text_length, layout.seq_len, 0)]

        h = self.model._embed_and_pack(
            latent, None, context, layout, {}, transformer_options
        )
        # PDD publishes float32 t = 1 - sigma. Converting its exact float32
        # value through Python and back retains that bit pattern here.
        t_value = float(timestep.to(device="cpu", dtype=torch.float32).item())
        t_emb_fp32 = self.model._time_embedding([t_value], first, torch.float32)
        if t_emb_fp32.dtype != torch.float32:
            raise RuntimeError(f"The timestep MLP must remain FP32, got {t_emb_fp32.dtype}")
        t_emb = functional.silu(t_emb_fp32).to(h.dtype)
        if t_emb.dtype != torch.bfloat16:
            raise RuntimeError(f"The AdaLN projection input must be BF16, got {t_emb.dtype}")
        rope = rope_rotation_table(self.model.rope_freqs(layout.position_ids, first), h.dtype)

        block_rows = []
        current_device = first
        for block_index, block in enumerate(self.model.blocks):
            if self.mode == "four_static":
                desired = self.devices[block_device(self.plan, block_index)]
                if desired != current_device:
                    h = self._move(h, desired, "hidden_states", step_index)
                    t_emb = self._move(t_emb, desired, "timestep_embedding", step_index)
                    rope = self._move(rope, desired, "rope_table", step_index)
                    current_device = desired
            else:
                load = self._load_offload_block(block_index)
            self.metrics.set_phase(
                "dit_block",
                step=step_index,
                block=block_index,
                logical_device=current_device.index,
            )
            started = time.perf_counter()
            transformer_options["block_index"] = block_index
            # comfy-kitchen's HIP/DLPack kernels validate the process-current
            # device as well as each tensor's device.  Tensor movement alone
            # does not update that state, so make every complete-layer segment
            # explicit rather than relying on CUDA/HIP's default device.
            with torch.cuda.device(current_device):
                h = block(h, t_emb, mod_segments, rope, transformer_options=transformer_options)
                torch.cuda.synchronize(current_device)
            row = {
                "block": block_index,
                "logical_device": current_device.index,
                "compute_seconds": time.perf_counter() - started,
            }
            if self.mode == "single_offload":
                row["weight_bytes"] = load["bytes"]
                unload_module_to_meta(block)
            block_rows.append(row)

        final_device = self.devices[-1]
        if current_device != final_device:
            h = self._move(h, final_device, "hidden_states", step_index)
            t_emb = self._move(t_emb, final_device, "timestep_embedding", step_index)
        latent_for_update = self._move(latent, final_device, "latent_for_scheduler", step_index)
        self.metrics.set_phase("dit_pdd_head", step=step_index, logical_device=final_device.index)
        started = time.perf_counter()
        with torch.cuda.device(final_device):
            image_start, image_stop, _ = layout.segments[-1]
            shift, scale = self.model.final_layer.adaln_proj(t_emb)
            image_hidden = self.model.final_layer._modulate(
                h, shift, scale, (image_start, image_stop, 0)
            )
            head = self.model.final_layer.video_out
            weight = head.weight.view(6, 96, self.model.hidden_size)[step_index]
            bias = head.bias.view(6, 96)[step_index]
            velocity_rows = functional.linear(image_hidden, weight, bias)
            velocity = unpatchify_video(
                velocity_rows,
                latent_t,
                latent_h // 2,
                latent_w // 2,
                self.model.latents_dim,
                self.model.patch_size,
            )
            torch.cuda.synchronize(final_device)
        final_seconds = time.perf_counter() - started
        if not bool(torch.isfinite(velocity).all().item()):
            raise RuntimeError(f"Non-finite DiT velocity at step {step_index}")
        return velocity, latent_for_update, {
            "step": step_index,
            "timestep": t_value,
            "sequence_length": layout.seq_len,
            "block_compute_seconds": sum(row["compute_seconds"] for row in block_rows),
            "pdd_head_seconds": final_seconds,
            "blocks": block_rows,
            "velocity_dtype": str(velocity.dtype),
            "velocity_shape": list(velocity.shape),
            "time_embedding_mlp_dtype": str(t_emb_fp32.dtype),
            "time_embedding_adaln_input_dtype": str(t_emb.dtype),
        }

    def release_first_device(self) -> int:
        released = 0
        if self.mode == "single_offload":
            released += unload_module_to_meta(self.model)
        else:
            for module in (
                self.model.video_patch_proj,
                self.model.condition_proj,
                self.model.time_embedder,
                self.model.rope,
                self.model.token_refiner,
            ):
                released += unload_module_to_meta(module)
            first_range = self.plan["ranges"][0]
            for index in range(first_range["start"], first_range["stop"]):
                released += unload_module_to_meta(self.model.blocks[index])
        gc.collect()
        with torch.cuda.device(self.devices[0]):
            torch.cuda.empty_cache()
        return released

    def release_all(self) -> int:
        released = unload_module_to_meta(self.model)
        gc.collect()
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return released
