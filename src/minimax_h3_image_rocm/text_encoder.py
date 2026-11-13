"""Exact text-only Qwen3-VL layer-50 streaming encoder."""
from __future__ import annotations

import gc
from pathlib import Path
import re
import time

import torch
from transformers import Qwen2TokenizerFast

from .bootstrap import materialized_module_construction
from .weights import SafetensorIndex, meta_tensors, unload_module_to_meta


class StreamingTextEncoder:
    def __init__(self, model_root: Path, tokenizer_root: Path, device: torch.device, metrics):
        self.index = SafetensorIndex(Path(model_root), "model.safetensors.index.json")
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            str(tokenizer_root), local_files_only=True
        )
        self.device = torch.device(device)
        self.metrics = metrics
        self.embed_key = self._find_embed_key()
        self.layer_prefix = self._find_layer_prefix()

    def _find_embed_key(self) -> str:
        candidates = [
            name
            for name in self.index.specs
            if name.endswith("language_model.embed_tokens.weight")
            or name.endswith("language_model.model.embed_tokens.weight")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Could not identify one language embedding tensor: {candidates}")
        return candidates[0]

    def _find_layer_prefix(self) -> str:
        candidates = set()
        pattern = re.compile(r"^(.*layers\.)0\.self_attn\.q_proj\.weight$")
        for name in self.index.specs:
            match = pattern.match(name)
            if match:
                candidates.add(match.group(1))
        if len(candidates) != 1:
            raise RuntimeError(f"Could not identify one language layer prefix: {sorted(candidates)}")
        return candidates.pop()

    def _layer_mapping(self, layer: torch.nn.Module, index: int) -> dict[str, str]:
        expected = set(layer.state_dict().keys())
        mapping = {
            f"{self.layer_prefix}{index}.{target_name}": target_name for target_name in expected
        }
        missing = sorted(set(mapping) - set(self.index.specs))
        if missing:
            raise RuntimeError(f"Text layer {index} mapping is incomplete: {missing}")
        return mapping

    @torch.no_grad()
    def encode(self, prompt: str, max_length: int = 4096) -> tuple[torch.Tensor, dict]:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("The official text-only presentation requires one non-empty prompt")
        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"][:max_length]
        if not token_ids:
            raise ValueError("Prompt tokenization produced no tokens")
        ids = torch.tensor(token_ids, dtype=torch.long)

        from comfy.ldm.modules.attention import optimized_attention_for_device
        from comfy.text_encoders.llama import (
            Qwen3VL_32BConfig,
            TransformerBlock,
            precompute_freqs_cis,
        )
        import comfy.ops

        config = Qwen3VL_32BConfig()
        started = time.perf_counter()
        self.metrics.set_phase("text_embedding_rows", tokens=len(token_ids))
        selected = self.index.select_rows_cpu(self.embed_key, ids)
        embedding_read_seconds = time.perf_counter() - started
        transfer_started = time.perf_counter()
        hidden = selected.to(device=self.device, dtype=torch.bfloat16, non_blocking=False).unsqueeze(0)
        torch.cuda.synchronize(self.device)
        self.metrics.transfer(
            "host_to_device_tensor",
            hidden.numel() * hidden.element_size(),
            time.perf_counter() - transfer_started,
            component="text_selected_embedding_rows",
            logical_devices=[self.device.index],
        )
        del selected

        position_ids = torch.arange(len(token_ids), device=self.device, dtype=torch.long).unsqueeze(0)
        freqs_cis = precompute_freqs_cis(
            config.head_dim,
            position_ids,
            config.rope_theta,
            config.rope_scale,
            config.rope_dims,
            interleaved_mrope=config.interleaved_mrope,
            device=self.device,
        )
        mask = torch.empty(
            len(token_ids), len(token_ids), dtype=hidden.dtype, device=self.device
        ).fill_(torch.finfo(hidden.dtype).min / 4).triu_(1)
        attention = optimized_attention_for_device(self.device, mask=True, small_input=True)

        layer_rows = []
        for layer_index in range(config.num_hidden_layers):
            self.metrics.set_phase(
                "text_encoder_layer", layer=layer_index, total_layers=config.num_hidden_layers
            )
            with materialized_module_construction(), torch.device("meta"):
                layer = TransformerBlock(
                    config,
                    index=layer_index,
                    device=torch.device("meta"),
                    dtype=torch.bfloat16,
                    ops=comfy.ops.disable_weight_init,
                )
            mapping = self._layer_mapping(layer, layer_index)
            load = self.index.load_mapping(
                layer,
                mapping,
                lambda _source, _target: self.device,
                metrics=self.metrics,
                component=f"text_layer_{layer_index:02d}",
            )
            unresolved = meta_tensors(layer)
            if unresolved:
                raise RuntimeError(f"Text layer {layer_index} has unresolved tensors: {unresolved}")
            compute_started = time.perf_counter()
            hidden, _ = layer(
                hidden,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=attention,
                past_key_value=None,
            )
            torch.cuda.synchronize(self.device)
            compute_seconds = time.perf_counter() - compute_started
            layer_rows.append(
                {
                    "layer": layer_index,
                    "weight_bytes": load["bytes"],
                    "compute_seconds": compute_seconds,
                }
            )
            unload_module_to_meta(layer)
            del layer
            if layer_index % 5 == 4:
                gc.collect()

        if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (1, len(token_ids), 5120):
            raise RuntimeError(f"Unexpected layer-50 hidden state: {hidden.dtype} {tuple(hidden.shape)}")
        if not bool(torch.isfinite(hidden).all().item()):
            raise RuntimeError("Text encoder produced non-finite conditioning")
        torch.cuda.empty_cache()
        report = {
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "embedding_source": self.embed_key,
            "embedding_read_seconds": embedding_read_seconds,
            "layers_consumed": config.num_hidden_layers,
            "final_norm_applied": False,
            "dtype": str(hidden.dtype),
            "shape": list(hidden.shape),
            "layers": layer_rows,
        }
        self.metrics.report["text_encoder"] = report
        self.metrics.write()
        return hidden, report
