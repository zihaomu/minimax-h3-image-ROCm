"""Exact safetensors loading, byte accounting, and contiguous DiT placement."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import time
from typing import Callable

import torch
from safetensors import safe_open


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int


def _header(path: Path) -> dict:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size < 64 * 1024 * 1024:
            raise RuntimeError(f"Invalid safetensors header length in {path}: {size}")
        return json.loads(stream.read(size))


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _resolve(module: torch.nn.Module, name: str):
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf in parent._parameters:
        return parent, leaf, parent._parameters[leaf], "parameter"
    if leaf in parent._buffers:
        return parent, leaf, parent._buffers[leaf], "buffer"
    raise KeyError(f"Target tensor is not registered: {name}")


def assign_tensor(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    parent, leaf, prior, kind = _resolve(module, name)
    if prior is not None:
        if tuple(prior.shape) != tuple(tensor.shape):
            raise RuntimeError(f"Shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(prior.shape)}")
        if prior.dtype != tensor.dtype:
            raise RuntimeError(f"Dtype mismatch for {name}: {tensor.dtype} != {prior.dtype}")
    if kind == "parameter":
        parent._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        parent._buffers[leaf] = tensor


def raw_head_major_qkv_to_component_major(
    tensor: torch.Tensor, heads: int = 56, head_dim: int = 128
) -> torch.Tensor:
    """Convert the released raw H3 QKV rows to Comfy's Q|K|V layout.

    The original checkpoint packs rows as ``[head, qkv, channel]``.  Comfy's
    fused projection splits its output into three contiguous tensors and hence
    consumes ``[qkv, head, channel]``.  This is a pure row permutation; dtype,
    values and shape are otherwise unchanged.
    """
    if tensor.ndim != 2 or tensor.shape[0] != 3 * heads * head_dim:
        raise RuntimeError(
            f"Unexpected raw QKV shape {tuple(tensor.shape)} for heads={heads}, head_dim={head_dim}"
        )
    return (
        tensor.reshape(heads, 3, head_dim, tensor.shape[1])
        .permute(1, 0, 2, 3)
        .reshape(tensor.shape)
        .contiguous()
    )


class SafetensorIndex:
    def __init__(self, directory: Path, index_name: str = "model.safetensors.index.json"):
        self.directory = Path(directory)
        payload = json.loads((self.directory / index_name).read_text())
        self.weight_map: dict[str, str] = payload["weight_map"]
        self.declared_total_bytes = int(payload.get("metadata", {}).get("total_size", 0))
        self.specs: dict[str, TensorSpec] = {}
        for shard in sorted(set(self.weight_map.values())):
            path = self.directory / shard
            if not path.is_file():
                # A partial text-encoder download intentionally omits shards that
                # only hold layers after the Image model's layer-50 read point.
                continue
            header = _header(path)
            for name, row in header.items():
                if name == "__metadata__":
                    continue
                shape = tuple(int(value) for value in row["shape"])
                dtype = row["dtype"]
                self.specs[name] = TensorSpec(
                    name=name,
                    shard=shard,
                    dtype=dtype,
                    shape=shape,
                    bytes=_numel(shape) * DTYPE_BYTES[dtype],
                )
        missing_local = sorted(name for name, shard in self.weight_map.items() if (self.directory / shard).is_file() and name not in self.specs)
        if missing_local:
            raise RuntimeError(f"Index/header disagreement in {self.directory}: {missing_local[:5]}")

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(name for name in self.specs if name.startswith(prefix))

    def bytes_for(self, keys) -> int:
        return sum(self.specs[name].bytes for name in keys)

    def select_rows_cpu(self, name: str, indices: torch.Tensor) -> torch.Tensor:
        spec = self.specs[name]
        with safe_open(self.directory / spec.shard, framework="pt", device="cpu") as tensors:
            return tensors.get_tensor(name).index_select(0, indices.to("cpu")).clone()

    def load_mapping(
        self,
        target: torch.nn.Module,
        mapping: dict[str, str],
        device_for: Callable[[str, str], torch.device],
        metrics=None,
        component: str = "weights",
        tensor_transform: Callable[[str, str, torch.Tensor], torch.Tensor] | None = None,
    ) -> dict:
        unknown = sorted(set(mapping) - set(self.specs))
        if unknown:
            raise RuntimeError(f"Requested tensors are absent from downloaded shards: {unknown[:10]}")
        grouped = defaultdict(list)
        for source_name, target_name in mapping.items():
            grouped[self.specs[source_name].shard].append((source_name, target_name))

        total_bytes = 0
        transformed = []
        rows = []
        for shard in sorted(grouped):
            destinations = set()
            shard_bytes = 0
            started = time.perf_counter()
            with safe_open(self.directory / shard, framework="pt", device="cpu") as tensors:
                for source_name, target_name in grouped[shard]:
                    source = tensors.get_tensor(source_name)
                    prepared = (
                        tensor_transform(source_name, target_name, source)
                        if tensor_transform is not None
                        else source
                    )
                    if prepared is not source:
                        if prepared.shape != source.shape or prepared.dtype != source.dtype:
                            raise RuntimeError(
                                f"Tensor transform changed shape/dtype for {source_name}: "
                                f"{tuple(source.shape)}/{source.dtype} -> {tuple(prepared.shape)}/{prepared.dtype}"
                            )
                        transformed.append(source_name)
                    device = torch.device(device_for(source_name, target_name))
                    moved = prepared.to(device=device, non_blocking=False)
                    assign_tensor(target, target_name, moved)
                    shard_bytes += moved.numel() * moved.element_size()
                    destinations.add(device.index)
                    del source, prepared, moved
            for index in sorted(destinations):
                torch.cuda.synchronize(index)
            seconds = time.perf_counter() - started
            total_bytes += shard_bytes
            row = {
                "shard": shard,
                "tensor_count": len(grouped[shard]),
                "bytes": shard_bytes,
                "logical_devices": sorted(destinations),
                "seconds": seconds,
            }
            rows.append(row)
            if metrics is not None:
                metrics.transfer(
                    "host_to_device_weights",
                    shard_bytes,
                    seconds,
                    component=component,
                    shard=shard,
                    logical_devices=sorted(destinations),
                )
        return {
            "bytes": total_bytes,
            "shards": rows,
            "tensor_count": len(mapping),
            "transformed_tensor_count": len(transformed),
            "transformed_tensors": sorted(transformed),
        }


def load_single_safetensors(
    target: torch.nn.Module,
    path: Path,
    device: torch.device,
    metrics=None,
    component: str = "weights",
) -> dict:
    expected = set(target.state_dict().keys())
    header = _header(path)
    source_keys = {name for name in header if name != "__metadata__"}
    if source_keys != expected:
        missing = sorted(expected - source_keys)
        extra = sorted(source_keys - expected)
        raise RuntimeError(f"State mapping mismatch for {path}: missing={missing[:10]} extra={extra[:10]}")
    started = time.perf_counter()
    byte_count = 0
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for name in sorted(source_keys):
            source = tensors.get_tensor(name)
            moved = source.to(device=device, non_blocking=False)
            assign_tensor(target, name, moved)
            byte_count += moved.numel() * moved.element_size()
            del source, moved
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if metrics is not None:
        metrics.transfer(
            "host_to_device_weights",
            byte_count,
            seconds,
            component=component,
            shard=path.name,
            logical_devices=[device.index],
        )
    return {"bytes": byte_count, "seconds": seconds, "tensor_count": len(source_keys)}


def load_absolute_pdd_heads(target: torch.nn.Module, path: Path, device: torch.device, metrics=None) -> dict:
    """Load the six absolute official heads into Comfy's stacked head storage.

    Comfy's generic PDD path stores one base head plus offsets so it can merge
    skipped intervals. H3-Image's ``pdd_fused_heads`` instead contains six
    already-fused absolute heads, selected one-for-one by the official step
    index. We preserve those values and select them directly in our forward.
    """
    started = time.perf_counter()
    byte_count = 0
    with safe_open(path, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("video_out.weight")
        bias = tensors.get_tensor("video_out.bias")
        if tuple(weight.shape[:2]) != (6, 96) or tuple(bias.shape) != (6, 96):
            raise RuntimeError(f"Unexpected PDD head shapes: {tuple(weight.shape)}, {tuple(bias.shape)}")
        weight = weight.flatten(0, 1).to(device=device, non_blocking=False)
        bias = bias.flatten(0, 1).to(device=device, non_blocking=False)
        assign_tensor(target, "final_layer.video_out.weight", weight)
        assign_tensor(target, "final_layer.video_out.bias", bias)
        byte_count = weight.numel() * weight.element_size() + bias.numel() * bias.element_size()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if metrics is not None:
        metrics.transfer(
            "host_to_device_weights",
            byte_count,
            seconds,
            component="dit_pdd_absolute_heads",
            shard=path.name,
            logical_devices=[device.index],
        )
    return {"bytes": byte_count, "seconds": seconds, "head_count": 6}


def unload_module_to_meta(module: torch.nn.Module) -> int:
    released = 0
    for name, parameter in list(module.named_parameters(recurse=True)):
        if not parameter.is_meta:
            released += parameter.numel() * parameter.element_size()
        assign_tensor(module, name, torch.empty(parameter.shape, dtype=parameter.dtype, device="meta"))
    for name, buffer in list(module.named_buffers(recurse=True)):
        if buffer is not None and not buffer.is_meta:
            released += buffer.numel() * buffer.element_size()
            assign_tensor(module, name, torch.empty(buffer.shape, dtype=buffer.dtype, device="meta"))
    return released


def meta_tensors(module: torch.nn.Module) -> list[str]:
    rows = [name for name, value in module.named_parameters() if value.is_meta]
    rows.extend(name for name, value in module.named_buffers() if value is not None and value.is_meta)
    return sorted(rows)


def plan_dit(index: SafetensorIndex, device_count: int = 4, pdd_bytes: int = 12_388_608) -> dict:
    if device_count != 4:
        raise ValueError("The current native placement gate is defined for four devices")
    block_re = re.compile(r"^blocks\.(\d+)\.")
    block_bytes = [0] * 50
    front_bytes = 0
    final_bytes = pdd_bytes
    skipped_raw_head_bytes = 0
    for name, spec in index.specs.items():
        match = block_re.match(name)
        if match:
            block_bytes[int(match.group(1))] += spec.bytes
        elif name.startswith("final_layer.video_out."):
            skipped_raw_head_bytes += spec.bytes
        elif name.startswith("final_layer."):
            final_bytes += spec.bytes
        else:
            front_bytes += spec.bytes
    if not all(block_bytes):
        raise RuntimeError("Could not account for all 50 DiT blocks")

    prefix = [0]
    for value in block_bytes:
        prefix.append(prefix[-1] + value)
    best = None
    for first in range(1, 48):
        for second in range(first + 1, 49):
            for third in range(second + 1, 50):
                loads = [
                    front_bytes + prefix[first],
                    prefix[second] - prefix[first],
                    prefix[third] - prefix[second],
                    prefix[50] - prefix[third] + final_bytes,
                ]
                score = (max(loads), max(loads) - min(loads), sum(abs(loads[i] - loads[i - 1]) for i in range(1, 4)))
                if best is None or score < best[0]:
                    best = (score, (first, second, third), loads)
    boundaries = (0, *best[1], 50)
    ranges = [
        {"logical_device": index_, "start": boundaries[index_], "stop": boundaries[index_ + 1], "bytes": best[2][index_]}
        for index_ in range(4)
    ]
    return {
        "ranges": ranges,
        "front_bytes": front_bytes,
        "final_and_pdd_bytes": final_bytes,
        "skipped_raw_output_head_bytes": skipped_raw_head_bytes,
        "block_bytes": block_bytes,
        "maximum_device_weight_bytes": max(best[2]),
    }


def block_device(plan: dict, block_index: int) -> int:
    for row in plan["ranges"]:
        if row["start"] <= block_index < row["stop"]:
            return int(row["logical_device"])
    raise IndexError(block_index)
