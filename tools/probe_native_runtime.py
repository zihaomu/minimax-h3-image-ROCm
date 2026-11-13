#!/usr/bin/env python3
"""Probe the pinned ROCm foundation, four-device mapping, and 4K-shape BF16 SDPA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / ".runtime/engines/ComfyUI-image"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def set_phase(run: Path, name: str, **extra: object) -> None:
    atomic_json(run / "phase.json", {"name": name, "time_ns": time.time_ns(), **extra})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def synchronize_all(torch) -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-4k", action="store_true")
    parser.add_argument("--copy-mib", type=int, default=64)
    args = parser.parse_args()
    if not 8 <= args.copy_mib <= 512:
        parser.error("--copy-mib must be between 8 and 512")

    run = Path(os.environ["H3I_RUN_DIR"])
    set_phase(run, "runtime_import")
    os.environ.update(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
            "OCL_SET_SVM_SIZE": "4194304",
        }
    )
    sys.path.insert(0, str(ENGINE))
    sys.argv = [
        str(ENGINE / "main.py"),
        "--disable-all-custom-nodes",
        "--disable-partner-nodes",
        "--disable-assets",
        "--preview-method",
        "none",
        "--use-pytorch-cross-attention",
        "--disable-xformers",
    ]

    import torch
    import comfy.model_management
    from comfy.ldm.minimax.model import MiniMaxH3ImageModel
    from comfy.text_encoders.llama import Qwen3VL_32BConfig

    lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())
    foundation = lock["foundation"]
    if torch.__version__ != foundation["torch"]:
        raise RuntimeError(f"Torch identity mismatch: {torch.__version__}")
    if importlib.metadata.version("triton") != foundation["triton"]:
        raise RuntimeError("Triton identity mismatch")
    if platform.python_version() != foundation["python"]:
        raise RuntimeError("Python identity mismatch")
    if torch.cuda.device_count() not in (1, 4):
        raise RuntimeError(f"Expected one or four visible devices, got {torch.cuda.device_count()}")

    devices = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        architecture = prop.gcnArchName.split(":")[0]
        if architecture != "gfx1201":
            raise RuntimeError(f"Logical GPU{index} is {architecture}, expected gfx1201")
        devices.append(
            {
                "logical_id": index,
                "name": prop.name,
                "architecture": prop.gcnArchName,
                "uuid": str(prop.uuid),
                "pci_bus_id": prop.pci_bus_id,
                "total_memory_bytes": prop.total_memory,
            }
        )

    report = {
        "schema_version": 1,
        "status": "running",
        "started_time_ns": time.time_ns(),
        "physical_devices_requested": os.environ.get("H3I_PHYSICAL_GPUS"),
        "visible_environment": {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "foundation": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": importlib.metadata.version("triton"),
            "hip": torch.version.hip,
        },
        "devices": devices,
        "source": {
            "comfyui_commit": subprocess.check_output(
                ["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True
            ).strip(),
            "files": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    ENGINE / "comfy/ldm/minimax/model.py",
                    ENGINE / "comfy/ldm/minimax/sampling.py",
                    ENGINE / "comfy/ldm/minimax/vae.py",
                    ENGINE / "comfy/text_encoders/minimax.py",
                    ENGINE / "comfy/text_encoders/llama.py",
                )
            },
        },
        "architecture_probe": {
            "dit_class": MiniMaxH3ImageModel.__qualname__,
            "text_hidden_size": Qwen3VL_32BConfig.hidden_size,
            "text_layers_consumed": Qwen3VL_32BConfig.num_hidden_layers,
            "text_final_norm": Qwen3VL_32BConfig.final_norm,
        },
        "copies": [],
        "attention_4k": None,
    }
    atomic_json(run / "native-runtime-probe.json", report)

    set_phase(run, "device_copy_probe", bytes=args.copy_mib * 1024**2)
    elements = args.copy_mib * 1024**2 // 2
    host = torch.zeros(elements, dtype=torch.bfloat16, pin_memory=True)
    resident = []
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
        started = time.perf_counter()
        tensor = host.to(torch.device("cuda", index), non_blocking=True)
        torch.cuda.synchronize(index)
        report["copies"].append(
            {
                "kind": "host_to_device",
                "destination_logical_id": index,
                "bytes": tensor.numel() * tensor.element_size(),
                "seconds": time.perf_counter() - started,
            }
        )
        resident.append(tensor)

    if torch.cuda.device_count() > 1:
        for source in range(torch.cuda.device_count()):
            destination = (source + 1) % torch.cuda.device_count()
            synchronize_all(torch)
            started = time.perf_counter()
            copied = resident[source].to(torch.device("cuda", destination), non_blocking=True)
            torch.cuda.synchronize(destination)
            report["copies"].append(
                {
                    "kind": "device_to_device",
                    "source_logical_id": source,
                    "destination_logical_id": destination,
                    "torch_reports_peer_access": bool(torch.cuda.can_device_access_peer(source, destination)),
                    "bytes": copied.numel() * copied.element_size(),
                    "seconds": time.perf_counter() - started,
                    "transport_qualification": "copy_completed; transport path not inferred from timing",
                }
            )
            del copied
    del resident, host
    synchronize_all(torch)
    atomic_json(run / "native-runtime-probe.json", report)

    if args.attention_4k:
        set_phase(run, "attention_4k_bf16", tokens=16384, heads=56, head_dim=128)
        device = torch.device("cuda", 0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        shape = (1, 56, 16384, 128)
        torch.manual_seed(20261113)
        q = torch.randn(shape, device=device, dtype=torch.bfloat16)
        k = torch.randn(shape, device=device, dtype=torch.bfloat16)
        v = torch.randn(shape, device=device, dtype=torch.bfloat16)
        from torch.nn.attention import SDPBackend, sdpa_kernel

        params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
        eligibility = {
            "flash": bool(torch.backends.cuda.can_use_flash_attention(params, True)),
            "efficient": bool(torch.backends.cuda.can_use_efficient_attention(params, True)),
            "cudnn": bool(torch.backends.cuda.can_use_cudnn_attention(params, True)),
        }
        started = time.perf_counter()
        # Force the only backend accepted for this capacity gate. A missing or
        # failed memory-efficient kernel must not silently materialize S^2 math.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        finite = bool(torch.isfinite(output[:, :, :1]).all().item())
        report["attention_4k"] = {
            "status": "passed" if finite else "failed_non_finite",
            "shape": list(shape),
            "dtype": str(q.dtype),
            "backend_forced": "FLASH_ATTENTION",
            "backend_eligibility": eligibility,
            "seconds": seconds,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "finite_sample": finite,
        }
        if not finite:
            raise RuntimeError("4K-shape BF16 attention produced non-finite output")
        del q, k, v, output
        torch.cuda.empty_cache()

    report["torch_peaks"] = {
        str(index): {
            "allocated_bytes": torch.cuda.max_memory_allocated(index),
            "reserved_bytes": torch.cuda.max_memory_reserved(index),
        }
        for index in range(torch.cuda.device_count())
    }
    report["status"] = "complete"
    report["finished_time_ns"] = time.time_ns()
    set_phase(run, "complete")
    atomic_json(run / "native-runtime-probe.json", report)
    print(json.dumps({"status": report["status"], "devices": devices, "attention_4k": report["attention_4k"]}), flush=True)


if __name__ == "__main__":
    main()
