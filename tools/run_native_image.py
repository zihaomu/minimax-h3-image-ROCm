#!/usr/bin/env python3
"""Run one measured original-precision H3-Image qualification request."""
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
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimax_h3_image_rocm.bootstrap import ENGINE, configure_comfy, require_flash_sdpa


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path) -> dict:
    return {
        "path": str(path),
        "commit": subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def validate_environment(torch) -> dict:
    lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())
    expected = lock["foundation"]
    actual = {
        "image_id": os.environ.get("DEEPINFER_IMAGE_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "hip_build": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "physical_gpus": os.environ.get("H3I_PHYSICAL_GPUS", "").split(","),
    }
    checks = {
        "image_id": expected["image_id"],
        "python": expected["python"],
        "torch": expected["torch"],
        "triton": expected["triton"],
        "hip_build": expected["hip_build"],
    }
    mismatches = {key: (actual[key], value) for key, value in checks.items() if actual[key] != value}
    if mismatches:
        raise RuntimeError(f"Foundation identity mismatch: {mismatches}")
    expected_commit = lock["application"]["comfyui_commit"]
    comfy = source_identity(ENGINE)
    if comfy["commit"] != expected_commit:
        raise RuntimeError(f"ComfyUI commit mismatch: {comfy['commit']} != {expected_commit}")
    actual["comfyui"] = comfy
    actual["devices"] = [
        {
            "logical_id": index,
            "name": torch.cuda.get_device_name(index),
            "architecture": torch.cuda.get_device_properties(index).gcnArchName,
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]
    if any(row["architecture"].split(":", 1)[0] != "gfx1201" for row in actual["devices"]):
        raise RuntimeError(f"A visible device is not gfx1201: {actual['devices']}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single-offload", "four-static"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native_t2i.json")
    parser.add_argument("--width", type=int, choices=(1024, 2048, 4096))
    parser.add_argument("--height", type=int, choices=(1024, 2048, 4096))
    parser.add_argument("--prompt", help="explicit qualification prompt override")
    parser.add_argument("--seed", type=int, help="explicit non-negative qualification seed override")
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()

    run = Path(os.environ["H3I_RUN_DIR"]).resolve()
    if ROOT / "runs" not in run.parents:
        raise RuntimeError("H3I_RUN_DIR must be under this project's runs directory")
    config = json.loads(args.config.read_text())
    if args.width is not None:
        config["width"] = args.width
    if args.height is not None:
        config["height"] = args.height
    if args.prompt is not None:
        if not args.prompt.strip():
            parser.error("--prompt must not be empty")
        config["prompt"] = args.prompt
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        config["seed"] = args.seed
    reference = args.reference_run.resolve() if args.reference_run else None
    if reference is not None and ROOT / "runs" not in reference.parents:
        raise RuntimeError("The comparison reference must be under this project's runs directory")

    configure_comfy()
    import torch

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    environment = validate_environment(torch)
    require_flash_sdpa()

    from minimax_h3_image_rocm.metrics import RunMetrics
    from minimax_h3_image_rocm.pipeline import NativeImagePipeline

    internal_mode = args.mode.replace("-", "_")
    project_sources = [
        ROOT / "tools/run_native_image.py",
        ROOT / "tools/encode_native_conditioning.py",
        ROOT / "tools/run_rocm10.py",
        ROOT / "configs/native_t2i.json",
        *sorted((ROOT / "src/minimax_h3_image_rocm").glob("*.py")),
    ]
    verification = ROOT / "models/.download/native/verification.json"
    metadata = {
        "environment": environment,
        "mode": internal_mode,
        "config_path": str(args.config.resolve()),
        "config": config,
        "reference_run": str(reference) if reference else None,
        "project_git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "project_source_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in project_sources
        ],
        "model_verification_report": {
            "path": str(verification),
            "sha256": sha256(verification),
        },
        "numeric_policy": {
            "quantization": "disabled",
            "cpu_offload": (
                "text encoder is layer-streamed in an isolated child process and conditioning "
                "crosses a bit-preserving CPU artifact/process boundary in both modes; DiT is "
                "layer-streamed only in single_offload"
            ),
            "silent_attention_fallback": False,
            "comfy_graph_runtime": False,
            "vllm_omni": False,
        },
    }
    metrics = RunMetrics(run, torch, metadata)
    pipeline = NativeImagePipeline(ROOT, run, config, internal_mode, metrics)
    try:
        result = pipeline.run_request(reference)
        print(json.dumps({"event": "request_complete", "output": result["output"]}), flush=True)
    except BaseException as error:
        metrics.fail(error)
        (run / "traceback.txt").write_text(traceback.format_exc())
        print(json.dumps({"event": "request_failed", "error": repr(error)}), flush=True)
        raise


if __name__ == "__main__":
    main()
