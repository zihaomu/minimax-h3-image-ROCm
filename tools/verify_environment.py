#!/usr/bin/env python3
"""Verify native Image imports and a small HIP ConvRot operation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from comfy_runtime import ROOT, ENGINE, bootstrap


def main():
    run = Path(os.environ["H3I_RUN_DIR"])
    bootstrap(run)
    import comfy_kitchen as ck
    import comfy.memory_management
    import folder_paths
    import torch
    from comfy.ldm.modules import attention
    from comfy_kitchen.backends import hip

    target = json.loads((ROOT / "configs/runtime-target.json").read_text())
    foundation = target["foundation"]
    assert torch.__version__ == foundation["torch"]
    assert importlib.metadata.version("triton") == foundation["triton"]
    assert platform.python_version() == foundation["python"]
    assert torch.cuda.device_count() == 1
    prop = torch.cuda.get_device_properties(0)
    assert prop.gcnArchName.split(":")[0] == "gfx1201"
    assert ck.registry.is_available("hip"), ck.list_backends()
    assert comfy.memory_management.aimdo_enabled, "Expected native DynamicVRAM initialization"
    # Validate a small real ConvRot execution; full model accuracy remains P2/P3 work.
    torch.manual_seed(20261113)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-16, 16, (128, 256), device="cuda", dtype=torch.int8)
    scale = torch.full((128,), 0.01, device="cuda", dtype=torch.float32)
    kwargs = dict(x=x, weight=weight, weight_scale=scale, out_dtype=torch.bfloat16,
                  convrot=True, convrot_groupsize=256)
    assert ck.registry.get_capable_backend("int8_linear", kwargs) == "hip"
    y = ck.int8_linear(**kwargs)
    torch.cuda.synchronize()
    assert y.shape == (64, 128) and torch.isfinite(y).all() and y.abs().max() > 0
    torch.testing.assert_close(torch.arange(16, device="cuda", dtype=torch.float64).sum().cpu(), torch.tensor(120., dtype=torch.float64), rtol=0, atol=0)

    manifest = json.loads((ROOT / "configs/runtime-lock.json").read_text())["models"]
    paths = []
    for file in manifest:
        category, filename = file["path"].split("/", 1)
        actual = Path(folder_paths.get_full_path_or_raise(category, filename)).resolve()
        assert actual == (ROOT / "models" / file["path"]).resolve()
        assert actual.stat().st_size == file["size_bytes"]
        paths.append({"path": file["path"], "sha256": file["sha256"], "size_bytes": file["size_bytes"]})
    hip_libs = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libamdhip64.so" in line}
    assert len(hip_libs) == 1, hip_libs
    extensions = [Path(hip._C.__file__), Path(importlib.metadata.distribution("comfy-aimdo").locate_file("comfy_aimdo/aimdo_rocm.so"))]
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "triton", "torchvision", "comfy-kitchen", "comfy-aimdo", "comfyui-workflow-templates", "comfyui-workflow-templates-json", "transformers", "tokenizers", "safetensors")}
    report = {"passed": True, "foundation": foundation, "versions": versions,
              "comfyui_commit": subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip(),
              "device": {"name": prop.name, "architecture": prop.gcnArchName, "uuid": str(prop.uuid),
                         "pci_bus_id": prop.pci_bus_id, "total_memory": prop.total_memory},
              "backends": ck.list_backends(), "int8_convrot_backend": "hip", "attention": attention.optimized_attention.__name__,
              "dynamic_vram": comfy.memory_management.aimdo_enabled, "cli": sys.argv,
              "models": paths, "hip_libraries": sorted(hip_libs),
              "native_extensions": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in extensions],
              "smoke": {"convrot_shape": list(y.shape), "finite": True, "fp64_sum": 120}}
    (run / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "device": report["device"], "int8_backend": "hip", "attention": report["attention"]}), flush=True)


if __name__ == "__main__":
    main()
