#!/usr/bin/env python3
"""Prepare the project venv inside the locked foundation container."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / ".runtime/engines/ComfyUI-image"
VENV = ROOT / ".runtime/venv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    run = Path(os.environ["H3I_RUN_DIR"])
    target = json.loads((ROOT / "configs/runtime-target.json").read_text())
    if os.environ.get("DEEPINFER_IMAGE_ID") != target["foundation"]["image_id"]:
        raise RuntimeError("Use tools/run_rocm10.py with the locked foundation")
    head = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip()
    if head != target["application"]["comfyui_candidate_commit"]:
        raise RuntimeError("Unexpected ComfyUI checkout")
    if not (VENV / "bin/python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        site = VENV / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        (site / "foundation.pth").write_text(sysconfig.get_path("purelib") + "\n")
    constraints = run / "foundation-constraints.txt"
    protected = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                 if d.metadata["Name"].lower() in ("torch", "triton", "torchvision")
                 or "rocm" in d.metadata["Name"].lower()}
    constraints.write_text("".join(f"{k}=={v}\n" for k, v in sorted(protected.items())))
    python = str(VENV / "bin/python")
    pip = [python, "-m", "pip", "install", "-c", str(constraints), "-r", str(ENGINE / "requirements.txt")]
    overlay_lock = ROOT / "configs/requirements-image.lock"
    if overlay_lock.exists():
        pip.extend(["-r", str(overlay_lock)])
    report = run / "dependency-plan.json"
    subprocess.run([*pip, "--dry-run", "--report", str(report)], check=True)
    planned = json.loads(report.read_text())["install"]
    for item in planned:
        name = item["metadata"]["name"]
        if name.lower() in {k.lower() for k in protected}:
            raise RuntimeError(f"Dependency resolver would replace protected package: {name}")
    print("Project installs:", [(x["metadata"]["name"], x["metadata"]["version"]) for x in planned], flush=True)
    if args.install:
        # Install exactly the artifacts reviewed above, with dependency resolution already done.
        wheels = [item["download_info"]["url"] for item in planned]
        if wheels:
            subprocess.run([python, "-m", "pip", "install", "--no-deps", *wheels], check=True)
        installed = json.loads(subprocess.check_output([python, "-m", "pip", "list", "--format=json"], text=True))
        actual = {x["name"].lower(): x["version"] for x in installed}
        for name, version in protected.items():
            if actual.get(name.lower()) != version:
                raise RuntimeError(f"Foundation package changed: {name}")
        (run / "installed-packages.json").write_text(json.dumps(installed, indent=2) + "\n")
        freeze = subprocess.check_output([python, "-m", "pip", "freeze", "--all"], text=True)
        (run / "requirements-resolved.txt").write_text(freeze)
        result = subprocess.run([python, "-m", "pip", "check"], text=True, capture_output=True)
        (run / "pip-check.txt").write_text(result.stdout + result.stderr)
        print("pip check returncode:", result.returncode, flush=True)


if __name__ == "__main__":
    main()
