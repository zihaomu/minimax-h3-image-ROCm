#!/usr/bin/env python3
"""Build the pinned upstream HIP extensions in the project foundation venv."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "comfy-kitchen": "cbc44a5b7cc3b8f910ef76ae5251c94971eb05d1",
    "comfy-aimdo": "68088de9dce9f5e32231fccf230938769c5562da",
}


def main():
    run = Path(os.environ["H3I_RUN_DIR"])
    target = json.loads((ROOT / "configs/runtime-target.json").read_text())
    if os.environ.get("DEEPINFER_IMAGE_ID") != target["foundation"]["image_id"]:
        raise RuntimeError("Use the project ROCm 10 launcher")
    for name, expected in SOURCES.items():
        source = ROOT / ".runtime/sources" / name
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if head != expected:
            raise RuntimeError(f"Unexpected {name} source: {head}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-c", str(ROOT / "configs/requirements-image.lock"),
                    "nanobind>=2.0.0", "cmake>=3.26", "setuptools-scm>=8"], check=True)
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
           "COMFY_KITCHEN_BUILD_HIP": "1", "COMFY_HIP_ARCHS": "gfx1201"}
    kitchen = ROOT / ".runtime/sources/comfy-kitchen"
    # Upstream setup.py uses os.cpu_count for CMake parallelism; bound this build to eight workers.
    subprocess.run([sys.executable, "-c", "import os,runpy; os.cpu_count=lambda:8; runpy.run_path('setup.py',run_name='__main__')",
                    "--hip", "--no-cuda", "bdist_wheel"], cwd=kitchen, env=env, check=True)
    aimdo = ROOT / ".runtime/sources/comfy-aimdo"
    subprocess.run(["bash", "scripts/build-linux-aimdo.sh"], cwd=aimdo, env=env, check=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", "dist", "."],
                   cwd=aimdo, env=env, check=True)
    wheels = [next((ROOT / ".runtime/sources" / name / "dist").glob("*.whl")) for name in SOURCES]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", *map(str, wheels)], check=True)
    report = {"sources": SOURCES, "gpu_arch": "gfx1201", "image_id": os.environ["DEEPINFER_IMAGE_ID"],
              "build_dependencies": {name: importlib.metadata.version(name) for name in ("nanobind", "cmake", "ninja", "setuptools-scm", "setuptools", "wheel")},
              "wheels": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in wheels]}
    (run / "native-build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
