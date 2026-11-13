"""Shared bootstrap for the pinned native ComfyUI workflow and smoke check."""
import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / ".runtime/engines/ComfyUI-image"


def bootstrap(run):
    run = Path(run)
    for name in ("input", "output", "temp", "user"):
        (run / name).mkdir(parents=True, exist_ok=True)
    os.environ.update({"HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                       "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1", "OCL_SET_SVM_SIZE": "4194304"})
    sys.path.insert(0, str(ENGINE))
    sys.argv = [str(ENGINE / "main.py"), "--models-directory", str(ROOT / "models"),
                "--input-directory", str(run / "input"), "--output-directory", str(run / "output"),
                "--temp-directory", str(run / "temp"), "--user-directory", str(run / "user"),
                "--cache-none", "--disable-comfy-compiler", "--disable-all-custom-nodes",
                "--disable-partner-nodes", "--disable-assets", "--preview-method", "none",
                "--use-pytorch-cross-attention", "--disable-xformers"]
    import main
    import nodes
    ok = asyncio.run(nodes.load_custom_node(str(ENGINE / "comfy_extras/nodes_minimax_h3.py"), module_parent="comfy_extras"))
    if not ok or "MiniMaxH3ImageConditioning" not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("Native H3-Image conditioning node did not load")
    return main, nodes
