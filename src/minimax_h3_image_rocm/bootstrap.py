"""Configure the fixed ComfyUI source tree as an architecture/kernel dependency."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / ".runtime/engines/ComfyUI-image"


def configure_comfy() -> None:
    """Set deterministic CLI policy before the first ComfyUI module import."""
    os.environ.update(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
            "OCL_SET_SVM_SIZE": "4194304",
        }
    )
    engine = str(ENGINE)
    if engine not in sys.path:
        sys.path.insert(0, engine)
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


_SDPA_GUARD = None


def require_flash_sdpa(minimum_sequence_length: int = 4096):
    """Make large Comfy PyTorch-SDPA calls fail unless FLASH_ATTENTION runs.

    H3-Image's 2K/4K DiT sequences cannot safely fall back to the quadratic
    math backend. Short text-encoder and tiled-VAE calls retain PyTorch's
    ordinary dispatcher because the released FP32 VAE is not the large-memory
    risk this guard addresses. The selected policy and call shapes are exposed
    through :func:`attention_policy_report`.
    """
    import torch
    import comfy.ops
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original = torch.nn.functional.scaled_dot_product_attention

    calls = {"minimum_sequence_length": minimum_sequence_length, "strict_flash_calls": 0,
             "ordinary_calls": 0, "strict_shapes": [], "ordinary_shapes": []}

    def guarded_sdpa(query, key, value, *args, **kwargs):
        shape = {"q": list(query.shape), "k": list(key.shape), "dtype": str(query.dtype)}
        strict = max(query.shape[-2], key.shape[-2]) >= minimum_sequence_length
        if strict:
            calls["strict_flash_calls"] += 1
            if len(calls["strict_shapes"]) < 16:
                calls["strict_shapes"].append(shape)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return original(query, key, value, *args, **kwargs)
        calls["ordinary_calls"] += 1
        if len(calls["ordinary_shapes"]) < 16:
            calls["ordinary_shapes"].append(shape)
        return original(query, key, value, *args, **kwargs)

    comfy.ops.scaled_dot_product_attention = guarded_sdpa
    global _SDPA_GUARD
    _SDPA_GUARD = calls
    return guarded_sdpa


def attention_policy_report() -> dict:
    return dict(_SDPA_GUARD or {})


@contextmanager
def materialized_module_construction():
    """Disable Comfy's AIMDO lazy-``None`` parameters while creating meta modules.

    The independent loader assigns every registered parameter itself. It needs
    ordinary meta tensors with shapes and dtypes, not Comfy's ModelPatcher
    file-backed placeholders.
    """
    import comfy.memory_management

    prior = comfy.memory_management.aimdo_enabled
    comfy.memory_management.aimdo_enabled = False
    try:
        yield
    finally:
        comfy.memory_management.aimdo_enabled = prior
