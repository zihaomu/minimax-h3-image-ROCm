#!/usr/bin/env python3
"""Encode one H3-Image prompt in an isolated GPU process and save BF16 conditioning."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimax_h3_image_rocm.bootstrap import configure_comfy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--stage-run", type=Path, required=True)
    args = parser.parse_args()

    outer_run = Path(os.environ["H3I_RUN_DIR"]).resolve()
    request_path = args.request.resolve()
    stage_run = args.stage_run.resolve()
    expected_stage = outer_run / "stages/text-encoder"
    if stage_run != expected_stage or request_path.parent != stage_run:
        raise RuntimeError("The isolated text stage must stay under the owned run directory")
    request = json.loads(request_path.read_text())
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("The isolated text stage requires one non-empty prompt")

    configure_comfy()
    import torch

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.device_count() not in (1, 4):
        raise RuntimeError(
            "The isolated text stage requires the same one- or four-device visibility as the request"
        )

    from minimax_h3_image_rocm.metrics import RunMetrics
    from minimax_h3_image_rocm.pipeline import tensor_artifact
    from minimax_h3_image_rocm.text_encoder import StreamingTextEncoder

    metrics = RunMetrics(
        stage_run,
        torch,
        {
            "purpose": "process-isolated original-BF16 H3-Image text encoding",
            "outer_run": str(outer_run),
            "physical_gpus": os.environ.get("H3I_PHYSICAL_GPUS", "").split(","),
            "quantization": "disabled",
            "output_residency": "CPU BF16 safetensors artifact",
        },
    )
    encoder = None
    context = None
    try:
        with metrics.phase("text_encode", prompt_characters=len(prompt)):
            encoder = StreamingTextEncoder(
                ROOT / "models/native/minimax-h3/text_encoder",
                ROOT / "models/native/minimax-h3-image/tokenizer",
                torch.device("cuda", 0),
                metrics,
            )
            context, text_report = encoder.encode(prompt)
            del encoder
            encoder = None

        with metrics.phase("save_conditioning_cpu_artifact"):
            artifact = tensor_artifact(
                metrics,
                stage_run / "conditioning.safetensors",
                context,
                "isolated_conditioning",
            )
        del context
        context = None
        gc.collect()
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
        cleanup = {str(index): torch.cuda.memory_allocated(index) for index in range(torch.cuda.device_count())}
        metrics.finish(text_encoder=text_report, conditioning=artifact, cleanup=cleanup)
        print(json.dumps({"event": "conditioning_complete", "artifact": artifact}), flush=True)
    except BaseException as error:
        if context is not None:
            del context
        if encoder is not None:
            del encoder
        gc.collect()
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
        metrics.fail(error)
        (stage_run / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
