#!/usr/bin/env python3
"""Execute the native ComfyUI API graph and retain first-image diagnostics."""
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

from comfy_runtime import ROOT, bootstrap


def main():
    run = Path(os.environ["H3I_RUN_DIR"])
    workflow = ROOT / "configs/workflows/t2i_2048_api.json"
    prompt = json.loads(workflow.read_text())
    shutil.copy2(workflow, run / "workflow_api.json")
    bootstrap(run)
    import comfy_kitchen as ck
    import comfy.sample
    import comfy.samplers
    import execution
    import latent_preview
    import numpy as np
    from PIL import Image
    from safetensors.torch import save_file
    import torch
    import torch.nn.functional as functional

    rows, artifacts, step_records, sdpa_records = [], [], [], []
    dispatch = Counter()
    current_node = None
    report = {"status": "running", "scope": "instrumented first complete request; not a warm performance baseline",
              "workflow_sha256": hashlib.sha256(workflow.read_bytes()).hexdigest(),
              "node_timings": rows, "artifacts": artifacts, "steps": step_records, "sdpa": sdpa_records}

    def write_report():
        report["kitchen_dispatch_calls"] = dict(dispatch)
        (run / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def capture(name, value):
        tensors = {}

        def walk(item, key):
            if isinstance(item, torch.Tensor):
                tensors[key] = item.detach().to("cpu").contiguous().clone()
            elif isinstance(item, dict):
                for k, v in item.items():
                    walk(v, f"{key}.{k}")
            elif isinstance(item, (tuple, list)):
                for i, v in enumerate(item):
                    walk(v, f"{key}.{i}")

        walk(value, "value")
        if not tensors:
            return
        for key, tensor in tensors.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite tensor: {name}/{key}")
        path = run / f"{name}.safetensors"
        save_file(tensors, str(path))
        artifacts.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype)} for k, t in tensors.items()}})

    class LocalEvents:
        client_id = "h3-image-local"
        last_node_id = None
        sockets_metadata = {}

        def send_sync(self, event, data, sid=None):
            if isinstance(event, str):
                with (run / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps({"time_ns": time.time_ns(), "event": event, "data": data}, default=str) + "\n")
            if event == "execution_cached" and data["nodes"]:
                raise RuntimeError("Unexpected node-result cache hit")

        def queue_updated(self):
            pass

    original_output = execution.get_output_data
    original_noise = comfy.sample.prepare_noise
    original_sigmas = comfy.samplers.calculate_sigmas
    original_callback = latent_preview.prepare_callback
    original_impl = ck.registry.get_implementation
    original_sdpa = functional.scaled_dot_product_attention

    async def measured_output(prompt_id, unique_id, obj, input_data_all, *args, **kwargs):
        nonlocal current_node
        current_node = unique_id
        torch.cuda.synchronize()
        started = time.perf_counter()
        print(f"NODE_START {unique_id} {type(obj).__name__}", flush=True)
        output = await original_output(prompt_id, unique_id, obj, input_data_all, *args, **kwargs)
        torch.cuda.synchronize()
        row = {"node": unique_id, "class_type": prompt[unique_id]["class_type"],
               "seconds": time.perf_counter() - started,
               "torch_allocated_bytes": torch.cuda.memory_allocated(), "torch_reserved_bytes": torch.cuda.memory_reserved(),
               "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "host_rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        rows.append(row)
        print("NODE_COMPLETE " + json.dumps(row), flush=True)
        if unique_id in ("5", "7", "8"):
            capture({"5": "conditioning", "7": "final_latent", "8": "decoded_image"}[unique_id], output[0])
        write_report()
        return output

    def prepare_noise(*args, **kwargs):
        noise = original_noise(*args, **kwargs)
        capture("initial_noise", noise)
        return noise

    def calculate_sigmas(*args, **kwargs):
        sigmas = original_sigmas(*args, **kwargs)
        capture("sigmas", sigmas)
        return sigmas

    def prepare_callback(*args, **kwargs):
        callback = original_callback(*args, **kwargs)

        def captured_callback(step, x0, x, total_steps):
            capture(f"step_{step:02d}", {"x0": x0, "x": x})
            step_records.append({"step": step, "total_steps": total_steps, "time_ns": time.time_ns()})
            write_report()
            return callback(step, x0, x, total_steps)

        return captured_callback

    def implementation(name, backend=None, kwargs=None):
        impl = original_impl(name, backend=backend, kwargs=kwargs)
        module = impl.__module__
        dispatch[f"{name}:{module}"] += 1
        if name.startswith("int8_linear") and module != "comfy_kitchen.backends.hip":
            raise RuntimeError(f"INT8 linear selected unexpected backend: {module}")
        return impl

    def sdpa(*args, **kwargs):
        # Inspect one actual SDPA call per node; retain the profiling overhead label.
        if not any(row["node"] == current_node for row in sdpa_records):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                output = original_sdpa(*args, **kwargs)
            operations = [event.key for event in trace.key_averages() if "attention" in event.key]
            sdpa_records.append({"node": current_node, "q_shape": list(args[0].shape),
                                 "dtype": str(args[0].dtype), "operators": operations})
            print("SDPA " + json.dumps(sdpa_records[-1]), flush=True)
            return output
        return original_sdpa(*args, **kwargs)

    execution.get_output_data = measured_output
    comfy.sample.prepare_noise = prepare_noise
    comfy.samplers.calculate_sigmas = calculate_sigmas
    latent_preview.prepare_callback = prepare_callback
    ck.registry.get_implementation = implementation
    functional.scaled_dot_product_attention = sdpa
    try:
        valid = asyncio.run(execution.validate_prompt("h3-image-first", prompt, None))
        (run / "validation.json").write_text(json.dumps(valid, default=str, indent=2) + "\n")
        if not valid[0]:
            raise RuntimeError(f"Invalid workflow: {valid}")
        executor = execution.PromptExecutor(LocalEvents(), cache_type=execution.CacheType.NONE,
                                           cache_args={"ram": 10, "ram_inactive": 128})
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        executor.execute(prompt, "h3-image-first", {"client_id": "h3-image-local"}, valid[2])
        torch.cuda.synchronize()
        report["request_seconds"] = time.perf_counter() - started
        if not executor.success:
            raise RuntimeError("Native workflow failed; inspect events.jsonl and console.log")
        if [s["step"] for s in step_records] != list(range(6)):
            raise RuntimeError("Did not execute exactly six sampler steps")
        files = sorted((run / "output").glob("*.png"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one PNG, got {len(files)}")
        image = Image.open(files[0]).convert("RGB")
        pixels = np.asarray(image)
        if image.size != (2048, 2048) or pixels.std() < 1:
            raise RuntimeError("Output has wrong dimensions or is nearly constant")
        report.update(status="complete", output={"path": str(files[0].relative_to(run)),
                      "sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(), "size": list(image.size),
                      "pixel_mean": float(pixels.mean()), "pixel_std": float(pixels.std())},
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      host_rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print("IMAGE_COMPLETE " + json.dumps(report["output"]), flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write_report()
        execution.get_output_data = original_output
        comfy.sample.prepare_noise = original_noise
        comfy.samplers.calculate_sigmas = original_sigmas
        latent_preview.prepare_callback = original_callback
        ck.registry.get_implementation = original_impl
        functional.scaled_dot_product_attention = original_sdpa


if __name__ == "__main__":
    main()
