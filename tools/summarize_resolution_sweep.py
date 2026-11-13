#!/usr/bin/env python3
"""Validate matched native-resolution runs and print a publishable JSON summary.

CPU-only: this reads completed local runs and never launches inference.
Raw tensors, images, host paths, UUIDs and process records are not exported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = (1024, 2048, 4096)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    groups = {size: [] for size in RESOLUTIONS}
    common = None
    model_report = None
    seen = set()

    for directory in args.runs:
        run = directory.resolve()
        require(ROOT / "runs" in run.parents and run not in seen, f"Invalid/duplicate run: {run}")
        seen.add(run)
        result = read(run / "result.json")
        summary = read(run / "summary.json")
        execution = read(run / "execution.json")
        hardware = read(run / "hardware.json")
        metadata, request = result["metadata"], result["request"]
        config = metadata["config"]
        size = request["width"]
        require(result["status"] == summary["status"] == execution["status"] == "complete", run.name)
        require(execution["returncode"] == 0, f"Nonzero exit: {run.name}")
        require(size in RESOLUTIONS and request["height"] == size, f"Invalid size: {run.name}")
        require(request["steps"] == 6 and request["cfg"] == 1 and not request["guidance_branch"], run.name)
        require(request["mode"] == "four_static" and len(request["denoise"]) == 6, run.name)
        require(request["vae"]["output_dtype"] == "torch.float32", run.name)
        require(request["text_encoder_stage"]["process_isolation"] is True, run.name)
        require(metadata["numeric_policy"]["quantization"] == "disabled", run.name)
        require(not metadata["numeric_policy"]["vllm_omni"], run.name)
        require(all(row["status"] == "complete" for row in result["phases"]), run.name)
        require(request["attention_policy"]["minimum_sequence_length"] == 4096, run.name)
        expected_flash = 0 if size == 1024 else 300
        require(request["attention_policy"]["strict_flash_calls"] == expected_flash, run.name)
        mapping = [row["physical_id"] for row in summary["mapping"]]
        require(len(mapping) == 4 and 3 not in mapping, run.name)
        require(summary["gpu"]["whole_card_sample_interval_seconds"] == 0.5, run.name)
        require(len(read(run / "postflight.json")) == 4, run.name)
        for row in read(run / "postflight.json"):
            require(not row["external_processes"] and not row["owned_kfd_pids"], run.name)
            require(row["vram_used_bytes"] < 1024**3, f"Unreleased VRAM: {run.name}")
        with (run / "monitor.jsonl").open() as stream:
            for line in stream:
                require(all(not gpu["external_processes"] for gpu in json.loads(line)["gpus"]), run.name)

        environment = {key: value for key, value in metadata["environment"].items() if key != "comfyui"}
        environment["comfyui_commit"] = metadata["environment"]["comfyui"]["commit"]
        identity = {
            "config": {key: value for key, value in config.items() if key not in ("width", "height")},
            "environment": environment,
            "physical_gpu_order": mapping,
            "project_git_base": metadata["project_git_head"],
            "source_files": metadata["project_source_files"],
            "model_verification_sha256": metadata["model_verification_report"]["sha256"],
            "host_total_ram_bytes": hardware["baseline_host_memory"]["mem_total_bytes"],
        }
        if common is None:
            common = identity
            model_report = read(Path(metadata["model_verification_report"]["path"]))
            require(digest(Path(metadata["model_verification_report"]["path"])) == identity["model_verification_sha256"], "Model report changed")
            for source in identity["source_files"]:
                require(digest(ROOT / source["path"]) == source["sha256"], f"Source changed: {source['path']}")
        require(identity == common, f"Unmatched source, configuration, hardware or runtime: {run.name}")

        hashes = {}
        for artifact in request["artifacts"]:
            path = Path(artifact["path"]).resolve()
            require(run in path.parents and digest(path) == artifact["sha256"], run.name)
            hashes[path.name] = artifact["sha256"]
        require(len(hashes) == 8, f"Expected eight tensor artifacts: {run.name}")
        output = result["output"]
        require(output["nonconstant"] and output["width"] == output["height"] == size, run.name)
        require(digest(Path(output["path"])) == output["sha256"], run.name)
        ram = summary["cpu_ram"]
        require(ram["peak_process_tree_rss_bytes"] > 0 and ram["peak_process_tree_sum_vm_hwm_bytes"] > 0, run.name)
        require(ram["peak_host_used_delta_from_baseline_bytes"] is not None, run.name)
        peaks = {key: value["peak_vram_used_bytes"] for key, value in summary["gpu"]["physical"].items()}
        require(len(peaks) == 4 and all(value > 0 for value in peaks.values()), run.name)
        groups[size].append({
            "run_id": run.name,
            "started_at_utc": datetime.fromtimestamp(result["started_time_ns"] / 1e9, timezone.utc).isoformat(),
            "request_seconds": result["request_seconds"],
            "launcher_seconds": execution["elapsed_seconds"],
            "denoise_seconds": summary["duration"]["denoise_total_seconds"],
            "phase_seconds": {name: sum(row["seconds"] for row in result["phases"] if row["name"] == name)
                              for name in ("text_encode_isolated_process", "dit_load", "denoise_step", "vae_load", "vae_decode", "save_png")},
            "peak_vram_bytes_by_physical_gpu": peaks,
            "peak_process_tree_rss_bytes": ram["peak_process_tree_rss_bytes"],
            "peak_process_tree_sum_vm_hwm_bytes": ram["peak_process_tree_sum_vm_hwm_bytes"],
            "peak_host_used_delta_bytes": ram["peak_host_used_delta_from_baseline_bytes"],
            "strict_flash_dit_calls": expected_flash,
            "tensor_sha256": hashes,
            "png_sha256": output["sha256"],
            "evidence_sha256": {name: digest(run / name) for name in ("result.json", "summary.json", "execution.json", "monitor-summary.json", "monitor.jsonl", "postflight.json")},
        })

    resolutions = []
    for size, samples in groups.items():
        require(len(samples) >= 2, f"Need at least two samples at {size}")
        require(all(row["tensor_sha256"] == samples[0]["tensor_sha256"] and row["png_sha256"] == samples[0]["png_sha256"] for row in samples), f"A/A mismatch: {size}")
        resolutions.append({
            "width": size, "height": size, "sample_count": len(samples),
            "request_seconds": stats([row["request_seconds"] for row in samples]),
            "launcher_seconds": stats([row["launcher_seconds"] for row in samples]),
            "denoise_seconds": stats([row["denoise_seconds"] for row in samples]),
            "max_single_gpu_vram_gib": max(max(row["peak_vram_bytes_by_physical_gpu"].values()) for row in samples) / 1024**3,
            "max_process_tree_rss_gib": max(row["peak_process_tree_rss_bytes"] for row in samples) / 1024**3,
            "max_process_tree_sum_vm_hwm_gib": max(row["peak_process_tree_sum_vm_hwm_bytes"] for row in samples) / 1024**3,
            "max_host_used_delta_gib": max(row["peak_host_used_delta_bytes"] for row in samples) / 1024**3,
            "aa_tensor_and_png_exact": True,
            "samples": samples,
        })
    sources = sorted({(row["repository"], row["revision"]) for row in model_report["files"]})
    print(json.dumps({
        "schema_version": 1,
        "suite": "R9700 original-precision matched resolution sweep",
        "generator": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))},
        "measurement": {
            "batch_size": 1, "monitor_interval_seconds": 0.5,
            "request_scope": "pipeline including text encoding, weight loading, six steps, VAE, PNG and cleanup; excludes process/container bootstrap",
            "cache_policy": "fresh process per sample; OS file and kernel caches may be warm; no conditioning reuse",
            "memory_scope": "whole-card VRAM samples; summed process RSS/HWM; whole-host delta includes unrelated activity and page cache",
            "timing_summary": "arithmetic mean and observed range; no throughput or statistical confidence claim",
        },
        "identity": common,
        "model_sources": [{"repository": repo, "revision": revision} for repo, revision in sources],
        "resolutions": resolutions,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
