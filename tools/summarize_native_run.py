#!/usr/bin/env python3
"""Create a compact duration/VRAM/RAM report from one native run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def gib(value):
    return None if value is None else value / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if ROOT / "runs" not in run.parents:
        parser.error("run must be under this project's runs directory")

    result = read(run / "result.json")
    monitor = read(run / "monitor-summary.json")
    execution = read(run / "execution.json")
    hardware = read(run / "hardware.json")
    phases = result.get("phases", [])
    phase_rows = [
        {
            "name": row.get("name"),
            "step": row.get("step"),
            "status": row.get("status"),
            "seconds": row.get("seconds"),
        }
        for row in phases
    ]
    denoise = [row.get("seconds", 0.0) for row in phases if row.get("name") == "denoise_step"]
    report = {
        "schema_version": 1,
        "run": str(run),
        "status": result.get("status", execution.get("status", "unknown")),
        "duration": {
            "launcher_wall_seconds": execution.get("elapsed_seconds"),
            "measured_request_seconds": result.get("request_seconds"),
            "denoise_total_seconds": sum(denoise) if denoise else None,
            "denoise_step_seconds": denoise,
            "phases": phase_rows,
        },
        "gpu": {
            "whole_card_sample_interval_seconds": monitor.get("sampling_interval_seconds"),
            "whole_card_sample_count": monitor.get("sample_count"),
            "physical": {
                key: {
                    **value,
                    "peak_vram_used_gib": gib(value.get("peak_vram_used_bytes")),
                }
                for key, value in monitor.get("gpus", {}).items()
            },
            "torch_logical": result.get("torch_global_peaks", {}),
        },
        "cpu_ram": {
            "peak_process_tree_rss_bytes": monitor.get("peak_process_tree_rss_bytes"),
            "peak_process_tree_rss_gib": gib(monitor.get("peak_process_tree_rss_bytes")),
            "peak_process_tree_sum_vm_hwm_bytes": monitor.get("peak_process_tree_sum_vm_hwm_bytes"),
            "peak_process_tree_sum_vm_hwm_gib": gib(
                monitor.get("peak_process_tree_sum_vm_hwm_bytes")
            ),
            "baseline_host_used_from_available_bytes": monitor.get("baseline_host_memory", {}).get(
                "mem_used_from_available_bytes"
            ),
            "peak_host_used_from_available_bytes": monitor.get(
                "peak_host_used_from_available_bytes"
            ),
            "peak_host_used_delta_from_baseline_bytes": monitor.get(
                "peak_host_used_delta_from_baseline_bytes"
            ),
            "peak_host_used_delta_from_baseline_gib": gib(
                monitor.get("peak_host_used_delta_from_baseline_bytes")
            ),
            "minimum_host_available_bytes": monitor.get("minimum_host_available_bytes"),
            "minimum_swap_free_bytes": monitor.get("minimum_swap_free_bytes"),
            "scope_note": monitor.get("measurement_notes", {}).get("host"),
        },
        "mapping": hardware.get("logical_mapping"),
        "output": result.get("output"),
        "error": result.get("error") or execution.get("error"),
    }
    (run / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Native run: {run.name}",
        "",
        f"- Status: `{report['status']}`",
        f"- Launcher wall: `{report['duration']['launcher_wall_seconds']}` s",
        f"- Measured request: `{report['duration']['measured_request_seconds']}` s",
        f"- Six-step denoise total: `{report['duration']['denoise_total_seconds']}` s",
        "- Whole-card sampled VRAM peaks:",
    ]
    for physical, row in report["gpu"]["physical"].items():
        lines.append(
            f"  - physical GPU{physical}: `{row.get('peak_vram_used_bytes')}` bytes "
            f"(`{row.get('peak_vram_used_gib')}` GiB)"
        )
    ram = report["cpu_ram"]
    lines.extend(
        [
            f"- Process-tree RSS sampled peak: `{ram['peak_process_tree_rss_bytes']}` bytes "
            f"(`{ram['peak_process_tree_rss_gib']}` GiB)",
            f"- Whole-host used-RAM peak delta from launch baseline: "
            f"`{ram['peak_host_used_delta_from_baseline_bytes']}` bytes "
            f"(`{ram['peak_host_used_delta_from_baseline_gib']}` GiB)",
            "",
            "The whole-host delta includes concurrent host activity; the process-tree RSS can count shared pages "
            "more than once. GPU peaks are 0.5 s samples unless the run records another interval.",
            "",
        ]
    )
    (run / "summary.md").write_text("\n".join(lines))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
