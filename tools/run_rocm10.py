#!/usr/bin/env python3
"""Run one owned job on reserved R9700 devices through the ROCm 10.1 foundation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LEASES = WORKSPACE / "deepinfer_workspace/artifacts/amd/rocm10"
GPU_LOCKS = WORKSPACE / "minimax_h3/.standard-opt-state/locks"
ALLOWED_GPUS = frozenset((0, 1, 2, 4, 5, 6, 7))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def reservations():
    rows = []
    for path in LEASES.glob("gpu-reservation*.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "active":
            rows.append(row)
    return rows


def device(index):
    inventory = json.loads(
        subprocess.check_output(["rocm-smi", "--showbus", "--showuniqueid", "--json"], text=True)
    )
    row = inventory[f"card{index}"]
    pci = row["PCI Bus"].lower()
    unique_id = int(row["Unique ID"], 16)
    uuid = f"GPU-{unique_id:016x}"
    for node in Path("/sys/class/kfd/kfd/topology/nodes").iterdir():
        props = dict(line.split(maxsplit=1) for line in (node / "properties").read_text().splitlines())
        if int(props.get("unique_id", "0")) == unique_id:
            if props.get("gfx_target_version") != "120001":
                raise RuntimeError("Selected device is not gfx1201")
            pci_path = Path("/sys/bus/pci/devices") / pci
            total_path = pci_path / "mem_info_vram_total"
            return {
                "physical_id": index,
                "pci": pci,
                "uuid": uuid,
                "kfd_gpu_id": int((node / "gpu_id").read_text()),
                "vram_total_bytes": int(total_path.read_text()) if total_path.exists() else None,
            }
    raise RuntimeError("Could not match ROCm device to KFD identity")


def snapshot(gpu, own_pids=(), own_container=None):
    """Take a whole-device sample and classify KFD users by ownership.

    Keep this single-device interface stable because the ownership regression
    fixtures exercise it without real hardware.
    """
    gid = gpu["kfd_gpu_id"]
    processes, owned = [], []
    own_pids = set(own_pids)
    kfd_root = Path("/sys/class/kfd/kfd/proc")
    for folder in kfd_root.iterdir():
        try:
            pid = int(folder.name)
            cgroup = Path(f"/proc/{pid}/cgroup").read_text()
            if pid in own_pids or (own_container and own_container in cgroup):
                owned.append(pid)
                continue
            queues = any(int(path.read_text()) == gid for path in (folder / "queues").glob("*/gpuid"))
            memory = folder / f"vram_{gid}"
            allocated = int(memory.read_text()) if memory.exists() else 0
            if queues or allocated:
                processes.append(
                    {
                        "pid": pid,
                        "queues": queues,
                        "vram_bytes": allocated,
                        "cgroup": cgroup.strip(),
                    }
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
    pci = Path("/sys/bus/pci/devices") / gpu["pci"]
    return {
        "time_ns": time.time_ns(),
        "physical_id": gpu.get("physical_id"),
        "pci": gpu["pci"],
        "external_processes": processes,
        "owned_kfd_pids": sorted(owned),
        "vram_used_bytes": int((pci / "mem_info_vram_used").read_text()),
        "vram_total_bytes": gpu.get("vram_total_bytes"),
        "gpu_busy_percent": int((pci / "gpu_busy_percent").read_text()),
    }


def container_pids(cid_file):
    if not cid_file.exists():
        return []
    cid = cid_file.read_text().strip()
    result = subprocess.run(
        ["docker", "top", cid, "-eo", "pid"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return [int(value.strip()) for value in result.stdout.splitlines()[1:] if value.strip().isdigit()]


def process_memory(pids):
    rows = []
    for pid in sorted(set(pids)):
        try:
            status = {}
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    status[key] = value.strip()
            rss_kib = int(status.get("VmRSS", "0 kB").split()[0])
            hwm_kib = int(status.get("VmHWM", "0 kB").split()[0])
            rows.append(
                {
                    "pid": pid,
                    "name": status.get("Name"),
                    "rss_bytes": rss_kib * 1024,
                    "vm_hwm_bytes": hwm_kib * 1024,
                }
            )
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
    return {
        "pid_count": len(rows),
        "rss_bytes": sum(row["rss_bytes"] for row in rows),
        "sum_vm_hwm_bytes": sum(row["vm_hwm_bytes"] for row in rows),
        "processes": rows,
    }


def host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.strip().split()[0]) * 1024
    total = values["MemTotal"]
    available = values["MemAvailable"]
    return {
        "mem_total_bytes": total,
        "mem_available_bytes": available,
        "mem_used_from_available_bytes": total - available,
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def read_phase(run):
    path = run / "phase.json"
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {"value": value}
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def monitor_sample(gpus, pids, cid, run, elapsed):
    return {
        "time_ns": time.time_ns(),
        "elapsed_seconds": elapsed,
        "phase": read_phase(run),
        "gpus": [snapshot(gpu, pids, cid) for gpu in gpus],
        "process_tree_memory": process_memory(pids),
        "host_memory": host_memory(),
    }


def initial_summary(gpus, baseline, interval):
    return {
        "schema_version": 1,
        "sampling_interval_seconds": interval,
        "sample_count": 0,
        "measurement_notes": {
            "gpu": "Whole-card sysfs samples; includes any use not rejected by ownership checks.",
            "process_tree": "Sum of current RSS for docker-top PIDs; shared pages can be counted more than once.",
            "host": "MemTotal minus MemAvailable for the whole host; concurrent host activity is not attributable.",
            "sampling": "Sampled peaks can understate transients shorter than the sampling interval.",
        },
        "baseline_host_memory": baseline,
        "gpus": {
            str(gpu["physical_id"]): {
                "physical_id": gpu["physical_id"],
                "pci": gpu["pci"],
                "uuid": gpu["uuid"],
                "vram_total_bytes": gpu.get("vram_total_bytes"),
                "peak_vram_used_bytes": None,
                "peak_gpu_busy_percent": None,
            }
            for gpu in gpus
        },
        "peak_process_tree_rss_bytes": None,
        "peak_process_tree_sum_vm_hwm_bytes": None,
        "peak_host_used_from_available_bytes": baseline["mem_used_from_available_bytes"],
        "peak_host_used_delta_from_baseline_bytes": 0,
        "minimum_host_available_bytes": baseline["mem_available_bytes"],
        "minimum_swap_free_bytes": baseline.get("swap_free_bytes"),
    }


def update_summary(summary, sample):
    summary["sample_count"] += 1
    for gpu in sample["gpus"]:
        row = summary["gpus"][str(gpu["physical_id"])]
        for target, source in (
            ("peak_vram_used_bytes", "vram_used_bytes"),
            ("peak_gpu_busy_percent", "gpu_busy_percent"),
        ):
            value = gpu[source]
            row[target] = value if row[target] is None else max(row[target], value)
    memory = sample["process_tree_memory"]
    for target, source in (
        ("peak_process_tree_rss_bytes", "rss_bytes"),
        ("peak_process_tree_sum_vm_hwm_bytes", "sum_vm_hwm_bytes"),
    ):
        value = memory[source]
        summary[target] = value if summary[target] is None else max(summary[target], value)
    host = sample["host_memory"]
    summary["peak_host_used_from_available_bytes"] = max(
        summary["peak_host_used_from_available_bytes"], host["mem_used_from_available_bytes"]
    )
    baseline_used = summary["baseline_host_memory"]["mem_used_from_available_bytes"]
    summary["peak_host_used_delta_from_baseline_bytes"] = max(
        summary["peak_host_used_delta_from_baseline_bytes"],
        host["mem_used_from_available_bytes"] - baseline_used,
    )
    summary["minimum_host_available_bytes"] = min(
        summary["minimum_host_available_bytes"], host["mem_available_bytes"]
    )
    if host.get("swap_free_bytes") is not None:
        prior = summary.get("minimum_swap_free_bytes")
        summary["minimum_swap_free_bytes"] = (
            host["swap_free_bytes"] if prior is None else min(prior, host["swap_free_bytes"])
        )


def stop_container(cid_file):
    if not cid_file.exists():
        return
    cid = cid_file.read_text().strip()
    if len(cid) not in (12, 64) or any(character not in "0123456789abcdef" for character in cid):
        raise RuntimeError("Invalid owned container identity")
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.Config.WorkingDir}}", cid],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip() == str(ROOT):
        subprocess.run(["docker", "stop", "--time", "10", cid], capture_output=True)


def parse_gpu_list(args, parser):
    if args.gpus is not None and args.gpu is not None:
        parser.error("Use either --gpu or --gpus, not both")
    raw = args.gpus if args.gpus is not None else str(args.gpu if args.gpu is not None else 1)
    try:
        result = [int(value) for value in raw.split(",")]
    except ValueError:
        parser.error("GPU IDs must be comma-separated integers")
    if not result or len(result) != len(set(result)) or any(value not in ALLOWED_GPUS for value in result):
        parser.error("Select unique non-GPU3 physical devices from 0,1,2,4,5,6,7")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None, help="one physical device (legacy form)")
    parser.add_argument("--gpus", help="ordered comma-separated physical devices")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--purpose", default="H3-Image independent native-precision validation")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    selected = parse_gpu_list(args, parser)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or not 0.1 <= args.monitor_interval <= 10:
        parser.error("Provide a command; timeout must be positive and monitor interval 0.1..10 seconds")

    run = args.run_dir.resolve()
    if ROOT / "runs" not in run.parents:
        parser.error("--run-dir must be under this project's runs/")
    run.mkdir(parents=True, exist_ok=False)
    cid_file = run / "container-id"
    lease = LEASES / f"gpu-reservation.h3-image-{os.getpid()}.json"
    row = {
        "owner": f"h3-image-{os.getpid()}",
        "pid": os.getpid(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "status": "active",
        "physical_gpus": selected,
        "evidence": str(run),
        "purpose": args.purpose,
    }
    process = None
    outcome = {"status": "failed", "returncode": None}
    lease_created = False

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with ExitStack() as stack:
        for physical_id in sorted(selected):
            lock = stack.enter_context((GPU_LOCKS / f"gpu{physical_id}.lock").open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = stack.enter_context((LEASES / "gpu-reservation.json.lock").open("a+"))
        fcntl.flock(registry, fcntl.LOCK_EX)
        gpus = [device(physical_id) for physical_id in selected]
        active = reservations()
        conflicts = [
            reservation
            for reservation in active
            if set(map(int, reservation.get("physical_gpus", []))) & set(selected)
        ]
        if conflicts:
            raise RuntimeError(f"Selected GPUs have an active shared reservation: {conflicts}")
        preflight = [snapshot(gpu) for gpu in gpus]
        occupied = [
            sample
            for sample in preflight
            if sample["external_processes"] or sample["vram_used_bytes"] >= 1024**3
        ]
        if occupied:
            raise RuntimeError(f"Selected GPU is occupied: {occupied}")
        baseline_host = host_memory()
        write(
            run / "hardware.json",
            {
                "ordered_physical_gpus": selected,
                "logical_mapping": [
                    {"logical_id": logical_id, **gpu} for logical_id, gpu in enumerate(gpus)
                ],
                "preflight": preflight,
                "baseline_host_memory": baseline_host,
            },
        )
        write(lease, {**row, "updated_at": datetime.now(timezone.utc).isoformat()})
        lease_created = True
        fcntl.flock(registry, fcntl.LOCK_UN)
        try:
            cache = ROOT / ".runtime/cache/rocm10-image"
            inner_env = {
                "HIP_VISIBLE_DEVICES": ",".join(str(index) for index in range(len(gpus))),
                "ROCR_VISIBLE_DEVICES": ",".join(gpu["uuid"] for gpu in gpus),
                "H3I_PHYSICAL_GPUS": ",".join(map(str, selected)),
                "H3I_RUN_DIR": str(run),
                "H3I_ROOT": str(ROOT),
                "TRITON_CACHE_DIR": str(cache / "triton"),
                "TORCH_EXTENSIONS_DIR": str(cache / "torch"),
                "XDG_CACHE_HOME": str(cache / "xdg"),
                "HF_HOME": str(cache / "huggingface"),
                "PIP_CACHE_DIR": str(cache / "pip"),
                "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "16",
                "TOKENIZERS_PARALLELISM": "false",
            }
            argv = [
                str(WORKSPACE / "rocm10-launch.sh"),
                "env",
                *[f"{key}={value}" for key, value in inner_env.items()],
                "bash",
                "-c",
                'cat /etc/hostname > "$H3I_RUN_DIR/container-id"; exec "$@"',
                "bash",
                *command,
            ]
            write(
                run / "launch.json",
                {
                    "command": argv,
                    "physical_gpus": selected,
                    "logical_mapping": [
                        {"logical_id": logical_id, **gpu} for logical_id, gpu in enumerate(gpus)
                    ],
                    "host_hip_visible_devices": ",".join(map(str, selected)),
                    "monitor_interval_seconds": args.monitor_interval,
                },
            )
            started = time.monotonic()
            summary = initial_summary(gpus, baseline_host, args.monitor_interval)
            write(run / "monitor-summary.json", summary)
            with (run / "console.log").open("x") as log, (run / "monitor.jsonl").open("x") as monitor:
                process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "HIP_VISIBLE_DEVICES": ",".join(map(str, selected))},
                    start_new_session=True,
                )
                print(f"GPU{selected} reserved; output: {run}", flush=True)
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > args.timeout:
                        raise TimeoutError("Owned job exceeded timeout")
                    # The immutable foundation verifier executes before the wrapper writes the CID.
                    if cid_file.exists():
                        cid = cid_file.read_text().strip()
                        pids = container_pids(cid_file)
                        current = monitor_sample(gpus, pids, cid, run, elapsed)
                        monitor.write(json.dumps(current, sort_keys=True) + "\n")
                        monitor.flush()
                        update_summary(summary, current)
                        write(run / "monitor-latest.json", current)
                        write(run / "monitor-summary.json", summary)
                        external = [
                            sample for sample in current["gpus"] if sample["external_processes"]
                        ]
                        if external:
                            raise RuntimeError(f"External GPU use detected: {external}")
                    time.sleep(args.monitor_interval)
                outcome = {
                    "status": "complete" if process.returncode == 0 else "failed",
                    "returncode": process.returncode,
                    "elapsed_seconds": time.monotonic() - started,
                }
        except BaseException as error:
            outcome["error"] = repr(error)
            raise
        finally:
            try:
                stop_container(cid_file)
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                write(run / "execution.json", outcome)
                write(run / "postflight.json", [snapshot(gpu) for gpu in gpus])
            finally:
                if lease_created:
                    write(
                        lease,
                        {**row, "status": "released", "updated_at": datetime.now(timezone.utc).isoformat()},
                    )
        raise SystemExit(outcome["returncode"] or 0)


if __name__ == "__main__":
    main()
