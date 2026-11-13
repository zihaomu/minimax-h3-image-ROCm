#!/usr/bin/env python3
"""Run one owned job on a reserved R9700 through the workspace foundation."""
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


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def reservations():
    rows = []
    for path in LEASES.glob("gpu-reservation*.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "active":
            rows.append(row)
    return rows


def device(index):
    inventory = json.loads(subprocess.check_output(
        ["rocm-smi", "--showbus", "--showuniqueid", "--json"], text=True))
    row = inventory[f"card{index}"]
    pci = row["PCI Bus"].lower()
    uuid = f"GPU-{int(row['Unique ID'], 16):016x}"
    for node in Path("/sys/class/kfd/kfd/topology/nodes").iterdir():
        props = dict(line.split(maxsplit=1) for line in (node / "properties").read_text().splitlines())
        if int(props.get("unique_id", "0")) == int(row["Unique ID"], 16):
            if props.get("gfx_target_version") != "120001":
                raise RuntimeError("Selected device is not gfx1201")
            return {"physical_id": index, "pci": pci, "uuid": uuid,
                    "kfd_gpu_id": int((node / "gpu_id").read_text())}
    raise RuntimeError("Could not match ROCm device to KFD identity")


def snapshot(gpu, own_pids=(), own_container=None):
    gid = gpu["kfd_gpu_id"]
    processes, owned = [], []
    for folder in Path("/sys/class/kfd/kfd/proc").iterdir():
        pid = int(folder.name)
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text()
            if pid in own_pids or (own_container and own_container in cgroup):
                owned.append(pid)
                continue
            queues = any(int(p.read_text()) == gid for p in (folder / "queues").glob("*/gpuid"))
            memory = folder / f"vram_{gid}"
            allocated = int(memory.read_text()) if memory.exists() else 0
            if queues or allocated:
                processes.append({"pid": pid, "queues": queues, "vram_bytes": allocated, "cgroup": cgroup.strip()})
        except FileNotFoundError:
            continue
    pci = Path("/sys/bus/pci/devices") / gpu["pci"]
    return {"time_ns": time.time_ns(), "external_processes": processes, "owned_kfd_pids": owned,
            "vram_used_bytes": int((pci / "mem_info_vram_used").read_text()),
            "gpu_busy_percent": int((pci / "gpu_busy_percent").read_text())}


def container_pids(cid_file):
    if not cid_file.exists():
        return []
    cid = cid_file.read_text().strip()
    result = subprocess.run(["docker", "top", cid, "-eo", "pid"], capture_output=True, text=True)
    return [int(x.strip()) for x in result.stdout.splitlines()[1:] if x.strip().isdigit()]


def stop_container(cid_file):
    if not cid_file.exists():
        return
    cid = cid_file.read_text().strip()
    if len(cid) not in (12, 64) or any(c not in "0123456789abcdef" for c in cid):
        raise RuntimeError("Invalid owned container identity")
    result = subprocess.run(["docker", "inspect", "--format", "{{.Config.WorkingDir}}", cid],
                            capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip() == str(ROOT):
        subprocess.run(["docker", "stop", "--time", "10", cid], capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.gpu not in (0, 1, 2, 4, 5, 6, 7) or not command:
        parser.error("Select one non-GPU3 physical device and a command after --")
    run = args.run_dir.resolve()
    if ROOT / "runs" not in run.parents:
        parser.error("--run-dir must be under this project's runs/")
    run.mkdir(parents=True, exist_ok=False)
    cid_file = run / "container-id"
    lease = LEASES / f"gpu-reservation.h3-image-{os.getpid()}.json"
    row = {"owner": f"h3-image-{os.getpid()}", "pid": os.getpid(),
           "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
           "status": "active", "physical_gpus": [args.gpu], "evidence": str(run),
           "purpose": "H3-Image independent environment / single GPU validation"}
    process = None
    outcome = {"status": "failed", "returncode": None}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with ExitStack() as stack:
        lock = stack.enter_context((GPU_LOCKS / f"gpu{args.gpu}.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = stack.enter_context((LEASES / "gpu-reservation.json.lock").open("a+"))
        fcntl.flock(registry, fcntl.LOCK_EX)
        gpu = device(args.gpu)
        if any(args.gpu in map(int, r.get("physical_gpus", [])) for r in reservations()):
            raise RuntimeError("Selected GPU has an active shared reservation")
        before = snapshot(gpu)
        if before["external_processes"] or before["vram_used_bytes"] >= 1024**3:
            raise RuntimeError(f"GPU is occupied: {before}")
        write(run / "hardware.json", {**gpu, "preflight": before})
        write(lease, {**row, "updated_at": datetime.now(timezone.utc).isoformat()})
        fcntl.flock(registry, fcntl.LOCK_UN)
        try:
            cache = ROOT / ".runtime/cache/rocm10-image"
            env = {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": gpu["uuid"],
                   "H3I_RUN_DIR": str(run), "H3I_ROOT": str(ROOT),
                   "TRITON_CACHE_DIR": str(cache / "triton"),
                   "TORCH_EXTENSIONS_DIR": str(cache / "torch"),
                   "XDG_CACHE_HOME": str(cache / "xdg"), "HF_HOME": str(cache / "huggingface"),
                   "PIP_CACHE_DIR": str(cache / "pip"), "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
                   "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "16", "TOKENIZERS_PARALLELISM": "false"}
            argv = [str(WORKSPACE / "rocm10-launch.sh"), "env",
                    *[f"{k}={v}" for k, v in env.items()], "bash", "-c",
                    'cat /etc/hostname > "$H3I_RUN_DIR/container-id"; exec "$@"', "bash", *command]
            write(run / "launch.json", {"command": argv, "physical_gpu": args.gpu,
                                        "host_hip_visible_devices": str(args.gpu)})
            started = time.monotonic()
            with (run / "console.log").open("x") as log, (run / "monitor.jsonl").open("x") as monitor:
                process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "HIP_VISIBLE_DEVICES": str(args.gpu)},
                                           start_new_session=True)
                print(f"GPU{args.gpu} {gpu['pci']} reserved; output: {run}", flush=True)
                while process.poll() is None:
                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError("Owned job exceeded timeout")
                    # The foundation verifier runs before the container-ID wrapper.
                    if cid_file.exists():
                        current = snapshot(gpu, container_pids(cid_file), cid_file.read_text().strip())
                        monitor.write(json.dumps(current) + "\n")
                        monitor.flush()
                        if current["external_processes"]:
                            raise RuntimeError(f"External GPU use detected: {current}")
                    time.sleep(2)
                outcome = {"status": "complete" if process.returncode == 0 else "failed",
                           "returncode": process.returncode, "elapsed_seconds": time.monotonic() - started}
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
                write(run / "postflight.json", snapshot(gpu))
            finally:
                write(lease, {**row, "status": "released", "updated_at": datetime.now(timezone.utc).isoformat()})
        raise SystemExit(outcome["returncode"] or 0)


if __name__ == "__main__":
    main()
