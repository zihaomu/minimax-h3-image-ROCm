"""Live in-process phase, transfer, and PyTorch memory measurements."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class RunMetrics:
    def __init__(self, run: Path, torch, metadata: dict):
        self.run = Path(run)
        self.torch = torch
        self.report = {
            "schema_version": 1,
            "status": "running",
            "started_time_ns": time.time_ns(),
            "metadata": metadata,
            "phases": [],
            "transfers": [],
            "torch_global_peaks": {},
        }
        self.started = time.perf_counter()
        self.write()

    def write(self) -> None:
        atomic_json(self.run / "result.json", self.report)

    def set_phase(self, name: str, **fields: object) -> None:
        atomic_json(self.run / "phase.json", {"name": name, "time_ns": time.time_ns(), **fields})

    def synchronize(self) -> None:
        for index in range(self.torch.cuda.device_count()):
            self.torch.cuda.synchronize(index)

    def reset_peaks(self) -> None:
        for index in range(self.torch.cuda.device_count()):
            self.torch.cuda.reset_peak_memory_stats(index)

    def device_memory(self) -> dict:
        return {
            str(index): {
                "allocated_bytes": self.torch.cuda.memory_allocated(index),
                "reserved_bytes": self.torch.cuda.memory_reserved(index),
                "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(index),
                "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(index),
            }
            for index in range(self.torch.cuda.device_count())
        }

    def merge_global_peaks(self, snapshot: dict) -> None:
        for device, values in snapshot.items():
            aggregate = self.report["torch_global_peaks"].setdefault(device, {})
            for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                aggregate[key] = max(int(aggregate.get(key, 0)), int(values[key]))

    @contextmanager
    def phase(self, name: str, **fields: object):
        self.synchronize()
        self.reset_peaks()
        self.set_phase(name, **fields)
        started = time.perf_counter()
        row = {"name": name, "status": "running", "started_time_ns": time.time_ns(), **fields}
        self.report["phases"].append(row)
        self.write()
        try:
            yield row
            self.synchronize()
            row.update(
                status="complete",
                seconds=time.perf_counter() - started,
                finished_time_ns=time.time_ns(),
                torch_memory=self.device_memory(),
            )
        except BaseException as error:
            row.update(
                status="failed",
                seconds=time.perf_counter() - started,
                finished_time_ns=time.time_ns(),
                error=repr(error),
                torch_memory=self.device_memory(),
            )
            raise
        finally:
            self.merge_global_peaks(row["torch_memory"])
            self.write()

    def transfer(self, kind: str, byte_count: int, seconds: float, **fields: object) -> None:
        self.report["transfers"].append(
            {"kind": kind, "bytes": int(byte_count), "seconds": float(seconds), **fields}
        )
        self.write()

    def finish(self, **fields: object) -> None:
        self.synchronize()
        self.merge_global_peaks(self.device_memory())
        self.report.update(
            status="complete",
            request_seconds=time.perf_counter() - self.started,
            finished_time_ns=time.time_ns(),
            **fields,
        )
        self.set_phase("complete")
        self.write()

    def fail(self, error: BaseException) -> None:
        current = self.device_memory()
        self.merge_global_peaks(current)
        self.report.update(
            status="failed",
            request_seconds=time.perf_counter() - self.started,
            finished_time_ns=time.time_ns(),
            error=repr(error),
        )
        self.set_phase("failed", error=repr(error))
        self.write()
