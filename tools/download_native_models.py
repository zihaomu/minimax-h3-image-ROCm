#!/usr/bin/env python3
"""Download the pinned original MiniMax-H3-Image components with resume and SHA checks."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"
STATE_ROOT = MODEL_ROOT / ".download/native"
IMAGE_REVISION = "5f90473023975e6271b0ed1dd84c4e88e4bbc6d3"
TEXT_REVISION = "0919fb684102ef115126612f3fbb84d1027e7317"

IMAGE_FILES = [
    "LICENSE",
    "model_index.json",
    "modular_model_index.json",
    "pdd_config.json",
    "pdd_fused_heads.safetensors",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/model.safetensors.index.json",
    *[f"transformer/model-{index:05d}-of-00012.safetensors" for index in range(1, 13)],
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

TEXT_FILES = [
    "LICENSE",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    *[f"text_encoder/model-{index:05d}-of-00014.safetensors" for index in range(1, 12)],
]

SOURCES = [
    {
        "repository": "MiniMaxAI/MiniMax-H3-Image",
        "revision": IMAGE_REVISION,
        "destination": "native/minimax-h3-image",
        "files": IMAGE_FILES,
    },
    {
        "repository": "MiniMaxAI/MiniMax-H3",
        "revision": TEXT_REVISION,
        "destination": "native/minimax-h3",
        "files": TEXT_FILES,
    },
]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_inventory(session: requests.Session, source: dict) -> list[dict]:
    url = f"https://huggingface.co/api/models/{source['repository']}/revision/{source['revision']}?blobs=true"
    response = session.get(url, timeout=(20, 120))
    response.raise_for_status()
    payload = response.json()
    if payload.get("sha") != source["revision"]:
        raise RuntimeError(f"Revision mismatch for {source['repository']}: {payload.get('sha')}")
    inventory = {row["rfilename"]: row for row in payload.get("siblings", [])}
    missing = sorted(set(source["files"]) - set(inventory))
    if missing:
        raise RuntimeError(f"Files missing at pinned revision: {missing}")
    specs = []
    for remote_path in source["files"]:
        row = inventory[remote_path]
        local_path = str(Path(source["destination"]) / remote_path)
        specs.append(
            {
                "repository": source["repository"],
                "revision": source["revision"],
                "remote_path": remote_path,
                "path": local_path,
                "size_bytes": int(row["size"]),
                "sha256": (row.get("lfs") or {}).get("sha256"),
                "url": (
                    f"https://huggingface.co/{source['repository']}/resolve/"
                    f"{source['revision']}/{remote_path}"
                ),
            }
        )
    return specs


def safetensors_header(path: Path) -> int | None:
    # During first verification the fully downloaded file still has its
    # atomic ``.part`` suffix. Do not accidentally skip structure checks.
    if not (path.name.endswith(".safetensors") or path.name.endswith(".safetensors.part")):
        return None
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size < 64 * 1024 * 1024:
            raise RuntimeError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(header_size))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    data_bytes = path.stat().st_size - 8 - header_size
    if not tensors or max(value["data_offsets"][1] for value in tensors) != data_bytes:
        raise RuntimeError(f"Invalid safetensors offsets: {path}")
    return len(tensors)


class Downloader:
    def __init__(self, specs: list[dict], workers: int, chunk_bytes: int):
        self.specs = specs
        self.workers = workers
        self.chunk_bytes = chunk_bytes
        self.lock = threading.Lock()
        self.state_path = STATE_ROOT / "segments.json"
        self.manifest_path = STATE_ROOT / "manifest.json"
        self.active: dict[str, int] = {}
        self.fds: dict[str, int] = {}
        self.started = time.monotonic()
        self.total = sum(spec["size_bytes"] for spec in specs)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            expected = {spec["path"]: spec["size_bytes"] for spec in specs}
            if self.state.get("files") is None or self.state.get("sizes") != expected:
                raise RuntimeError("Existing native download state does not match the pinned manifest")
        else:
            self.state = {
                "schema_version": 1,
                "sizes": {spec["path"]: spec["size_bytes"] for spec in specs},
                "files": {spec["path"]: {"ranges": {}, "complete": False} for spec in specs},
            }
            self.save_state()

    def save_state(self) -> None:
        atomic_json(self.state_path, self.state)

    def progress(self) -> None:
        with self.lock:
            completed = 0
            for spec in self.specs:
                record = self.state["files"][spec["path"]]
                if record["complete"]:
                    completed += spec["size_bytes"]
                else:
                    completed += sum(end - int(start) + 1 for start, end in record["ranges"].items())
            received = completed + sum(self.active.values())
        elapsed = time.monotonic() - self.started
        rate = received / elapsed if elapsed else 0.0
        remaining = (self.total - received) / rate if rate else None
        print(
            json.dumps(
                {
                    "event": "download_progress",
                    "received_gb": round(received / 1e9, 3),
                    "total_gb": round(self.total / 1e9, 3),
                    "elapsed_seconds": round(elapsed, 1),
                    "rate_mib_s": round(rate / 1024**2, 2),
                    "eta_seconds": round(remaining, 1) if remaining is not None else None,
                }
            ),
            flush=True,
        )

    def transfer(self, spec: dict, start: int, end: int) -> None:
        name = spec["path"]
        key = f"{name}:{start}"
        for attempt in range(6):
            with self.lock:
                self.active[key] = 0
            try:
                with requests.get(
                    spec["url"],
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "minimax-h3-image-rocm/1"},
                    stream=True,
                    timeout=(20, 60),
                ) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{spec['size_bytes']}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(
                            f"Unexpected range response {response.status_code} {response.headers.get('Content-Range')}"
                        )
                    offset = start
                    for block in response.iter_content(1024 * 1024):
                        if offset + len(block) > end + 1:
                            raise RuntimeError("Range overrun")
                        view = memoryview(block)
                        while view:
                            written = os.pwrite(self.fds[name], view, offset)
                            if written <= 0:
                                raise OSError("Short pwrite")
                            offset += written
                            view = view[written:]
                        with self.lock:
                            self.active[key] = offset - start
                    if offset != end + 1:
                        raise RuntimeError("Truncated range")
                with self.lock:
                    self.state["files"][name]["ranges"][str(start)] = end
                    self.active.pop(key, None)
                    self.save_state()
                return
            except Exception as error:
                with self.lock:
                    self.active.pop(key, None)
                if attempt == 5:
                    raise RuntimeError(f"{name} bytes {start}-{end}: {error}") from error
                time.sleep(min(2**attempt, 15))

    def download(self) -> list[dict]:
        required = sum(
            spec["size_bytes"]
            for spec in self.specs
            if not (MODEL_ROOT / spec["path"]).exists()
        )
        free = shutil.disk_usage(MODEL_ROOT).free
        if free < required + 20 * 1024**3:
            raise RuntimeError(f"Insufficient disk: need {required} bytes plus 20 GiB, have {free}")

        jobs = []
        for spec in self.specs:
            target = MODEL_ROOT / spec["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != spec["size_bytes"]:
                    raise RuntimeError(f"Existing file has wrong size: {target}")
                continue
            partial = target.with_name(target.name + ".part")
            fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
            self.fds[spec["path"]] = fd
            os.ftruncate(fd, spec["size_bytes"])
            ranges = self.state["files"][spec["path"]]["ranges"]
            for start in range(0, spec["size_bytes"], self.chunk_bytes):
                end = min(start + self.chunk_bytes, spec["size_bytes"]) - 1
                if ranges.get(str(start)) != end:
                    jobs.append((spec, start, end))

        errors = []
        try:
            with futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                pending = {pool.submit(self.transfer, *job) for job in jobs}
                next_report = time.monotonic()
                while pending:
                    done, pending = futures.wait(pending, timeout=2, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        try:
                            future.result()
                        except Exception as error:
                            errors.append(str(error))
                    if time.monotonic() >= next_report:
                        self.progress()
                        next_report = time.monotonic() + 20
                    if errors:
                        for future in pending:
                            future.cancel()
                        break
        finally:
            for fd in self.fds.values():
                os.close(fd)
        if errors:
            atomic_json(STATE_ROOT / "errors.json", errors)
            raise RuntimeError(errors[0])

        verified = []
        for spec in self.specs:
            target = MODEL_ROOT / spec["path"]
            partial = target.with_name(target.name + ".part")
            path = target if target.exists() else partial
            if path.stat().st_size != spec["size_bytes"]:
                raise RuntimeError(f"Size mismatch: {path}")
            print(json.dumps({"event": "verify_start", "path": spec["path"]}), flush=True)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(16 * 1024 * 1024):
                    digest.update(block)
            actual_sha = digest.hexdigest()
            if spec["sha256"] is not None and actual_sha != spec["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            tensors = safetensors_header(path)
            if path == partial:
                partial.replace(target)
            self.state["files"][spec["path"]]["complete"] = True
            self.save_state()
            verified.append(
                {
                    **spec,
                    "sha256": actual_sha,
                    "tensor_count": tensors,
                    "verified": True,
                }
            )
            print(json.dumps({"event": "verify_complete", "path": spec["path"]}), flush=True)
        return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or not 8 <= args.chunk_mib <= 256:
        parser.error("workers must be 1..64 and chunk-mib must be 8..256")

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        specs = []
        for source in SOURCES:
            specs.extend(source_inventory(session, source))
    manifest = {
        "schema_version": 1,
        "sources": SOURCES,
        "files": specs,
        "total_size_bytes": sum(spec["size_bytes"] for spec in specs),
    }
    atomic_json(STATE_ROOT / "target-manifest.json", manifest)
    downloader = Downloader(specs, args.workers, args.chunk_mib * 1024**2)
    verified = downloader.download()
    report = {
        "schema_version": 1,
        "status": "complete",
        "elapsed_seconds": time.monotonic() - downloader.started,
        "files": verified,
        "total_size_bytes": sum(spec["size_bytes"] for spec in specs),
    }
    atomic_json(STATE_ROOT / "verification.json", report)
    print(json.dumps({"event": "native_models_complete", **report}, default=str), flush=True)


if __name__ == "__main__":
    main()
