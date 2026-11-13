"""Ownership regression checks; all device/process files are temporary fixtures."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_rocm10", ROOT / "tools/run_rocm10.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.gpu = {"pci": "0000:23:00.0", "kfd_gpu_id": 8819}
        self.put("/sys/bus/pci/devices/0000:23:00.0/mem_info_vram_used", "1024")
        self.put("/sys/bus/pci/devices/0000:23:00.0/gpu_busy_percent", "1")
        self.put("/sys/class/kfd/kfd/proc/123/vram_8819", "512")

    def put(self, name, text):
        path = self.root / name.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def snapshot(self, cid):
        with patch.object(runner, "Path", side_effect=lambda name: self.root / str(name).lstrip("/")):
            return runner.snapshot(self.gpu, own_pids=[], own_container=cid)

    def test_new_owned_process_missing_from_docker_top_snapshot(self):
        self.put("/proc/123/cgroup", "0::/system.slice/docker-abcdef0123456789.scope\n")
        result = self.snapshot("abcdef012345")
        self.assertEqual(result["external_processes"], [])
        self.assertEqual(result["owned_kfd_pids"], [123])

    def test_other_container_still_blocks_execution(self):
        self.put("/proc/123/cgroup", "0::/system.slice/docker-fedcba543210.scope\n")
        result = self.snapshot("abcdef012345")
        self.assertEqual(result["external_processes"][0]["pid"], 123)

    def test_process_using_only_another_device_does_not_block(self):
        self.put("/proc/123/cgroup", "0::/system.slice/docker-fedcba543210.scope\n")
        self.put("/sys/class/kfd/kfd/proc/123/vram_8819", "0")
        self.put("/sys/class/kfd/kfd/proc/123/queues/0/gpuid", "64106")
        self.assertEqual(self.snapshot("abcdef012345")["external_processes"], [])


if __name__ == "__main__":
    unittest.main()
