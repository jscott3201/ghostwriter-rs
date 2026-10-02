"""Scoped single-device CUDA execution; CPU v1 runtime remains unchanged."""
from contextlib import contextmanager
import os
import platform
import random

from ..artifact import ContractError


def require_cuda():
    """Reject unsupported hosts and distributed/device overrides before model allocation."""
    import torch
    if (platform.system() != "Linux" or platform.machine() != "x86_64" or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or not torch.cuda.is_bf16_supported(including_emulation=False)
            or torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise ContractError("CUDA policy requires one Linux BF16-capable CUDA device and one process")
    for key, allowed in {"WORLD_SIZE": {"1"}, "RANK": {"0"}, "LOCAL_RANK": {"-1", "0"},
                         "ACCELERATE_USE_CPU": {"false", "False", "0"}}.items():
        if key in os.environ and os.environ[key] not in allowed:
            raise ContractError("ambient distributed or CPU execution override is unsupported")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ContractError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before initializing the CUDA process")
    return torch.device("cuda:0")


@contextmanager
def runtime(seed=0):
    """Restore Python, NumPy, CPU and the sole CUDA RNG plus all changed backend flags."""
    import numpy as np
    import torch
    device = require_cuda()
    threads = torch.get_num_threads()
    python, numpy = random.getstate(), np.random.get_state()
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    with torch.random.fork_rng(devices=[0]):
        try:
            torch.set_num_threads(1)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            yield device
        finally:
            torch.set_num_threads(threads)
            random.setstate(python); np.random.set_state(numpy)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[:2]
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[2:4]
            torch.use_deterministic_algorithms(previous[4], warn_only=previous[5])


def memory():
    """Observe allocator peaks, host RSS and this process's driver-reported GPU residency."""
    import subprocess
    import psutil
    import torch
    torch.cuda.synchronize(0)
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True)
        own_pids = {os.getpid()}
        # A driver may report an outer namespace PID; accept only IDs listed for this process.
        from pathlib import Path
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                own_pids.update(map(int, line.split()[1:]))
        matches = []
        for line in result.stdout.splitlines():
            pid, amount = (value.strip() for value in line.split(","))
            if int(pid) in own_pids:
                matches.append(int(amount) * 1024**2)
        if len(matches) != 1:
            raise ValueError("missing or multiple driver process entries")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ContractError("CUDA policy requires measured driver residency for this exact process") from error
    return {"allocated": torch.cuda.memory_allocated(0), "reserved": torch.cuda.memory_reserved(0),
            "peak_allocated": torch.cuda.max_memory_allocated(0), "peak_reserved": torch.cuda.max_memory_reserved(0),
            "host_rss": psutil.Process().memory_info().rss, "process_gpu_bytes": matches[0]}
