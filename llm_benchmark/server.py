"""Manage a vLLM OpenAI-compatible server as a background subprocess."""

import os
import signal
import subprocess
import time
from typing import List, Optional

import requests

from .config import BASE_URL, SERVER_PORT


def start_vllm_server(
    model_id: str,
    port: int = SERVER_PORT,
    **kwargs,
) -> subprocess.Popen:
    """Launch `vllm serve` as a background subprocess and return the Popen handle.

    Keyword args are CLI flags with underscores converted to hyphens
    (e.g. gpu_memory_utilization=0.90 -> --gpu-memory-utilization 0.90).
    Boolean True values become bare flags (e.g. enforce_eager=True -> --enforce-eager).
    """
    defaults = {
        "dtype": "float16",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "uni",
        "port": port,
        "trust_remote_code": True,
    }
    defaults.update(kwargs)

    cmd = ["vllm", "serve", model_id]
    for flag, value in defaults.items():
        flag = flag.replace("_", "-")
        if value is True:
            cmd.append(f"--{flag}")
        elif value is False or value is None:
            continue
        else:
            cmd.append(f"--{flag}")
            cmd.append(str(value))

    log_file = open(f"/content/vllm_server_{port}.log", "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach into its own process group
    )
    print("Launched:", " ".join(cmd))
    return proc


def wait_for_server(
    base_url: str = BASE_URL,
    timeout: int = 300,
    sleep: int = 5,
) -> bool:
    """Poll the /health endpoint until the server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {time.time() - start:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(sleep)
    print(f"WARNING: server not ready after {timeout}s")
    return False


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    """Terminate the server process group and wait for GPU memory to free."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    time.sleep(5)
    print("Server stopped.")
