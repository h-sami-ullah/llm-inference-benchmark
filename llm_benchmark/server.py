"""Manage a vLLM OpenAI-compatible server as a background subprocess."""

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional

import requests

from .config import BASE_URL, SERVER_PORT


def _log_dir() -> str:
    return "/content" if os.path.isdir("/content") else os.getcwd()


def vllm_cli(*args: str) -> List[str]:
    """Build a ``vllm <args...>`` invocation, resolving the executable robustly.

    Prefer the `vllm` console script if it is on PATH; otherwise fall back to
    running the vLLM module with the current Python interpreter
    (``python -m vllm ...``), which avoids ``FileNotFoundError: vllm`` when
    the entry point is not on PATH (e.g. inside a notebook kernel).
    """
    exe = shutil.which("vllm")
    if exe:
        return [exe, *args]
    return [sys.executable, "-m", "vllm", *args]


def _resolve_command(model_id: str, defaults: dict) -> List[str]:
    """Build the `vllm serve <model> ...` invocation with the given flags."""
    cmd = vllm_cli("serve", model_id)

    for flag, value in defaults.items():
        flag = flag.replace("_", "-")
        if value is True:
            cmd.append(f"--{flag}")
        elif value is False or value is None:
            continue
        else:
            cmd.append(f"--{flag}")
            cmd.append(str(value))
    return cmd


def _stream_output(proc, log_file, verbose: bool, prefix: str = "[vllm] ") -> None:
    """Read the server's stdout/stderr line by line, tee to log + (optionally) console."""
    try:
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            if verbose:
                sys.stdout.write(prefix + line)
                sys.stdout.flush()
    except Exception:
        pass
    finally:
        try:
            log_file.close()
        except Exception:
            pass


def start_vllm_server(
    model_id: str,
    port: int = SERVER_PORT,
    verbose: bool = True,
    **kwargs,
) -> subprocess.Popen:
    """Launch `vllm serve` as a background subprocess and return the Popen handle.

    Keyword args are CLI flags with underscores converted to hyphens
    (e.g. gpu_memory_utilization=0.90 -> --gpu-memory-utilization 0.90).
    Boolean True values become bare flags (e.g. enforce_eager=True -> --enforce-eager).

    With ``verbose=True`` (default), server logs are streamed to the console in
    addition to being written to ``<log_dir>/vllm_server_<port>.log``.

    Notes on defaults vs. current vLLM guidance
    (https://github.com/vllm-project/vllm):

    * ``enforce_eager=True`` disables CUDA graphs. vLLM enables them by default
      for higher throughput; we keep eager mode as the safe default for
      portability (e.g. Colab T4) — pass ``enforce_eager=False`` for max TPS.
    * Run ``vllm download-kernels`` once after installing/upgrading vLLM to avoid
      downloading and compiling kernels on first startup (adds minutes on
      Hopper+ GPUs).
    """
    defaults = {
        "dtype": "float16",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "uni",
        "port": port,
        "trust_remote_code": True,
        # Don't let the model's generation_config.json override our explicit
        # sampling params (e.g. temperature=0.0 greedy) — keeps results
        # reproducible across models.
        "generation_config": "vllm",
    }
    defaults.update(kwargs)

    cmd = _resolve_command(model_id, defaults)
    log_path = os.path.join(_log_dir(), f"vllm_server_{port}.log")

    print("=" * 60)
    print("Launching vLLM server (subprocess):")
    print("  " + " ".join(cmd))
    print(f"  Log file: {log_path}")
    print("=" * 60)

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # detach into its own process group
    )

    # Attach metadata used by stop_server() and wait_for_server().
    proc.log_file = log_file
    proc.log_path = log_path

    # Drain stdout/stderr continuously so the pipe never blocks.
    proc._reader = threading.Thread(
        target=_stream_output,
        args=(proc, log_file, verbose),
        daemon=True,
    )
    proc._reader.start()

    return proc


def _tail(path: str, n: int = 20) -> str:
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""


def wait_for_server(
    base_url: str = BASE_URL,
    timeout: int = 300,
    sleep: int = 5,
    verbose: bool = True,
    log_path: Optional[str] = None,
) -> bool:
    """Poll the /health endpoint until the server is ready."""
    start = time.time()
    if verbose:
        print(f"Waiting for server at {base_url} (timeout {timeout}s)...")

    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                if verbose:
                    print(f"Server ready after {time.time() - start:.0f}s")
                return True
        except Exception:
            pass

        if verbose:
            print(f"  ...still waiting ({time.time() - start:.0f}s elapsed)")
        time.sleep(sleep)

    print(f"ERROR: server not ready after {timeout}s")
    if log_path:
        tail = _tail(log_path)
        if tail:
            print("--- last log lines ---")
            print(tail)
            print("----------------------")
    return False


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    """Terminate the server process group and wait for GPU memory to free."""
    if proc is None:
        return

    print("Stopping vLLM server...")
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

    # Let the reader thread finish and close the log file.
    reader = getattr(proc, "_reader", None)
    if reader is not None:
        reader.join(timeout=5)

    time.sleep(5)
    print("Server stopped.")
