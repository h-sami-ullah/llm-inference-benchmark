"""Online serving benchmark via vLLM's built-in `vllm bench serve` CLI.

vLLM ships a built-in benchmark CLI (`pip install "vllm[bench]"`):

- ``vllm bench serve``      — API server throughput/latency (online)
- ``vllm bench throughput`` — engine throughput (offline)
- ``vllm bench latency``    — single-batch latency (offline)
- ``vllm bench forward``    — model forward-pass timing

This module wraps ``vllm bench serve`` with :func:`benchmark_model`, which
starts/stops the server for you. It replaces the previously vendored
``benchmark_serving.py`` (obsolete in current vLLM).
"""

import subprocess
from typing import Optional, Sequence

from llm_benchmark.server import start_vllm_server, stop_server, vllm_cli, wait_for_server

__all__ = ["benchmark_model"]


def bench_cmd(subcommand: str, *args: str) -> Sequence[str]:
    """Build a ``vllm bench <subcommand> <args...>`` command."""
    return vllm_cli("bench", subcommand, *args)


def benchmark_model(
    model_id: str,
    port: int = 8000,
    subcommand: str = "serve",
    bench_args: Optional[Sequence[str]] = None,
    **server_kwargs,
) -> None:
    """Start a vLLM server, run `vllm bench <subcommand>`, then stop the server.

    ``server_kwargs`` are forwarded to :func:`llm_benchmark.start_vllm_server`
    (e.g. ``quantization="awq"``, ``gpu_memory_utilization=0.90``).
    ``bench_args`` are appended verbatim after ``vllm bench <subcommand>``.

    Example:
        benchmark_model(
            "Qwen/Qwen2.5-3B-Instruct",
            bench_args=[
                "--backend", "openai",
                "--endpoint", "/v1/completions",
                "--model", "Qwen/Qwen2.5-3B-Instruct",
                "--dataset-name", "random",
                "--random-input-len", "1024",
                "--random-output-len", "256",
                "--num-prompts", "100",
                "--max-concurrency", "8",
                "--save-result",
            ],
        )
    """
    proc = start_vllm_server(model_id, port=port, **server_kwargs)
    try:
        wait_for_server(log_path=proc.log_path)
        cmd = list(bench_cmd(subcommand, *(bench_args or [])))
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)
    finally:
        stop_server(proc)
