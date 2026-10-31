"""Offline vLLM (`LLM` Python API) benchmark — the previous notebook converted to code.

This package benchmarks the same optimization stack as `llm_benchmark`, but using
vLLM's in-process `LLM` engine (`LLM.generate`) instead of a subprocess HTTP server.

It reuses the shared primitives (config, BenchmarkResult, storage, HuggingFace
baseline) from `llm_benchmark` and only adds the offline-vLLM specifics.

Also exposed here are thin wrappers around vLLM's built-in offline benchmarks
(``vllm bench throughput`` / ``vllm bench latency``), which supersede the
previously vendored standalone scripts.
"""

import subprocess
from typing import Optional, Sequence

from llm_benchmark.server import vllm_cli

from .pipeline import run_full_pipeline
from .vllm_offline import benchmark_prefix_caching, benchmark_vllm

__all__ = [
    "benchmark_latency",
    "benchmark_prefix_caching",
    "benchmark_throughput",
    "benchmark_vllm",
    "run_full_pipeline",
]


def _run_bench(subcommand: str, bench_args: Optional[Sequence[str]]) -> None:
    cmd = list(vllm_cli("bench", subcommand, *(bench_args or [])))
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def benchmark_throughput(bench_args: Optional[Sequence[str]] = None) -> None:
    """Run `vllm bench throughput` (offline engine throughput)."""
    _run_bench("throughput", bench_args)


def benchmark_latency(bench_args: Optional[Sequence[str]] = None) -> None:
    """Run `vllm bench latency` (offline single-batch latency)."""
    _run_bench("latency", bench_args)
