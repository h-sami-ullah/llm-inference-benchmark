"""Offline vLLM (`LLM` Python API) benchmark — the previous notebook converted to code.

This package benchmarks the same optimization stack as `llm_benchmark`, but using
vLLM's in-process `LLM` engine (`LLM.generate`) instead of a subprocess HTTP server.

It reuses the shared primitives (config, BenchmarkResult, storage, HuggingFace
baseline) from `llm_benchmark` and only adds the offline-vLLM specifics.

Also vendored here are vLLM's official offline benchmark scripts
(`benchmark_throughput.py`, `benchmark_latency.py`) with `cli()` entry points.
"""

from . import benchmark_latency, benchmark_throughput
from .pipeline import run_full_pipeline
from .vllm_offline import benchmark_prefix_caching, benchmark_vllm

__all__ = [
    "benchmark_latency",
    "benchmark_prefix_caching",
    "benchmark_throughput",
    "benchmark_vllm",
    "run_full_pipeline",
]
