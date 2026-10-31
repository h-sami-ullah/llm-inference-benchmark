"""LLM inference optimization benchmark.

Provides reusable building blocks to benchmark HuggingFace and vLLM (as an
OpenAI-compatible subprocess server) inference optimizations, plus a
`run_full_pipeline` orchestrator.
"""

from .benchmark import (
    BenchmarkResult,
    clear_memory,
    display_results,
    format_ttft,
    free_model,
    get_memory_usage,
    set_seed,
)
from .config import setup_environment
from .hf import benchmark_hf, load_hf_model
from .pipeline import run_full_pipeline
from .server import start_vllm_server, stop_server, wait_for_server
from .server_benchmark import benchmark_server, get_client
from .storage import load_results_from_file, save_results_to_file

__all__ = [
    "BenchmarkResult",
    "benchmark_hf",
    "benchmark_server",
    "clear_memory",
    "display_results",
    "format_ttft",
    "free_model",
    "get_client",
    "get_memory_usage",
    "load_hf_model",
    "load_results_from_file",
    "run_full_pipeline",
    "save_results_to_file",
    "set_seed",
    "setup_environment",
    "start_vllm_server",
    "stop_server",
    "wait_for_server",
]

__version__ = "0.1.0"
