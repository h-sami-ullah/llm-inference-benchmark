"""Orchestrate the offline vLLM benchmark via subprocess isolation.

Each stage (HuggingFace baseline and every vLLM ``LLM`` section) runs in its own
child process. When a child exits, its CUDA context is destroyed and all of its
GPU allocations are released, which is far more reliable than relying on Python
garbage collection — vLLM manages its own memory pools and ``empty_cache()``
cannot fully reclaim them.
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import List

from llm_benchmark.benchmark import BenchmarkResult, gpu_memory_report
from llm_benchmark.config import MODEL_ID
from llm_benchmark.hf import run_hf_baseline_isolated
from llm_benchmark.storage import save_results_to_file

OFFLINE_RESULTS_FILE_PATH = os.environ.get(
    "LLM_BENCHMARK_OFFLINE_RESULTS_PATH",
    "/content/benchmark_results_qwen.json",
)

# Order matches the original notebook's sections. "prefix" yields cold+warm;
# the orchestrator keeps only the warm result (cold is just cache warm-up).
VLLM_TASKS = ["fp16", "awq", "flash", "prefix", "spec", "combined"]


def _run_task_isolated(task: str) -> List[BenchmarkResult]:
    out_path = os.path.join(tempfile.gettempdir(), f"vllm_{task}_result.json")
    cmd = [
        sys.executable,
        "-m",
        "llm_benchmark_offline.worker",
        "--task",
        task,
        "--output",
        out_path,
    ]
    print(f"\n{'=' * 60}\nBenchmarking task: {task}\n{'=' * 60}")
    subprocess.run(cmd, check=True)
    with open(out_path) as f:
        data = json.load(f)
    return [BenchmarkResult(**d) for d in data]


def run_hf_baseline(results: List[BenchmarkResult]) -> BenchmarkResult:
    result = run_hf_baseline_isolated(MODEL_ID)
    results.append(result)
    save_results_to_file(results, OFFLINE_RESULTS_FILE_PATH)
    print(f"Memory after HF baseline: {gpu_memory_report()}")
    return result


def run_full_pipeline() -> List[BenchmarkResult]:
    """Run HF baseline + every vLLM section, each isolated in its own process."""
    results: List[BenchmarkResult] = []

    run_hf_baseline(results)

    for task in VLLM_TASKS:
        for r in _run_task_isolated(task):
            if r.name == "Prefix Cache (cold)":
                continue
            results.append(r)
            save_results_to_file(results, OFFLINE_RESULTS_FILE_PATH)

    save_results_to_file(results, OFFLINE_RESULTS_FILE_PATH)
    return results
