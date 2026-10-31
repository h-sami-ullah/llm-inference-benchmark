"""Run a single offline vLLM benchmark section in a fresh subprocess.

Each section loads its own vLLM ``LLM()`` engine, benchmarks it, writes the
result as JSON, and exits — so the OS reclaims vLLM's GPU memory pools (weights
+ KV cache) on process exit. This is more reliable than ``del llm`` +
``torch.cuda.empty_cache()``, which does not fully release vLLM's allocations
and lets one section's footprint leak into the next.
"""

import argparse
import json
from dataclasses import asdict
from typing import List

import torch
from vllm import LLM

from llm_benchmark.benchmark import BenchmarkResult, format_ttft
from llm_benchmark.config import (
    DRAFT_MODEL_ID,
    LONG_PROMPTS,
    MODEL_ID,
    MODEL_ID_AWQ,
    PREFIX_PROMPTS,
    SHARED_PREFIX,
    TEST_PROMPTS,
    setup_environment,
)
from .vllm_offline import benchmark_prefix_caching, benchmark_vllm


def _llm(**kwargs) -> LLM:
    kwargs.setdefault("dtype", "float16")
    return LLM(**kwargs)


def run_task(task: str) -> List[BenchmarkResult]:
    torch.cuda.reset_peak_memory_stats()

    if task == "fp16":
        llm = _llm(
            model=MODEL_ID,
            max_model_len=2048,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
        )
        return [benchmark_vllm(llm, TEST_PROMPTS, name="vLLM (FP16)")]

    if task == "awq":
        llm = _llm(
            model=MODEL_ID_AWQ,
            max_model_len=2048,
            quantization="awq",
            gpu_memory_utilization=0.80,
            enforce_eager=True,
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
        )
        return [benchmark_vllm(llm, TEST_PROMPTS, name="vLLM + AWQ")]

    if task == "flash":
        llm = _llm(
            model=MODEL_ID_AWQ,
            max_model_len=2048,
            quantization="awq",
            gpu_memory_utilization=0.80,
            enforce_eager=True,
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
        )
        return [benchmark_vllm(llm, LONG_PROMPTS, max_new_tokens=256, name="vLLM + AWQ + Flash")]

    if task == "prefix":
        llm = _llm(
            model=MODEL_ID_AWQ,
            quantization="awq",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enable_prefix_caching=True,
        )
        r = benchmark_prefix_caching(llm, PREFIX_PROMPTS, SHARED_PREFIX)
        return [r["cold"], r["warm"]]

    if task == "spec":
        llm = _llm(
            model=MODEL_ID_AWQ,
            quantization="awq",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            spec_model=DRAFT_MODEL_ID,
            spec_tokens=4,
        )
        return [benchmark_vllm(llm, TEST_PROMPTS, name="vLLM + AWQ + Spec")]

    if task == "combined":
        llm = _llm(
            model=MODEL_ID_AWQ,
            quantization="awq",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enable_prefix_caching=True,
            spec_model=DRAFT_MODEL_ID,
            spec_tokens=4,
        )
        return [benchmark_vllm(llm, TEST_PROMPTS, name="Combined Stack")]

    raise ValueError(f"Unknown task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    setup_environment()
    results = run_task(args.task)

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_res = torch.cuda.max_memory_reserved() / 1e9

    for r in results:
        print(f"[{r.name}] TTFT {format_ttft(r.ttft_ms)}, TPS {r.tps:.1f}, "
              f"TPM {r.tpm:.0f}, Memory {r.memory_gb:.2f} GB")
    print(f"Peak VRAM (torch): allocated {peak_alloc:.2f} GB, reserved {peak_res:.2f} GB")

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f)


if __name__ == "__main__":
    main()
