"""Orchestrate the offline vLLM (`LLM` API) optimization benchmark pipeline."""

import os
from typing import List

import torch
from vllm import LLM

from llm_benchmark.benchmark import (
    BenchmarkResult,
    clear_memory,
    format_ttft,
    free_model,
    get_memory_usage,
    gpu_memory_report,
)
from llm_benchmark.config import (
    DRAFT_MODEL_ID,
    LONG_PROMPTS,
    MODEL_ID,
    MODEL_ID_AWQ,
    PREFIX_PROMPTS,
    SHARED_PREFIX,
    TEST_PROMPTS,
)
from llm_benchmark.hf import benchmark_hf, load_hf_model
from llm_benchmark.storage import save_results_to_file

from .vllm_offline import benchmark_prefix_caching, benchmark_vllm

OFFLINE_RESULTS_FILE_PATH = os.environ.get(
    "LLM_BENCHMARK_OFFLINE_RESULTS_PATH",
    "/content/benchmark_results_qwen.json",
)


def run_hf_baseline(results: List[BenchmarkResult]) -> BenchmarkResult:
    print("Loading model with HuggingFace (FP16)...")
    clear_memory()
    model, tokenizer = load_hf_model(MODEL_ID)
    print(f"Model loaded. Memory: {get_memory_usage():.2f} GB")

    result = benchmark_hf(model, tokenizer, TEST_PROMPTS)
    results.append(result)
    save_results_to_file(results, OFFLINE_RESULTS_FILE_PATH)

    print(f"\n--- HuggingFace Baseline Results ---")
    print(f"TTFT: {format_ttft(result.ttft_ms)}")
    print(f"TPS: {result.tps:.1f} tokens/sec")
    print(f"TPM: {result.tpm:.0f} tokens/min")
    print(f"Memory: {result.memory_gb:.2f} GB")

    free_model(model)
    del model
    del tokenizer
    clear_memory()
    print(f"Memory after cleanup: {gpu_memory_report()}")
    return result


def run_vllm_fp16(results: List[BenchmarkResult]) -> BenchmarkResult:
    print("Loading model with vLLM (FP16)...")
    clear_memory()

    vllm_model = LLM(
        model=MODEL_ID,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
    )
    print(f"{MODEL_ID} loaded successfully with vLLM!")

    result = benchmark_vllm(vllm_model, TEST_PROMPTS, name="vLLM (FP16)")
    results.append(result)
    save_results_to_file(results, OFFLINE_RESULTS_FILE_PATH)

    print(f"\n--- vLLM Baseline Results ---")
    print(f"TTFT: {format_ttft(result.ttft_ms)}")
    print(f"TPS: {result.tps:.1f} tokens/sec")
    print(f"TPM: {result.tpm:.0f} tokens/min")
    print(f"Memory: {result.memory_gb:.2f} GB")

    baseline = next((r for r in results if r.name == "HF Baseline"), None)
    if baseline:
        print(f"\n--- Improvement over HF ---")
        print(f"TPS: {result.tps / baseline.tps:.2f}x faster")

    del vllm_model
    clear_memory()
    print(f"Memory after cleanup: {get_memory_usage():.2f} GB")
    return result


def run_awq_and_flash(results: List[BenchmarkResult]) -> None:
    print("Loading AWQ quantized model with vLLM...")
    clear_memory()

    vllm_awq = LLM(
        model=MODEL_ID_AWQ,
        dtype="float16",
        max_model_len=2048,
        quantization="awq",
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
    )
    print("AWQ model loaded.")

    # 3. AWQ
    awq_result = benchmark_vllm(vllm_awq, TEST_PROMPTS, name="vLLM + AWQ")
    results.append(awq_result)
    save_results_to_file(results, OFFLINE_RESULTS_FILE_PATH)

    print(f"\n--- AWQ Results ---")
    print(f"TTFT: {format_ttft(awq_result.ttft_ms)}")
    print(f"TPS: {awq_result.tps:.1f} tokens/sec")
    print(f"TPM: {awq_result.tpm:.0f} tokens/min")
    print(f"Memory: {awq_result.memory_gb:.2f} GB")

    fp16_baseline = next((r for r in results if r.name == "vLLM (FP16)"), None)
    if fp16_baseline:
        import pandas as pd
        memory_ratio = (fp16_baseline.memory_gb / awq_result.memory_gb) if awq_result.memory_gb > 0 else float("nan")
        print(f"\n--- Improvement ---")
        print(f"Memory: {memory_ratio:.2f}x less" if not pd.isna(memory_ratio) else "Memory: N/A (0 GB allocated)")
        print(f"TPS: {awq_result.tps / fp16_baseline.tps:.2f}x vs vLLM FP16")

    # 4. Flash Attention (longer sequences, same AWQ engine)
    flash_result = benchmark_vllm(
        vllm_awq, LONG_PROMPTS, max_new_tokens=256, name="vLLM + AWQ + Flash"
    )
    results.append(flash_result)

    print(f"\n--- Flash Attention Results (longer sequences) ---")
    print(f"TTFT: {format_ttft(flash_result.ttft_ms)}")
    print(f"TPS: {flash_result.tps:.1f} tokens/sec")
    print(f"TPM: {flash_result.tpm:.0f} tokens/min")
    print(f"Memory: {flash_result.memory_gb:.2f} GB")

    del vllm_awq
    clear_memory()


def run_prefix_caching(results: List[BenchmarkResult]) -> None:
    print("Loading model with prefix caching enabled...")
    clear_memory()

    vllm_prefix = LLM(
        model=MODEL_ID_AWQ,
        dtype="float16",
        quantization="awq",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=True,
    )
    print("Model loaded with prefix caching.")

    print("Running prefix caching benchmark...")
    prefix_results = benchmark_prefix_caching(vllm_prefix, PREFIX_PROMPTS, SHARED_PREFIX)
    results.append(prefix_results["warm"])

    print(f"\n--- Prefix Caching Results ---")
    print(f"Cold cache - TTFT: {format_ttft(prefix_results['cold'].ttft_ms)}, TPS: {prefix_results['cold'].tps:.1f}, Time: {prefix_results['cold'].total_time_s:.2f}s")
    print(f"Warm cache - TTFT: {format_ttft(prefix_results['warm'].ttft_ms)}, TPS: {prefix_results['warm'].tps:.1f}, Time: {prefix_results['warm'].total_time_s:.2f}s")
    print(f"\nSpeedup from cache: {prefix_results['cold'].total_time_s / prefix_results['warm'].total_time_s:.2f}x")

    del vllm_prefix
    clear_memory()


def run_speculative(results: List[BenchmarkResult]) -> None:
    print("Loading model with speculative decoding...")
    print(f"Target model: {MODEL_ID_AWQ}")
    print(f"Draft model: {DRAFT_MODEL_ID}")
    clear_memory()

    vllm_spec = LLM(
        model=MODEL_ID_AWQ,
        dtype="float16",
        quantization="awq",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        spec_model=DRAFT_MODEL_ID,
        spec_tokens=4,
    )
    print("Model loaded with speculative decoding.")

    spec_result = benchmark_vllm(vllm_spec, TEST_PROMPTS, name="vLLM + AWQ + Spec")
    results.append(spec_result)
    save_results_to_file(results, OFFLINE_RESULTS_FILE_PATH)

    print(f"\n--- Speculative Decoding Results ---")
    print(f"TTFT: {format_ttft(spec_result.ttft_ms)}")
    print(f"TPS: {spec_result.tps:.1f} tokens/sec")
    print(f"TPM: {spec_result.tpm:.0f} tokens/min")
    print(f"Memory: {spec_result.memory_gb:.2f} GB")

    awq_baseline = next((r for r in results if r.name == "vLLM + AWQ"), None)
    if awq_baseline:
        print(f"\n--- Improvement over AWQ baseline ---")
        print(f"TPS: {spec_result.tps / awq_baseline.tps:.2f}x faster")

    del vllm_spec
    clear_memory()


def run_combined(results: List[BenchmarkResult]) -> BenchmarkResult:
    print("Loading model with ALL optimizations...")
    clear_memory()

    vllm_combined = LLM(
        model=MODEL_ID_AWQ,
        dtype="float16",
        quantization="awq",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=True,
        spec_model=DRAFT_MODEL_ID,
        spec_tokens=4,
    )

    print("\nOptimizations enabled:")
    print("  [x] Continuous Batching (vLLM default)")
    print("  [x] PagedAttention (vLLM default)")
    print("  [x] AWQ 4-bit Quantization")
    print("  [x] Flash Attention (auto-enabled)")
    print("  [x] Prefix Caching")
    print("  [x] Speculative Decoding (4 tokens)")

    combined_result = benchmark_vllm(vllm_combined, TEST_PROMPTS, name="Combined Stack")
    results.append(combined_result)

    print(f"\n{'=' * 50}")
    print("COMBINED STACK RESULTS")
    print("=" * 50)
    ttft_status = ("PASS" if combined_result.ttft_ms < 1000 else "FAIL") if combined_result.ttft_ms is not None else "NOT MEASURED"
    print(f"TTFT: {format_ttft(combined_result.ttft_ms)} {ttft_status} (target: <1000ms)")
    print(f"TPS: {combined_result.tps:.1f} tokens/sec {'PASS' if combined_result.tps > 60 else 'FAIL'} (target: >60)")
    print(f"TPM: {combined_result.tpm:.0f} tokens/min")
    print(f"Memory: {combined_result.memory_gb:.2f} GB")
    print("=" * 50)

    del vllm_combined
    clear_memory()
    return combined_result


def run_full_pipeline() -> List[BenchmarkResult]:
    """Run every offline benchmark section in order and return the collected results."""
    results: List[BenchmarkResult] = []

    run_hf_baseline(results)
    run_vllm_fp16(results)
    run_awq_and_flash(results)
    run_prefix_caching(results)
    run_speculative(results)
    run_combined(results)

    save_results_to_file(results, OFFLINE_RESULTS_FILE_PATH)
    return results
