"""Orchestrate the full optimization benchmark pipeline."""

from typing import List

import torch

from .benchmark import (
    BenchmarkResult,
    clear_memory,
    format_ttft,
    free_model,
    get_memory_usage,
    gpu_memory_report,
)
from .config import (
    DRAFT_MODEL_ID,
    LONG_PROMPTS,
    MODEL_ID,
    MODEL_ID_AWQ,
    PREFIX_PROMPTS,
    SERVER_PORT,
    SHARED_PREFIX,
    TEST_PROMPTS,
)
from .hf import benchmark_hf, load_hf_model
from .server import start_vllm_server, stop_server, wait_for_server
from .server_benchmark import benchmark_server
from .storage import save_results_to_file


def run_hf_baseline(results: List[BenchmarkResult]) -> BenchmarkResult:
    print("Loading model with HuggingFace (FP16)...")
    clear_memory()
    model, tokenizer = load_hf_model(MODEL_ID)
    print(f"Model loaded. Memory: {get_memory_usage():.2f} GB")

    result = benchmark_hf(model, tokenizer, TEST_PROMPTS)
    results.append(result)
    save_results_to_file(results)

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
    proc = start_vllm_server(MODEL_ID)
    wait_for_server(log_path=proc.log_path)

    result = benchmark_server(SERVER_PORT, TEST_PROMPTS, name="vLLM Server (FP16)")
    results.append(result)
    save_results_to_file(results)

    print(f"\n--- vLLM Server (FP16) Results ---")
    print(f"TTFT: {format_ttft(result.ttft_ms)}")
    print(f"TPS: {result.tps:.1f} tokens/sec")
    print(f"TPM: {result.tpm:.0f} tokens/min")
    print(f"Memory: {result.memory_gb:.2f} GB")

    baseline = next((r for r in results if r.name == "HF Baseline"), None)
    if baseline:
        print(f"\n--- Improvement over HF ---")
        print(f"TPS: {result.tps / baseline.tps:.2f}x faster")

    stop_server(proc)
    clear_memory()
    return result


def run_awq_and_flash(results: List[BenchmarkResult]) -> None:
    proc = start_vllm_server(
        MODEL_ID_AWQ,
        quantization="awq",
        gpu_memory_utilization=0.80,
    )
    wait_for_server(log_path=proc.log_path)

    # 3. AWQ
    awq_result = benchmark_server(SERVER_PORT, TEST_PROMPTS, name="vLLM Server + AWQ")
    results.append(awq_result)
    save_results_to_file(results)

    print(f"\n--- AWQ Results ---")
    print(f"TTFT: {format_ttft(awq_result.ttft_ms)}")
    print(f"TPS: {awq_result.tps:.1f} tokens/sec")
    print(f"TPM: {awq_result.tpm:.0f} tokens/min")
    print(f"Memory: {awq_result.memory_gb:.2f} GB")

    fp16_baseline = next((r for r in results if r.name == "vLLM Server (FP16)"), None)
    if fp16_baseline:
        import pandas as pd
        memory_ratio = (fp16_baseline.memory_gb / awq_result.memory_gb) if awq_result.memory_gb > 0 else float("nan")
        print(f"\n--- Improvement ---")
        print(f"Memory: {memory_ratio:.2f}x less" if not pd.isna(memory_ratio) else "Memory: N/A (0 GB allocated)")
        print(f"TPS: {awq_result.tps / fp16_baseline.tps:.2f}x vs vLLM FP16")

    # 4. Flash Attention (longer sequences, same AWQ server)
    flash_result = benchmark_server(SERVER_PORT, LONG_PROMPTS, max_new_tokens=256, name="vLLM Server + AWQ + Flash")
    results.append(flash_result)

    print(f"\n--- Flash Attention Results (longer sequences) ---")
    print(f"TTFT: {format_ttft(flash_result.ttft_ms)}")
    print(f"TPS: {flash_result.tps:.1f} tokens/sec")
    print(f"TPM: {flash_result.tpm:.0f} tokens/min")
    print(f"Memory: {flash_result.memory_gb:.2f} GB")

    stop_server(proc)
    clear_memory()


def run_prefix_caching(results: List[BenchmarkResult]) -> None:
    proc = start_vllm_server(
        MODEL_ID_AWQ,
        quantization="awq",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=True,
    )
    wait_for_server(log_path=proc.log_path)

    print("Running prefix caching benchmark...")
    cold_result = benchmark_server(
        SERVER_PORT, PREFIX_PROMPTS, system_prompt=SHARED_PREFIX, num_runs=1, name="Prefix Cache (cold)"
    )
    warm_result = benchmark_server(
        SERVER_PORT, PREFIX_PROMPTS, system_prompt=SHARED_PREFIX, num_runs=1, name="Prefix Cache (warm)"
    )
    results.append(warm_result)

    print(f"\n--- Prefix Caching Results ---")
    print(f"Cold cache - TTFT: {format_ttft(cold_result.ttft_ms)}, TPS: {cold_result.tps:.1f}, Time: {cold_result.total_time_s:.2f}s")
    print(f"Warm cache - TTFT: {format_ttft(warm_result.ttft_ms)}, TPS: {warm_result.tps:.1f}, Time: {warm_result.total_time_s:.2f}s")
    print(f"\nSpeedup from cache: {cold_result.total_time_s / warm_result.total_time_s:.2f}x")

    stop_server(proc)
    clear_memory()


def run_speculative(results: List[BenchmarkResult]) -> None:
    print(f"Target model: {MODEL_ID_AWQ}")
    print(f"Draft model: {DRAFT_MODEL_ID}")

    proc = start_vllm_server(
        MODEL_ID_AWQ,
        quantization="awq",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        speculative_model=DRAFT_MODEL_ID,
        num_speculative_tokens=4,
    )
    wait_for_server(log_path=proc.log_path)

    spec_result = benchmark_server(SERVER_PORT, TEST_PROMPTS, name="vLLM Server + AWQ + Spec")
    results.append(spec_result)
    save_results_to_file(results)

    print(f"\n--- Speculative Decoding Results ---")
    print(f"TTFT: {format_ttft(spec_result.ttft_ms)}")
    print(f"TPS: {spec_result.tps:.1f} tokens/sec")
    print(f"TPM: {spec_result.tpm:.0f} tokens/min")
    print(f"Memory: {spec_result.memory_gb:.2f} GB")

    awq_baseline = next((r for r in results if r.name == "vLLM Server + AWQ"), None)
    if awq_baseline:
        print(f"\n--- Improvement over AWQ baseline ---")
        print(f"TPS: {spec_result.tps / awq_baseline.tps:.2f}x faster")

    stop_server(proc)
    clear_memory()


def run_combined(results: List[BenchmarkResult]) -> BenchmarkResult:
    proc = start_vllm_server(
        MODEL_ID_AWQ,
        quantization="awq",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=True,
        speculative_model=DRAFT_MODEL_ID,
        num_speculative_tokens=4,
    )
    wait_for_server(log_path=proc.log_path)

    print("\nOptimizations enabled:")
    print("  [x] Continuous Batching (vLLM default)")
    print("  [x] PagedAttention (vLLM default)")
    print("  [x] AWQ 4-bit Quantization")
    print("  [x] Flash Attention (auto-enabled)")
    print("  [x] Prefix Caching")
    print("  [x] Speculative Decoding (4 tokens)")

    combined_result = benchmark_server(SERVER_PORT, TEST_PROMPTS, name="Combined Stack")
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

    stop_server(proc)
    clear_memory()
    return combined_result


def run_full_pipeline() -> List[BenchmarkResult]:
    """Run every benchmark section in order and return the collected results."""
    results: List[BenchmarkResult] = []

    run_hf_baseline(results)
    run_vllm_fp16(results)
    run_awq_and_flash(results)
    run_prefix_caching(results)
    run_speculative(results)
    run_combined(results)

    save_results_to_file(results)
    return results
