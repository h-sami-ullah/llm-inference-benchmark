"""Offline vLLM (`LLM` Python API) benchmarking helpers."""

import statistics
import time
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams

from llm_benchmark.benchmark import BenchmarkResult, get_memory_usage
from llm_benchmark.config import SYSTEM_PROMPT


def format_vllm_prompts(llm: LLM, prompts: List[str], system_prompt: Optional[str] = None) -> List[str]:
    """Format prompts with the tokenizer/chat template belonging to this vLLM model."""
    system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
    tokenizer = llm.get_tokenizer()
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def get_vllm_ttft_ms(outputs) -> List[float]:
    """
    Read vLLM's native per-request first_token_latency.

    In vLLM 0.26 this is a duration in seconds (RequestOutput.metrics
    .first_token_latency), not an absolute timestamp. Returns None entries are
    filtered out; may return an empty list if the build attached no metrics.
    """
    ttft_values = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        latency_s = getattr(metrics, "first_token_latency", None) if metrics else None
        if latency_s is not None and latency_s > 0:
            ttft_values.append(latency_s * 1000.0)
    return ttft_values


def benchmark_vllm(
    llm: LLM,
    prompts: List[str],
    max_new_tokens: int = 128,
    num_runs: int = 3,
    name: str = "vLLM",
) -> BenchmarkResult:
    """
    Benchmark vLLM offline LLM.generate().

    TTFT: Mean native per-request first_token_latency from RequestOutput.metrics.
    TPS: Aggregate generated tokens / wall-clock batch duration.
    Memory: Physical GPU memory currently used (NVML).
    """
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=max_new_tokens,
    )

    formatted = format_vllm_prompts(llm, prompts)

    # Warm-up run (excluded from measurements).
    _ = llm.generate(
        formatted[:1],
        SamplingParams(temperature=0.0, max_tokens=8),
        use_tqdm=False,
    )

    all_ttft = []
    all_tps = []
    all_tokens = []
    all_times = []

    for _ in range(num_runs):
        start = time.perf_counter()

        outputs = llm.generate(formatted, sampling_params, use_tqdm=False)

        # LLM.generate() is synchronous and returns after all requests finish.
        total_time = time.perf_counter() - start

        total_new_tokens = sum(
            len(request_output.outputs[0].token_ids)
            for request_output in outputs
        )

        run_ttfts = get_vllm_ttft_ms(outputs)
        all_ttft.extend(run_ttfts)

        all_tps.append(total_new_tokens / total_time)
        all_tokens.append(total_new_tokens)
        all_times.append(total_time)

    avg_ttft = statistics.mean(all_ttft) if all_ttft else None
    avg_tps = statistics.mean(all_tps)
    avg_tokens = statistics.mean(all_tokens)
    avg_time = statistics.mean(all_times)

    return BenchmarkResult(
        name=name,
        ttft_ms=avg_ttft,
        tps=avg_tps,
        tpm=avg_tps * 60.0,
        memory_gb=get_memory_usage(),
        total_tokens=int(round(avg_tokens)),
        total_time_s=avg_time,
    )


def benchmark_prefix_caching(
    llm: LLM,
    prompts: List[str],
    shared_prefix: str,
    max_new_tokens: int = 128,
    num_runs: int = 3,
) -> Dict[str, BenchmarkResult]:
    """Compare first (cold) and subsequent (warm) prefix-cache passes."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    formatted = format_vllm_prompts(llm, prompts, system_prompt=shared_prefix)

    def run_once(label: str) -> BenchmarkResult:
        start = time.perf_counter()
        outputs = llm.generate(formatted, sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - start

        token_count = sum(
            len(request_output.outputs[0].token_ids)
            for request_output in outputs
        )

        ttft_values = get_vllm_ttft_ms(outputs)
        ttft_ms = statistics.mean(ttft_values) if ttft_values else None
        tps = token_count / elapsed

        return BenchmarkResult(
            name=label,
            ttft_ms=ttft_ms,
            tps=tps,
            tpm=tps * 60.0,
            memory_gb=get_memory_usage(),
            total_tokens=token_count,
            total_time_s=elapsed,
        )

    # First pass populates the cache; second pass can reuse shared-prefix KV blocks.
    cold_result = run_once("Prefix Cache (cold)")
    warm_result = run_once("Prefix Cache (warm)")

    return {"cold": cold_result, "warm": warm_result}
