"""Benchmark a running vLLM OpenAI-compatible server via the OpenAI client."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from openai import OpenAI

from .benchmark import BenchmarkResult, get_memory_usage
from .config import SERVER_PORT, SYSTEM_PROMPT


def get_client(port: int = SERVER_PORT) -> OpenAI:
    return OpenAI(base_url=f"http://localhost:{port}/v1", api_key="token-abc123")


def get_served_model_name(client: OpenAI) -> str:
    """vLLM serves the model under its HF id; resolve it via the models endpoint."""
    return client.models.list().data[0].id


def _messages(prompt: str, system_prompt: Optional[str] = None) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT if system_prompt is None else system_prompt},
        {"role": "user", "content": prompt},
    ]


def measure_ttft_streaming(
    client: OpenAI,
    model_name: str,
    messages: List[dict],
    max_tokens: int,
) -> Optional[float]:
    """Host-observed TTFT: time from request start to first non-empty content delta.

    The stream is always closed (via ``finally``) once the first token is seen —
    leaving it open leaks the HTTP connection and makes the server keep
    generating tokens for a request nobody is reading, which eventually exhausts
    the client's connection pool and hangs the benchmark.
    """
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )
    try:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                return (time.perf_counter() - start) * 1000.0
    finally:
        stream.close()
    return None


def _completion_tokens(
    client: OpenAI,
    model_name: str,
    messages: List[dict],
    max_tokens: int,
) -> int:
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.usage.completion_tokens


def run_concurrent_batch(
    client: OpenAI,
    model_name: str,
    messages_list: List[List[dict]],
    max_tokens: int,
) -> int:
    """Submit all requests concurrently to stress continuous batching."""
    with ThreadPoolExecutor(max_workers=len(messages_list)) as ex:
        tokens = list(
            ex.map(lambda m: _completion_tokens(client, model_name, m, max_tokens), messages_list)
        )
    return sum(tokens)


def benchmark_server(
    port: int,
    prompts: List[str],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 128,
    num_runs: int = 3,
    name: str = "vLLM Server",
) -> BenchmarkResult:
    """
    Benchmark a running vLLM OpenAI-compatible server.

    TTFT: Mean host-observed time to first token (streaming), one request at a time.
    TPS:  Aggregate generated tokens / wall-clock time with all requests submitted concurrently.
    Memory: Physical GPU memory currently used (NVML, device-wide).
    """
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    client = get_client(port)
    model_name = get_served_model_name(client)
    messages_list = [_messages(p, system_prompt) for p in prompts]

    # Warm-up (excluded from measurements)
    _ = client.chat.completions.create(
        model=model_name, messages=messages_list[0], max_tokens=8, temperature=0.0
    )

    all_ttft = []
    all_tps = []
    all_tokens = []
    all_times = []

    for _ in range(num_runs):
        # 1. TTFT: one prompt at a time via streaming
        run_ttfts = []
        for messages in messages_list:
            ttft = measure_ttft_streaming(client, model_name, messages, max_new_tokens)
            if ttft is not None:
                run_ttfts.append(ttft)
        all_ttft.extend(run_ttfts)

        # 2. TPS: concurrent batch, aggregate throughput
        start = time.perf_counter()
        total_new_tokens = run_concurrent_batch(
            client, model_name, messages_list, max_new_tokens
        )
        total_time = time.perf_counter() - start

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
