"""vLLM's official online-serving benchmark, vendored and wired to this package.

Contains two files vendored from the vLLM project (same commit/version so they
stay compatible):

- ``backend_request_func.py`` — async request functions + request dataclasses.
- ``benchmark_serving.py`` — the throughput/latency benchmark CLI.

Usage:

- CLI:      ``python -m llm_benchmark.serving.benchmark_serving --help``
- Console:  ``llm-benchmark-serve --help`` (after ``pip install -e .``)
- Programmatic: :func:`benchmark_model` (starts a server, benchmarks, tears down).
"""

import argparse
from typing import List, Optional

from . import benchmark_serving

__all__ = ["benchmark_model", "benchmark_serving"]


def _namespace(
    *,
    backend: str,
    host: str,
    port: int,
    endpoint: str,
    model: str,
    tokenizer: Optional[str],
    dataset_name: str,
    num_prompts: int,
    random_input_len: int,
    random_output_len: int,
    request_rate: float,
    best_of: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    save_result: bool,
    result_dir: Optional[str],
    seed: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        backend=backend,
        base_url=None,
        host=host,
        port=port,
        endpoint=endpoint,
        dataset=None,
        dataset_name=dataset_name,
        dataset_path=None,
        model=model,
        tokenizer=tokenizer,
        best_of=best_of,
        use_beam_search=use_beam_search,
        num_prompts=num_prompts,
        sharegpt_output_len=None,
        sonnet_input_len=550,
        sonnet_output_len=150,
        sonnet_prefix_len=200,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        random_range_ratio=1.0,
        request_rate=request_rate,
        seed=seed,
        trust_remote_code=trust_remote_code,
        disable_tqdm=False,
        save_result=save_result,
        metadata=None,
        result_dir=result_dir,
        result_filename=None,
    )


def benchmark_model(
    model_id: str,
    port: int = 8000,
    backend: str = "vllm",
    dataset_name: str = "random",
    num_prompts: int = 100,
    random_input_len: int = 1024,
    random_output_len: int = 128,
    request_rate: float = float("inf"),
    best_of: int = 1,
    use_beam_search: bool = False,
    trust_remote_code: bool = True,
    save_result: bool = True,
    result_dir: Optional[str] = None,
    seed: int = 0,
    **server_kwargs,
) -> None:
    """Start a vLLM server, run vLLM's official serving benchmark, then stop it.

    ``server_kwargs`` are forwarded to :func:`llm_benchmark.start_vllm_server`
    (e.g. ``quantization="awq"``, ``gpu_memory_utilization=0.90``).
    """
    from llm_benchmark.server import start_vllm_server, stop_server, wait_for_server

    proc = start_vllm_server(model_id, port=port, **server_kwargs)
    try:
        wait_for_server(log_path=proc.log_path)
        args = _namespace(
            backend=backend,
            host="localhost",
            port=port,
            endpoint="/v1/completions",
            model=model_id,
            tokenizer=None,
            dataset_name=dataset_name,
            num_prompts=num_prompts,
            random_input_len=random_input_len,
            random_output_len=random_output_len,
            request_rate=request_rate,
            best_of=best_of,
            use_beam_search=use_beam_search,
            trust_remote_code=trust_remote_code,
            save_result=save_result,
            result_dir=result_dir,
            seed=seed,
        )
        benchmark_serving.main(args)
    finally:
        stop_server(proc)
