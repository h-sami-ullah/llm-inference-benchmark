# LLM Inference Optimization Benchmark

Reusable Python package for benchmarking LLM inference optimizations, with a thin
notebook driver on top.

The vLLM sections launch `vllm serve` as a background subprocess (an
OpenAI-compatible HTTP server) and benchmark it through the OpenAI client — the
same pattern used in production deployments.

## Layout

```
llm_benchmark/
  config.py            # model ids, prompts, server settings, env setup
  benchmark.py         # BenchmarkResult, memory/seed helpers, table display
  storage.py           # save/load results as JSON
  hf.py                # HuggingFace baseline benchmark
  server.py            # start/wait/stop a `vllm serve` subprocess
  server_benchmark.py  # OpenAI-client benchmark (TTFT / TPS / TPM / memory)
  pipeline.py          # run_full_pipeline() orchestrator
notebooks/
  inference_optimization.ipynb   # thin driver notebook
```

## Install

```bash
pip install -e .
```

## Usage (in a notebook or script)

```python
from llm_benchmark import (
    setup_environment, set_seed, run_full_pipeline,
    display_results, save_results_to_file,
)

setup_environment()
set_seed(42)
results = run_full_pipeline()
display_results(results)
save_results_to_file(results)
```

Or piecemeal:

```python
from llm_benchmark import start_vllm_server, wait_for_server, stop_server, benchmark_server

proc = start_vllm_server("Qwen/Qwen2.5-3B-Instruct")
wait_for_server()
r = benchmark_server(8000, ["Explain quantum computing in one sentence."], name="FP16")
stop_server(proc)
```

## Notes

- Requires a CUDA GPU (defaults to device 0).
- Results persist to `/content/benchmark_results_qwen_server.json` by default;
  override with the `LLM_BENCHMARK_RESULTS_PATH` env var.
- Each server section tears down its process before the next; logs go to
  `/content/vllm_server_8000.log`.
