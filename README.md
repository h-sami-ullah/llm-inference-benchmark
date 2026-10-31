# LLM Inference Optimization Benchmark

Reusable Python package for benchmarking LLM inference optimizations, with a thin
notebook driver on top.

Two complementary packages benchmark the same optimization stack (HuggingFace
baseline, vLLM continuous batching/PagedAttention, AWQ, Flash Attention, prefix
caching, speculative decoding, combined):

- `llm_benchmark` — launches `vllm serve` as a background subprocess (an
  OpenAI-compatible HTTP server) and benchmarks it through the OpenAI client,
  mirroring production deployments.
- `llm_benchmark_offline` — uses vLLM's in-process `LLM` Python API
  (`LLM.generate`) for the same measurements.

## Layout

```
llm_benchmark/
  config.py            # model ids, prompts, server settings, env setup
  benchmark.py         # BenchmarkResult, memory/seed helpers, table display
  storage.py           # save/load results as JSON
  hf.py                # HuggingFace baseline benchmark
  server.py            # start/wait/stop a `vllm serve` subprocess
  server_benchmark.py  # OpenAI-client benchmark (TTFT / TPS / TPM / memory)
  pipeline.py          # run_full_pipeline() orchestrator (server)
llm_benchmark_offline/
  vllm_offline.py          # offline LLM.generate() benchmark helpers
  pipeline.py              # run_full_pipeline() orchestrator (offline)
  benchmark_throughput.py  # vLLM's offline throughput benchmark (vendored + CLI)
  benchmark_latency.py     # vLLM's single-batch latency benchmark (vendored + CLI)
llm_benchmark/serving/
  backend_request_func.py  # vLLM's async request functions (vendored)
  benchmark_serving.py     # vLLM's official serving benchmark (vendored + CLI)
  sonnet.txt               # poem-lines dataset for --dataset-name sonnet
notebooks/
  inference_optimization.ipynb          # thin driver (server)
  inference_optimization_offline.ipynb  # thin driver (offline)
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

Offline (`LLM` API) equivalent:

```python
from llm_benchmark_offline import run_full_pipeline

results = run_full_pipeline()
```

Or piecemeal (server):

```python
from llm_benchmark import start_vllm_server, wait_for_server, stop_server, benchmark_server

proc = start_vllm_server("Qwen/Qwen2.5-3B-Instruct")
wait_for_server()
r = benchmark_server(8000, ["Explain quantum computing in one sentence."], name="FP16")
stop_server(proc)
```

## Official vLLM serving benchmark

vLLM's own throughput/latency benchmark is vendored (same commit as your script,
so `backend_request_func.py` stays compatible) and wired to `start_vllm_server`.

Run it against a live server (server must already be running):

```bash
python -m llm_benchmark.serving.benchmark_serving \
    --backend vllm \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset-name random \
    --num-prompts 100 \
    --save-result
```

Or let the package start/stop the server for you:

```python
from llm_benchmark.serving import benchmark_model

benchmark_model(
    "Qwen/Qwen2.5-3B-Instruct",
    dataset_name="random",
    num_prompts=100,
    # forwarded to start_vllm_server
    quantization="awq",
    gpu_memory_utilization=0.90,
)
```

After `pip install -e .`, a console script is also available:
`llm-benchmark-serve --help`.

It reports TTFT, TPOT (time-per-output-token), ITL (inter-token latency), and
request/input/output throughput with mean/median/p99, supporting `sharegpt`,
`sonnet`, and `random` datasets plus configurable Poisson request rates.
The bundled `sonnet.txt` is used automatically for `dataset_name="sonnet"`.

## Official vLLM offline benchmarks

vLLM's offline (in-process `LLM`) throughput and latency scripts are also
vendored under `llm_benchmark_offline`:

```bash
# Offline throughput (requests/s and tokens/s), ShareGPT or synthetic input
python -m llm_benchmark_offline.benchmark_throughput \
    --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --input-len 1024 --output-len 128 --num-prompts 100 \
    --quantization awq

# Single-batch latency (percentiles of end-to-end latency)
python -m llm_benchmark_offline.benchmark_latency \
    --model Qwen/Qwen2.5-3B-Instruct --batch-size 8 \
    --input-len 32 --output-len 128
```

Or via the console scripts: `llm-benchmark-throughput` and
`llm-benchmark-latency`.

## Notes

- Requires a CUDA GPU (defaults to device 0).
- Server results persist to `/content/benchmark_results_qwen_server.json`
  (override `LLM_BENCHMARK_RESULTS_PATH`); offline results to
  `/content/benchmark_results_qwen.json` (override `LLM_BENCHMARK_OFFLINE_RESULTS_PATH`).
- Each server section tears down its process before the next; logs go to
  `/content/vllm_server_8000.log`.
