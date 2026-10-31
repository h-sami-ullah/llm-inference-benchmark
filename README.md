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
llm_benchmark/serving/
  __init__.py              # benchmark_model(): vllm bench serve wrapper
notebooks/
  inference_optimization.ipynb          # thin driver (server)
  inference_optimization_offline.ipynb  # thin driver (offline)
```

## Install

Requires Python 3.11+. The package's core dependencies are lightweight and
portable; the heavy GPU packages (`torch`, `vllm`, `autoawq`, ...) are optional
because they need platform-specific CUDA wheels.

### Core only (works anywhere, no GPU build)

```bash
uv sync          # or: pip install -e .
```

### With GPU support

Use a CUDA-capable PyTorch index so vLLM installs as a wheel instead of building
from source (which otherwise fails with "Cannot find CUDA_HOME"):

```bash
# uv
uv sync --extra gpu --torch-backend=auto

# pip
pip install -e '.[gpu]'
```

Then pre-fetch precompiled kernels for faster startup:

```bash
vllm download-kernels
```

The plain startup command is just `vllm serve <model>` (OpenAI-compatible,
`http://localhost:8000` by default). This package wraps that same command via
`start_vllm_server()`, adding stable defaults (see `llm_benchmark/server.py`).

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

## Official vLLM benchmarks (`vllm bench`)

vLLM ships a built-in benchmark CLI (installed via the `vllm[bench]` extra), which
supersedes the old standalone `benchmarks/*.py` scripts:

- `vllm bench serve` — API server throughput/latency (online)
- `vllm bench throughput` — engine throughput (offline)
- `vllm bench latency` — single-batch latency (offline)
- `vllm bench forward` — model forward-pass timing

### Online (`vllm bench serve`)

Start a server, then benchmark it in another terminal:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct

vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 100 \
  --max-concurrency 8 \
  --save-result
```

Or let the package start/stop the server and run `vllm bench serve` for you:

```python
from llm_benchmark.serving import benchmark_model

benchmark_model(
    "Qwen/Qwen2.5-3B-Instruct",
    bench_args=[
        "--backend", "openai",
        "--endpoint", "/v1/completions",
        "--model", "Qwen/Qwen2.5-3B-Instruct",
        "--dataset-name", "random",
        "--random-input-len", "1024",
        "--random-output-len", "256",
        "--num-prompts", "100",
        "--max-concurrency", "8",
        "--save-result",
    ],
    # forwarded to start_vllm_server
    quantization="awq",
    gpu_memory_utilization=0.90,
)
```

It reports request/input/output throughput plus TTFT, TPOT (time-per-output-token),
and ITL (inter-token latency) with mean/median/p99, and supports `--request-rate`,
`--burstiness`, `--max-concurrency`, `--save-result`, `--save-detailed`, and many
datasets (`random`, `sharegpt`, `sonnet`, `prefix_repetition`, `timed_trace`, ...).

### Offline (`vllm bench throughput` / `vllm bench latency`)

```bash
vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct \
  --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 100

vllm bench latency --model Qwen/Qwen2.5-3B-Instruct \
  --batch-size 8 --input-len 32 --output-len 128
```

Or via the thin wrappers:

```python
from llm_benchmark_offline import benchmark_throughput, benchmark_latency

benchmark_throughput(["--model", "Qwen/Qwen2.5-3B-Instruct", "--dataset-name", "random"])
benchmark_latency(["--model", "Qwen/Qwen2.5-3B-Instruct", "--batch-size", "8"])
```

For production load testing with live progress and automatic report generation,
see [GuideLLM](https://github.com/vllm-project/guidellm)
(`pip install "guidellm[recommended]"`).

## Notes

- Requires a CUDA GPU (defaults to device 0).
- Server results persist to `/content/benchmark_results_qwen_server.json`
  (override `LLM_BENCHMARK_RESULTS_PATH`); offline results to
  `/content/benchmark_results_qwen.json` (override `LLM_BENCHMARK_OFFLINE_RESULTS_PATH`).
- Each server section tears down its process before the next; logs go to
  `/content/vllm_server_8000.log`.
