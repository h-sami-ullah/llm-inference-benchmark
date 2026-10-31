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

Requires Python 3.11+.

vLLM and PyTorch are CUDA packages that must be installed with a matching
CUDA backend, so they are **not** Python dependencies here (declaring them makes
`uv sync`/`pip` try to resolve or build a vLLM wheel and fail). Install them
separately using vLLM's recommended command.

> Note: AWQ quantization is done natively by vLLM (`quantization="awq"`), so the
> `autoawq` package is **not** required for benchmarking. Only install it
> separately if you need to *create* AWQ-quantized models yourself.

### 1. Core package (light, works anywhere)

```bash
uv sync                 # or: pip install -e .
```

### 2. GPU stack (vLLM + CUDA PyTorch + transformers/accelerate)

```bash
uv pip install vllm --torch-backend=auto
uv pip install accelerate
vllm download-kernels   # optional: precompiled kernels for faster startup
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

## Results

Measured with the server (online) pipeline on a Google Colab **T4 (16 GB)**, model
`Qwen/Qwen2.5-3B-Instruct` (AWQ variant where noted), draft model
`Qwen/Qwen2.5-0.5B-Instruct`, CUDA graphs enabled (`enforce_eager=False`). The pipeline writes the raw data to
`benchmark_results_qwen_server.json`.

| Section | TTFT (ms) | TPS | TPM | Memory (GB) |
| --- | ---: | ---: | ---: | ---: |
| HF Baseline | 74.4 | 72.6 | 4,358 | 7.11 |
| vLLM Server (FP16) | 78.3 | 120.6 | 7,237 | 13.36 |
| vLLM Server + AWQ | 43.4 | **271.8** | 16,306 | 12.70 |
| vLLM Server + AWQ + Flash | 39.5 | 237.3 | 14,236 | 12.70 |
| Prefix Cache (warm) | 55.3 | **371.2** | 22,274 | 14.27 |
| vLLM Server + AWQ + Spec | 311.9 | 98.1 | 5,883 | 14.34 |
| Combined Stack | 320.9 | 102.7 | 6,160 | 14.84 |

Key takeaways:

- **vLLM FP16** is ~1.7× the HuggingFace baseline on throughput — continuous batching
  and PagedAttention are always on.
- **AWQ 4-bit** is the biggest single win: ~2.3× FP16 throughput at lower memory.
- **Flash Attention** keeps memory flat as sequences get longer.
- **Prefix caching** is the fastest for shared prompts (system prompt / RAG context).
- **Speculative decoding** with a 0.5B draft against a 3B target is *slower* than plain
  AWQ (98 vs 272 TPS): the draft's low acceptance rate costs more than it saves. A
  well-matched draft is required for a win.

## Notes

- Requires a CUDA GPU (defaults to device 0).
- The HuggingFace baseline runs in an isolated subprocess (`llm_benchmark/hf_worker.py`)
  so its GPU memory is reclaimed by the OS before vLLM starts — avoid holding HF and
  vLLM in the same process.
- Server results persist to `/content/benchmark_results_qwen_server.json`
  (override `LLM_BENCHMARK_RESULTS_PATH`); offline results to
  `/content/benchmark_results_qwen.json` (override `LLM_BENCHMARK_OFFLINE_RESULTS_PATH`).
- Each server section tears down its process before the next; logs go to
  `/content/vllm_server_8000.log`.
