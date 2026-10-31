"""HuggingFace Transformers baseline benchmark."""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .benchmark import BenchmarkResult, get_memory_usage
from .config import SYSTEM_PROMPT


class FirstTokenTimingStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that timestamps the first generated token ID."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_time = None

    def put(self, value):
        if (
            not self.next_tokens_are_prompt
            and self.first_token_time is None
            and value.numel() > 0
        ):
            self.first_token_time = time.perf_counter()
        super().put(value)


def _count_hf_generated_tokens(generated_sequences, eos_token_id) -> int:
    """Count generated token IDs, stopping each row at its first EOS."""
    total = 0
    for sequence in generated_sequences:
        if eos_token_id is not None:
            eos_positions = (sequence == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                total += int(eos_positions[0].item()) + 1
                continue
        total += int(sequence.numel())
    return total


def load_hf_model(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load directly onto the GPU (no `device_map="auto"`). `device_map` uses
    # accelerate dispatch hooks that prevent `del model` + `empty_cache()` from
    # actually freeing the GPU tensors, which OOMs the subsequent vLLM server.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    ).to("cuda")
    return model, tokenizer


def benchmark_hf(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    num_runs: int = 3,
) -> BenchmarkResult:
    """
    Benchmark Hugging Face Transformers.

    TTFT: Host-observed time until the first generated token ID reaches the streamer,
          measured one prompt at a time because TextIteratorStreamer is batch-size 1.
    TPS:  Aggregate output-token throughput using static batching.
    Memory: Physical GPU memory currently in use, measured with NVML.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    formatted = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # Warm-up (excluded from measurements).
    warmup_inputs = tokenizer(formatted[0], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    torch.cuda.synchronize()

    all_ttft = []
    all_tps = []
    all_tokens = []
    all_times = []

    for _ in range(num_runs):
        # 1. TTFT: one prompt at a time (streamer only supports B=1)
        run_ttfts = []
        for prompt_text in formatted:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            streamer = FirstTokenTimingStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=120,
            )

            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": pad_token_id,
            }

            torch.cuda.synchronize()
            start = time.perf_counter()

            thread = threading.Thread(
                target=model.generate,
                kwargs=generation_kwargs,
                daemon=True,
            )
            thread.start()

            for _ in streamer:
                pass

            thread.join()

            if streamer.first_token_time is not None:
                run_ttfts.append((streamer.first_token_time - start) * 1000.0)

        if run_ttfts:
            all_ttft.extend(run_ttfts)

        # 2. TPS: static batch, non-streaming generate()
        batch_inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(
            model.device
        )

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        generated_sequences = outputs[:, batch_inputs.input_ids.shape[1]:]
        total_new_tokens = _count_hf_generated_tokens(
            generated_sequences,
            tokenizer.eos_token_id,
        )

        all_tps.append(total_new_tokens / total_time)
        all_tokens.append(total_new_tokens)
        all_times.append(total_time)

    avg_ttft = statistics.mean(all_ttft) if all_ttft else None
    avg_tps = statistics.mean(all_tps)
    avg_tokens = statistics.mean(all_tokens)
    avg_time = statistics.mean(all_times)

    return BenchmarkResult(
        name="HF Baseline",
        ttft_ms=avg_ttft,
        tps=avg_tps,
        tpm=avg_tps * 60.0,
        memory_gb=get_memory_usage(),
        total_tokens=int(round(avg_tokens)),
        total_time_s=avg_time,
    )


def run_hf_baseline_isolated(model_id: str, num_runs: int = 3) -> BenchmarkResult:
    """Run the HF baseline in a subprocess and return its result.

    The model lives and dies inside ``llm_benchmark.hf_worker``, so its GPU
    memory is fully released by the OS when the subprocess exits — avoiding
    notebook hidden references (``Out``/``_``, tracebacks, device_map hooks)
    that otherwise keep the weights resident and OOM the next stage.
    """
    out_path = os.path.join(tempfile.gettempdir(), "hf_baseline_result.json")
    cmd = [
        sys.executable,
        "-m",
        "llm_benchmark.hf_worker",
        "--model",
        model_id,
        "--output",
        out_path,
        "--num-runs",
        str(num_runs),
    ]
    print("Running HuggingFace baseline in a subprocess (isolated GPU memory)...")
    subprocess.run(cmd, check=True)

    with open(out_path) as f:
        data = json.load(f)
    return BenchmarkResult(**data)
