"""Core benchmark primitives: result dataclass, memory/seed utilities, display."""

import gc
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pynvml
import torch


@dataclass
class BenchmarkResult:
    name: str
    ttft_ms: float | None
    tps: float
    tpm: float
    memory_gb: float
    total_tokens: int
    total_time_s: float


def get_memory_usage() -> float:
    """Get total physical VRAM currently allocated on GPU 0 in GB using NVML."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1e9
    except Exception:
        return torch.cuda.max_memory_reserved(0) / 1e9


def clear_memory() -> None:
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def free_model(model) -> None:
    """Release a model's GPU memory by moving it to CPU, then deleting it.

    A plain ``del model`` + ``torch.cuda.empty_cache()`` often leaves the weights
    resident on the GPU (especially with ``device_map="auto"``/accelerate, which
    keeps dispatch hooks alive). Moving to CPU first guarantees the GPU tensors
    are freed before the next stage (e.g. launching vLLM) starts.
    """
    if model is None:
        return
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_ttft(value: float | None) -> str:
    return f"{value:.1f} ms" if value is not None else "Not available"


def display_results(results: List[BenchmarkResult]) -> None:
    """Display results as a styled comparison table."""
    df = pd.DataFrame([vars(r) for r in results])
    df = df.set_index("name")
    display(df.style.format({
        "ttft_ms": "{:.1f}",
        "tps": "{:.1f}",
        "tpm": "{:.0f}",
        "memory_gb": "{:.2f}",
        "total_time_s": "{:.2f}",
    }).background_gradient(subset=["tps", "tpm"], cmap="Greens")
      .background_gradient(subset=["ttft_ms", "memory_gb"], cmap="Reds_r"))
