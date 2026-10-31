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
    """Release a model's GPU memory.

    Moves every parameter and buffer to CPU in-place (dropping the GPU tensors)
    before deleting. This is robust regardless of ``device_map`` dispatch hooks,
    lingering references, or the model object still being referenced by the
    caller — the GPU tensors are released as soon as ``p.data`` is reassigned.
    """
    if model is None:
        return
    try:
        for p in model.parameters():
            p.data = p.data.detach().to("cpu")
        for b in model.buffers():
            b.data = b.data.detach().to("cpu")
    except Exception:
        try:
            model.to("cpu")
        except Exception:
            pass
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gpu_memory_report() -> str:
    """Human-readable GPU memory breakdown (allocated / reserved / NVML used)."""
    try:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
    except Exception:
        allocated = reserved = float("nan")
    return (
        f"allocated {allocated:.2f} GB, reserved {reserved:.2f} GB, "
        f"nvml used {get_memory_usage():.2f} GB"
    )


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
