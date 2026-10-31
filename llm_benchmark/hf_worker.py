"""Run the HuggingFace baseline in a fresh subprocess.

Isolating the HF model in its own process guarantees the GPU memory is fully
reclaimed by the OS when the process exits. This avoids the hidden references
that Jupyter/Colab keeps (``Out``/``_`` variables, tracebacks, accelerate
dispatch hooks, ...) which otherwise leave the model resident on the GPU and
OOM the subsequent vLLM stage.
"""

import argparse
import json
from dataclasses import asdict

from .benchmark import format_ttft
from .config import TEST_PROMPTS, setup_environment
from .hf import benchmark_hf, load_hf_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    args = parser.parse_args()

    setup_environment()

    model, tokenizer = load_hf_model(args.model)
    result = benchmark_hf(model, tokenizer, TEST_PROMPTS, num_runs=args.num_runs)

    print("\n--- HuggingFace Baseline Results ---")
    print(f"TTFT: {format_ttft(result.ttft_ms)}")
    print(f"TPS: {result.tps:.1f} tokens/sec")
    print(f"TPM: {result.tpm:.0f} tokens/min")
    print(f"Memory: {result.memory_gb:.2f} GB")

    with open(args.output, "w") as f:
        json.dump(asdict(result), f)


if __name__ == "__main__":
    main()
