"""Shared configuration and constants for the LLM inference benchmark."""

import os

# Model configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_ID_AWQ = "Qwen/Qwen2.5-3B-Instruct-AWQ"
DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0

# Server configuration
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"

# Result persistence (Colab default; override with LLM_BENCHMARK_RESULTS_PATH)
RESULTS_FILE_PATH = os.environ.get(
    "LLM_BENCHMARK_RESULTS_PATH",
    "/content/benchmark_results_qwen_server.json",
)

# Test prompts
TEST_PROMPTS = [
    "Explain quantum computing in one sentence.",
    "Write a Python function to reverse a string.",
    "What are the three laws of thermodynamics?",
    "Describe the process of photosynthesis briefly.",
    "What is the difference between TCP and UDP?",
]

SYSTEM_PROMPT = "You are a helpful assistant. Be concise."

# Longer sequences to stress Flash Attention
LONG_PROMPTS = [
    "Write a detailed explanation of how neural networks learn, including the concepts of forward propagation, backpropagation, gradient descent, and the role of activation functions. Provide examples.",
    "Explain the complete history of artificial intelligence from its inception in the 1950s to modern large language models. Include key milestones, researchers, and breakthroughs.",
    "Describe the architecture of a transformer model in detail, including self-attention mechanisms, positional encoding, layer normalization, and how it differs from RNNs and LSTMs.",
]

# Prefix caching fixtures
SHARED_PREFIX = """You are an expert AI assistant specializing in software engineering,
machine learning, and system design. You provide detailed, accurate, and helpful responses.
Always structure your answers clearly with examples when appropriate."""

PREFIX_PROMPTS = [
    "What is the difference between a process and a thread?",
    "Explain what a deadlock is and how to prevent it.",
    "What are the SOLID principles in software design?",
    "How does garbage collection work in Python?",
    "What is the CAP theorem?",
]


def setup_environment() -> None:
    """Configure environment variables before importing vLLM or torch."""
    for key in (
        "VLLM_USE_V1",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_RAY",
        "VLLM_DISTRIBUTED_BACKEND",
    ):
        os.environ.pop(key, None)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
