"""Hermes Agent adapter — configure Hermes to route LLM calls through agent-compiler.

Hermes Agent (by Nous Research) supports many providers via its config system.
It communicates via OpenAI-compatible chat completions API internally, making
the agent-compiler proxy a drop-in replacement.

Quick setup (two options):

Option A — Environment variable (simplest):
    export LLM_API_BASE="http://127.0.0.1:8100/v1"
    export LLM_API_KEY="any-value"

Option B — Hermes config file:
    In your Hermes config, set the provider endpoint to:
    http://127.0.0.1:8100/v1

How it works:
    - Hermes calls the proxy for every LLM inference
    - Proxy checks L1 (rules) and L2 (semantic cache)
    - Cache hit → tool results returned as natural language
    - Cache miss → forwarded to DeepSeek/OpenAI/etc.
    - Hermes sees normal LLM responses, no code changes needed
"""


def generate_hermes_env(proxy_port: int = 8100) -> dict[str, str]:
    """Generate environment variables for Hermes to use the proxy."""
    return {
        "LLM_API_BASE": f"http://127.0.0.1:{proxy_port}/v1",
        "LLM_API_KEY": "agent-compiler-proxy",
        "LLM_MODEL": "agent-compiler",
    }
