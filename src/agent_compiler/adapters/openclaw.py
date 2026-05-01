"""OpenClaw adapter — configure OpenClaw to route LLM calls through agent-compiler.

OpenClaw uses a provider-based model config in ~/.openclaw/openclaw.json
or ~/openclaw.json. Each provider has a `baseUrl` that we redirect to the
agent-compiler proxy.

Quick setup:
    1. Start the proxy:  python -m agent_compiler.adapters.proxy
    2. In OpenClaw config, add a provider pointing to localhost:8100:

```json
{
  "models": {
    "providers": {
      "agent-compiler": {
        "baseUrl": "http://127.0.0.1:8100/v1",
        "apiKey": "any-value",
        "models": ["agent-compiler"]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": "agent-compiler"
    }
  }
}
```

How it works:
    - OpenClaw sends chat completion requests to the proxy
    - Proxy checks L1 (rules) and L2 (semantic cache)
    - Cache hit → tools execute locally, result formatted as LLM response
    - Cache miss → forwarded to actual LLM (DeepSeek/OpenAI/etc.)
    - OpenClaw sees normal chat completions — no code changes needed
"""

import json
from pathlib import Path

OPENCLAW_CONFIG_SNIPPET = {
    "models": {
        "providers": {
            "agent-compiler": {
                "baseUrl": "http://127.0.0.1:8100/v1",
                "apiKey": "agent-compiler-proxy",
                "models": ["agent-compiler"],
            }
        }
    },
    "agents": {
        "defaults": {
            "model": "agent-compiler",
        }
    },
}


def generate_openclaw_config(proxy_port: int = 8100) -> dict:
    """Generate an OpenClaw provider config snippet pointing to the proxy."""
    cfg = json.loads(json.dumps(OPENCLAW_CONFIG_SNIPPET))
    cfg["models"]["providers"]["agent-compiler"]["baseUrl"] = \
        f"http://127.0.0.1:{proxy_port}/v1"
    return cfg


def find_openclaw_config_paths() -> list[Path]:
    """Find existing OpenClaw config files on this machine."""
    candidates = [
        Path.home() / "openclaw.json",
        Path.home() / ".openclaw" / "openclaw.json",
    ]
    return [p for p in candidates if p.exists()]
