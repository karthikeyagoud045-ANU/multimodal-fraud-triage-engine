"""
utils/llm_clients.py — Centralised LLM/VLM Client Factory.

This module is the SINGLE source of truth for all AI client initialization.
Every agent in the pipeline imports its instructor-patched client from here.

ARCHITECTURE DECISION — Why OpenAI SDK for BOTH providers?
-----------------------------------------------------------
The `instructor` library patches an OpenAI client object to intercept
completion responses and validate them against Pydantic schemas. By using
the OpenAI SDK's `base_url` parameter to point to alternate endpoints,
we get a completely uniform patching surface:

    instructor.from_openai(openai_client) → works identically for:
      - OpenAI (api.openai.com)
      - Groq   (api.groq.com/openai/v1)    ← text agents
      - Nvidia (integrate.api.nvidia.com/v1) ← vision agents

This means:
  ✅ One patching strategy (`instructor.from_openai`) for all providers.
  ✅ No conditional logic per-provider in the agents.
  ✅ Trivially swap Groq ↔ OpenAI ↔ Together AI by changing base_url only.
  ✅ `instructor.Mode.JSON` enforced consistently for guaranteed schema output.

SINGLETON PATTERN:
------------------
Both instructor clients (`groq_instructor_client`, `nvidia_instructor_client`)
are module-level singletons initialized once at import time. This avoids the
overhead of recreating AsyncOpenAI connection pools on every agent call.

They are ASYNC clients (AsyncOpenAI) so they integrate correctly with
the asyncio.gather() in main.py without blocking the event loop.
"""
from __future__ import annotations

import instructor
from openai import AsyncOpenAI

from config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Groq Client — Text Agents (text_extractor.py, scribe.py)
# ─────────────────────────────────────────────────────────────────────────────

import itertools

# The Groq endpoint is OpenAI API-compatible at a different base_url.
# Models available: llama-3.3-70b-versatile, llama-3.1-8b-instant, etc.
# Rate limits (free tier): 12,000 TPM / 30 RPM for 70B model.
_groq_api_keys = settings.groq_api_key_list()
if not _groq_api_keys:
    _groq_api_keys = ["dummy_groq_key"]

_groq_async_clients = [
    AsyncOpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        timeout=30.0,
        max_retries=0,
    ) for key in _groq_api_keys
]

_groq_instructor_clients = [
    instructor.from_openai(client, mode=instructor.Mode.JSON)
    for client in _groq_async_clients
]

class RoundRobinInstructor:
    def __init__(self, clients):
        self.clients = clients
        self._cycle = itertools.cycle(clients)

    @property
    def chat(self):
        return next(self._cycle).chat

groq_instructor_client = RoundRobinInstructor(_groq_instructor_clients)
"""Instructor-patched Groq async client.

Uses round-robin across multiple API keys to distribute load and mitigate rate limits.
Use this for all text-based LLM calls (text_extractor.py, scribe.py).

Example:
    from utils.llm_clients import groq_instructor_client
    from models import TextExtractorOutput

    result = await groq_instructor_client.chat.completions.create(
        model=settings.groq_model,
        response_model=TextExtractorOutput,
        messages=[...],
        max_retries=2,
    )
"""


# ─────────────────────────────────────────────────────────────────────────────
# Nvidia NIM Client — Vision Agent (vlm_inspector.py)
# ─────────────────────────────────────────────────────────────────────────────

# Nvidia NIM exposes an OpenAI-compatible endpoint that accepts vision messages
# (image_url content type) in addition to standard text messages.
# Models available: meta/llama-3.2-90b-vision-instruct, etc.
# Rate limits (free tier): 40 RPM / generous TPM (images are billed separately).
_nvidia_async_client = AsyncOpenAI(
    api_key=settings.nvidia_api_key_str(),
    base_url=settings.nvidia_base_url,
    # Vision inference takes longer than text — increase timeout accordingly.
    # A 1024px image with complex damage may take 10-20s on first inference.
    timeout=60.0,
    max_retries=0,  # tenacity owns all rate-limit retries
)

nvidia_instructor_client = instructor.from_openai(
    _nvidia_async_client,
    mode=instructor.Mode.JSON,
)
"""Instructor-patched Nvidia NIM async client.

Use this for all vision-based LLM calls (vlm_inspector.py).

Example:
    from utils.llm_clients import nvidia_instructor_client
    from models import VLMInspectorOutput

    result = await nvidia_instructor_client.chat.completions.create(
        model=settings.nvidia_model,
        response_model=VLMInspectorOutput,
        messages=[
            {"role": "system", "content": "..."},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyse this image."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            ]},
        ],
        max_retries=2,
    )
"""


# ─────────────────────────────────────────────────────────────────────────────
# Health check (for debugging — not called in production pipeline)
# ─────────────────────────────────────────────────────────────────────────────

async def ping_groq() -> bool:
    """Quick liveness check for the Groq endpoint.

    Returns True if Groq responds, False otherwise.
    Does not raise — safe to call in a startup health check.
    """
    try:
        # List models is a cheap GET request with no token cost
        await _groq_async_clients[0].models.list()
        return True
    except Exception:
        return False


async def ping_nvidia() -> bool:
    """Quick liveness check for the Nvidia NIM endpoint.

    Returns True if Nvidia NIM responds, False otherwise.
    Does not raise — safe to call in a startup health check.
    """
    try:
        await _nvidia_async_client.models.list()
        return True
    except Exception:
        return False
