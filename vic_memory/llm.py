"""Optional adapters for any OpenAI-compatible endpoint (OpenAI, DeepSeek, Qwen,
Kimi, Ollama, vLLM, LM Studio...). Pure stdlib (urllib). Nothing here is required.

    export VIC_LLM_BASE_URL=https://api.deepseek.com/v1
    export VIC_LLM_API_KEY=sk-...
    export VIC_LLM_MODEL=deepseek-chat
    export VIC_EMBED_MODEL=text-embedding-3-small   # optional
"""
from __future__ import annotations

import json
import os
import urllib.request


def _post(path: str, payload: dict) -> dict:
    base = os.environ.get("VIC_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ.get('VIC_LLM_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def openai_embedder(model: str | None = None):
    model = model or os.environ.get("VIC_EMBED_MODEL", "text-embedding-3-small")

    def embed(text: str):
        return _post("/embeddings", {"model": model, "input": text})["data"][0]["embedding"]
    return embed


SUMMARY_PROMPT = (
    "You consolidate an AI agent's episodic memories into ONE durable semantic memory, "
    "like the brain does during sleep. Keep stable facts, preferences, patterns and lessons; "
    "drop one-off chatter. Keep names, numbers and dates exact. Answer in the language of the "
    "memories, one or two sentences, no preamble.\n\nEpisodes:\n"
)


def openai_summarizer(model: str | None = None):
    model = model or os.environ.get("VIC_LLM_MODEL", "gpt-4o-mini")

    def summarize(texts: list[str]) -> str:
        body = SUMMARY_PROMPT + "\n".join(f"- {t}" for t in texts)
        r = _post("/chat/completions", {"model": model, "temperature": 0,
                                        "messages": [{"role": "user", "content": body}]})
        return r["choices"][0]["message"]["content"].strip()
    return summarize


def from_env():
    """Return (embedder, summarizer) based on environment variables, or (None, None)."""
    if not os.environ.get("VIC_LLM_API_KEY") and not os.environ.get("VIC_LLM_BASE_URL"):
        return None, None
    emb = openai_embedder() if os.environ.get("VIC_EMBED_MODEL") else None
    return emb, openai_summarizer()
