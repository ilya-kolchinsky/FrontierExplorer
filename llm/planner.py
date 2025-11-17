import asyncio
import json
import re
from typing import List, Optional

import httpx

from config import settings

SYSTEM_PROMPT = """You are a research search query planner.
Given a user's message, produce a concise search query, or multiple queries if needed, that will retrieve the state of the art across papers and repos.
Keep each query <= 8 words. Prefer key phrases over long sentences.
Avoid quotes unless necessary. Prefer neutral, general terms.
Return ONLY a JSON array of strings.
"""


def _simple_expand(user_intent: str, k: int = 5) -> List[str]:
    """Fallback when no LLM creds are available."""
    text = re.sub(r"[^a-zA-Z0-9\s\-_/]", " ", user_intent.lower())
    words = [w for w in text.split() if len(w) > 2]
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
    base = " ".join(uniq[:6]) or user_intent
    candidates = [
        base,
        f"{base} benchmarks",
        f"{base} openreview",
        f"{base} github",
        f"{base} arxiv",
    ]
    out = []
    for c in candidates:
        c = " ".join(c.split())
        if c and c not in out:
            out.append(c)
    return out[:k]


async def _chat_completions(
    base_url: str,
    api_key: Optional[str],
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 256,
    timeout: float = 20.0,
) -> str:
    """Call any OpenAI-compatible /v1/chat/completions endpoint (OpenAI or vLLM)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = base_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    # OpenAI & vLLM both return choices[0].message.content
    return data["choices"][0]["message"]["content"]


def _parse_queries(text: str, n: int) -> List[str]:
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            qs = [q.strip() for q in arr if isinstance(q, str) and q.strip()]
            return qs[:n] if qs else []
    except Exception:
        pass
    # fallback: extract lines or quoted phrases if not valid JSON
    lines = [ln.strip("-•* \t") for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if len(ln.split()) <= 12]
    return lines[:n]


# add llm_base_url / llm_model overrides to the signature
def generate_queries(
    user_intent: str,
    n: int = 5,
    *,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> List[str]:
    """
    Synchronous facade for Streamlit. Uses OpenAI or vLLM (OpenAI-compatible)
    if configured, else falls back to heuristic expansion.

    - If llm_base_url is provided, we treat it as an OpenAI-compatible endpoint (vLLM/LiteLLM/etc.).
    - If llm_base_url is empty/None and OPENAI_API_KEY is set, we use OpenAI.
    - llm_model overrides the backend's default model if provided.
    """
    base_url = None
    api_key = None
    model = None

    if llm_base_url:                           # explicit URL → treat as OpenAI-compatible (vLLM/gateway)
        base_url = llm_base_url
        api_key = settings.VLLM_API_KEY                 # optional
        model = llm_model or settings.DEFAULT_VLLM_MODEL
    elif settings.OPENAI_API_KEY:                       # OpenAI path
        base_url = settings.OPENAI_BASE_URL
        api_key = settings.OPENAI_API_KEY
        model = llm_model or settings.DEFAULT_OPENAI_MODEL
    elif settings.VLLM_BASE_URL:                        # env-configured vLLM fallback
        base_url = settings.VLLM_BASE_URL
        api_key = settings.VLLM_API_KEY
        model = llm_model or settings.DEFAULT_VLLM_MODEL
    else:
        return _simple_expand(user_intent, k=n)

    async def _run():
        raw = await _chat_completions(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system=SYSTEM_PROMPT,
            user=user_intent,
        )
        qs = _parse_queries(raw, n)
        return qs or _simple_expand(user_intent, k=n)

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()
