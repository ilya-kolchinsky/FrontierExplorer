from __future__ import annotations

import asyncio
import httpx
from typing import List, Optional, Dict, Tuple

from analysis.index import RAGIndex
from config import settings

SYSTEM = (
    "You are a precise research assistant. Answer only using the given context. "
    "Compare and contrast works, call out limitations, and be specific. "
    "If the answer is not in the context, say you don't have evidence."
)


async def _chat(base_url: str, api_key: Optional[str], model: str, system: str, messages: List[Dict], max_tokens=600):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": [{"role": "system", "content": system}] + messages,
               "temperature": 0.2, "max_tokens": max_tokens}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


def _pick_backend(llm_base_url: Optional[str], llm_model: Optional[str]):
    if llm_base_url:
        return {"base_url": llm_base_url, "api_key": settings.VLLM_API_KEY, "model": llm_model or settings.DEFAULT_VLLM_MODEL}
    if settings.OPENAI_API_KEY:
        return {"base_url": settings.OPENAI_BASE_URL, "api_key": settings.OPENAI_API_KEY, "model": llm_model or settings.DEFAULT_OPENAI_MODEL}
    if settings.VLLM_BASE_URL:
        return {"base_url": settings.VLLM_BASE_URL, "api_key": settings.VLLM_API_KEY, "model": llm_model or settings.DEFAULT_VLLM_MODEL}
    return None


def answer_with_selected(
        question: str,
        index: RAGIndex,
        *,
        llm_base_url: Optional[str],
        llm_model: Optional[str],
        top_k: int = 8,
) -> Tuple[str, List[Dict]]:
    hits = index.search(question, top_k=top_k)
    if not hits:
        return "I don't have enough context from the selected items to answer.", []

    # assemble context with per-chunk source footers
    parts = []
    citations = []
    for rank, (i, sim) in enumerate(hits, 1):
        m = index.meta[i]
        chunk = index.chunks[i]
        parts.append(f"[{rank}] {m['title']} — {m['url']}\n{chunk}")
        citations.append({"rank": rank, "title": m["title"], "url": m["url"], "work_id": m["work_id"]})

    context = "\n\n---\n\n".join(parts[:top_k])
    user = f"Question: {question}\n\nUse ONLY this context (numbered):\n\n{context}\n\nWhen you claim something, cite [rank] numbers."

    backend = _pick_backend(llm_base_url, llm_model)
    if not backend:
        # crude fallback: return the top chunk titles
        text = "No LLM configured. Top matches:\n" + "\n".join(f"[{c['rank']}] {c['title']}" for c in citations)
        return text, citations

    async def _run():
        return await _chat(backend["base_url"], backend["api_key"], backend["model"], SYSTEM,
                           [{"role": "user", "content": user}], max_tokens=4096)

    try:
        txt = asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            txt = loop.run_until_complete(_run())
        finally:
            loop.close()
    return txt, citations
