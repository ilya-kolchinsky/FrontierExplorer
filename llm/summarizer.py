import asyncio
import httpx
import json
from typing import List, Optional

from config import settings
from search.types import WorkItem

SYSTEM = (
    "Your goal is to concisely summarize the given list of materials discussing a common topic.\n"
    "The materials may include research papers, public repositories, HuggingFace models, datasets, etc.\n"
    "Write 5–8 short bullets max, grouping related work and noting trends.\n"
    "Prefer concrete signals: dates, venues, citations/stars/downloads.\n"
    "If relevant, suggest a short next-step like 'compare A vs B on X'\n"
    "Be neutral. No fluff. No first-person. No marketing tone."
)


def _pick_backend(base_url_override: Optional[str], model_override: Optional[str]):
    # If the caller provided a base URL, use it (vLLM/gateway); else OpenAI (if key) else env vLLM; else None.
    if base_url_override:
        return {
            "base_url": base_url_override.rstrip("/"),
            "api_key": settings.VLLM_API_KEY,   # optional
            "model": model_override or settings.DEFAULT_VLLM_MODEL,
        }
    if settings.OPENAI_API_KEY:
        return {
            "base_url": settings.OPENAI_BASE_URL.rstrip("/"),
            "api_key": settings.OPENAI_API_KEY,
            "model": model_override or settings.DEFAULT_OPENAI_MODEL,
        }
    if settings.VLLM_BASE_URL:
        return {
            "base_url": settings.VLLM_BASE_URL.rstrip("/"),
            "api_key": settings.VLLM_API_KEY,
            "model": model_override or settings.DEFAULT_VLLM_MODEL,
        }
    return None


def _serialize_items(works: List[WorkItem], max_items: int | None = None) -> str:
    """Compact JSON list for the prompt: keeps titles + key signals only."""
    rows = []
    works_to_consider = works[:max_items] if max_items is not None else works
    for w in works_to_consider:
        rows.append({
            "title": w.title,
            "source": w.source,
            "venue": w.venue,
            "year": w.year,
            "citations": w.citations,
            "stars": w.stars,
            "meta": w.meta or {},
        })
    return json.dumps(rows, ensure_ascii=False)


async def _chat(base_url: str, api_key: Optional[str], model: str, system: str, user: str):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


def summarize_results(
    works: List[WorkItem],
    *,
    query_or_goal: str,
    llm_base_url: Optional[str],
    llm_model: Optional[str],
    max_items: int | None = None,
) -> str:
    backend = _pick_backend(llm_base_url, llm_model)
    if not backend or not works:
        # Heuristic fallback: list top titles as bullets
        bullets = [f"- {w.title} ({w.venue or w.source}, {w.year or 'n/a'})" for w in works[:min(8, max_items)]]
        return "Quick summary (no LLM configured):\n" + "\n".join(bullets)

    items_json = _serialize_items(works, max_items=max_items)
    user = (
        f"Topic: {query_or_goal}\n"
        f"Summarize the following items:\n{items_json}\n"
    )

    async def _run():
        return await _chat(
            base_url=backend["base_url"],
            api_key=backend["api_key"],
            model=backend["model"],
            system=SYSTEM,
            user=user,
        )

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()
