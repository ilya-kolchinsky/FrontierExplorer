from typing import Dict, List, Optional
import asyncio
from core.types import WorkItem
from analysis.loaders import load_fulltext_for_work
from analysis.index import RAGIndex


async def _load_all(works: List[WorkItem], github_token: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}

    async def _one(w: WorkItem):
        t = await load_fulltext_for_work(w, github_token=github_token)
        print(t)
        if t and (w.id or ""):
            out[w.id] = t
    await asyncio.gather(*[_one(w) for w in works])
    return out


def build_knowledge_index(works: List[WorkItem], github_token: Optional[str]) -> RAGIndex:
    # Run async fetch in sync context
    try:
        full_texts = asyncio.run(_load_all(works, github_token))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            full_texts = loop.run_until_complete(_load_all(works, github_token))
        finally:
            loop.close()
    idx = RAGIndex()
    idx.build(works, full_texts)
    return idx
