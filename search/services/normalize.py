from typing import Iterable, Dict, List
from core.types import ProviderResult, WorkItem


def provider_result_to_work(r: ProviderResult) -> WorkItem:
    # Stable ID namespace: "{source}:{provider_id}"
    wid = f"{r.source}:{r.id}"
    return WorkItem(
        id=wid,
        title=r.title.strip(),
        url=r.url,
        abstract=r.abstract.strip(),
        source=r.source,
        authors=r.authors,
        year=r.year,
        venue=r.venue,
        stars=r.stars,
        citations=r.citations,
        updated_iso=r.updated_iso,
        meta=r.meta or {},
    )


def dedupe(works: Iterable[WorkItem]) -> List[WorkItem]:
    seen: Dict[str, WorkItem] = {}
    for w in works:
        if w.id not in seen:
            seen[w.id] = w
    return list(seen.values())
