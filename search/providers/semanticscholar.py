from typing import List, Optional
import asyncio
from search.providers.base import Provider
from search.types import ProviderResult
from search.utils import get_http_client
from config import settings

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarProvider(Provider):
    name = "semanticscholar"

    @staticmethod
    async def _fetch_page(client, params, headers):
        backoff = settings.S2_BACKOFF_SECONDS
        for attempt in range(settings.S2_MAX_RETRIES):
            r = await client.get(S2_API, params=params, headers=headers)
            # Retry on rate limit or server errors
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                await asyncio.sleep(wait)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        # final try raises or returns empty
        r.raise_for_status()  # will raise the last response error
        return {"data": []}

    async def search(
        self,
        query: str,
        limit: int,
        *,
        earliest_date: Optional[str] = None,   # "YYYY-MM-DD"
        min_popularity: Optional[int] = None,  # citations threshold
    ) -> List[ProviderResult]:
        # Build base params
        params = {
            "query": query,
            "fields": (
                "title,abstract,year,venue,authors,"
                "citationCount,influentialCitationCount,externalIds,url"
            ),
            "offset": 0,
            "limit": limit,
        }
        # year filter: open-ended range "YYYY-"
        if earliest_date and earliest_date[:4].isdigit():
            params["year"] = f"{earliest_date[:4]}-"

        headers = {}
        if settings.S2_API_KEY:
            headers["x-api-key"] = settings.S2_API_KEY

        out: List[ProviderResult] = []
        seen_ids = set()

        async with get_http_client() as client:
            while len(out) < limit:
                data = await self._fetch_page(client, params, headers)
                rows = data.get("data", [])
                if not rows:
                    break

                for p in rows:
                    # Popularity filter = citations
                    cites = int(p.get("citationCount") or 0)
                    if min_popularity is not None and cites < min_popularity:
                        continue
                    # Client-side cutoff (in case server-side year was absent later)
                    if earliest_date and p.get("year"):
                        try:
                            if int(p["year"]) < int(earliest_date[:4]):
                                continue
                        except Exception:
                            pass

                    ext = p.get("externalIds") or {}
                    doi = (ext.get("DOI") or "").strip()
                    pid = doi or (p.get("paperId") or "").strip()
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    out.append(ProviderResult(
                        id=pid,
                        title=p.get("title", "") or "",
                        url=p.get("url", "") or "",
                        abstract=p.get("abstract") or "",
                        source="semanticscholar",
                        authors=[a.get("name","") for a in p.get("authors", []) if a.get("name")],
                        year=p.get("year"),
                        venue=p.get("venue"),
                        citations=cites,
                        meta={"influential": p.get("influentialCitationCount")}
                    ))
                    if len(out) >= limit:
                        break

                # paginate
                params["offset"] = params.get("offset", 0) + params["limit"]

                # be polite between pages to avoid bursts (esp. without an API key)
                await asyncio.sleep(0.2)

        return out[:limit]
