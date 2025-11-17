from typing import List, Optional
from datetime import datetime, timezone
import asyncio

from search.providers.base import Provider
from search.types import ProviderResult
from search.utils import get_http_client

API_BASE = "https://api.openreview.net"

def _to_iso_utc_ms(ms: int | None) -> Optional[str]:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

def _parse_cutoff(earliest_date: Optional[str]) -> Optional[datetime]:
    if not earliest_date:
        return None
    try:
        return datetime.strptime(earliest_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None

class OpenReviewProvider(Provider):
    """
    Full-text search via OpenReview's Elasticsearch-backed endpoint:
      POST /notes/search { term, content, group, source, limit, offset }

    Notes:
      - No citations -> min_popularity is ignored.
      - We sort by tmdate desc locally.
      - If /notes/search fails, fall back to recent venue submissions and filter locally.
    """
    name = "openreview"

    async def _search_es(self, query: str, limit: int) -> List[dict]:
        url = f"{API_BASE}/notes/search"
        body = {
            "term": query,
            "content": "all",
            "group": "all",
            "source": "all",
            "limit": min(max(limit, 20), 200),
            "offset": 0,
        }
        async with get_http_client() as client:
            r = await client.post(url, json=body, headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json()
        # Normalize variants
        if isinstance(data, dict) and isinstance(data.get("notes"), list):
            return data["notes"]
        if isinstance(data, list):
            return data
        return []

    async def _fallback_recent_submissions(self, limit: int) -> List[dict]:
        """
        Pull recent submissions from a few popular venues using /notes and sort by tmdate desc.
        Not a full-text search; caller will do a lightweight client-side filter.
        """
        venues = [
            "ICLR.cc/2025/Conference/-/Blind_Submission",
            "NeurIPS.cc/2025/Conference/-/Submission",
            "ICML.cc/2025/Conference/-/Blind_Submission",
        ]

        def _rows(payload) -> list[dict]:
            # Normalize OpenReview responses
            if isinstance(payload, dict):
                if "notes" in payload and isinstance(payload["notes"], list):
                    return payload["notes"]
                # Some endpoints may return {"items": [...]}
                if "items" in payload and isinstance(payload["items"], list):
                    return payload["items"]
                return []
            if isinstance(payload, list):
                return payload
            return []

        out: list[dict] = []
        async with get_http_client() as client:
            for inv in venues:
                params = {
                    "invitation": inv,
                    "select": "id,content.title,content.abstract,content.venue,tmdate,tcdate,forum,forumUrl,url",
                    "sort": "tmdate:desc",
                    "limit": 100,
                    "offset": 0,
                }
                r = await client.get(f"{API_BASE}/notes", params=params)
                if r.status_code >= 400:
                    continue
                rows = _rows(r.json())
                out.extend(rows)
                await asyncio.sleep(0.1)

        # Deduplicate by id and trim
        seen, unique = set(), []
        for n in out:
            if not isinstance(n, dict):
                continue
            nid = n.get("id") or n.get("forum")
            if not nid or nid in seen:
                continue
            seen.add(nid)
            unique.append(n)

        unique.sort(key=lambda n: (n.get("tmdate") or 0), reverse=True)
        return unique[:limit]

    async def search(
        self,
        query: str,
        limit: int,
        *,
        earliest_date: Optional[str] = None,
        min_popularity: Optional[int] = None,  # ignored (no citations)
    ) -> List[ProviderResult]:
        cutoff = _parse_cutoff(earliest_date)

        # Try ES-backed search first
        try:
            notes = await self._search_es(query, limit=limit * 2)  # overfetch for client filtering
        except Exception:
            # Fallback path
            notes = await self._fallback_recent_submissions(limit=limit * 3)

        # Normalize & filter
        out: List[ProviderResult] = []
        for n in notes:
            content = n.get("content") or {}
            title = (content.get("title") or "").strip()
            abstract = (content.get("abstract") or "").strip()

            # If we're on the fallback path, do a simple client-side relevance gate
            if title or abstract:
                qlow = query.lower()
                text = (title + " " + abstract).lower()
                if "/notes/search" not in str(n):  # cheap way to not double-filter ES results
                    # allow weak contains check
                    if qlow and (qlow not in text):
                        continue

            tm_ms = n.get("tmdate") or n.get("tcdate")
            updated_iso = _to_iso_utc_ms(tm_ms)
            if cutoff and tm_ms:
                if datetime.fromtimestamp(tm_ms / 1000.0, tz=timezone.utc) < cutoff:
                    continue

            nid = n.get("id") or n.get("forum")
            venue = content.get("venue") or content.get("venueid") or "OpenReview"
            url = n.get("forumUrl") or n.get("url") or (f"https://openreview.net/forum?id={n.get('forum', nid)}")

            out.append(ProviderResult(
                id=str(nid),
                title=title or "(untitled)",
                url=url,
                abstract=abstract,
                source="openreview",
                authors=(content.get("authors") or []),
                year=(datetime.fromisoformat(updated_iso).year if updated_iso else None),
                venue=venue,
                citations=None,
                stars=None,
                forks=None,
                updated_iso=updated_iso,
                meta={}
            ))
            if len(out) >= limit:
                break

        # Sort freshest first (tmdate desc) to mirror the site
        out.sort(key=lambda w: (w.updated_iso or ""), reverse=True)
        return out[:limit]
