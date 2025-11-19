from typing import List, Optional
from datetime import datetime, timezone
from urllib.parse import urlencode

from search.providers.base import Provider
from core.types import ProviderResult
from core.utils import get_http_client

HF_DATASETS_API = "https://huggingface.co/api/datasets"


def _parse_iso(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


class HuggingFaceDatasetsProvider(Provider):
    """
    Search Hugging Face Hub datasets.
    Popularity -> downloads (or likes). Earliest date -> lastModified cutoff.
    """
    name = "huggingface_datasets"

    async def search(
        self,
        query: str,
        limit: int,
        *,
        earliest_date: Optional[str] = None,   # "YYYY-MM-DD"
        min_popularity: Optional[int] = None,  # downloads or likes threshold
    ) -> List[ProviderResult]:
        per_page = min(max(limit, 20), 100)
        params = {
            "search": query,
            "limit": per_page,
            "sort": "downloads",   # or "likes"
            "direction": "-1",
        }
        url = f"{HF_DATASETS_API}?{urlencode(params)}"

        out: List[ProviderResult] = []
        async with get_http_client() as client:
            r = await client.get(url, headers={"Accept": "application/json"})
            r.raise_for_status()
            items = r.json()

        cutoff = None
        if earliest_date:
            try:
                cutoff = datetime.strptime(earliest_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass

        for d in items:
            did = d.get("id")
            if not did:
                continue
            title = did.split("/")[-1]
            url = f"https://huggingface.co/datasets/{did}"
            desc = d.get("description") or ""
            lastmod_iso = _parse_iso(d.get("lastModified"))
            year = datetime.fromisoformat(lastmod_iso).year if lastmod_iso else None

            downloads = int(d.get("downloads") or 0)
            likes = int(d.get("likes") or 0)
            popularity = max(downloads, likes)

            if min_popularity is not None and popularity < min_popularity:
                continue
            if cutoff and lastmod_iso:
                if datetime.fromisoformat(lastmod_iso) < cutoff:
                    continue

            out.append(ProviderResult(
                id=did,
                title=title,
                url=url,
                abstract=desc,
                source="huggingface_datasets",
                authors=[],
                year=year,
                venue="Hugging Face (Datasets)",
                citations=None,
                stars=popularity,   # map popularity to stars for unified ranking
                forks=None,
                updated_iso=lastmod_iso,
                meta={"downloads": downloads, "likes": likes, "tags": d.get("tags", [])}
            ))

            if len(out) >= limit:
                break

        return out[:limit]
