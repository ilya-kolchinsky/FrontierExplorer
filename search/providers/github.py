from typing import List, Optional
from urllib.parse import urlencode
from search.providers.base import Provider
from search.types import ProviderResult
from search.utils import get_http_client
from config import settings

GITHUB_API = "https://api.github.com/search/repositories"


class GitHubProvider(Provider):
    name = "github"

    async def search(
        self,
        query: str,
        limit: int = 30,
        *,
        earliest_date: Optional[str] = None,   # "YYYY-MM-DD"
        min_popularity: Optional[int] = None
    ) -> List[ProviderResult]:
        qualifiers = []
        if min_popularity is not None:
            qualifiers.append(f"stars:>={min_popularity}")
        if earliest_date:
            # Prefer most recent activity; fallback to created if you want
            qualifiers.append(f"pushed:>={earliest_date}")

        q = " ".join([query] + qualifiers)

        per_page = min(50, limit)
        params = {
            "q": q,
            "sort": "stars",      # still sort by stars
            "order": "desc",
            "per_page": per_page,
            "page": 1,
        }
        headers = {"Accept": "application/vnd.github+json"}
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

        async with get_http_client(headers=headers) as client:
            r = await client.get(f"{GITHUB_API}?{urlencode(params)}")
            r.raise_for_status()
            data = r.json()

        items = data.get("items", [])[:limit]
        out: List[ProviderResult] = []
        for it in items:
            full_name = it.get("full_name")
            out.append(ProviderResult(
                id=full_name,
                title=it.get("name") or full_name,
                url=it.get("html_url"),
                abstract=it.get("description") or "",
                source="github",
                authors=[it.get("owner", {}).get("login", "")],
                year=(int(it["created_at"][:4]) if it.get("created_at") else None),
                venue="GitHub",
                stars=it.get("stargazers_count"),
                forks=it.get("forks_count"),
                citations=None,
                updated_iso=it.get("pushed_at") or it.get("updated_at"),
                meta={"topics": it.get("topics", [])}
            ))
        return out
