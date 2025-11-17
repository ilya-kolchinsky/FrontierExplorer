import datetime as dt
from typing import List, Optional
from xml.etree import ElementTree
from urllib.parse import urlencode
from search.providers.base import Provider
from search.types import ProviderResult
from search.utils import get_http_client

ARXIV_API = "https://export.arxiv.org/api/query"


class ArxivProvider(Provider):
    name = "arxiv"

    async def search(
        self,
        query: str,
        limit: int = 40,
        *,
        earliest_date: Optional[str] = None,
        min_popularity: Optional[int] = None,  # ignored as arxiv does not provide citations data
    ) -> List[ProviderResult]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urlencode(params)}"

        async with get_http_client() as client:
            r = await client.get(url)
            r.raise_for_status()
            xml = ElementTree.fromstring(r.text)

        cutoff: Optional[dt.datetime] = None
        if earliest_date:
            cutoff = dt.datetime.strptime(earliest_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

        ns = {"a": "http://www.w3.org/2005/Atom"}
        out: List[ProviderResult] = []
        for entry in xml.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            link = ""
            for link_str in entry.findall("a:link", ns):
                if link_str.attrib.get("type") == "text/html":
                    link = link_str.attrib.get("href", "")
            if not link:
                link = entry.findtext("a:id", default="", namespaces=ns) or ""

            authors = [a.findtext("a:name", default="", namespaces=ns) or "" for a in entry.findall("a:author", ns)]
            published = entry.findtext("a:published", default="", namespaces=ns)
            year = None
            updated_iso = None
            include = True
            try:
                if published:
                    dtp = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
                    year = dtp.year
                    updated_iso = dtp.isoformat()
                    if cutoff and dtp < cutoff:
                        include = False
            except Exception:
                pass
            if not include:
                continue

            arxiv_id = (entry.findtext("a:id", default="", namespaces=ns) or link).split("/")[-1]
            out.append(ProviderResult(
                id=arxiv_id, title=title, url=link, abstract=summary, source="arxiv",
                authors=[a for a in authors if a], year=year, venue="arXiv",
                citations=None, stars=None, forks=None, updated_iso=updated_iso, meta={}
            ))
        return out
