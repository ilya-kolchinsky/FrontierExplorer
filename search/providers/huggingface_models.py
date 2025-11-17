from typing import List, Optional, Union
from datetime import datetime, timezone
from urllib.parse import urlencode

from search.providers.base import Provider
from search.types import ProviderResult
from search.utils import get_http_client

HF_MODELS_API = "https://huggingface.co/api/models"


def _parse_iso(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v]
    return [str(x)]


def _summarize_model(m: dict) -> str:
    """
    Build a short, human-readable summary from HF metadata so cards aren't empty
    even when description/README isn't included in the API response.
    """
    pt = m.get("pipeline_tag")
    lib = m.get("library_name")
    card = m.get("cardData") or {}
    lic = card.get("license") or m.get("license")
    langs = ", ".join(_as_list(card.get("language")))
    ds = _as_list(card.get("datasets"))
    ds_str = (", ".join(ds[:3]) + ("…" if len(ds) > 3 else "")) if ds else ""
    bits = []
    if pt: bits.append(f"pipeline: {pt}")
    if lib: bits.append(f"lib: {lib}")
    if lic: bits.append(f"license: {lic}")
    if langs: bits.append(f"lang: {langs}")
    if ds_str: bits.append(f"datasets: {ds_str}")
    return " · ".join(bits)


class HuggingFaceModelsProvider(Provider):
    """
    Search Hugging Face Hub models. Popularity -> max(downloads, likes).
    Earliest date -> lastModified cutoff.
    """
    name = "huggingface_models"

    async def search(
        self,
        query: str,
        limit: int,
        *,
        earliest_date: Optional[str] = None,   # "YYYY-MM-DD", checked against lastModified
        min_popularity: Optional[int] = None,  # downloads/likes threshold
    ) -> List[ProviderResult]:
        per_page = min(max(limit, 20), 100)
        params = {
            "search": query,
            "limit": per_page,
            "sort": "downloads",      # or "likes"
            "direction": "-1",
            "full": "true",           # <-- get richer fields (pipeline_tag, library_name, etc.)
            "cardData": "true",       # <-- include parsed README metadata (license, language, datasets, …)
        }
        url = f"{HF_MODELS_API}?{urlencode(params)}"

        headers = {"Accept": "application/json"}
        # If you configured HF_TOKEN, you can add: headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"

        out: List[ProviderResult] = []
        async with get_http_client() as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            items = r.json()

        cutoff = None
        if earliest_date:
            try:
                cutoff = datetime.strptime(earliest_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass

        for m in items:
            rid = m.get("id")
            if not rid:
                continue
            title = rid.split("/")[-1]
            page_url = f"https://huggingface.co/{rid}"
            lastmod_iso = _parse_iso(m.get("lastModified"))
            year = datetime.fromisoformat(lastmod_iso).year if lastmod_iso else None

            downloads = int(m.get("downloads") or 0)
            likes = int(m.get("likes") or 0)
            popularity = max(downloads, likes)

            if min_popularity is not None and popularity < min_popularity:
                continue
            if cutoff and lastmod_iso and datetime.fromisoformat(lastmod_iso) < cutoff:
                continue

            # Prefer provided description; otherwise synthesize from metadata
            desc = m.get("description") or _summarize_model(m)

            out.append(ProviderResult(
                id=rid,
                title=title,
                url=page_url,
                abstract=desc or "",
                source="huggingface_models",
                authors=[],
                year=year,
                venue="Hugging Face",
                citations=None,
                stars=popularity,  # unified popularity field
                forks=None,
                updated_iso=lastmod_iso,
                meta={
                    "downloads": downloads,
                    "likes": likes,
                    "tags": m.get("tags", []),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "library_name": m.get("library_name"),
                    "license": (m.get("cardData") or {}).get("license") or m.get("license"),
                    "language": (m.get("cardData") or {}).get("language"),
                }
            ))

            if len(out) >= limit:
                break

        return out[:limit]
