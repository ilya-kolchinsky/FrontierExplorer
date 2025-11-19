from __future__ import annotations

import os
from typing import Optional
import re
import base64
import io
from urllib.parse import urlparse

from PyPDF2 import PdfReader

from core.types import WorkItem
from core.utils import get_http_client


# --- helpers ---
def work_id_from_global_id(wid: str) -> str:
    return wid.split(':')[0].strip()

def _arxiv_id_from_url(url: str) -> Optional[str]:
    # handles .../abs/2501.01234v2 or .../pdf/2501.01234.pdf
    m = re.search(r"/(abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", url)
    return m.group(2) if m else None


async def _fetch(url: str, headers: dict | None = None) -> str:
    async with get_http_client(headers=headers or {}) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


async def _fetch_bytes(url: str, headers: dict | None = None) -> bytes:
    async with get_http_client(headers=headers or {}) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


def _md_from_readme_api_json(j: dict) -> Optional[str]:
    # GitHub "readme" API returns base64 by default unless Accept: raw
    c = j.get("content")
    if c and j.get("encoding") == "base64":
        try:
            return base64.b64decode(c.encode()).decode("utf-8", errors="ignore")
        except Exception:
            return None
    return None


# --- extractors per source ---


async def load_fulltext_arxiv(w: WorkItem) -> Optional[str]:
    arx_id = (_arxiv_id_from_url(w.url or "") or
              _arxiv_id_from_url("https://arxiv.org/abs/" + work_id_from_global_id(w.id)))
    if not arx_id:
        return None

    try:
        html = await _fetch(f"https://arxiv.org/html/{arx_id}")
        # crude strip
        return re.sub(r"<[^>]+>", " ", html)
    except Exception:
        # resort to PDF if no other choice
        pdf_bytes = await _fetch_bytes(f"https://arxiv.org/pdf/{arx_id}")
        # lightweight extraction via PyPDF2 (good enough for PoC)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or None


async def load_fulltext_openreview(w: WorkItem) -> Optional[str]:
    # OpenReview PDF: https://openreview.net/pdf?id=<forum_or_id>
    fid = work_id_from_global_id(w.id)
    pdf_url = f"https://openreview.net/pdf?id={fid}"
    try:
        pdf_bytes = await _fetch_bytes(pdf_url)
        import io
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or None
    except Exception:
        return (w.abstract or "").strip() or None


async def load_fulltext_github(w: WorkItem, github_token: Optional[str]) -> Optional[str]:
    # derive "owner/repo"
    repo = work_id_from_global_id(w.id)
    if "/" not in repo:
        # try to parse from URL like https://github.com/owner/repo
        try:
            u = urlparse(w.url or "")
            m = re.match(r"^/([^/]+)/([^/]+)", u.path or "")
            if m:
                repo = f"{m.group(1)}/{m.group(2).replace('.git', '')}"
        except Exception:
            pass
    if "/" not in repo:
        return None  # can't identify repo

    # base headers
    base_headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FrontierScanner/1.0",
    }
    if github_token:
        base_headers["Authorization"] = f"Bearer {github_token}"

    async with get_http_client(headers=base_headers) as client:
        # --- 1) try the raw readme endpoint (content negotiation) ---
        raw_headers = dict(base_headers)
        raw_headers["Accept"] = "application/vnd.github.v3.raw"
        r = await client.get(f"https://api.github.com/repos/{repo}/readme", headers=raw_headers)
        if r.status_code == 200:
            return r.text
        # If rate-limited, surface a clear message in logs (optional)
        if r.status_code in (403, 429):
            # You can log r.headers.get("x-ratelimit-remaining"), etc.
            pass

        # --- 2) fallback: list repo root and find a README.* file ---
        r2 = await client.get(f"https://api.github.com/repos/{repo}/contents")
        if r2.status_code != 200:
            return None
        items = r2.json() or []
        # prioritize typical names
        candidates = []
        for it in items:
            if it.get("type") != "file":
                continue
            name = (it.get("name") or "").lower()
            if name.startswith("readme"):
                candidates.append(it)

        # sort preference: README.md > README.MD > README.rst > README
        def score(n: str) -> int:
            n = n.lower()
            if n == "readme.md":
                return 100
            if n.endswith(".md"):
                return 90
            if n.endswith(".rst"):
                return 80
            if n == "readme":
                return 70
            return 0

        candidates.sort(key=lambda it: score(it.get("name", "")), reverse=True)

        for it in candidates:
            dl = it.get("download_url")
            if not dl:
                # fallback: use the raw endpoint path
                path = it.get("path")
                if not path:
                    continue
                dl = f"https://raw.githubusercontent.com/{repo}/HEAD/{path}"
            r3 = await client.get(dl, headers={"User-Agent": base_headers["User-Agent"]})
            if r3.status_code == 200 and r3.text.strip():
                return r3.text

    return None


async def load_fulltext_hf_model(w: WorkItem) -> Optional[str]:
    # Hugging Face model card raw markdown
    # GET https://huggingface.co/api/models/{repo_id}/readme?raw=1
    rid = work_id_from_global_id(w.id)
    try:
        md = await _fetch(f"https://huggingface.co/api/models/{rid}/readme?raw=1",
                          headers={"Accept": "text/plain"})
        return md
    except Exception:
        return (w.abstract or "").strip() or None


async def load_fulltext_hf_dataset(w: WorkItem) -> Optional[str]:
    rid = work_id_from_global_id(w.id)
    try:
        md = await _fetch(f"https://huggingface.co/api/datasets/{rid}/readme?raw=1",
                          headers={"Accept": "text/plain"})
        return md
    except Exception:
        return (w.abstract or "").strip() or None


async def load_fulltext_semanticscholar(w: WorkItem) -> Optional[str]:
    """
    Fetch full text for a Semantic Scholar paper:
      1) Hit S2 paper endpoint to get open-access PDF URL
      2) Download PDF and extract text
      3) Fallback: arXiv (if arXivId present) -> arXiv loader
      4) Fallback: title + abstract
    """
    S2_API = "https://api.semanticscholar.org/graph/v1"
    s2_api_key = os.getenv("S2_API_KEY")

    # Figure out the best identifier: DOI or S2 paperId
    pid = work_id_from_global_id(w.id)
    if not pid and w.url:
        # try to extract a paper id from the URL if possible
        m = re.search(r"/paper/([0-9a-fA-F]{40})", w.url)  # legacy pid heuristic
        if m:
            pid = m.group(1)

    # Prefer DOI route if the id looks like a DOI (starts with '10.')
    if pid.startswith("10."):
        ident = f"DOI:{pid}"
    else:
        ident = pid  # assume paperId; if empty, we’ll just fallback later

    # 0) If we already have an arXiv-like URL in meta or url, try arXiv directly
    #    (your provider may have populated externalIds; if so, store arXiv in WorkItem.meta)
    arx = None
    if w.meta and isinstance(w.meta, dict):
        arx = w.meta.get("arxivId") or w.meta.get("ArXiv") or w.meta.get("arxiv")
    if not arx:
        # try to spot arXiv in the URL
        if (w.url or "").find("arxiv.org") >= 0:
            arx = True  # signal to try arXiv fallback below

    # 1) Query S2 for open-access PDF
    pdf_bytes = None
    if ident:
        headers = {"Accept": "application/json"}
        if s2_api_key:
            headers["x-api-key"] = s2_api_key
        params = {"fields": "title,year,venue,externalIds,openAccessPdf,url"}
        async with get_http_client(headers=headers) as client:
            r = await client.get(f"{S2_API}/paper/{ident}", params=params)
            if r.status_code == 200:
                data = r.json()
                # if arXiv present, remember it for fallback
                ext = data.get("externalIds") or {}
                if not arx:
                    arx = ext.get("ArXiv") or ext.get("arXiv")
                oapdf = (data.get("openAccessPdf") or {}).get("url")
                if oapdf:
                    # fetch the PDF
                    r2 = await client.get(oapdf)
                    if r2.status_code == 200 and r2.content:
                        pdf_bytes = r2.content

    # 2) PDF -> text
    if pdf_bytes:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                return text.strip()
        except Exception:
            pass  # fall through to arXiv/abstract fallback

    # 3) arXiv fallback (if we have an arXiv id or URL)
    if arx:
        t = await load_fulltext_arxiv(w)
        if t and t.strip():
            return t.strip()

    # 4) Last resort
    return await load_fulltext_generic(w)


async def load_fulltext_generic(w: WorkItem) -> Optional[str]:
    # Fallback: return title + abstract
    return f"{w.title or ''}\n\n{w.abstract or ''}".strip() or None


async def load_fulltext_for_work(w: WorkItem, github_token: Optional[str] = None) -> Optional[str]:
    src = (w.source or "").lower().strip()
    if src == "arxiv":
        return await load_fulltext_arxiv(w)
    if src == "openreview":
        return await load_fulltext_openreview(w)
    if src == "github":
        return await load_fulltext_github(w, github_token)
    if src == "huggingface_models":
        return await load_fulltext_hf_model(w)
    if src == "huggingface_datasets":
        return await load_fulltext_hf_dataset(w)
    if src == "semanticscholar":
        return await load_fulltext_semanticscholar(w)

    return await load_fulltext_generic(w)
