from datetime import datetime, timezone
from typing import List, Literal, Optional, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import math

from core.types import WorkItem
from config import settings

RankingMode = Literal["relevance", "recency", "popularity", "custom"]


def _tokenize(s: str) -> list[str]:
    # super simple tokenizer is fine for BM25 here
    return [tok.lower() for tok in s.split()]


def _signals(query: str, works: List[WorkItem]) -> Tuple[list[float], list[float], list[float]]:
    """Compute relevance (BM25), recency boost, and popularity for each work."""
    # Relevance (BM25) on title+abstract
    corpus = [f"{w.title} {w.abstract}" for w in works]
    tokenized_corpus = [_tokenize(c) for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens = _tokenize(query)
    relevance = list(map(float, bm25.get_scores(q_tokens)))

    # Recency (half-life)
    now = datetime.now(timezone.utc)
    half_life = max(1, settings.RECENCY_HALFLIFE_DAYS)
    recency: list[float] = []
    for w in works:
        if w.updated_iso:
            try:
                dt_ = datetime.fromisoformat(w.updated_iso.replace("Z", "+00:00"))
            except Exception:
                dt_ = now
        elif w.year:
            dt_ = datetime(w.year, 1, 1, tzinfo=timezone.utc)
        else:
            dt_ = now
        days = max(0.0, (now - dt_).total_seconds() / 86400.0)
        recency.append(0.5 ** (days / half_life))

    # Popularity (GitHub stars or citations)
    popularity: list[float] = []
    for w in works:
        s = float(w.stars or 0)
        c = float(w.citations or 0)
        popularity.append(math.log1p(max(s, c)))

    return relevance, recency, popularity


def _normalize_weights(w: Dict[str, float]) -> Tuple[float, float, float]:
    r = max(0.0, float(w.get("relevance", 0.0)))
    t = max(0.0, float(w.get("recency", 0.0)))
    p = max(0.0, float(w.get("popularity", 0.0)))
    s = r + t + p
    if s == 0:
        return 1.0, 0.0, 0.0  # default to pure relevance
    return r / s, t / s, p / s


def rank_works(
    query: str,
    works: List[WorkItem],
    mode: Union[str, RankingMode] = "relevance",
    weights: Optional[Dict[str, float]] = None,
) -> List[WorkItem]:
    if not works:
        return works

    relevance, recency, popularity = _signals(query, works)

    # Choose scoring strategy
    scores: list[Tuple[float, int]] = []
    if mode == "relevance":
        scores = [(relevance[i], i) for i in range(len(works))]
    elif mode == "recency":
        scores = [(recency[i], i) for i in range(len(works))]
    elif mode == "popularity":
        scores = [(popularity[i], i) for i in range(len(works))]
    elif mode == "custom":
        w_rel, w_rec, w_pop = _normalize_weights(weights or {})
        scores = [(
            w_rel * relevance[i] + w_rec * recency[i] + w_pop * popularity[i],
            i
        ) for i in range(len(works))]
    else:
        # fallback
        scores = [(relevance[i], i) for i in range(len(works))]

    # Sort by score desc; use relevance as a stable tie-breaker
    scores.sort(key=lambda x: x[0], reverse=True)
    return [works[i] for _, i in scores]


def compute_work_scores(
    query: str,
    works: List[WorkItem],
    mode: Union[str, RankingMode] = "relevance",
    weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Return a score per work under the chosen ranking mode/weights."""
    if not works:
        return []
    relevance, recency, popularity = _signals(query, works)
    if mode == "relevance":
        return relevance
    if mode == "recency":
        return recency
    if mode == "popularity":
        return popularity
    # custom
    w_rel, w_rec, w_pop = _normalize_weights(weights or {})
    return [
        w_rel * relevance[i] + w_rec * recency[i] + w_pop * popularity[i]
        for i in range(len(works))
    ]