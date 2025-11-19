from typing import List, Tuple, Optional, Dict, Any, Union
import asyncio
import numpy as np

from .providers.huggingface_datasets import HuggingFaceDatasetsProvider
from .providers.huggingface_models import HuggingFaceModelsProvider
from .providers.openreview import OpenReviewProvider
from .providers.semanticscholar import SemanticScholarProvider
from core.types import WorkItem, Cluster
from config import settings
from .providers.arxiv import ArxivProvider
from .providers.github import GitHubProvider
from .services.normalize import provider_result_to_work, dedupe
from .services.ranking import rank_works, compute_work_scores
from core.embeddings import embed_texts
from .services.clustering import cluster_works

PROVIDERS = {
    "Arxiv": ArxivProvider(),
    "Github": GitHubProvider(),
    "Semantic Scholar": SemanticScholarProvider(),
    "OpenReview": OpenReviewProvider(),
    "Hugging Face Models": HuggingFaceModelsProvider(),
    "Hugging Face Datasets": HuggingFaceDatasetsProvider(),
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    # assume vectors are normalized by embed_texts; still guard:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float(np.dot(a, b.T).squeeze())


async def _fetch_from_providers(
    query: str,
    selected: List[str],
    cap: int,
    earliest_date: Optional[str],
    min_popularity: Optional[int],
) -> Tuple[List[Any], List[Dict[str, str | Any]]]:
    async def fetch(pname: str):
        prov = PROVIDERS[pname]
        return await prov.search(
            query, cap,
            earliest_date=earliest_date, min_popularity=min_popularity
        )
    tasks = [asyncio.create_task(fetch(p)) for p in selected]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items, errors = [], []
    for pname, res in zip(selected, results):
        if isinstance(res, Exception):
            errors.append({"provider": pname, "error": f"{type(res).__name__}: {str(res)}"})
        else:
            items.extend(res)
    return items, errors


async def _search_async(
    query_or_queries: Union[str, List[str]],
    providers: List[str],
    k: Optional[int],
    earliest_date: Optional[str],   # "YYYY-MM-DD" or None
    min_popularity: Optional[int],       # None means don’t filter by popularity
    max_results: Optional[int],
    ranking_mode: str,
    ranking_weights: Optional[Dict[str, float]],
) -> Tuple[List[WorkItem], List[Cluster], List[Dict[str, str]]]:
    selected = [p for p in providers if p in PROVIDERS] or list(PROVIDERS.keys())
    cap = max_results or settings.DEFAULT_MAX_RESULTS

    queries: List[str] = [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return [], [], [{"provider": "planner", "error": "No queries provided"}]

    merged_raw = []
    all_errors: List[Dict[str, str]] = []
    for q in queries:
        items, errs = await _fetch_from_providers(q, selected, cap, earliest_date, min_popularity)
        merged_raw.extend(items)
        all_errors.extend(errs)

    works = dedupe(provider_result_to_work(r) for r in merged_raw)
    if not works:
        return [], [], all_errors

    # Use a combined text for relevance (BM25) so multiple queries help ranking
    combined_query = " ".join(queries)

    # rank first, then take top N overall
    works = rank_works(combined_query, works, mode=ranking_mode, weights=ranking_weights)
    works = works[:cap]

    texts = [f"{w.title}. {w.abstract}" for w in works]
    vecs = embed_texts(texts)

    clusters, works_out = cluster_works(works, vecs, k=k)

    # cluster scoring & ordering
    # 1) per-work scores under the chosen mode
    per_work_scores = compute_work_scores(combined_query, works_out, mode=ranking_mode, weights=ranking_weights)
    wid_to_score = {w.id: per_work_scores[i] for i, w in enumerate(works_out)}

    # 2) query embedding for centroid similarity (blend factor is small)
    qv = embed_texts([combined_query])[0]  # normalized by embed_texts
    centroid_weight = 0.25  # tweak: 0..1 small influence
    member_weight = 0.75  # main signal from members

    scored_clusters: List[Cluster] = []
    for cl in clusters:
        # sort members inside cluster by per-work score (desc)
        member_scores = [(wid_to_score.get(wid, 0.0), wid) for wid in cl.work_ids]
        member_scores.sort(reverse=True)
        ordered_wids = [wid for _, wid in member_scores]

        # mean of top-3 member scores to avoid big-cluster bias
        topk = [s for s, _ in member_scores[:3]] or [0.0]
        member_score = float(np.mean(topk))

        # centroid similarity
        cent = np.array(cl.centroid, dtype="float32") if cl.centroid is not None else None
        centroid_sim = _cosine(qv, cent) if cent is not None else 0.0

        # blended cluster score
        cl_score = member_weight * member_score + centroid_weight * centroid_sim

        scored_clusters.append(Cluster(
            id=cl.id,
            label=cl.label,
            work_ids=ordered_wids,
            centroid=cl.centroid,
            score=cl_score,
        ))

    # 3) order clusters by score desc
    scored_clusters.sort(key=lambda c: (c.score or 0.0), reverse=True)

    return works_out, clusters, all_errors


def frontier_search(
    query_or_queries: Union[str, List[str]],
    providers: List[str] | None = None,
    k: Optional[int] = None,
    earliest_date: Optional[str] = None,
    min_popularity: Optional[int] = None,
    max_results: Optional[int] = None,
    ranking_mode: str = "relevance",
    ranking_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[WorkItem], List[Cluster], List[Dict[str, str]]]:
    return asyncio.run(
        _search_async(
            query_or_queries,
            providers or list(PROVIDERS.keys()),
            k,
            earliest_date,
            min_popularity,
            max_results,
            ranking_mode,
            ranking_weights,
        )
    )
