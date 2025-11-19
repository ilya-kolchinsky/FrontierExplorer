from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from core.types import WorkItem, Cluster
from config import settings

# Domain stopwords we never want in labels (extend as needed)
DOMAIN_STOP = {
    "paper", "study", "approach", "method", "methods", "framework", "system", "model", "models",
    "task", "tasks", "result", "results", "analysis", "dataset", "datasets", "using", "with",
    "based", "across", "towards", "toward", "state", "art", "soa", "performance", "evaluation",
    "application", "applications", "problem", "problems", "novel", "new", "proposed"
}

# Whitelist short technical terms that should NOT be filtered even if short
KEEP_SHORT = {"llm", "gpt", "rl", "nlp", "rag", "rlhf", "api", "mlp", "gan", "cnn", "rnn", "lstm", "mcts"}


def _clean_terms(terms: np.ndarray) -> np.ndarray:
    """Lowercase; keep alphabetic or known tech acronyms; drop tiny junk."""
    out = []
    for t in terms:
        s = t.lower().strip()
        if s in KEEP_SHORT:
            out.append(s)
            continue
        # prefer alphabetic tokens or 2-word phrases of alphabetic tokens
        if " " in s:
            a, b = s.split(" ", 1)
            if a.isalpha() and b.isalpha() and len(a) > 2 and len(b) > 2:
                out.append(s)
        else:
            if s.isalpha() and len(s) > 2:
                out.append(s)
    return np.array(out)


def _cluster_labels(texts: List[str], groups: List[List[int]], title_weights: List[float]) -> List[str]:
    if not texts:
        return ["Results"] * len(groups)

    # Vectorizer tuned for informative n-grams
    vec = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS.union(DOMAIN_STOP)),
        ngram_range=(1, 2),
        max_features=4000,
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        min_df=2,  # drop rare noise
        max_df=0.85  # drop overly common terms
    )
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    terms = _clean_terms(terms)

    # Build a mapping back to column indices after cleaning
    col_map = {t: i for i, t in enumerate(vec.get_feature_names_out())}
    valid_cols = [col_map[t] for t in terms if t in col_map]
    if not valid_cols:
        return ["Results"] * len(groups)
    Xv = X[:, valid_cols]
    terms_v = np.array([t for t in terms if t in col_map])

    labels: List[str] = []
    for idxs in groups:
        if not idxs:
            labels.append("Misc")
            continue

        # Sum tf-idf in the cluster; add a small preference for bigrams and title-weighted docs
        weights = np.array([title_weights[i] for i in idxs])[:, None]  # shape (n_docs,1)
        sub = Xv[idxs].multiply(weights).sum(axis=0).A1  # weighted sum
        # boost bigrams slightly
        bonus = np.array([1.25 if " " in t else 1.0 for t in terms_v])
        scored = sub * bonus

        # pick top 3 distinct terms
        top_idx = np.argsort(scored)[-3:][::-1]
        top_terms = [terms_v[i] for i in top_idx if scored[i] > 0]

        # fallback if something goes wrong
        label = " / ".join(top_terms) if top_terms else "Results"
        labels.append(label.title())  # Title Case for nice UI
    return labels


def cluster_works(works: List[WorkItem], vectors: np.ndarray, k: int | None = None) -> Tuple[List[Cluster], List[WorkItem]]:
    if len(works) == 0:
        return [], []

    k_eff = min(max(1, k or settings.CLUSTERS_K), len(works))

    # KMeans on L2-normalized vectors approximates cosine clustering
    kmeans = KMeans(n_clusters=k_eff, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(vectors)

    groups: List[List[int]] = [[] for _ in range(k_eff)]
    for i, lab in enumerate(labels):
        groups[lab].append(i)

    # Build weighted text: give titles extra weight to steer labels
    texts = []
    title_weights = []
    for w in works:
        title = (w.title or "").strip()
        abstract = (w.abstract or "").strip()
        # Repeat title to up-weight it in TF-IDF; keep a scalar for cluster weighting
        texts.append(f"{title}. {title}. {abstract}")
        title_weights.append(1.5)  # scalar boost per doc; tweak if desired

    friendly = _cluster_labels(texts, groups, title_weights)

    clusters: List[Cluster] = []
    for cid, idxs in enumerate(groups):
        wid_list = [works[i].id for i in idxs]
        centroid = kmeans.cluster_centers_[cid].tolist()
        clusters.append(Cluster(id=f"c{cid}", label=friendly[cid], work_ids=wid_list, centroid=centroid))

    return clusters, works
