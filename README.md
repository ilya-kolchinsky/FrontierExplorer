# Frontier Explorer

A lightweight, Streamlit-based **frontier scanner** that finds cutting-edge papers, repos, models, and datasets — then lets you **chat with the selected works** using a local or hosted LLM.

- Multi-source search (arXiv, GitHub, Semantic Scholar, OpenReview, Hugging Face Models/Datasets/Spaces)
- Smart ranking (relevance / recency / popularity or custom weights)
- Clustering with query-aware cluster ranking
- LLM query planner: turn a natural request into multiple targeted search queries
- LLM summary: concise bullet summary of the top results
- **Chat with selected works**: fetch full papers/READMEs/model cards and ask deep questions

---

## Demo (quick tour)

1. Choose **Direct** query or **LLM (generate queries)** and providers.  
2. Filter by **earliest date** and **min popularity** (stars/citations/downloads/likes).  
3. Rank by relevance/recency/popularity or custom weights.  
4. Browse clusters (auto-sorted by relevance to your query).  
5. **Select** up to *N* items (sidebar-controlled cap).  
6. (Optional) Generate a **summary** and/or **chat** with the selected works.

> The chat uses full text when possible (PDFs via arXiv/OpenReview/Semantic Scholar OA, GitHub READMEs, Hugging Face model cards/dataset cards).

---

## Features

- **Providers**
  - arXiv (papers)
  - GitHub (repos; popularity = stars)
  - Semantic Scholar (papers; popularity = citations; retries & backoff supported)
  - OpenReview (ICLR/NeurIPS/ICML submissions; freshest venue content)
  - Hugging Face **Models** (popularity = max(downloads, likes))
  - Hugging Face **Datasets** (popularity = max(downloads, likes))
  - Hugging Face **Spaces** (popularity = likes)
- **Ranking modes**: `relevance`, `recency`, `popularity`, or **custom weights**.
- **Clustering**: query-aware cluster scoring (member scores + centroid cosine), single-cluster auto-flatten.
- **LLM planner**: OpenAI **or** any OpenAI-compatible endpoint (vLLM/LiteLLM/etc.). UI fields for **Base URL** + **Model**.
- **LLM summary**: 5–8 crisp bullets; show **Results only**, **Summary only**, or **Both**.
- **RAG chat**: builds an in-memory index of chunks from full texts; answers cite sources as `[1]`, `[2]`, …  

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Environment

Create `.env` (or export as env vars):

```bash
# LLMs (planner + summary + chat) — pick OpenAI or an OpenAI-compatible endpoint (vLLM/LiteLLM)
OPENAI_API_KEY=sk-...                 # optional if using vLLM only
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini                 # default OpenAI model

VLLM_BASE_URL=http://localhost:8000/v1  # e.g. vLLM or a gateway; leave empty to prefer OpenAI
VLLM_API_KEY=                          # optional if your gateway requires it
VLLM_MODEL=qwen2.5-7b-instruct

# Provider tokens / keys (optional but recommended)
GITHUB_TOKEN=ghp_...                  # avoids rate limits
S2_API_KEY=...                        # Semantic Scholar key; avoids 429s
HF_TOKEN=...                          # only needed for private HF repos or stricter limits
```

### 4) Run

```bash
python -m streamlit run ui/app.py
```

Open the local URL Streamlit prints.

---

## Providers & popularity mapping

| Provider              | Type       | Popularity used           | Notes                                              |
|-----------------------|------------|---------------------------|----------------------------------------------------|
| arXiv                 | Papers     | *(none)*                  | Recency + relevance only                           |
| GitHub                | Repos      | **stars**                 | README fetched via API                             |
| Semantic Scholar (S2) | Papers     | **citationCount**         | Uses details endpoint; retries/backoff             |
| OpenReview            | Papers     | *(none)*                  | Full-text search endpoint; fallback to venue lists |
| HF **Models**         | Models     | **max(downloads, likes)** | Model card used as fulltext                        |
| HF **Datasets**       | Datasets   | **max(downloads, likes)** | Dataset card used as fulltext                      |
| HF **Spaces**         | Apps/Demos | **likes**                 | Optional                                           |

---

## Extending: add a new provider

1. **Create a provider** in `core/providers/yourprovider.py` implementing:
   ```python
   from typing import List
   from search.providers.base import Provider
   from core.types import ProviderResult
   
   class YourProvider(Provider):
       name = "yourprovider"
       async def search(self, query, limit, *, earliest_date=None, min_popularity=None) -> List[ProviderResult]: ...
   ```
2. Map its **popularity** signal to `ProviderResult.stars` or `.citations` (the ranking uses `log1p(max(stars, citations))`).
3. Add it to `PROVIDERS` in `core/orchestrator.py`.
4. Add to the Streamlit **Providers** multiselect.
5. (Optional) Implement a **fulltext loader** in `core/ingest/loaders.py` and dispatch it in `load_fulltext_for_work`.

---

## Troubleshooting

- **Semantic Scholar 429**: set `S2_API_KEY`; our provider retries/backoffs but a key helps a lot.
- **GitHub README is None**: ensure `WorkItem.id` is `owner/repo` or a valid `github.com/owner/repo` URL; set `GITHUB_TOKEN`.
- **OpenReview 400**: we use `/notes/search` (POST) for full-text; fallback pulls recent venue submissions.

---

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

---

## Acknowledgements

- Thanks to the arXiv, Semantic Scholar, OpenReview, GitHub, and Hugging Face teams for their public APIs.
- This project was built as a fast, pragmatic research tool — feedback and PRs are welcome!
