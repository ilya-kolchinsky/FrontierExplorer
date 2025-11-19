import os

import streamlit as st
from typing import List, Optional
from datetime import date, timedelta

from analysis.knowledge import build_knowledge_index
from config import settings
from llm.analyzer import answer_with_selected
from llm.summarizer import summarize_results
from search.orchestrator import frontier_search, PROVIDERS
from llm.planner import generate_queries

st.set_page_config(page_title="Frontier Explorer", layout="wide")
st.title("Frontier Explorer")
st.caption("Search state-of-the-art papers and repos")

# Session state: keep latest results + params
if "results" not in st.session_state:
    st.session_state["results"] = None
if "params" not in st.session_state:
    st.session_state["params"] = {}

# Chat state
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "rag_for_ids" not in st.session_state:
    st.session_state.rag_for_ids = set()
if "chat_answer" not in st.session_state:
    st.session_state.chat_answer = None
if "chat_citations" not in st.session_state:
    st.session_state.chat_citations = []
if "chat_question" not in st.session_state:
    st.session_state.chat_question = "Compare the provided works."


def clear_selections():
    # remove any previous selection checkboxes
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("sel:"):
            del st.session_state[key]


with st.sidebar:
    st.header("Main Options")

    query_mode = st.radio(
        "Query input mode",
        options=["Direct", "LLM (generate queries)"],
        index=0,
        help="Use your exact search string, or ask an LLM to craft multiple search queries."
    )

    if query_mode == "Direct":
        query = st.text_input("Search string", value="semantic routing LLM")
        planned_intent = None
    else:
        planned_intent = st.text_area(
            "Describe what you want to find",
            value="Give me the latest and best methods for semantically routing a query to the most fitting language model.",
            height=90
        )
        query = None  # not used in this mode

    valid_providers = PROVIDERS.keys()
    providers = st.multiselect("Select the sources to use", options=valid_providers, default=valid_providers)

    show_mode = st.radio(
        "Display",
        options=["Results only", "Summary only", "Both"],
        index=2,
        help="Choose whether to show raw results, an LLM-generated summary, or both."
    )

    st.divider()
    st.subheader("Filters")

    max_results: int = st.number_input(
        "Max results",
        min_value=10, max_value=500, value=settings.DEFAULT_MAX_RESULTS, step=10,
        help="Caps the total results after ranking/dedup; providers will also fetch at most this many."
    )

    default_earliest = date.today() - timedelta(days=365)  # default: last 12 months
    earliest_date: Optional[date] = st.date_input(
        "Earliest publication/activity date",
        value=default_earliest,
        help="arXiv: published date; GitHub: last push date"
    )

    min_popularity: Optional[int] = st.number_input(
        "Min popularity (stars/citations/downloads)",
        min_value=0, max_value=1_000_000, value=10, step=1,
        help="GitHub: stars; papers: citations; other: downloads or likes."
    )

    st.divider()
    st.subheader("Group and Sort")

    ranking_mode = st.radio(
        "Rank by",
        options=["relevance", "recency", "popularity", "custom"],
        index=0,
        help="Choose a single criterion or a custom blend."
    )

    ranking_weights = None
    if ranking_mode == "custom":
        st.caption("Set custom weights (we'll normalize them).")
        w_rel = st.slider("Relevance weight", 0.0, 3.0, 1.0, 0.1)
        w_rec = st.slider("Recency weight",   0.0, 3.0, 0.6, 0.1)
        w_pop = st.slider("Popularity weight",0.0, 3.0, 0.4, 0.1)
        ranking_weights = {"relevance": w_rel, "recency": w_rec, "popularity": w_pop}

    k = st.slider("Clusters (k)", min_value=1, max_value=10, value=1)

    st.divider()
    st.subheader("LLM Settings")
    llm_base_url = st.text_input(
        "LLM API Base URL",
        value=settings.VLLM_BASE_URL,
        placeholder="e.g. http://localhost:8000/v1  (leave empty to use OpenAI)",
        help="Any OpenAI-compatible /v1/chat/completions endpoint (vLLM, LiteLLM, etc.)."
    )
    llm_model = st.text_input(
        "LLM model name",
        value=settings.DEFAULT_VLLM_MODEL,
        help="Model name to send to the endpoint. For vLLM, use the served model ID."
    )

    st.divider()
    st.subheader("Search Results Analysis")

    selection_limit: int = st.number_input(
        "Max Selected Results",
        min_value=1, max_value=50, value=5, step=1,
        help="Hard cap on the number of items you can select at the same time."
    )

    st.divider()
    run = st.button("Run search", type="primary")

if run:
    # Plan queries if needed
    if query_mode == "LLM (generate queries)":
        if not planned_intent or not planned_intent.strip():
            st.error("Please describe your goal for the LLM to generate queries.")
            st.stop()
        with st.spinner("Asking LLM to craft search queries..."):
            planned_queries = generate_queries(
                planned_intent,
                n=5,
                llm_base_url=(llm_base_url or None),
                llm_model=(llm_model or None),
            )
        st.session_state["queries_used"] = planned_queries
        query_or_queries = planned_queries
    else:
        if not query or not query.strip():
            st.error("Please enter a search string.")
            st.stop()
        st.session_state["queries_used"] = [query]
        query_or_queries = query

    with st.spinner("Searching..."):
        dstr = earliest_date.strftime("%Y-%m-%d") if earliest_date else None
        works, clusters, errors = frontier_search(
            query_or_queries,
            providers=providers,
            k=k,
            earliest_date=dstr,
            min_popularity=min_popularity if min_popularity is not None else None,
            max_results=max_results,
            ranking_mode=ranking_mode,
            ranking_weights=ranking_weights,
        )

    if show_mode != "Results only":
        query_or_goal = (planned_intent if query_mode.startswith("LLM") else (query or ""))
        with st.spinner("Summarizing search results..."):
            summary_text = summarize_results(
                works,
                query_or_goal=query_or_goal,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
            )
    else:
        summary_text = None

    # Store results + params in session state and clear old selections
    st.session_state.results = {
        "works": works,
        "clusters": clusters,
        "errors": errors,
        "summary": summary_text,
    }
    st.session_state["params"] = {
        "query": query, "k": k, "providers": providers,
        "earliest": dstr, "min_popularity": min_popularity, "max_results": max_results,
        "ranking_mode": ranking_mode, "ranking_weights": ranking_weights,
    }
    clear_selections()

data = st.session_state["results"]
if not data:
    st.info("Enter a query on the left and click **Run search**.")
    st.stop()

# Show the queries used (for transparency & quick edits)
queries_used = st.session_state.get("queries_used", [])
if queries_used:
    st.caption("Queries used:")
    st.write(", ".join([f"`{q}`" for q in queries_used]))

works = data["works"]
clusters = data["clusters"]
errors = data.get("errors", []) or []
summary_text = data.get("summary")

for err in errors:
    st.toast(f"Provider **{err.get('provider','?')}** failed: {err.get('error','Unknown error')}", icon="⚠️")

work_by_id = {w.id: w for w in works}


def selected_ids_from_state() -> list[str]:
    """Scan session_state for current selections for this search token."""
    out = []
    for k, v in st.session_state.items():
        if isinstance(k, str) and k.startswith("sel:") and v is True:
            # key format: sel:{work_id}
            parts = k.split(":")
            if len(parts) == 2:
                out.append(parts[1])
    return out


def render_work_card(w, limit: int, num_current_selected: int) -> None:
    """Render a single result with a checkbox that respects the selection cap.

    If the cap is reached, disable checkboxes for *unselected* items.
    Already-selected items remain enabled so users can unselect them.
    """
    limit_reached = num_current_selected >= limit
    is_selected = f"sel:{w.id}" in st.session_state and bool(st.session_state[f"sel:{w.id}"])
    disable_new_select = (limit_reached and not is_selected)

    cols = st.columns([6, 1])
    with cols[0]:
        st.markdown(f"**[{w.title}]({w.url})**")
        meta = []
        if w.source:
            meta.append(w.source)
        if w.venue:
            meta.append(w.venue)
        if w.year:
            meta.append(str(w.year))
        if w.stars:
            meta.append(f"★ {w.stars}")
        st.caption(" • ".join(meta))
        if w.abstract:
            st.write(w.abstract[:400] + ("..." if len(w.abstract) > 400 else ""))

        # HF models: show extra chips if you have them
        if w.source == "huggingface_models" and (w.meta or {}):
            m = w.meta or {}
            chips = []
            if m.get("pipeline_tag"):
                chips.append(f"pipeline: {m['pipeline_tag']}")
            if m.get("library_name"):
                chips.append(f"lib: {m['library_name']}")
            if m.get("license"):
                chips.append(f"license: {m['license']}")
            lang = m.get("language")
            if lang:
                chips.append("lang: " + (", ".join(lang) if isinstance(lang, list) else str(lang)))
            if chips:
                st.caption(" · ".join(chips))
            dl = m.get("downloads")
            lk = m.get("likes")
            if dl or lk:
                st.caption(f"⬇ {dl or 0}   ♥ {lk or 0}")

    with cols[1]:
        st.checkbox(
            "Select",
            key=f"sel:{w.id}",
            disabled=disable_new_select,
            help=("Selection limit reached" if disable_new_select else None),
        )


selected_ids: List[str] = selected_ids_from_state()


def render_summary():
    st.subheader("LLM Summary of Search Results")
    st.markdown(summary_text)


def render_results():
    st.subheader(f"Results ({len(works)})")
    if len(clusters) == 1:
        # Only one cluster exists - render flat
        for wid in clusters[0].work_ids:
            render_work_card(work_by_id[wid], selection_limit, len(selected_ids))
            if st.session_state.get(f"sel:{wid}"):
                selected_ids.append(wid)
    else:
        for cl in clusters:
            label = f"{cl.label}  ({len(cl.work_ids)})"
            if getattr(cl, "score", None) is not None:
                label += f" — {cl.score:.2f}"
            with st.expander(label, expanded=False):
                for wid in cl.work_ids:
                    render_work_card(work_by_id[wid], selection_limit, len(selected_ids))
                    if st.session_state.get(f"sel:{wid}"):
                        selected_ids.append(wid)


if show_mode == "Summary only":
    render_summary()
elif show_mode == "Both":
    render_summary()
    st.markdown("---")
    render_results()
else:  # Results only
    render_results()

st.markdown("---")
st.subheader("Chat with Selected Works")

selected_items = [work_by_id[wid] for wid in selected_ids if wid in work_by_id]

if not selected_ids:
    st.caption("No works selected yet.")
else:
    # Pretty list with links
    st.write(f"{len(selected_ids)} works selected (max {selection_limit} is possible):")
    for wid in selected_ids:
        w = work_by_id.get(wid)
        if not w:  # defensive
            continue
        st.markdown(f"- [{w.title}]({w.url})  _{w.venue or w.source}{', ' + str(w.year) if w.year else ''}_")

    # build index on demand or if selection changed
    sel_set = set(selected_ids)
    if st.session_state.rag_index is None or st.session_state.rag_for_ids != sel_set:
        with st.spinner("Fetching full texts and indexing selected items..."):
            idx = build_knowledge_index(selected_items, github_token=os.getenv("GITHUB_TOKEN"))
            st.session_state.rag_index = idx
            st.session_state.rag_for_ids = sel_set
        st.success(f"Indexed {len(st.session_state.rag_index.chunks)} chunks from {len(sel_set)} items.")

    # Controls (reuse LLM fields you already added in sidebar)
    question = st.text_input("Your question", value="Compare the methodology and evaluation protocols across these works.")
    st.caption("Ask detailed questions. The model sees full papers/repos/cards for the selected items.")
    run_llm_btn = st.button("Chat with your items!", type="primary",
                            disabled=(not question or st.session_state.rag_index is None or len(st.session_state.rag_index.chunks) == 0))
    if run_llm_btn and question.strip():
        llm_base_url = llm_base_url if 'llm_base_url' in locals() else None
        llm_model = llm_model if 'llm_model' in locals() else None
        with st.spinner("Thinking..."):
            answer, _ = answer_with_selected(
                question,
                st.session_state.rag_index,
                llm_base_url=llm_base_url or None,
                llm_model=llm_model or None,
                top_k=8,
            )
        st.markdown("### Answer")
        st.markdown(answer)
