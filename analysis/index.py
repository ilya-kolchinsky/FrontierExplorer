from typing import List, Tuple, Dict
import numpy as np
import re
from core.types import WorkItem
from core.embeddings import embed_texts


def _split_markdown(text: str, max_chars: int = 1500) -> List[str]:
    # simple splitter that respects paragraphs and headings
    blocks = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for b in blocks:
        if len(cur) + len(b) + 2 <= max_chars:
            cur = (cur + "\n\n" + b) if cur else b
        else:
            if cur:
                chunks.append(cur.strip())
            if len(b) <= max_chars:
                cur = b
            else:
                # hard wrap long block
                for i in range(0, len(b), max_chars):
                    part = b[i:i + max_chars]
                    chunks.append(part.strip())
                cur = ""
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if c]


class RAGIndex:
    def __init__(self):
        self.chunks: List[str] = []
        self.meta: List[Dict] = []
        self.vecs: np.ndarray | None = None

    def build(self, works: List[WorkItem], full_texts: Dict[str, str]) -> None:
        self.chunks.clear()
        self.meta.clear()
        self.vecs = None
        for w in works:
            t = full_texts.get(w.id or "", "")
            if not t:
                continue
            for ch in _split_markdown(t):
                self.chunks.append(ch)
                self.meta.append({
                    "work_id": w.id, "title": w.title, "url": w.url, "source": w.source
                })
        if not self.chunks:
            self.vecs = np.zeros((0, 384), dtype="float32")
            return
        self.vecs = embed_texts(self.chunks)  # expect L2-normalized

    def search(self, query: str, top_k: int = 8, mmr_lambda: float = 0.5) -> List[Tuple[int, float]]:
        if self.vecs is None or len(self.chunks) == 0:
            return []
        qv = embed_texts([query])[0]
        sims = np.dot(self.vecs, qv)  # cosine if vecs are normalized
        # MMR: select diverse top_k
        selected, cand = [], set(range(len(self.chunks)))
        while len(selected) < min(top_k, len(self.chunks)) and cand:
            if not selected:
                i = int(np.argmax(sims))
                selected.append(i)
                cand.remove(i)
                continue
            # penalize redundancy
            cand_list = list(cand)
            cand_sims = sims[cand_list]
            div = np.max(np.dot(self.vecs[cand_list], self.vecs[selected].T), axis=1)
            mmr = mmr_lambda * cand_sims - (1 - mmr_lambda) * div
            i_local = int(np.argmax(mmr))
            i = cand_list[i_local]
            selected.append(i)
            cand.remove(i)
        return [(i, float(sims[i])) for i in selected]
