from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
from typing import Iterable
from config import settings


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(settings.EMBED_MODEL)


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    model = _model()
    vecs = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")
