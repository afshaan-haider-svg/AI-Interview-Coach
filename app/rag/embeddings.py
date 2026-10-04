import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Avoid network overhead and timeout retries when running local cached model
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.config import settings

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Singleton getter to load SentenceTransformer model once on CPU and reuse across calls."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device="cpu",
        )
    return _embedding_model


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Generates normalized vector embeddings for a list of document texts using batch inference."""
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [vec.tolist() for vec in embeddings]


def embed_query(query: str) -> List[float]:
    """Generates a normalized vector embedding for a single search query."""
    if not query or not query.strip():
        return []

    model = get_embedding_model()
    embedding = model.encode(
        query.strip(),
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embedding.tolist()
