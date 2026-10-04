"""RAG (Retrieval-Augmented Generation) package for local vector retrieval."""

from app.rag.chunker import TextChunk, chunk_text

__all__ = ["TextChunk", "chunk_text"]
