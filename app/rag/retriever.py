"""Semantic retrieval service with strict user isolation and document filtering."""

from typing import Any, Dict, List, Optional

from app.rag.embeddings import embed_query
from app.rag.vector_store import get_collection
from app.schemas.rag import RetrievedChunk


def _extract_chunks_from_results(results: Dict[str, Any]) -> List[RetrievedChunk]:
    """Helper to convert Chroma query results into RetrievedChunk objects."""
    if not results or not results["ids"] or not results["ids"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved: List[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, distances):
        # In cosine distance: similarity = 1 - distance
        similarity = 1.0 - float(dist) if dist is not None else 0.0
        source_type = str(meta.get("source_type", "unknown"))
        source_id = int(
            meta.get("resume_id" if source_type == "resume" else "job_description_id", 0)
        )
        chunk_index = int(meta.get("chunk_index", 0))

        retrieved.append(
            RetrievedChunk(
                text=doc,
                source_type=source_type,
                source_id=source_id,
                chunk_index=chunk_index,
                metadata=dict(meta),
                score=round(similarity, 4),
            )
        )
    return retrieved


def retrieve_context(
    query: str,
    user_id: int,
    resume_id: Optional[int] = None,
    job_description_id: Optional[int] = None,
    top_k: int = 4,
) -> List[RetrievedChunk]:
    """Performs semantic similarity search against local Chroma vector store

    with strict user isolation and optional document filtering.
    """
    if not query or not query.strip():
        return []

    query_vector = embed_query(query)
    if not query_vector:
        return []

    collection = get_collection()

    # Case 1: Both resume_id and job_description_id provided
    # Retrieve top chunks for each document independently so context covers both documents
    if resume_id is not None and job_description_id is not None:
        where_resume = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"resume_id": {"$eq": resume_id}},
            ]
        }
        where_jd = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"job_description_id": {"$eq": job_description_id}},
            ]
        }

        res_resume = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_resume,
            include=["documents", "metadatas", "distances"],
        )
        res_jd = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_jd,
            include=["documents", "metadatas", "distances"],
        )

        combined = _extract_chunks_from_results(res_resume) + _extract_chunks_from_results(res_jd)
        # Sort by similarity score descending and cap at top_k
        combined.sort(key=lambda c: c.score, reverse=True)
        return combined[:top_k]

    # Case 2: Only resume_id provided
    if resume_id is not None:
        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"resume_id": {"$eq": resume_id}},
            ]
        }
    # Case 3: Only job_description_id provided
    elif job_description_id is not None:
        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"job_description_id": {"$eq": job_description_id}},
            ]
        }
    # Case 4: No specific document filter; strictly scoped to user_id
    else:
        where_filter = {"user_id": {"$eq": user_id}}

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    return _extract_chunks_from_results(results)


def retrieve_chunks_by_sources(
    sources: List[Any],
    user_id: int,
) -> List[RetrievedChunk]:
    """Retrieves specific chunks identified by source coordinates, strictly verifying user ownership.

    If a source chunk does not exist or belongs to another user, it is safely excluded.
    """
    if not sources:
        return []

    collection = get_collection()
    retrieved: List[RetrievedChunk] = []

    for src in sources:
        if isinstance(src, dict):
            src_type = src.get("source_type")
            src_id = src.get("source_id")
            chunk_idx = src.get("chunk_index")
        else:
            src_type = getattr(src, "source_type", None)
            src_id = getattr(src, "source_id", None)
            chunk_idx = getattr(src, "chunk_index", None)

        if not src_type or src_id is None or chunk_idx is None:
            continue

        chunk_id = (
            f"resume_{src_id}_chunk_{chunk_idx}"
            if src_type == "resume"
            else f"job_{src_id}_chunk_{chunk_idx}"
        )

        try:
            res = collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if res and res["ids"] and len(res["ids"]) > 0:
                doc = res["documents"][0]
                meta = res["metadatas"][0] if res["metadatas"] else {}
                # Strictly verify user isolation
                if meta.get("user_id") == user_id:
                    retrieved.append(
                        RetrievedChunk(
                            text=doc,
                            source_type=src_type,
                            source_id=int(src_id),
                            chunk_index=int(chunk_idx),
                            metadata=dict(meta),
                            score=1.0,
                        )
                    )
        except Exception:
            continue

    return retrieved

