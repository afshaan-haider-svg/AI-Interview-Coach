"""Lightweight text chunker respecting paragraph, sentence, and newline boundaries."""

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """Represents a text chunk with its index, content, and character count."""

    chunk_index: int
    text: str
    character_count: int


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[TextChunk]:
    """Splits text into readable chunks respecting paragraph, sentence, and word boundaries

    while strictly maintaining chunk size and overlap constraints.
    """
    if not text or not text.strip():
        return []

    clean_text = text.strip()

    # If text is already within chunk_size, return single chunk
    if len(clean_text) <= chunk_size:
        return [
            TextChunk(
                chunk_index=0,
                text=clean_text,
                character_count=len(clean_text),
            )
        ]

    chunks: List[str] = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Look backwards from end down to halfway for a natural break
            search_start = start + chunk_size // 2
            for sep in ["\n\n", "\n", ". ", "? ", "! ", " "]:
                idx = clean_text.rfind(sep, search_start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        chunk = clean_text[start:end].strip()
        if chunk and (not chunks or chunk != chunks[-1]):
            chunks.append(chunk)

        if end >= text_len:
            break

        # Move start forward respecting overlap
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return [
        TextChunk(
            chunk_index=idx,
            text=chk,
            character_count=len(chk),
        )
        for idx, chk in enumerate(chunks)
        if chk.strip()
    ]
