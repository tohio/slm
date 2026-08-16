"""Deterministic, non-overlapping segmentation for long prose records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
# Ordered from strongest semantic boundary to weakest fallback. Selecting the
# first available boundary preserves a complete paragraph or sentence even
# when a later comma or space exists closer to the hard size limit.
_BOUNDARIES = ("\n", ". ", "! ", "? ", "; ", ", ", " ")
_HEADING_MAX_CHARS = 120
_HEADING_MAX_WORDS = 16


def _cut_position(text: str, limit: int, minimum: int) -> int:
    """Find a boundary in [minimum, limit], preferring semantic strength."""
    for boundary in _BOUNDARIES:
        position = text.rfind(boundary, minimum, limit + 1)
        if position >= minimum:
            return position + len(boundary.rstrip())
    return limit


def _looks_like_standalone_heading(text: str) -> bool:
    """Return True for a short paragraph that should lead the next chunk."""
    stripped = text.strip()
    if not stripped or "\n" in stripped:
        return False
    words = stripped.split()
    return (
        len(stripped) <= _HEADING_MAX_CHARS
        and len(words) <= _HEADING_MAX_WORDS
        and any(character.isalpha() for character in stripped)
        and not stripped.endswith((".", "!", "?", ";", ","))
    )


def _move_orphaned_headings(
    chunks: list[str],
    *,
    max_chars: int,
    min_chars: int,
) -> list[str]:
    """Move a trailing section heading to the following chunk when safe."""
    for index in range(len(chunks) - 1):
        paragraphs = _PARAGRAPH_BREAK.split(chunks[index])
        if len(paragraphs) < 2:
            continue
        heading = paragraphs[-1].strip()
        if not _looks_like_standalone_heading(heading):
            continue
        left = "\n\n".join(paragraphs[:-1]).strip()
        right = f"{heading}\n\n{chunks[index + 1]}"
        if len(left) >= min_chars and len(right) <= max_chars:
            chunks[index] = left
            chunks[index + 1] = right
    return chunks


def _split_oversized_block(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = _cut_position(remaining, max_chars, max_chars // 2)
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_long_text(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    """Split text at paragraph/sentence boundaries without overlap."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if min_chars < 1 or min_chars > max_chars:
        raise ValueError("min_chars must be in [1, max_chars]")

    normalized = text.strip()
    if len(normalized) <= max_chars:
        return [normalized]

    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(normalized):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units.extend(_split_oversized_block(paragraph, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = unit
    if current:
        chunks.append(current)

    # A tiny final tail is part of the source document, not a low-quality
    # standalone record. Rebalance the last two chunks around their midpoint
    # so both can pass the existing minimum-length check.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        combined = f"{chunks[-2]}\n\n{chunks[-1]}"
        midpoint = (len(combined) + 1) // 2
        cut = _cut_position(combined, midpoint, min_chars)
        left = combined[:cut].strip()
        right = combined[cut:].strip()
        if (
            min_chars <= len(left) <= max_chars
            and min_chars <= len(right) <= max_chars
        ):
            chunks[-2:] = [left, right]

    chunks = _move_orphaned_headings(
        chunks,
        max_chars=max_chars,
        min_chars=min_chars,
    )

    if not chunks or any(not chunk or len(chunk) > max_chars for chunk in chunks):
        raise RuntimeError("Long-document segmentation violated its size contract")
    return chunks


def segment_long_document(
    record: dict,
    *,
    eligible_sources: Collection[str],
    max_chars: int,
    min_chars: int,
) -> list[dict]:
    """Return an unchanged record or deterministic long-prose segments."""
    source = str(record.get("source", ""))
    text = record.get("text", "")
    if not isinstance(text, str):
        raise RuntimeError("Record text must be a string before segmentation")
    if source not in eligible_sources or len(text) <= max_chars:
        return [record]
    if "long_document_segment" in record:
        raise RuntimeError("Raw record already contains long_document_segment")

    texts = split_long_text(text, max_chars=max_chars, min_chars=min_chars)
    parent_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    segments = []
    for index, segment_text in enumerate(texts):
        segment = dict(record)
        segment["text"] = segment_text
        segment["long_document_segment"] = {
            "parent_sha256": parent_sha256,
            "index": index,
            "count": len(texts),
            "original_characters": len(text),
        }
        segments.append(segment)
    return segments
