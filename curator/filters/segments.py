"""Deterministic, non-overlapping segmentation for long prose records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_BOUNDARIES = ("\n", ". ", "! ", "? ", "; ", ", ", " ")


def _cut_position(text: str, limit: int, minimum: int) -> int:
    """Find the latest readable boundary in [minimum, limit]."""
    best = -1
    best_width = 0
    for boundary in _BOUNDARIES:
        position = text.rfind(boundary, minimum, limit + 1)
        if position > best:
            best = position
            best_width = len(boundary.rstrip())
    if best < minimum:
        return limit
    return best + best_width


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
