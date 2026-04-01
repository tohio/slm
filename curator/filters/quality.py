"""
curator/filters/quality.py
---------------------------
Quality heuristic filters for pretraining data.

Applies a set of rule-based quality signals to filter out low-quality
documents before tokenization. These heuristics are adapted from
FineWeb, RedPajama, and Dolma — the most widely used open LLM
pretraining pipelines.

Filters are composable — each returns True to keep, False to discard.
The QualityFilter class runs all filters and tracks rejection reasons.

Heuristics applied:
    - Minimum/maximum document length
    - Minimum mean word length (filters gibberish)
    - Maximum symbol-to-word ratio (filters SEO spam)
    - Maximum bullet point ratio (filters list-heavy content)
    - Maximum ellipsis ratio (filters truncated content)
    - Minimum alphabetic character ratio (filters numeric/code spam)
    - Repeated line deduplication within document
    - Stop word presence (filters non-English content that slipped through)

Reference:
    FineWeb: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
    Gopher: Rae et al. (2021) — https://arxiv.org/abs/2112.11446
"""

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# English stop words — presence indicates natural language
EN_STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have",
    "it", "for", "not", "on", "with", "he", "as", "you", "do",
    "at", "this", "but", "his", "by", "from", "they", "we",
    "say", "her", "she", "or", "an", "will", "my", "one", "all",
}


@dataclass
class QualityConfig:
    """Configuration for quality filter thresholds."""
    # Document length
    min_chars: int = 200
    max_chars: int = 100_000

    # Word-level signals
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0

    # Symbol ratio — symbols / words
    max_symbol_to_word_ratio: float = 0.1

    # Bullet point ratio — lines starting with bullet / total lines
    max_bullet_ratio: float = 0.9

    # Ellipsis ratio — lines ending with ... / total lines
    max_ellipsis_ratio: float = 0.3

    # Alphabetic character ratio — alpha chars / total chars
    min_alpha_ratio: float = 0.7

    # Repeated lines — fraction of lines that are duplicates
    max_repeated_line_ratio: float = 0.3

    # Stop word check — minimum number of EN stop words in first 100 words
    min_stop_words: int = 2

    # Sources that skip certain filters
    skip_alpha_ratio_sources: list[str] = field(default_factory=lambda: ["code"])
    skip_stop_word_sources: list[str] = field(default_factory=lambda: ["code"])


class QualityFilter:
    """
    Applies heuristic quality filters to a document.

    Args:
        config: QualityConfig with filter thresholds.

    Example::

        filter = QualityFilter()
        record = {"text": "...", "source": "common_crawl"}
        kept, reason = filter.check(record)
        if not kept:
            print(f"Rejected: {reason}")
    """

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()
        self.stats = {
            "total": 0,
            "kept": 0,
            "rejected": {},
        }

    def check(self, record: dict) -> tuple[bool, str | None]:
        """
        Run all quality filters on a document.

        Args:
            record: Dict with at least a "text" key and optional "source".

        Returns:
            (True, None) if the document passes all filters.
            (False, reason) if the document is rejected.
        """
        self.stats["total"] += 1
        text = record.get("text", "")
        source = record.get("source", "")

        checks = [
            self._check_length,
            self._check_mean_word_length,
            self._check_symbol_ratio,
            self._check_bullet_ratio,
            self._check_ellipsis_ratio,
            self._check_repeated_lines,
        ]

        # Source-conditional checks
        if source not in self.config.skip_alpha_ratio_sources:
            checks.append(self._check_alpha_ratio)
        if source not in self.config.skip_stop_word_sources:
            checks.append(self._check_stop_words)

        for check in checks:
            passed, reason = check(text)
            if not passed:
                self.stats["rejected"][reason] = self.stats["rejected"].get(reason, 0) + 1
                return False, reason

        self.stats["kept"] += 1
        return True, None

    def filter_batch(self, records: list[dict]) -> list[dict]:
        """Filter a list of records, returning only those that pass."""
        return [r for r in records if self.check(r)[0]]

    def reset_stats(self) -> None:
        """Reset filter statistics."""
        self.stats = {"total": 0, "kept": 0, "rejected": {}}

    def report(self) -> str:
        """Return a human-readable filter report."""
        total = self.stats["total"]
        kept = self.stats["kept"]
        rejected = total - kept
        lines = [
            f"Quality filter report:",
            f"  Total:    {total:>10,}",
            f"  Kept:     {kept:>10,}  ({100 * kept / max(total, 1):.1f}%)",
            f"  Rejected: {rejected:>10,}  ({100 * rejected / max(total, 1):.1f}%)",
            f"  Rejection reasons:",
        ]
        for reason, count in sorted(self.stats["rejected"].items(), key=lambda x: -x[1]):
            lines.append(f"    {reason:<40} {count:>8,}  ({100 * count / max(total, 1):.1f}%)")
        return "\n".join(lines)

    # ── Individual filter methods ──────────────────────────────────────────────

    def _check_length(self, text: str) -> tuple[bool, str | None]:
        n = len(text)
        if n < self.config.min_chars:
            return False, "too_short"
        if n > self.config.max_chars:
            return False, "too_long"
        return True, None

    def _check_mean_word_length(self, text: str) -> tuple[bool, str | None]:
        words = text.split()
        if not words:
            return False, "no_words"
        mean_len = sum(len(w) for w in words) / len(words)
        if mean_len < self.config.min_mean_word_length:
            return False, "mean_word_too_short"
        if mean_len > self.config.max_mean_word_length:
            return False, "mean_word_too_long"
        return True, None

    def _check_symbol_ratio(self, text: str) -> tuple[bool, str | None]:
        words = text.split()
        if not words:
            return False, "no_words"
        symbols = sum(1 for c in text if c in "#$%&*+/<=>@\\^_`|~")
        ratio = symbols / len(words)
        if ratio > self.config.max_symbol_to_word_ratio:
            return False, "high_symbol_ratio"
        return True, None

    def _check_bullet_ratio(self, text: str) -> tuple[bool, str | None]:
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return False, "no_lines"
        bullet_lines = sum(
            1 for l in lines
            if l.strip().startswith(("•", "-", "*", "·", "–", "—", "▪", "◦"))
        )
        ratio = bullet_lines / len(lines)
        if ratio > self.config.max_bullet_ratio:
            return False, "high_bullet_ratio"
        return True, None

    def _check_ellipsis_ratio(self, text: str) -> tuple[bool, str | None]:
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return False, "no_lines"
        ellipsis_lines = sum(1 for l in lines if l.rstrip().endswith("..."))
        ratio = ellipsis_lines / len(lines)
        if ratio > self.config.max_ellipsis_ratio:
            return False, "high_ellipsis_ratio"
        return True, None

    def _check_alpha_ratio(self, text: str) -> tuple[bool, str | None]:
        if not text:
            return False, "empty"
        alpha = sum(1 for c in text if c.isalpha())
        ratio = alpha / len(text)
        if ratio < self.config.min_alpha_ratio:
            return False, "low_alpha_ratio"
        return True, None

    def _check_repeated_lines(self, text: str) -> tuple[bool, str | None]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 4:
            return True, None
        seen = set()
        duplicates = 0
        for line in lines:
            if line in seen:
                duplicates += 1
            seen.add(line)
        ratio = duplicates / len(lines)
        if ratio > self.config.max_repeated_line_ratio:
            return False, "high_repeated_lines"
        return True, None

    def _check_stop_words(self, text: str) -> tuple[bool, str | None]:
        words = set(text.lower().split()[:100])
        count = len(words & EN_STOP_WORDS)
        if count < self.config.min_stop_words:
            return False, "insufficient_stop_words"
        return True, None