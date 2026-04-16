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
    - Boilerplate detection (filters cookie notices, privacy policies, etc.)
    - Language detection via fasttext (filters non-English documents)

Source-conditional filter skips:
    code source skips: symbol_ratio, mean_word_length, alpha_ratio,
                       stop_words, boilerplate, language
    These filters are designed for natural language and incorrectly reject
    valid code — symbol-heavy syntax, long identifiers, and lack of stop
    words are all normal properties of code, not quality signals.

FastText language detection:
    Requires the fasttext lid.176.ftz model — download once via:
        make download-fasttext-model
    Model path defaults to DATA_DIR/models/lid.176.ftz.
    If the model is not found, language detection is skipped with a warning
    and the stop word fallback is used instead.

Reference:
    FineWeb: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
    Gopher: Rae et al. (2021) — https://arxiv.org/abs/2112.11446
    CC-Net: Wenzek et al. (2019) — https://arxiv.org/abs/1911.00359
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# English stop words — presence indicates natural language.
# Used as fallback when fasttext model is not available.
EN_STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have",
    "it", "for", "not", "on", "with", "he", "as", "you", "do",
    "at", "this", "but", "his", "by", "from", "they", "we",
    "say", "her", "she", "or", "an", "will", "my", "one", "all",
}

# Boilerplate patterns — common web page fragments with zero training value.
# Matched case-insensitively against the full document text.
BOILERPLATE_PATTERNS = [
    r"cookie policy",
    r"privacy policy",
    r"terms of service",
    r"terms and conditions",
    r"all rights reserved",
    r"subscribe to our newsletter",
    r"javascript is (?:disabled|required)",
    r"please enable javascript",
    r"click here to (?:accept|enable|continue)",
    r"we use cookies",
    r"your (?:privacy|cookie) (?:settings|preferences|choices)",
]
_BOILERPLATE_RE = re.compile(
    "|".join(BOILERPLATE_PATTERNS),
    re.IGNORECASE,
)

# FastText model path — resolved at import time from DATA_DIR env var.
# The model is loaded lazily on first use and cached as a module-level
# singleton so subprocess workers each load it once, not per document.
_FASTTEXT_MODEL_PATH = Path(
    os.environ.get("DATA_DIR", "data")
) / "models" / "lid.176.ftz"

_fasttext_model = None        # module-level cache
_fasttext_warned = False      # emit missing-model warning once per process


def _get_fasttext_model():
    """
    Load and cache the fasttext language identification model.

    Returns the model if available, None otherwise. Emits a warning
    once per process if the model file is not found.
    """
    global _fasttext_model, _fasttext_warned
    if _fasttext_model is not None:
        return _fasttext_model

    if not _FASTTEXT_MODEL_PATH.exists():
        if not _fasttext_warned:
            log.warning(
                f"FastText model not found at {_FASTTEXT_MODEL_PATH}. "
                f"Language detection disabled — run 'make download-fasttext-model'. "
                f"Stop word fallback will be used instead."
            )
            _fasttext_warned = True
        return None

    try:
        import fasttext
        # suppress fasttext's noisy stderr output
        fasttext.FastText.eprint = lambda *args, **kwargs: None
        _fasttext_model = fasttext.load_model(str(_FASTTEXT_MODEL_PATH))
        log.info(f"FastText language model loaded from {_FASTTEXT_MODEL_PATH}")
    except ImportError:
        if not _fasttext_warned:
            log.warning(
                "fasttext not installed — language detection disabled. "
                "Install with: pip install fasttext-wheel"
            )
            _fasttext_warned = True

    return _fasttext_model


@dataclass
class QualityConfig:
    """Configuration for quality filter thresholds."""

    # Document length
    min_chars: int = 500           # up from 200 — removes very short fragments
    max_chars: int = 50_000        # down from 100_000 — removes spam/boilerplate

    # Word-level signals
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0

    # Symbol ratio — symbols / words
    max_symbol_to_word_ratio: float = 0.08   # tightened from 0.1

    # Bullet point ratio — lines starting with bullet / total lines
    max_bullet_ratio: float = 0.9

    # Ellipsis ratio — lines ending with ... / total lines
    max_ellipsis_ratio: float = 0.3

    # Alphabetic character ratio — alpha chars / total chars
    min_alpha_ratio: float = 0.75            # tightened from 0.70

    # Repeated lines — fraction of lines that are duplicates
    max_repeated_line_ratio: float = 0.2     # tightened from 0.3

    # Stop word check — minimum number of EN stop words in first 100 words.
    # Used as fallback when fasttext model is unavailable.
    min_stop_words: int = 3                  # tightened from 2

    # Boilerplate — number of boilerplate pattern matches to trigger rejection
    max_boilerplate_matches: int = 2

    # Language detection — minimum fasttext confidence to accept as English
    min_language_score: float = 0.65

    # Sources that skip certain filters.
    # Code skips NL-specific filters because symbol-heavy syntax, long
    # identifiers, lack of stop words, and boilerplate patterns are all
    # normal properties of code, not quality signals.
    skip_symbol_ratio_sources: list[str] = field(
        default_factory=lambda: ["code"]
    )
    skip_mean_word_length_sources: list[str] = field(
        default_factory=lambda: ["code"]
    )
    skip_alpha_ratio_sources: list[str] = field(
        default_factory=lambda: ["code"]
    )
    skip_stop_word_sources: list[str] = field(
        default_factory=lambda: ["code"]
    )
    skip_boilerplate_sources: list[str] = field(
        default_factory=lambda: ["code"]
    )
    skip_language_sources: list[str] = field(
        default_factory=lambda: ["code"]
    )


class QualityFilter:
    """
    Applies heuristic quality filters to a document.

    Filters run in cheapest-first order so expensive checks (language
    detection) only run on documents that passed all heuristic checks.

    Args:
        config: QualityConfig with filter thresholds.

    Example::

        qf = QualityFilter()
        record = {"text": "...", "source": "common_crawl"}
        kept, reason = qf.check(record)
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

        Filters execute in cheapest-first order:
            1. Length check (O(1))
            2. Structural checks — bullet, ellipsis, repeated lines (O(lines))
            3. Word-level checks — mean length, symbol ratio, alpha ratio (O(words))
            4. Boilerplate regex (O(text))
            5. Stop words (O(100 words)) — NL fallback only
            6. Language detection via fasttext (O(text[:500])) — most expensive

        Args:
            record: Dict with at least a "text" key and optional "source".

        Returns:
            (True, None) if the document passes all filters.
            (False, reason) if the document is rejected.
        """
        self.stats["total"] += 1
        text = record.get("text", "")
        source = record.get("source", "")

        # ── Tier 1: cheap structural checks (all sources) ─────────────────────
        checks = [
            self._check_length,
            self._check_bullet_ratio,
            self._check_ellipsis_ratio,
            self._check_repeated_lines,
        ]

        # ── Tier 2: word-level checks (skipped for code) ──────────────────────
        if source not in self.config.skip_mean_word_length_sources:
            checks.append(self._check_mean_word_length)
        if source not in self.config.skip_symbol_ratio_sources:
            checks.append(self._check_symbol_ratio)
        if source not in self.config.skip_alpha_ratio_sources:
            checks.append(self._check_alpha_ratio)

        # ── Tier 3: content checks (skipped for code) ─────────────────────────
        if source not in self.config.skip_boilerplate_sources:
            checks.append(self._check_boilerplate)

        # ── Tier 4: language checks (skipped for code) ────────────────────────
        # Use fasttext if available, fall back to stop word check otherwise.
        if source not in self.config.skip_language_sources:
            model = _get_fasttext_model()
            if model is not None:
                checks.append(self._check_language_fasttext)
            else:
                checks.append(self._check_stop_words)

        for check in checks:
            passed, reason = check(text)
            if not passed:
                self.stats["rejected"][reason] = (
                    self.stats["rejected"].get(reason, 0) + 1
                )
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
        for reason, count in sorted(
            self.stats["rejected"].items(), key=lambda x: -x[1]
        ):
            lines.append(
                f"    {reason:<40} {count:>8,}  "
                f"({100 * count / max(total, 1):.1f}%)"
            )
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
        seen: set[str] = set()
        duplicates = 0
        for line in lines:
            if line in seen:
                duplicates += 1
            seen.add(line)
        ratio = duplicates / len(lines)
        if ratio > self.config.max_repeated_line_ratio:
            return False, "high_repeated_lines"
        return True, None

    def _check_boilerplate(self, text: str) -> tuple[bool, str | None]:
        """
        Reject documents containing multiple boilerplate patterns.

        Cookie notices, privacy policies, and similar web page fragments
        pass all heuristic checks but have zero training value. Two or
        more matches in a single document is a strong signal the document
        is navigation/legal boilerplate rather than content.
        """
        matches = len(_BOILERPLATE_RE.findall(text))
        if matches >= self.config.max_boilerplate_matches:
            return False, "boilerplate"
        return True, None

    def _check_language_fasttext(self, text: str) -> tuple[bool, str | None]:
        """
        Reject non-English documents using fasttext language identification.

        Scores the first 500 characters — sufficient for language ID and
        avoids processing full documents for the language check.
        fasttext returns predictions as [("__label__en", 0.98), ...].
        """
        model = _get_fasttext_model()
        if model is None:
            return True, None  # model not available — skip

        # fasttext expects single-line input
        sample = text[:500].replace("\n", " ")
        try:
            labels, scores = model.predict(sample, k=1)
            lang = labels[0].replace("__label__", "")
            score = float(scores[0])
            if lang != "en" or score < self.config.min_language_score:
                return False, "non_english"
        except Exception:
            pass  # on any prediction error, pass the document through

        return True, None

    def _check_stop_words(self, text: str) -> tuple[bool, str | None]:
        """
        Fallback language check using English stop word presence.

        Used when the fasttext model is not available. Less accurate than
        fasttext but catches obvious non-English documents.
        """
        words = set(text.lower().split()[:100])
        count = len(words & EN_STOP_WORDS)
        if count < self.config.min_stop_words:
            return False, "insufficient_stop_words"
        return True, None