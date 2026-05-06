"""
curator/filters/quality.py
---------------------------
Quality heuristic filters for pretraining data.

Applies rule-based quality signals to filter out low-quality documents
before tokenization. Adapted from FineWeb, RedPajama, and Dolma.

Filters are composable — each returns True to keep, False to discard.
The QualityFilter class runs all filters and tracks rejection reasons.

Source-conditional filter skips:
    Code-adjacent sources skip: symbol_ratio, mean_word_length, alpha_ratio,
                                stop_words, boilerplate, language.
    These filters are designed for natural language and incorrectly reject
    valid code.

    The set of code-adjacent sources is defined at module level as
    CODE_SOURCES. Any source tag in this set inherits the default code
    skip behavior. Adding a new code-adjacent source is a single-line
    change to that constant.

    Length filters (min_chars / max_chars) have separate per-source skip
    lists because a source may legitimately fail one bound without the
    other — pg19 books exceed max_chars, conala NL→code pairs fall
    below min_chars, and jupyter notebooks can fall on either side.

FastText language detection:
    Requires the fasttext lid.176.ftz model — download once via:
        make download-fasttext-model
    Model path defaults to DATA_DIR/models/lid.176.ftz.
    If the model is not found, language detection is skipped with a warning
    and the stop word fallback is used instead.

Reference:
    FineWeb: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
    Gopher: Rae et al. (2021) — https://arxiv.org/abs/2112.11446
"""

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Source tags that represent code or code-adjacent content. These sources
# bypass English-prose-oriented quality filters (symbol ratio, alpha ratio,
# stop words, etc.) which incorrectly reject valid code.
#
# Mixed-content sources (jupyter, conala) are included here because their
# prose components would pass these filters but their code components
# would fail them, and per-chunk filtering isn't feasible at the source
# level. Accepting the trade-off: some non-English content in these
# sources won't be language-filtered.
CODE_SOURCES: frozenset[str] = frozenset({
    "synthetic_arithmetic",
    "codesearchnet",
    "stack_smol",
    "stack_v1",
    "jupyter",
    "conala",
})

# English stop words — fallback when fasttext is unavailable.
EN_STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have",
    "it", "for", "not", "on", "with", "he", "as", "you", "do",
    "at", "this", "but", "his", "by", "from", "they", "we",
    "say", "her", "she", "or", "an", "will", "my", "one", "all",
}

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
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# FastText model path
_FASTTEXT_MODEL_PATH = Path(
    os.environ.get("DATA_DIR", "data")
) / "models" / "lid.176.ftz"

_fasttext_model = None
_fasttext_warned = False


def _get_fasttext_model():
    """Load and cache the fasttext language identification model."""
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

    # Document length (defaults). Per-source skip lists below let specific
    # sources opt out of the min or max bound independently — see
    # skip_min_length_sources / skip_max_length_sources.
    min_chars: int = 500
    max_chars: int = 50_000

    # Word-level signals
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0

    # Symbol ratio — symbols / words
    max_symbol_to_word_ratio: float = 0.08

    # Bullet / ellipsis / alpha ratios
    max_bullet_ratio: float = 0.9
    max_ellipsis_ratio: float = 0.3
    min_alpha_ratio: float = 0.75

    # Repeated lines (duplicate occurrences) / total lines
    max_repeated_line_ratio: float = 0.2

    # Stop word fallback threshold (counts DISTINCT stop words in first 100
    # words). Only used when fasttext model is unavailable.
    min_stop_words: int = 3

    # Boilerplate matches to trigger rejection
    max_boilerplate_matches: int = 2

    # Language detection threshold
    min_language_score: float = 0.65

    # Per-filter skip lists. Default to CODE_SOURCES for filters that reject
    # code. Kept as per-filter fields (rather than one shared set) so future
    # tuning can customize which filters skip which sources without schema
    # changes — e.g. to still run boilerplate on jupyter but skip language.
    skip_symbol_ratio_sources: frozenset[str] = field(
        default_factory=lambda: CODE_SOURCES
    )
    skip_mean_word_length_sources: frozenset[str] = field(
        default_factory=lambda: CODE_SOURCES
    )
    skip_alpha_ratio_sources: frozenset[str] = field(
        default_factory=lambda: CODE_SOURCES
    )
    skip_stop_word_sources: frozenset[str] = field(
        default_factory=lambda: CODE_SOURCES
    )
    skip_boilerplate_sources: frozenset[str] = field(
        default_factory=lambda: CODE_SOURCES
    )
    skip_language_sources: frozenset[str] = field(
        default_factory=lambda: CODE_SOURCES
    )

    # Length filter skip lists. Separate min/max so sources can opt out
    # of one bound without the other:
    #   pg19     — books run 200k–1M chars; skip max only (keep min).
    #   conala   — NL→code pairs run 50–500 chars; skip min only (keep max).
    #   jupyter  — wide range; skip both bounds.
    skip_min_length_sources: frozenset[str] = field(
        default_factory=lambda: frozenset({"conala", "jupyter"})
    )
    skip_max_length_sources: frozenset[str] = field(
        default_factory=lambda: frozenset({"pg19", "jupyter"})
    )


class QualityFilter:
    """
    Applies heuristic quality filters to a document.

    Filters run in cheapest-first order so expensive checks only run on
    documents that passed cheaper ones.
    """

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()
        self.stats = {
            "total": 0,
            "kept": 0,
            "rejected": {},
        }
        # Track fasttext prediction failures — previously these were silently
        # debug-logged, which hides a systemic issue at scale.
        self._fasttext_errors = 0

    def check(self, record: dict) -> tuple[bool, str | None]:
        """
        Run all quality filters on a document.

        Returns:
            (True, None) if the document passes all filters.
            (False, reason) if the document is rejected.
        """
        self.stats["total"] += 1
        text = record.get("text", "")
        source = record.get("source", "")

        # Length uses per-source skip lists, so call it separately from
        # the uniform-signature checks below.
        passed, reason = self._check_length(text, source)
        if not passed:
            self.stats["rejected"][reason] = (
                self.stats["rejected"].get(reason, 0) + 1
            )
            return False, reason

        # Tier 1: cheap structural checks (all sources)
        checks = [
            self._check_bullet_ratio,
            self._check_ellipsis_ratio,
            self._check_repeated_lines,
        ]

        # Tier 2: word-level (skipped for code sources)
        if source not in self.config.skip_mean_word_length_sources:
            checks.append(self._check_mean_word_length)
        if source not in self.config.skip_symbol_ratio_sources:
            checks.append(self._check_symbol_ratio)
        if source not in self.config.skip_alpha_ratio_sources:
            checks.append(self._check_alpha_ratio)

        # Tier 3: content (skipped for code sources)
        if source not in self.config.skip_boilerplate_sources:
            checks.append(self._check_boilerplate)

        # Tier 4: language (skipped for code sources)
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
        self.stats = {"total": 0, "kept": 0, "rejected": {}}
        self._fasttext_errors = 0

    def report(self) -> str:
        total = self.stats["total"]
        kept = self.stats["kept"]
        rejected = total - kept
        lines = [
            "Quality filter report:",
            f"  Total:    {total:>10,}",
            f"  Kept:     {kept:>10,}  ({100 * kept / max(total, 1):.1f}%)",
            f"  Rejected: {rejected:>10,}  ({100 * rejected / max(total, 1):.1f}%)",
            "  Rejection reasons:",
        ]
        for reason, count in sorted(
            self.stats["rejected"].items(), key=lambda x: -x[1]
        ):
            lines.append(
                f"    {reason:<40} {count:>8,}  "
                f"({100 * count / max(total, 1):.1f}%)"
            )
        if self._fasttext_errors > 0:
            lines.append(
                f"  FastText prediction errors: {self._fasttext_errors:,} "
                f"(documents passed through without language check)"
            )
        return "\n".join(lines)

    # ── Individual filter methods ──────────────────────────────────────────────

    def _check_length(self, text: str, source: str = "") -> tuple[bool, str | None]:
        """
        Length check with per-source skip lists.

        A source in skip_min_length_sources bypasses the min_chars check;
        a source in skip_max_length_sources bypasses the max_chars check.
        The two lists are independent because real sources fail them
        asymmetrically (conala: always short, pg19: always long).
        """
        n = len(text)
        if (
            source not in self.config.skip_min_length_sources
            and n < self.config.min_chars
        ):
            return False, "too_short"
        if (
            source not in self.config.skip_max_length_sources
            and n > self.config.max_chars
        ):
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
        counts = Counter(lines)
        duplicate_occurrences = sum(c - 1 for c in counts.values() if c > 1)
        ratio = duplicate_occurrences / len(lines)
        if ratio > self.config.max_repeated_line_ratio:
            return False, "high_repeated_lines"
        return True, None

    def _check_boilerplate(self, text: str) -> tuple[bool, str | None]:
        matches = len(_BOILERPLATE_RE.findall(text))
        if matches >= self.config.max_boilerplate_matches:
            return False, "boilerplate"
        return True, None

    def _check_language_fasttext(self, text: str) -> tuple[bool, str | None]:
        """Reject non-English documents using fasttext."""
        model = _get_fasttext_model()
        if model is None:
            return True, None

        sample = text[:500].replace("\n", " ")
        try:
            labels, scores = model.predict(sample, k=1)
            lang = labels[0].replace("__label__", "")
            score = float(scores[0])
            if lang != "en" or score < self.config.min_language_score:
                return False, "non_english"
        except Exception as e:
            # Count and sample-log prediction errors. Previously silently logged
            # to DEBUG, which hid systemic fasttext issues at scale.
            self._fasttext_errors += 1
            if self._fasttext_errors <= 5 or self._fasttext_errors % 10_000 == 0:
                log.warning(
                    f"fasttext prediction error #{self._fasttext_errors} "
                    f"(passing through): {e}"
                )

        return True, None

    def _check_stop_words(self, text: str) -> tuple[bool, str | None]:
        """
        Fallback check using distinct English stop word count in the first
        100 tokens. Less accurate than fasttext but catches obvious non-English.
        """
        words = set(text.lower().split()[:100])
        count = len(words & EN_STOP_WORDS)
        if count < self.config.min_stop_words:
            return False, "insufficient_stop_words"
        return True, None