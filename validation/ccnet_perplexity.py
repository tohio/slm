"""CCNet-compatible text preparation and perplexity scoring.

The pretrained ``*.arpa.bin`` language models published by CCNet were trained
on normalized SentencePiece output, not raw whitespace-delimited text.  The
matching ``*.sp.model`` must therefore be applied before KenLM scoring.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


_UNICODE_PUNCT = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "、": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "１": '"',
        "」": '"',
        "「": '"',
        "《": '"',
        "》": '"',
        "´": "'",
        "∶": ":",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "；": ";",
        "–": "-",
        "—": " - ",
        "．": ". ",
        "～": "~",
        "’": "'",
        "…": "...",
        "━": "-",
        "〈": "<",
        "〉": ">",
        "【": "[",
        "】": "]",
        "％": "%",
        "►": "-",
    }
)
_DIGIT_RE = re.compile(r"\d")
_NON_PRINTING_RE = re.compile(
    f"[{''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))}]"
)


def normalize_ccnet_text(text: str) -> str:
    """Reproduce the normalization used to train CCNet language models."""
    text = text.strip().lower()
    if not text:
        return text
    decomposed = unicodedata.normalize("NFD", text)
    text = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    text = _DIGIT_RE.sub("0", text)
    text = text.translate(_UNICODE_PUNCT)
    return _NON_PRINTING_RE.sub("", text)


class CCNetPerplexityScorer:
    """Score text with a matched CCNet SentencePiece and KenLM model pair."""

    def __init__(
        self,
        language_model_path: Path,
        sentencepiece_model_path: Path,
    ) -> None:
        try:
            import kenlm
        except ImportError as exc:
            raise RuntimeError(
                "kenlm is required when perplexity measurement is enabled"
            ) from exc
        try:
            import sentencepiece
        except ImportError as exc:
            raise RuntimeError(
                "sentencepiece is required for CCNet KenLM preprocessing"
            ) from exc

        self.language_model = kenlm.Model(str(language_model_path))
        self.sentencepiece_model = sentencepiece.SentencePieceProcessor()
        if not self.sentencepiece_model.load(str(sentencepiece_model_path)):
            raise RuntimeError(
                f"Could not load SentencePiece model: {sentencepiece_model_path}"
            )

    def perplexity(self, text: str) -> float:
        """Return perplexity after CCNet normalization and tokenization."""
        normalized = normalize_ccnet_text(text)
        pieces = self.sentencepiece_model.encode(normalized, out_type=str)
        if not pieces:
            raise ValueError("CCNet preprocessing produced no tokens")
        return self.language_model.perplexity(" ".join(pieces))
