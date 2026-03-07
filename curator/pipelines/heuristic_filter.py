"""
Stage 3: Heuristic Filter
--------------------------
Applies rule-based quality filters inspired by the Gopher and CCNet papers.
Fast, interpretable, and effective at removing the majority of low-quality web text
before the slower quality classifier runs.

Filters:
  - Document length (chars, words)
  - Symbol-to-word ratio (spam/SEO signals)
  - Bullet point ratio (list-heavy SEO pages)
  - Ellipsis ratio
  - Non-alpha word ratio
  - N-gram repetition detection (duplicate spans)
  - Stop word presence (ensures natural language)

Input/Output: JSONL
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger("curator.heuristic_filter")

# Precompile regexes
RE_WORD = re.compile(r"\b\w+\b")
RE_ALPHA_WORD = re.compile(r"\b[a-zA-Z]+\b")
RE_SYMBOLS = re.compile(r"[#\$%\^&\*@~`<>{}|\[\]\\]")
RE_BULLET_LINE = re.compile(r"^\s*[-•·▪▸►*]\s+", re.MULTILINE)
RE_ELLIPSIS = re.compile(r"\.{3,}|…")
RE_LINE = re.compile(r"\n")


class HeuristicFilter:
    """
    Stateless filter — instantiate once, call check() per document.
    Returns (passed: bool, reason: str) for each document.
    """

    def __init__(self, cfg: dict):
        self.min_doc_length = cfg.get("min_doc_length", 200)
        self.max_doc_length = cfg.get("max_doc_length", 100000)
        self.min_word_count = cfg.get("min_word_count", 50)
        self.max_word_count = cfg.get("max_word_count", 100000)
        self.max_symbol_ratio = cfg.get("max_symbol_to_word_ratio", 0.1)
        self.max_bullet_ratio = cfg.get("max_bullet_start_ratio", 0.9)
        self.max_ellipsis_ratio = cfg.get("max_ellipsis_ratio", 0.1)
        self.max_non_alpha_ratio = cfg.get("max_non_alpha_words_ratio", 0.8)
        self.min_stop_words = cfg.get("min_stop_word_count", 2)
        self.stop_words = set(cfg.get("stop_words", ["the", "be", "to", "of", "and"]))

        # N-gram repetition config
        ngram_cfg = cfg.get("max_duplicated_ngram_char_frac", {})
        self.ngram_n = ngram_cfg.get("n", 5)
        self.ngram_max_frac = ngram_cfg.get("fraction", 0.15)

        top_ngram_cfg = cfg.get("max_top_ngram_char_frac", {})
        self.top_ngram_n = top_ngram_cfg.get("n", 2)
        self.top_ngram_max_frac = top_ngram_cfg.get("fraction", 0.20)

    def check(self, text: str) -> tuple[bool, str]:
        """
        Run all heuristic checks on a document.
        Returns (True, "") if passed, (False, reason) if filtered.
        Short-circuits on first failure for speed.
        """

        # --- Length checks ---
        char_count = len(text)
        if char_count < self.min_doc_length:
            return False, f"too_short ({char_count} chars)"
        if char_count > self.max_doc_length:
            return False, f"too_long ({char_count} chars)"

        words = RE_WORD.findall(text)
        word_count = len(words)
        if word_count < self.min_word_count:
            return False, f"too_few_words ({word_count})"
        if word_count > self.max_word_count:
            return False, f"too_many_words ({word_count})"

        # --- Symbol ratio ---
        symbol_count = len(RE_SYMBOLS.findall(text))
        symbol_ratio = symbol_count / word_count if word_count > 0 else 1.0
        if symbol_ratio > self.max_symbol_ratio:
            return False, f"high_symbol_ratio ({symbol_ratio:.3f})"

        # --- Bullet ratio ---
        lines = text.split("\n")
        if lines:
            bullet_lines = sum(1 for l in lines if RE_BULLET_LINE.match(l))
            bullet_ratio = bullet_lines / len(lines)
            if bullet_ratio > self.max_bullet_ratio:
                return False, f"high_bullet_ratio ({bullet_ratio:.3f})"

        # --- Ellipsis ratio ---
        ellipsis_count = len(RE_ELLIPSIS.findall(text))
        ellipsis_ratio = ellipsis_count / word_count if word_count > 0 else 1.0
        if ellipsis_ratio > self.max_ellipsis_ratio:
            return False, f"high_ellipsis_ratio ({ellipsis_ratio:.3f})"

        # --- Non-alpha word ratio ---
        alpha_words = RE_ALPHA_WORD.findall(text)
        non_alpha_ratio = 1.0 - (len(alpha_words) / word_count) if word_count > 0 else 1.0
        if non_alpha_ratio > self.max_non_alpha_ratio:
            return False, f"high_non_alpha_ratio ({non_alpha_ratio:.3f})"

        # --- Stop word check ---
        lower_words = [w.lower() for w in alpha_words]
        stop_word_count = sum(1 for w in lower_words if w in self.stop_words)
        if stop_word_count < self.min_stop_words:
            return False, f"insufficient_stop_words ({stop_word_count})"

        # --- N-gram repetition (Gopher method) ---
        # Detects documents with many repeated n-gram spans (low info content)
        if len(words) >= self.ngram_n:
            ngrams = [" ".join(words[i:i+self.ngram_n]) for i in range(len(words) - self.ngram_n + 1)]
            ngram_counts = Counter(ngrams)
            duplicated_char_count = sum(
                len(ng) * (count - 1)
                for ng, count in ngram_counts.items()
                if count > 1
            )
            dup_frac = duplicated_char_count / char_count if char_count > 0 else 0
            if dup_frac > self.ngram_max_frac:
                return False, f"high_ngram_repetition ({dup_frac:.3f})"

        # --- Top n-gram dominance ---
        if len(words) >= self.top_ngram_n:
            top_ngrams = [" ".join(words[i:i+self.top_ngram_n]) for i in range(len(words) - self.top_ngram_n + 1)]
            top_counts = Counter(top_ngrams)
            if top_counts:
                top_ng, top_count = top_counts.most_common(1)[0]
                top_frac = (len(top_ng) * top_count) / char_count if char_count > 0 else 0
                if top_frac > self.top_ngram_max_frac:
                    return False, f"dominant_top_ngram ({top_frac:.3f}: '{top_ng}')"

        return True, ""


def process_jsonl_file(
    input_file: Path,
    output_file: Path,
    filter_obj: HeuristicFilter,
) -> dict:
    """Process a single JSONL file through heuristic filters."""
    kept = 0
    total = 0
    filter_reasons = Counter()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            doc = json.loads(line)
            text = doc.get("text", "")

            passed, reason = filter_obj.check(text)

            if passed:
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
            else:
                filter_reasons[reason.split(" ")[0]] += 1

    return {
        "file": input_file.name,
        "total": total,
        "kept": kept,
        "filter_reasons": dict(filter_reasons),
    }


def run_heuristic_filter(input_path: Path, output_path: Path, cfg: dict):
    """Main heuristic filter entry point."""
    if not cfg.get("enabled", True):
        logger.info("Heuristic filter disabled — symlinking input to output")
        output_path.symlink_to(input_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    filter_obj = HeuristicFilter(cfg)

    input_files = sorted(input_path.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {input_path}")

    logger.info(f"Heuristic filtering {len(input_files)} files")

    stats_list = []
    aggregated_reasons = Counter()

    for input_file in input_files:
        output_file = output_path / input_file.name
        stats = process_jsonl_file(input_file, output_file, filter_obj)
        stats_list.append(stats)
        aggregated_reasons.update(stats["filter_reasons"])
        logger.debug(f"{stats['file']}: {stats['kept']}/{stats['total']} kept")

    total_in = sum(s["total"] for s in stats_list)
    total_out = sum(s["kept"] for s in stats_list)
    retention = (total_out / total_in * 100) if total_in > 0 else 0

    logger.info(f"Heuristic filter complete: {total_out}/{total_in} documents retained ({retention:.1f}%)")
    logger.info(f"Filter reasons (top 10): {aggregated_reasons.most_common(10)}")
