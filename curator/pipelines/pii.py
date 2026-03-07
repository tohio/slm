"""
Stage 7: PII Redaction
-----------------------
Detects and redacts Personally Identifiable Information (PII)
from document text before tokenization and training.

Uses regex-based detection for common PII patterns:
  - Email addresses
  - Phone numbers (US format + international)
  - IP addresses (v4 and v6)
  - URLs (optional — config-controlled)

Why regex over NER models here:
  - NER-based PII detection (e.g. spaCy, Presidio) is more accurate
    but much slower. At our scale, regex handles the high-confidence
    cases cheaply, and the training objective (next token prediction)
    is robust to occasional missed PII in pre-training data.
  - For SFT/DPO datasets (smaller, higher quality bar), upgrade to
    Presidio for more thorough detection.

Input/Output: JSONL
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("curator.pii")

REPLACEMENT = "<PII>"

# ─────────────────────────────────────────────
# Regex Patterns
# ─────────────────────────────────────────────

PATTERNS = {
    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"""
        (?:
            \+?1[\s.\-]?          # optional US country code
        )?
        (?:\(?\d{3}\)?[\s.\-]?)   # area code
        \d{3}[\s.\-]?             # prefix
        \d{4}                     # line number
        """,
        re.VERBOSE,
    ),
    "IP_ADDRESS": re.compile(
        r"""
        \b
        (?:
            # IPv4
            (?:\d{1,3}\.){3}\d{1,3}
            |
            # IPv6 (simplified)
            (?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}
        )
        \b
        """,
        re.VERBOSE,
    ),
    "URL": re.compile(
        r"https?://[^\s<>\"'\])]+"
    ),
}


def redact_text(text: str, entities: list[str], replacement: str = REPLACEMENT) -> tuple[str, dict]:
    """
    Redact PII from text. Returns (redacted_text, counts_by_entity_type).
    """
    counts = {}
    for entity in entities:
        pattern = PATTERNS.get(entity)
        if pattern is None:
            logger.warning(f"Unknown entity type: {entity}. Skipping.")
            continue
        matches = pattern.findall(text)
        counts[entity] = len(matches)
        text = pattern.sub(replacement, text)

    return text, counts


def process_jsonl_file(
    input_file: Path,
    output_file: Path,
    entities: list[str],
    replacement: str,
) -> dict:
    """Process a single JSONL file, redacting PII from all documents."""
    total = 0
    total_redacted_counts = {e: 0 for e in entities}

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

            redacted_text, counts = redact_text(text, entities, replacement)
            doc["text"] = redacted_text

            # Track if any PII was found in this doc
            pii_found = sum(counts.values()) > 0
            if pii_found:
                doc["pii_redacted"] = True

            for entity, count in counts.items():
                total_redacted_counts[entity] += count

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return {
        "file": input_file.name,
        "total": total,
        "redacted_counts": total_redacted_counts,
    }


def run_pii_redaction(input_path: Path, output_path: Path, cfg: dict):
    """Main PII redaction entry point."""
    if not cfg.get("enabled", True):
        logger.info("PII redaction disabled — symlinking input to output")
        output_path.symlink_to(input_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)

    entities = cfg.get("entities", ["EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"])
    replacement = cfg.get("replacement_string", REPLACEMENT)

    input_files = sorted(input_path.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No JSONL files in {input_path}")

    logger.info(f"PII redaction: {len(input_files)} files, entities={entities}")

    stats_list = []
    for input_file in input_files:
        output_file = output_path / input_file.name
        stats = process_jsonl_file(input_file, output_file, entities, replacement)
        stats_list.append(stats)

    # Aggregate counts
    total_docs = sum(s["total"] for s in stats_list)
    aggregated = {e: 0 for e in entities}
    for s in stats_list:
        for e, c in s["redacted_counts"].items():
            aggregated[e] = aggregated.get(e, 0) + c

    logger.info(f"PII redaction complete: {total_docs:,} documents processed")
    for entity, count in aggregated.items():
        logger.info(f"  {entity}: {count:,} instances redacted")
