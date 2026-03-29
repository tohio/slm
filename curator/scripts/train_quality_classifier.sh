#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# curator/scripts/train_quality_classifier.sh
#
# Trains a fastText binary classifier to distinguish high-quality (hq)
# from low-quality (lq) documents using weak supervision from heuristic
# filter scores.
#
# Strategy:
#   Documents that passed the heuristic filter are not equally good.
#   We use their heuristic scores as a weak supervision signal:
#     - Docs that passed comfortably (top quartile of scores) → __label__hq
#     - Docs that barely passed (bottom quartile of scores)   → __label__lq
#   This gives us ~free labels without manual annotation.
#
# Output:
#   /data/models/quality_classifier.bin   ← used by pipeline quality_filter stage
#
# Usage:
#   bash train_quality_classifier.sh
#   bash train_quality_classifier.sh --input-dir /data/curated/stages/pii \
#                                    --output-model /data/models/quality_classifier.bin \
#                                    --n-samples 50000
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT_DIR="/data/curated/stages/pii"
HEURISTIC_DIR="/data/curated/stages/heuristic_filter"
OUTPUT_MODEL="/data/models/quality_classifier.bin"
N_SAMPLES=50000
VAL_SPLIT=0.1
MIN_QUALITY_SCORE=0.3   # must match curator.yaml quality_filter.min_quality_score

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)     INPUT_DIR="$2";     shift 2 ;;
        --heuristic-dir) HEURISTIC_DIR="$2"; shift 2 ;;
        --output-model)  OUTPUT_MODEL="$2";  shift 2 ;;
        --n-samples)     N_SAMPLES="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$OUTPUT_MODEL")"

LOG_FILE="/tmp/train_quality_classifier_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== Quality Classifier Training ==="
log "  Input (PII stage):       $INPUT_DIR"
log "  Input (heuristic stage): $HEURISTIC_DIR"
log "  Output model:            $OUTPUT_MODEL"
log "  N samples:               $N_SAMPLES"

# ── Validate inputs ───────────────────────────────────────────────────────────
JSONL_COUNT=$(find "$INPUT_DIR" -name "*.jsonl" 2>/dev/null | wc -l)
if [[ "$JSONL_COUNT" -eq 0 ]]; then
    echo "ERROR: No JSONL files found in $INPUT_DIR"
    echo "  Run 'make docker-curate' (pass 1) first."
    exit 1
fi
log "Found $JSONL_COUNT JSONL files in $INPUT_DIR"

# ── Train classifier in Python ────────────────────────────────────────────────
log "Preparing training data and training classifier..."

python3 - <<PYEOF
import json
import random
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("quality_classifier")

INPUT_DIR     = Path("$INPUT_DIR")
HEURISTIC_DIR = Path("$HEURISTIC_DIR")
OUTPUT_MODEL  = Path("$OUTPUT_MODEL")
N_SAMPLES     = int("$N_SAMPLES")
VAL_SPLIT     = float("$VAL_SPLIT")

random.seed(42)

# ── Load documents ─────────────────────────────────────────────────────────────
# Try to use heuristic_filter output which contains quality scores.
# Fall back to PII output if heuristic dir not available.
source_dir = HEURISTIC_DIR if HEURISTIC_DIR.exists() and any(HEURISTIC_DIR.glob("*.jsonl")) else INPUT_DIR
logger.info(f"Loading documents from {source_dir}")

all_docs = []
for f in sorted(source_dir.glob("*.jsonl")):
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_docs.append(json.loads(line))

logger.info(f"Loaded {len(all_docs):,} documents")

if len(all_docs) < 1000:
    logger.error("Too few documents to train a classifier. Need at least 1000.")
    sys.exit(1)

random.shuffle(all_docs)
sample = all_docs[:N_SAMPLES]

# ── Compute quality proxy score ────────────────────────────────────────────────
# Use word count and character count as a simple quality proxy if
# heuristic scores are not stored in the doc metadata.
# Docs with more content and better structure tend to be higher quality.
def quality_proxy(doc: dict) -> float:
    text = doc.get("text", "")
    words = text.split()
    n_words = len(words)
    n_chars = len(text)

    if n_words == 0:
        return 0.0

    # Signals of quality
    avg_word_len   = n_chars / n_words
    has_punct      = sum(1 for c in text if c in ".!?,;:") / max(n_words, 1)
    alpha_ratio    = sum(c.isalpha() for c in text) / max(n_chars, 1)
    newline_ratio  = text.count("\n") / max(n_words, 1)

    # Length sweet spot: 100-1000 words is ideal
    length_score = min(n_words / 100, 1.0) if n_words < 100 else min(1000 / n_words, 1.0) if n_words > 1000 else 1.0

    score = (
        0.35 * length_score +
        0.25 * min(alpha_ratio, 1.0) +
        0.20 * min(has_punct * 10, 1.0) +
        0.10 * min(avg_word_len / 6, 1.0) +
        0.10 * (1.0 - min(newline_ratio * 5, 1.0))
    )

    # Override with stored language score if available
    if "language_score" in doc:
        score = score * 0.7 + float(doc["language_score"]) * 0.3

    return round(score, 4)

logger.info("Computing quality proxy scores...")
scored = [(quality_proxy(doc), doc) for doc in sample]
scored.sort(key=lambda x: x[0], reverse=True)

# ── Label using quartiles ──────────────────────────────────────────────────────
# Top 25% → __label__hq, Bottom 25% → __label__lq
# Middle 50% excluded — ambiguous signal
n = len(scored)
top_quartile    = scored[:n // 4]
bottom_quartile = scored[-(n // 4):]

labeled = (
    [("__label__hq", doc) for _, doc in top_quartile] +
    [("__label__lq", doc) for _, doc in bottom_quartile]
)

random.shuffle(labeled)
n_val   = max(1, int(len(labeled) * VAL_SPLIT))
val     = labeled[:n_val]
train   = labeled[n_val:]

logger.info(f"Labeled: {len(train):,} train, {n_val:,} val")
logger.info(f"  HQ (train): {sum(1 for l,_ in train if l=='__label__hq'):,}")
logger.info(f"  LQ (train): {sum(1 for l,_ in train if l=='__label__lq'):,}")

# ── Write fastText training format ─────────────────────────────────────────────
# fastText expects: __label__X <text>
def to_fasttext_line(label: str, doc: dict) -> str:
    text = doc.get("text", "").replace("\n", " ").strip()[:2000]
    # Remove fastText special chars
    text = text.replace("__label__", "")
    return f"{label} {text}"

train_file = Path("/tmp/quality_train.txt")
val_file   = Path("/tmp/quality_val.txt")

with open(train_file, "w", encoding="utf-8") as f:
    for label, doc in train:
        f.write(to_fasttext_line(label, doc) + "\n")

with open(val_file, "w", encoding="utf-8") as f:
    for label, doc in val:
        f.write(to_fasttext_line(label, doc) + "\n")

logger.info(f"Training data written to {train_file}")

# ── Train fastText classifier ──────────────────────────────────────────────────
import fasttext

logger.info("Training fastText classifier...")
model = fasttext.train_supervised(
    input=str(train_file),
    epoch=10,
    lr=0.5,
    wordNgrams=2,
    dim=100,
    loss="softmax",
    thread=32,
    verbose=2,
)

# ── Evaluate on validation set ─────────────────────────────────────────────────
logger.info("Evaluating on validation set...")
result = model.test(str(val_file))
n_samples, precision, recall = result
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

logger.info(f"  Validation samples: {n_samples:,}")
logger.info(f"  Precision:          {precision:.3f}")
logger.info(f"  Recall:             {recall:.3f}")
logger.info(f"  F1:                 {f1:.3f}")

if f1 < 0.6:
    logger.warning(
        f"F1={f1:.3f} is low — classifier may not add much signal over heuristics. "
        f"Consider increasing n-samples or reviewing labeling strategy."
    )

# ── Save model ─────────────────────────────────────────────────────────────────
OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(OUTPUT_MODEL))
logger.info(f"Model saved to {OUTPUT_MODEL}")
logger.info(f"  Size: {OUTPUT_MODEL.stat().st_size / 1024 / 1024:.1f}MB")

# Quick smoke test
test_hq = "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability."
test_lq = "buy now best price click here limited offer!!!! cheap discount sale"
for text, expected in [(test_hq, "hq"), (test_lq, "lq")]:
    labels, probs = model.predict(text)
    label = labels[0].replace("__label__", "")
    prob  = probs[0]
    status = "✓" if label == expected else "✗"
    logger.info(f"  {status} Smoke test ({expected}): predicted={label} ({prob:.3f})")

logger.info("Quality classifier training complete.")
PYEOF

log "Quality classifier trained successfully"
log "  Model: $OUTPUT_MODEL"
log "  Size:  $(du -sh "$OUTPUT_MODEL" | cut -f1)"
log ""
log "Next steps:"
log "  make docker-curate    # pass 2 — quality_filter and tokenize now run automatically"