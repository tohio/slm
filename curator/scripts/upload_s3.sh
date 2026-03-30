#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# upload_s3.sh
# Upload curated dataset to S3 for use on GPU training instance.
#
# Uploads all artifacts produced by the curation pipeline:
#   - Curated JSONL files       (curated/stages/pii/)
#   - Tokenized mmap files      (curated/tokenized/*.bin, *.idx)
#   - Tokenizer model           (tokenizer/)
#   - Quality classifier model  (models/quality_classifier.bin)
#
# Usage:
#   bash upload_s3.sh --bucket my-slm-bucket [--prefix slm/data] [--dry-run]
#   bash upload_s3.sh --bucket my-slm-bucket --skip-bin --skip-classifier
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Load .env if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../../.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a && source "$ENV_FILE" && set +a
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
S3_BUCKET=""
S3_PREFIX="slm/data"
LOCAL_JSONL_DIR="/data/curated/stages/pii"
LOCAL_BIN_DIR="/data/curated/tokenized"
TOKENIZER_DIR="/data/tokenizer"
CLASSIFIER_MODEL="/data/models/quality_classifier.bin"
DRY_RUN=false
SKIP_JSONL=false
SKIP_BIN=false
SKIP_TOKENIZER=false
SKIP_CLASSIFIER=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket)           S3_BUCKET="$2";        shift 2 ;;
        --prefix)           S3_PREFIX="$2";        shift 2 ;;
        --jsonl-dir)        LOCAL_JSONL_DIR="$2";  shift 2 ;;
        --bin-dir)          LOCAL_BIN_DIR="$2";    shift 2 ;;
        --tokenizer-dir)    TOKENIZER_DIR="$2";    shift 2 ;;
        --classifier-model) CLASSIFIER_MODEL="$2"; shift 2 ;;
        --skip-jsonl)       SKIP_JSONL=true;       shift ;;
        --skip-bin)         SKIP_BIN=true;         shift ;;
        --skip-tokenizer)   SKIP_TOKENIZER=true;   shift ;;
        --skip-classifier)  SKIP_CLASSIFIER=true;  shift ;;
        --dry-run)          DRY_RUN=true;          shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$S3_BUCKET" ]]; then
    echo "ERROR: --bucket is required"
    echo "Usage: $0 --bucket my-slm-bucket [--prefix slm/data]"
    exit 1
fi

S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"
LOG_FILE="/tmp/upload_s3_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

AWS_FLAGS="--no-progress"
[[ "$DRY_RUN" == "true" ]] && AWS_FLAGS="$AWS_FLAGS --dryrun"

log "=== SLM S3 Upload ==="
log "  Destination: $S3_BASE"
log "  Dry run:     $DRY_RUN"
log ""
log "  Skip flags:"
log "    --skip-jsonl:       $SKIP_JSONL"
log "    --skip-bin:         $SKIP_BIN"
log "    --skip-tokenizer:   $SKIP_TOKENIZER"
log "    --skip-classifier:  $SKIP_CLASSIFIER"

# ── Upload curated JSONL ───────────────────────────────────────────────────────
if [[ "$SKIP_JSONL" == "true" ]]; then
    log "[SKIP] curated JSONL (--skip-jsonl)"
else
    JSONL_COUNT=$(find "$LOCAL_JSONL_DIR" -name "*.jsonl" 2>/dev/null | wc -l)
    if [[ "$JSONL_COUNT" -eq 0 ]]; then
        log "WARNING: No JSONL files found in $LOCAL_JSONL_DIR — skipping"
    else
        log "Uploading $JSONL_COUNT curated JSONL files..."
        aws s3 sync \
            "$LOCAL_JSONL_DIR" \
            "${S3_BASE}/curated/pii/" \
            --exclude "*" \
            --include "*.jsonl" \
            $AWS_FLAGS
        log "✓ JSONL upload complete"
    fi
fi

# ── Upload tokenized mmap files ────────────────────────────────────────────────
if [[ "$SKIP_BIN" == "true" ]]; then
    log "[SKIP] tokenized mmap files (--skip-bin)"
else
    BIN_COUNT=$(find "$LOCAL_BIN_DIR" -name "*.bin" 2>/dev/null | wc -l)
    IDX_COUNT=$(find "$LOCAL_BIN_DIR" -name "*.idx" 2>/dev/null | wc -l)
    if [[ "$BIN_COUNT" -eq 0 ]]; then
        log "WARNING: No .bin files found in $LOCAL_BIN_DIR — skipping"
        log "  Pass 2 (tokenize stage) may not have completed yet."
        log "  Run 'make curate-full' to completion before uploading mmap files."
    else
        log "Uploading $BIN_COUNT .bin and $IDX_COUNT .idx files..."
        aws s3 sync \
            "$LOCAL_BIN_DIR" \
            "${S3_BASE}/curated/tokenized/" \
            --exclude "*" \
            --include "*.bin" \
            --include "*.idx" \
            $AWS_FLAGS
        log "✓ Mmap files upload complete"
    fi
fi

# ── Upload tokenizer ───────────────────────────────────────────────────────────
if [[ "$SKIP_TOKENIZER" == "true" ]]; then
    log "[SKIP] tokenizer (--skip-tokenizer)"
else
    if [[ ! -d "$TOKENIZER_DIR" ]]; then
        log "WARNING: Tokenizer directory not found at $TOKENIZER_DIR — skipping"
    else
        log "Uploading tokenizer..."
        aws s3 sync \
            "$TOKENIZER_DIR" \
            "${S3_BASE}/tokenizer/" \
            $AWS_FLAGS
        log "✓ Tokenizer upload complete"
    fi
fi

# ── Upload quality classifier ──────────────────────────────────────────────────
if [[ "$SKIP_CLASSIFIER" == "true" ]]; then
    log "[SKIP] quality classifier (--skip-classifier)"
else
    if [[ ! -f "$CLASSIFIER_MODEL" ]]; then
        log "WARNING: Quality classifier not found at $CLASSIFIER_MODEL — skipping"
        log "  Run 'make train-quality-classifier' to generate it."
    else
        log "Uploading quality classifier..."
        aws s3 cp \
            "$CLASSIFIER_MODEL" \
            "${S3_BASE}/models/quality_classifier.bin" \
            $AWS_FLAGS
        log "✓ Quality classifier upload complete"
    fi
fi

# ── Summary ────────────────────────────────────────────────────────────────────
log ""
log "=== Upload complete ==="
log "  S3 location: $S3_BASE"
log "  Log:         $LOG_FILE"
log ""
log "On your GPU instance, run:"
log "  make setup-instance"