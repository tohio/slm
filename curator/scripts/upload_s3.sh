#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# upload_s3.sh
# Upload curated, tokenized dataset to S3 for use on GPU training instance.
#
# Uploads the final memory-mapped dataset (.bin/.idx files) from the
# tokenization stage, plus the tokenizer model and configs.
#
# Usage:
#   bash upload_s3.sh --bucket my-slm-bucket [--prefix slm/data] [--dry-run]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
S3_BUCKET=""
S3_PREFIX="slm/data"
LOCAL_DATA_DIR="/data/curated/stages/tokenize"
TOKENIZER_DIR="/data/tokenizer"
CONFIG_DIR="$(dirname "$0")/../configs"
DRY_RUN=false
COMPRESS=true

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket)     S3_BUCKET="$2";     shift 2 ;;
        --prefix)     S3_PREFIX="$2";     shift 2 ;;
        --data-dir)   LOCAL_DATA_DIR="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;        shift ;;
        --no-compress) COMPRESS=false;    shift ;;
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
dry() { [[ "$DRY_RUN" == "true" ]] && echo "[DRY-RUN] $*" || true; }

AWS_FLAGS="--no-progress"
[[ "$DRY_RUN" == "true" ]] && AWS_FLAGS="$AWS_FLAGS --dryrun"

log "Upload configuration:"
log "  Source (tokenized data): $LOCAL_DATA_DIR"
log "  Source (tokenizer):      $TOKENIZER_DIR"
log "  Destination:             $S3_BASE"
log "  Dry run:                 $DRY_RUN"

# ── Upload tokenized dataset ───────────────────────────────────────────────────
log "Uploading tokenized dataset (.bin/.idx files)..."

BIN_COUNT=$(find "$LOCAL_DATA_DIR" -name "*.bin" 2>/dev/null | wc -l)
IDX_COUNT=$(find "$LOCAL_DATA_DIR" -name "*.idx" 2>/dev/null | wc -l)
log "  Found: $BIN_COUNT .bin files, $IDX_COUNT .idx files"

aws s3 sync \
    "$LOCAL_DATA_DIR" \
    "${S3_BASE}/tokenized/" \
    --exclude "*" \
    --include "*.bin" \
    --include "*.idx" \
    $AWS_FLAGS

log "Tokenized dataset upload complete"

# ── Upload tokenizer ───────────────────────────────────────────────────────────
log "Uploading tokenizer model..."

if [[ -d "$TOKENIZER_DIR" ]]; then
    aws s3 sync \
        "$TOKENIZER_DIR" \
        "${S3_BASE}/tokenizer/" \
        $AWS_FLAGS
    log "Tokenizer uploaded"
else
    log "WARNING: Tokenizer directory not found at $TOKENIZER_DIR — skipping"
fi

# ── Upload configs ─────────────────────────────────────────────────────────────
log "Uploading curator config..."

if [[ -d "$CONFIG_DIR" ]]; then
    aws s3 sync \
        "$CONFIG_DIR" \
        "${S3_BASE}/configs/curator/" \
        $AWS_FLAGS
    log "Config uploaded"
fi

# ── Generate manifest ─────────────────────────────────────────────────────────
MANIFEST_FILE="/tmp/dataset_manifest.json"

log "Generating dataset manifest..."
python3 - <<EOF
import json, os
from pathlib import Path

data_dir = Path("$LOCAL_DATA_DIR")
files = []
for f in sorted(data_dir.glob("*.bin")):
    idx_file = f.with_suffix(".idx")
    files.append({
        "bin": str(f.name),
        "idx": str(idx_file.name) if idx_file.exists() else None,
        "size_bytes": f.stat().st_size,
    })

manifest = {
    "dataset": "slm_pretrain",
    "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "s3_base": "$S3_BASE/tokenized",
    "files": files,
    "total_files": len(files),
}

with open("$MANIFEST_FILE", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest: {len(files)} dataset files")
EOF

aws s3 cp "$MANIFEST_FILE" "${S3_BASE}/manifest.json" $AWS_FLAGS
log "Manifest uploaded to ${S3_BASE}/manifest.json"

# ── Summary ────────────────────────────────────────────────────────────────────
log ""
log "Upload complete!"
log "  S3 location:  $S3_BASE"
log "  Log:          $LOG_FILE"
log ""
log "On your GPU instance, sync with:"
log "  aws s3 sync ${S3_BASE}/tokenized/ /data/pretrain/"
log "  aws s3 sync ${S3_BASE}/tokenizer/ /data/tokenizer/"
