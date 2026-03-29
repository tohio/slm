#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_cc.sh
# Download a subset of Common Crawl WARC files from S3.
#
# Common Crawl stores WARCs on S3 at:
#   s3://commoncrawl/crawl-data/<snapshot>/segments/<segment>/warc/
#
# We download a controlled number of WARC files to keep the dataset
# manageable. Each WARC is ~1GB compressed. For SLM, 20-50 WARCs
# gives us enough raw data to yield a few billion tokens after filtering.
#
# Usage:
#   bash download_cc.sh [--snapshot CC-MAIN-2024-10] [--n-files 20] [--output-dir /data/raw]
#
# Requirements:
#   aws CLI (configured with access to public S3, no credentials needed for CC)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SNAPSHOT="CC-MAIN-2024-10"          # Common Crawl snapshot ID
N_FILES=20                           # number of WARC files to download
OUTPUT_DIR="/data/raw/common_crawl"
PARALLEL_DOWNLOADS=4                 # concurrent downloads (watch bandwidth)

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --snapshot)   SNAPSHOT="$2";    shift 2 ;;
        --n-files)    N_FILES="$2";     shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        --parallel)   PARALLEL_DOWNLOADS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/download.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ── Fetch WARC paths index ────────────────────────────────────────────────────
WARC_PATHS_URL="https://data.commoncrawl.org/crawl-data/${SNAPSHOT}/warc.paths.gz"
WARC_PATHS_FILE="$OUTPUT_DIR/warc.paths.gz"

log "Snapshot: $SNAPSHOT"
log "Downloading WARC paths index from $WARC_PATHS_URL"

if [[ ! -f "$WARC_PATHS_FILE" ]]; then
    curl -sS "$WARC_PATHS_URL" -o "$WARC_PATHS_FILE"
    log "WARC paths index downloaded"
else
    log "WARC paths index already exists — skipping download"
fi

# ── Select subset of WARC paths ───────────────────────────────────────────────
SELECTED_PATHS_FILE="$OUTPUT_DIR/selected_warc_paths.txt"

log "Selecting $N_FILES WARC files from index"

# NOTE: || true suppresses SIGPIPE (exit 141) from zcat when head exits early
# after reading N_FILES lines. This is expected behavior with set -euo pipefail.
zcat "$WARC_PATHS_FILE" | head -n "$N_FILES" > "$SELECTED_PATHS_FILE" || true

ACTUAL_N=$(wc -l < "$SELECTED_PATHS_FILE")

# Verify we actually got the expected number of paths
if [[ "$ACTUAL_N" -eq 0 ]]; then
    log "ERROR: No WARC paths selected — index file may be corrupt"
    exit 1
fi

if [[ "$ACTUAL_N" -lt "$N_FILES" ]]; then
    log "WARNING: Only $ACTUAL_N paths available (requested $N_FILES)"
fi

log "Selected $ACTUAL_N WARC paths"

# ── Download WARCs in parallel ────────────────────────────────────────────────
log "Starting parallel download ($PARALLEL_DOWNLOADS concurrent)"

download_warc() {
    local relative_path="$1"
    local filename
    filename=$(basename "$relative_path")
    local output_file="$OUTPUT_DIR/$filename"

    if [[ -f "$output_file" ]]; then
        echo "[SKIP] $filename already exists"
        return 0
    fi

    echo "[DL] $filename"
    # Using curl via HTTPS instead of aws s3 cp:
    # Common Crawl's S3 bucket is public but aws s3 cp --no-sign-request
    # returns 403 when instance IAM credentials are present and interfere.
    # curl via https://data.commoncrawl.org bypasses this entirely.
    curl -fsSL \
        "https://data.commoncrawl.org/${relative_path}" \
        -o "$output_file"

    if [[ $? -eq 0 ]]; then
        echo "[OK] $filename ($(du -sh "$output_file" | cut -f1))"
    else
        echo "[FAIL] $filename"
        rm -f "$output_file"   # remove partial download
        return 1
    fi
}

export -f download_warc
export OUTPUT_DIR

# Use xargs for parallel downloads
cat "$SELECTED_PATHS_FILE" | \
    xargs -P "$PARALLEL_DOWNLOADS" -I{} bash -c 'download_warc "$@"' _ {}

# ── Summary ───────────────────────────────────────────────────────────────────
DOWNLOADED=$(find "$OUTPUT_DIR" -name "*.warc.gz" | wc -l)
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" --exclude="*.txt" --exclude="*.gz" 2>/dev/null | cut -f1 || echo "unknown")

log "Download complete"
log "  WARC files downloaded: $DOWNLOADED"
log "  Output directory:      $OUTPUT_DIR"
log ""
log "Next step: run the curator pipeline"
log "  make docker-curate"