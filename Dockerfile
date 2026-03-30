# =============================================================================
# SLM — NeMo Framework Container
#
# Base: nvcr.io/nvidia/nemo:25.02  (NGC — requires NGC_API_KEY)
# Includes pre-compiled: NeMo 2.2.1, NeMo-Aligner 0.7.0, megatron-core 0.11.1,
#                        Apex, Transformer Engine, NeMo Curator 0.7.1
#
# This image adds only the curation-specific dependencies that are not
# already present in the base container.
#
# Serves both pipeline roles:
#   make docker-shell-cpu   →  data curation  (no GPU required)
#   make docker-shell-gpu   →  pretrain / SFT / DPO
#
# Build:
#   make docker-build   (NGC login handled by setup_gpu_instance.sh)
# =============================================================================
FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /workspace

# -----------------------------------------------------------------------------
# System packages not present in the NeMo base image
#   mecab + unidic : required by nemo-curator (Japanese tokenizer)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Curation pipeline dependencies not in the NeMo base image
#
# Already present in nemo:25.02 — do NOT reinstall:
#   nemo-toolkit, nemo-aligner, nemo-curator, megatron-core
#   apex, transformer-engine, torch, boto3, wandb, sentencepiece
#   datasets, numpy, scipy, pyyaml, tqdm, psutil
#
# Adding only what the curator pipeline needs beyond the base:
#   trafilatura     : HTML → clean text extraction
#   langdetect      : language detection fallback
#   datasketch      : MinHash fuzzy deduplication
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    trafilatura==1.9.0 \
    langdetect==1.0.9 \
    datasketch==1.6.4

# -----------------------------------------------------------------------------
# Workspace directories
# /data and /results are bind-mounted at runtime by the Makefile.
# Creating them here ensures they exist if running without mounts.
# -----------------------------------------------------------------------------
RUN mkdir -p /workspace/slm /data /results /logs

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV NEMO_RESULTS_DIR=/results

CMD ["/bin/bash"]