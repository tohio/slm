# =============================================================================
# SLM — Self-Contained NeMo Image (no NGC auth required)
#
# Base: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime  (public DockerHub)
# Python: 3.10  |  CUDA: 12.1  |  PyTorch: 2.1.2
#
# Serves both:
#   make docker-shell-cpu   →  data curation  (no GPU required)
#   make docker-shell-gpu   →  pretrain / SFT / DPO
#
# Build:
#   make docker-build                        # uses DOCKER_IMAGE=slm:latest
#   docker build -t slm:latest .
# =============================================================================
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# -----------------------------------------------------------------------------
# System packages
#   libgomp1    : fasttext runtime dependency (OpenMP)
#   unzip       : AWS CLI v2 installer
#   curl        : AWS CLI v2 download + healthchecks
#   mecab + unidic : required by nemo-curator (Japanese tokenizer)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    unzip \
    libgomp1 \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# STEP 1 — NeMo core (sets the dependency floor for everything else)
#
# Version compatibility matrix (verified against PyPI metadata):
#
#   nemo_toolkit==2.2.0  [core] extra:
#     transformers>=4.48.0,<=4.48.3  ← exact match with nemo-curator requirement
#     pytorch-lightning > 2.2.1
#     hydra-core  >1.3,<=1.3.2
#     omegaconf   <=2.3
#     sentencepiece < 1.0.0
#     NO mamba-ssm in [core] — safe to use without nvcc
#
#   nemo-aligner==0.7.0:
#     deps: jsonlines, Jinja2 only — no nemo_toolkit[nlp] pin
#     nemo-aligner 0.4.0 pulled nemo_toolkit[nlp] → mamba-ssm (nvcc required)
#     0.7.0 eliminates this entirely
#
#   nemo-curator==0.7.1:
#     transformers>=4.48.0 — satisfied by nemo_toolkit 2.2.0's pin
#     fasttext==0.9.3  ← use 0.9.3 not 0.9.2
#     numpy<2
#     NOTE: nemo-curator 0.5.0 referenced in earlier docs never existed on PyPI
#
# Install order: NeMo first → aligner → curator → everything else
# This prevents pip from backtracking against already-resolved constraints.
# -----------------------------------------------------------------------------
# nemo_toolkit 2.2.0: pins transformers>=4.48.0,<=4.48.3 — compatible with
# nemo-curator 0.7.1 (needs >=4.48.0) and avoids BasicTokenizer removal in 4.46+.
# nemo-aligner 0.7.0: no nemo_toolkit[nlp] dependency — avoids mamba-ssm entirely.
# No --no-deps workarounds needed with this version combination.
RUN pip install --no-cache-dir \
    "nemo_toolkit[core]==2.2.0" \
    "nemo-aligner==0.7.0" \
    "dask[distributed]==2024.4.1"

# -----------------------------------------------------------------------------
# STEP 1b — NeMo NLP extras needed for GPT/SFT/DPO (mamba-ssm excluded)
#
# These are the [nlp] extras minus mamba-ssm and anything that requires
# nvcc or a GPU at install time. All needed for the SLM training pipeline.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    datasets \
    einops \
    "sentencepiece<1.0.0" \
    "tiktoken==0.7.0" \
    nltk \
    sacrebleu \
    sacremoses \
    rouge-score \
    pandas \
    h5py \
    ftfy \
    faiss-cpu \
    sentence-transformers \
    rapidfuzz \
    zarr \
    ijson \
    inflect \
    jieba \
    "opencc<1.1.7" \
    pangu \
    gdown \
    markdown2 \
    "matplotlib>=3.3.2" \
    "webdataset>=0.2.86"

# -----------------------------------------------------------------------------
# STEP 2 — NeMo Curator (CPU-only install — no cudf/dask-cuda needed)
#
# nemo-curator 0.7.1 is the earliest version on PyPI.
# Installing without extras avoids pulling in cudf/dask-cuda which
# require the NVIDIA PyPI index and an NGC-adjacent setup.
# The CPU pipeline (text extraction, dedup, filtering) works without them.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "nemo-curator==0.7.1"

# -----------------------------------------------------------------------------
# STEP 3 — Data curation pipeline dependencies
#
# Notes:
#   fasttext==0.9.3    : nemo-curator pins this; 0.9.2 will conflict
#   lxml_html_clean    : required by nemo-curator (not lxml alone)
#   trafilatura        : HTML → clean text (your curator pipeline)
#   warcio             : already pulled by curator but pin for clarity
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    trafilatura==1.9.0 \
    langdetect==1.0.9 \
    datasketch==1.6.4

# -----------------------------------------------------------------------------
# STEP 4 — Training, evaluation, and logging dependencies
#
# Dropped from your original list:
#   huggingface-hub  : owned by nemo_toolkit (>=0.24 resolved automatically)
#   transformers     : owned by nemo-curator (>=4.48.0 resolved automatically)
#   pytorch-lightning: owned by nemo_toolkit (>2.2.1 resolved automatically)
#   omegaconf        : owned by nemo_toolkit
#   hydra-core       : owned by nemo_toolkit
#   sentencepiece    : owned by nemo_toolkit (<1.0.0)
#   datasets         : owned by nemo-curator
#   awscli v1        : replaced with v2 binary below (avoids botocore conflict)
#
# numpy<2 is enforced by nemo-curator — do not pin numpy>=2
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    einops==0.8.0 \
    "numpy>=1.24,<2.0" \
    scipy==1.13.0 \
    wandb==0.17.0 \
    boto3==1.34.101 \
    "lm-eval==0.4.3" \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    "regex==2024.4.16" \
    psutil==5.9.8

# -----------------------------------------------------------------------------
# STEP 5 — AWS CLI v2 (standalone binary, zero pip dependencies)
#
# awscli v1 hard-pins its own botocore version and reliably conflicts
# with boto3==1.34.101. The v2 binary is self-contained — no pip conflict.
# Used by: curator/scripts/upload_s3.sh, infra/setup_gpu_instance.sh
# -----------------------------------------------------------------------------
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
        -o /tmp/awscliv2.zip \
    && unzip /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws \
    # The pytorch base image ships awscli v1 at /opt/conda/bin/aws.
    # That v1 has a botocore mismatch with our installed boto3.
    # Shadow it with the v2 binary so PATH always resolves to v2.
    && ln -sf /usr/local/bin/aws /opt/conda/bin/aws \
    && /usr/local/bin/aws --version

# -----------------------------------------------------------------------------
# Workspace directories
# /data and /results are bind-mounted at runtime by the Makefile:
#   docker run -v /data:/data -v /results:/results ...
# Creating them here ensures they exist if running without mounts.
# -----------------------------------------------------------------------------
RUN mkdir -p /workspace/slm /data /results

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1

# Predictable HuggingFace cache location inside the container
ENV HF_HOME=/workspace/.cache/huggingface

# NeMo experiment manager default output path (matches Makefile RESULTS_DIR)
ENV NEMO_RESULTS_DIR=/results

# CUDA_VISIBLE_DEVICES intentionally NOT set here.
# Control GPU access at runtime:
#   make docker-shell-gpu  →  docker run --gpus all ...
#   make docker-shell-cpu  →  no --gpus flag (CPU only)

CMD ["/bin/bash"]