# SLM Data Curation & Training Environment
# Base: PyTorch CUDA 12.1 (public image, no auth required)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies (Python already in base image)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install NeMo and dependencies (Python 3.12 compatible)
RUN pip install --no-cache-dir \
    nemo_toolkit[all]==2.0.0 \
    nemo-aligner>=1.0.0 \
    nemo-curator>=1.1.0 \
    dask[distributed]==2024.4.1 \
    trafilatura==1.9.0 \
    warcio==1.7.4 \
    beautifulsoup4==4.12.3 \
    lxml==5.2.1 \
    fasttext==0.9.2 \
    langdetect==1.0.9 \
    datasketch==1.6.4 \
    sentencepiece==0.2.0 \
    datasets==2.19.1 \
    huggingface-hub==0.23.0 \
    transformers==4.41.0 \
    pytorch-lightning==2.2.4 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    einops==0.8.0 \
    wandb==0.17.0 \
    boto3==1.34.101 \
    awscli==1.32.101 \
    lm-eval==0.4.3 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    regex==2024.4.16 \
    psutil==5.9.8

# Set up workspace
RUN mkdir -p /workspace/slm /data /results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["/bin/bash"]