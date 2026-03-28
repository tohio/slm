# SLM Data Curation & Training Environment
# Based on NVIDIA NeMo official image with project-specific setup

FROM nvcr.io/nvidia/nemo:latest

# Set working directory
WORKDIR /workspace

# Install additional Python dependencies from project requirements
# NeMo base image already has torch, cuda, nemo-toolkit, etc.
# We add data curation specific tools and project dependencies

RUN pip install --no-cache-dir \
    dask-cuda==24.4.0 \
    cudf-cu12==24.4.0 \
    boto3==1.34.101 \
    awscli==1.32.101 \
    wandb==0.17.0 \
    lm-eval==0.4.3

# Clone/mount the SLM repo at runtime, but set up the workspace
RUN mkdir -p /workspace/slm /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["/bin/bash"]