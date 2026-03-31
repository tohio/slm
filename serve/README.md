# serve

Production serving for SLM using vLLM on Kubernetes. Exposes an OpenAI-compatible REST API via PagedAttention for high-throughput inference. Deployed via [ai-infra](https://github.com/tohio/ai-infra) using ArgoCD.

---

## Local Serving

```bash
# Install vLLM
pip install vllm

# Serve slm-125m from Hub (default)
./serve/serve.sh

# Serve a specific model or size
./serve/serve.sh --model tohio/slm-350m
./serve/serve.sh --model results/slm-125m-dpo/final

# Custom port
./serve/serve.sh --port 8080

# Multi-GPU tensor parallelism
./serve/serve.sh --model tohio/slm-1b --tp 2
```

---

## Query the API

The vLLM server exposes an OpenAI-compatible API:

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-125m",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is a transformer?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# List available models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health
```

**Python client:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="slm-125m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a neural network is."},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

---

## Kubernetes Deployment

Manifests are deployed via [ai-infra](https://github.com/tohio/ai-infra) using ArgoCD. Apply manually:

```bash
# Create namespace and secret
kubectl create namespace inference
kubectl create secret generic hf-credentials \
  --from-literal=token=$HF_TOKEN \
  -n inference

# Apply manifests
kubectl apply -f serve/manifests/

# Check status
kubectl get pods -n inference
kubectl logs -f deployment/slm-125m -n inference
```

---

## Files

```
serve/
├── manifests/
│   ├── deployment.yaml     vLLM deployment (1 GPU, slm-125m)
│   ├── service.yaml        ClusterIP service on port 8000
│   └── hpa.yaml            HPA — scales 1–4 replicas on CPU utilization
├── serve.sh                local vLLM launch script
└── README.md
```

---

## Key Design Decisions

**Why vLLM?** PagedAttention enables continuous batching and near-optimal KV cache utilization. At inference time this gives 10–50× higher throughput compared to naive HuggingFace generation. The OpenAI-compatible API means any existing OpenAI client works out of the box.

**Why Kubernetes via ArgoCD?** Declarative GitOps deployment — the cluster state is always what's in the repo. ArgoCD syncs automatically on commit. Rollbacks are a git revert.

**Why ClusterIP?** The inference service is internal — exposed to other services in the cluster, not directly to the internet. An ingress or gateway layer handles external traffic and auth.