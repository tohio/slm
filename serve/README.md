# Serving

This directory contains a vLLM launch wrapper and Kubernetes manifests for
serving native SLM exports through chat, completion, model-list, health, and
metrics HTTP endpoints.

## Contents

| Path | Purpose |
|---|---|
| `serve.sh` | Build and launch the vLLM server command |
| `manifests/deployment.yaml` | Single-GPU 125M deployment template |
| `manifests/service.yaml` | Cluster service |
| `manifests/pvc.yaml` | Persistent Hugging Face cache |
| `manifests/hpa.yaml` | Optional request-metric autoscaler |

Serving uses the native Llama artifacts produced by `export/`.

## Local serving

Install a vLLM build compatible with the host's NVIDIA driver and CUDA stack,
then serve the published chat variant:

```bash
make serve SIZE=125m
```

Build and serve a local export:

```bash
make export-chat-local SIZE=125m
make serve-local SIZE=125m
```

Direct examples:

```bash
./serve/serve.sh --model tohio/slm-125m-chat

./serve/serve.sh \
  --model results/exports/125m/chat \
  --host 0.0.0.0 \
  --port 8080

./serve/serve.sh \
  --model tohio/slm-1b-chat \
  --tp 2
```

Runtime settings may also be supplied as environment variables:

```bash
MODEL=tohio/slm-125m-chat \
PORT=8000 \
DTYPE=bfloat16 \
GPU_MEMORY_UTILIZATION=0.90 \
./serve/serve.sh
```

`MAX_MODEL_LEN` is unset by default so vLLM reads the trained context length
from `config.json`. Set it only to impose an explicit shorter limit.
`TENSOR_PARALLEL_SIZE`/`--tp` must not exceed the available GPU count.

## HTTP examples

List the served models and check health:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/health
```

Send a chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-125m-chat",
    "messages": [
      {"role": "user", "content": "What is a transformer?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

The `model` field must match the served model name printed by `serve.sh`.
Hub IDs use the repository name; local exports use the leaf directory name,
or the parent directory when the selected path ends in `final`.

## Kubernetes

The checked-in deployment is a 125M, one-GPU template. Review the model ID,
served name, resource requests, storage class, image tag, namespace, and GPU
node policy before applying it to another cluster or model size.

Create the namespace and Hub credential:

```bash
kubectl create namespace inference

kubectl create secret generic hf-credentials \
  --from-literal=token="$HF_TOKEN" \
  --namespace inference
```

Apply the base resources:

```bash
kubectl apply -f serve/manifests/pvc.yaml
kubectl apply -f serve/manifests/deployment.yaml
kubectl apply -f serve/manifests/service.yaml

kubectl rollout status deployment/slm-125m --namespace inference
kubectl logs -f deployment/slm-125m --namespace inference
```

The optional HPA depends on a Prometheus adapter that exposes vLLM's running
request metric through the Kubernetes custom-metrics API. Verify that metric
before applying:

```bash
kubectl get --raw \
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/inference/pods/*/vllm:num_requests_running"

kubectl apply -f serve/manifests/hpa.yaml
```

The model cache PVC avoids downloading weights on every pod restart. vLLM
exposes Prometheus metrics at `/metrics`; use them to validate latency,
queueing, cache utilization, and autoscaling thresholds under representative
load.
