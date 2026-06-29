# serve

vLLM serving assets for SLM. The server exposes an OpenAI-compatible API for exported Hub models or local checkpoints.

---

## Files

```text
serve/
├── manifests/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── pvc.yaml
├── serve.sh
└── README.md
```

---

## Local serving

Default:

```bash
./serve/serve.sh
```

Serve a Hub model:

```bash
./serve/serve.sh --model tohio/slm-125m-chat
./serve/serve.sh --model tohio/slm-350m-chat
```

Serve a local checkpoint:

```bash
./serve/serve.sh --model results/runs/125m/dpo_chat/final
```

Custom port:

```bash
./serve/serve.sh --port 8080
```

Tensor parallelism:

```bash
./serve/serve.sh --model tohio/slm-1b-chat --tp 2
```

Environment variables:

```bash
MODEL=tohio/slm-125m-chat PORT=8000 ./serve/serve.sh
MAX_MODEL_LEN=2048 ./serve/serve.sh --model tohio/slm-1b-chat
```

`MAX_MODEL_LEN` is unset by default so vLLM can read the context length from the model config.

---

## Query the API

```bash
curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "slm-125m-chat",
    "messages": [
      {"role": "user", "content": "What is a transformer?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

List models:

```bash
curl http://localhost:8000/v1/models
```

Health check:

```bash
curl http://localhost:8000/health
```

Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="slm-125m-chat",
    messages=[{"role": "user", "content": "Explain what a neural network is."}],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

---

## Kubernetes manifests

Manifests live in `serve/manifests/` and can be deployed directly or through the external `ai-infra` repo.

```bash
kubectl create namespace inference
kubectl create secret generic hf-credentials   --from-literal=token="$HF_TOKEN"   -n inference

kubectl apply -f serve/manifests/
kubectl get pods -n inference
kubectl logs -f deployment/slm-125m -n inference
```

---

## Notes

- `serve.sh` always passes `--trust-remote-code`.
- Local model names are derived from the checkpoint directory.
- Hub model names are stripped to the repo name for the served model ID.
- Tensor parallelism is a vLLM serving feature and is separate from the training pipeline.
