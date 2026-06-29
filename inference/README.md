# inference

Local inference utilities for SLM checkpoints and exported Hub models.

---

## Responsibility

`inference/` owns:

- interactive chat CLI
- raw prompt generation
- chat-template formatting
- special-token resolution at runtime
- stopping behavior for generation

Serving through vLLM lives in `serve/`.

---

## Files

```text
inference/
├── chat.py
├── generate.py
├── utils.py
└── README.md
```

---

## Checkpoint names

Local checkpoints:

```text
results/runs/<size>/pretrain/final
results/runs/<size>/sft_instruct/final
results/runs/<size>/dpo_chat/final
results/runs/<size>/sft_code/final
```

Hub models:

```text
tohio/slm-<size>
tohio/slm-<size>-instruct
tohio/slm-<size>-chat
tohio/slm-<size>-code
```

---

## Chat CLI

Local chat checkpoint:

```bash
python inference/chat.py --model results/runs/125m/dpo_chat/final
```

Hub chat model:

```bash
python inference/chat.py --model tohio/slm-125m-chat
```

Custom system prompt:

```bash
python inference/chat.py   --model tohio/slm-125m-chat   --system "You are a concise assistant."
```

---

## Batch / prompt generation

Raw base-model completion:

```bash
echo "The capital of France is" | python inference/generate.py   --model results/runs/125m/pretrain/final   --max-new-tokens 30   --greedy
```

Chat-formatted generation:

```bash
echo "Explain attention in one sentence." | python inference/generate.py   --model results/runs/125m/dpo_chat/final   --chat   --max-new-tokens 80
```

From a file:

```bash
python inference/generate.py   --model tohio/slm-125m-chat   --input prompts.txt   --chat
```

---

## Special tokens

Runtime code resolves special token IDs from the loaded tokenizer instead of importing training-time constants. This prevents silent token ID drift when loading exported checkpoints.

---

## BOS / chat behavior

- Raw completion prepends BOS by default.
- `--no-bos` disables BOS for continuation-style generation.
- `--chat` formats the prompt as a user message with the tokenizer chat template.
- Chat/instruct models should normally be used with `--chat` or `chat.py`.

---

## Serving

Use `serve/` for an OpenAI-compatible vLLM server.
