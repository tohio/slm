# Supervised Fine-Tuning (SFT)

Adapts the pre-trained base model to follow instructions and produce well-formed responses. SFT runs in two sequential stages — general chat first, then coding — each building on the previous checkpoint.

## Why Sequential SFT?

The alternative is blending chat and coding data into a single SFT run. Sequential is preferred here for two reasons:

**Debuggability.** Each stage produces an independently evaluable checkpoint. If the code SFT degrades general conversation quality (a real risk), it's immediately visible by comparing the chat and code SFT checkpoints. With a blended run, diagnosing regressions is harder.

**Specialization control.** The code SFT uses a lower learning rate (5e-6 vs 1e-5) specifically to reduce catastrophic forgetting of chat capabilities learned in stage 1. This is harder to reason about with a blended dataset.

## Stage 1 — General Chat

**Dataset:** OpenAssistant OASST1 + Dolly 15k (~25k examples combined)
**Goal:** Instruction following, multi-turn conversation, appropriate response length and tone.

**Key config:** `answer_only_loss: true` — cross-entropy loss is computed only on the assistant's response tokens, not the human prompt. This is essential: without it the model learns to predict user messages, which wastes capacity and produces odd behavior.

**Format:**
```json
{
  "conversations": [
    {"from": "system",    "value": "You are a helpful assistant."},
    {"from": "human",     "value": "What is Python?"},
    {"from": "assistant", "value": "Python is a high-level..."}
  ]
}
```

## Stage 2 — Coding

**Dataset:** CodeSearchNet Python (~100k examples, two formats)
**Goal:** Code generation, code explanation, debugging assistance.

**Two example formats per function** are generated from CodeSearchNet:
- *Write*: "Write a function that does X" → function body
- *Explain*: "Explain what this code does" → docstring

This gives the model diversity in coding tasks — it learns to produce code and to reason about code, not just to pattern-match function signatures.

**Lower learning rate (5e-6)** reduces catastrophic forgetting of the chat capabilities from stage 1. The model is already well-initialized for instruction following; the code stage needs to add capability, not replace it.

## Design Decisions

**Full fine-tuning over LoRA**
At 125M parameters, the model fits comfortably in GPU memory for full fine-tuning. LoRA adds complexity (rank selection, target module selection, adapter merging) without meaningful benefit at this scale. LoRA becomes the right choice at 7B+ where full fine-tuning VRAM requirements become prohibitive.

**No system prompt variation**
Using a fixed system prompt during SFT keeps things simple. Varying system prompts during SFT (as done in models like Alpaca or Vicuna) can improve robustness to different prompts at inference, but adds dataset complexity. Worth revisiting if the model shows brittleness to prompt phrasing.

## Usage

All commands run inside the GPU container:

```bash
# Start GPU container
make docker-shell-gpu

# Inside the container:

# Prepare datasets (downloads from HuggingFace)
make prepare-sft-data

# Run both stages sequentially
make sft

# Or run individually
bash finetune/scripts/train_sft.sh --stage chat
bash finetune/scripts/train_sft.sh --stage code

# Evaluate
make eval-sft
```

## What to Watch

After chat SFT, generation samples should show:
- Responses that stop at the right place (don't bleed into next user turn)
- Appropriate response length — not one word, not 500 words for a simple question
- No repetition loops (a common failure mode in poorly trained small models)

After code SFT:
- Code blocks are fenced correctly (` ```python `)
- Generated functions are syntactically valid Python
- Explanations are coherent and reference the actual code
- General conversation quality is not visibly degraded vs chat checkpoint