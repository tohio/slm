"""
eval/win_rate.py
----------------
Evaluates DPO alignment quality by measuring win rate of the
DPO model vs the SFT reference model.

Method:
  1. Sample prompts from the DPO validation set
  2. Generate a response from the DPO (policy) model
  3. Generate a response from the SFT (reference) model
  4. Judge which response is better using a reward model or
     rule-based heuristics

Judging approaches (in order of preference):
  A. Reward model scoring  — if you have a trained RM, use it
  B. Rule-based heuristics — response length, refusal detection,
                             code validity for coding prompts
     (used here as the primary judge since we're not training a RM)

Win rate interpretation:
  > 60%  — DPO is clearly improving over SFT
  50-60% — Modest improvement
  ~50%   — No meaningful change (check beta, dataset quality)
  < 50%  — DPO is degrading the model (reduce beta, check data)

Note: For a rigorous win rate you'd use GPT-4 as judge
(LLM-as-judge approach). We implement heuristic judging here
since GPT-4 API costs add up. The heuristic is honest about
its limitations.
"""

import logging
import re
import json
import random
from pathlib import Path

import torch

logger = logging.getLogger("eval.win_rate")


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic judge
# ─────────────────────────────────────────────────────────────────────────────

def is_refusal(text: str) -> bool:
    """Detect if a response is an unhelpful refusal or non-answer."""
    refusal_patterns = [
        r"i (can'?t|cannot|won'?t|will not) (help|assist|answer|provide)",
        r"i('m| am) (not able|unable)",
        r"i don'?t (know|have|understand)",
        r"sorry,? i",
        r"as an ai",
        r"i must (decline|refuse)",
    ]
    lower = text.lower()
    return any(re.search(p, lower) for p in refusal_patterns)


def has_valid_code_block(text: str) -> bool:
    """Check if response contains a fenced code block."""
    return bool(re.search(r"```[\w]*\n.*?```", text, re.DOTALL))


def response_quality_score(response: str, prompt: str) -> float:
    """
    Heuristic quality score for a response. Returns float in [0, 1].

    Factors:
      - Length: too short is bad, very long may be padding
      - Refusal: penalize unhelpful refusals
      - Code presence: reward code blocks for coding prompts
      - Structure: reward responses with clear structure
    """
    score = 0.5  # neutral baseline

    n_words = len(response.split())

    # Length scoring — sweet spot is 50-400 words
    if n_words < 10:
        score -= 0.3
    elif n_words < 30:
        score -= 0.1
    elif 50 <= n_words <= 400:
        score += 0.1
    elif n_words > 800:
        score -= 0.1   # likely padding/rambling

    # Penalize refusals (unless it's a safety prompt)
    if is_refusal(response):
        is_safety_prompt = any(
            kw in prompt.lower()
            for kw in ["hack", "illegal", "dangerous", "harmful", "weapon"]
        )
        if not is_safety_prompt:
            score -= 0.3
        else:
            score += 0.2  # appropriate refusal on safety question

    # Reward code blocks for coding prompts
    is_code_prompt = any(
        kw in prompt.lower()
        for kw in ["python", "code", "function", "write a", "implement", "debug", "bug"]
    )
    if is_code_prompt and has_valid_code_block(response):
        score += 0.2
    elif is_code_prompt and not has_valid_code_block(response):
        score -= 0.1

    # Reward structured responses (newlines suggest organized content)
    if "\n" in response and n_words > 30:
        score += 0.05

    return max(0.0, min(1.0, score))


def judge_pair(
    prompt: str,
    policy_response: str,
    ref_response: str,
    margin: float = 0.05,
) -> str:
    """
    Judge which response is better.
    Returns: "policy_wins", "ref_wins", or "tie"
    """
    policy_score = response_quality_score(policy_response, prompt)
    ref_score = response_quality_score(ref_response, prompt)

    diff = policy_score - ref_score
    if diff > margin:
        return "policy_wins"
    elif diff < -margin:
        return "ref_wins"
    else:
        return "tie"


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    """Generate a response from a NeMo model."""
    input_ids = tokenizer.text_to_ids(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_tensor,
            max_length=len(input_ids) + max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_id,
            eos_token_id=tokenizer.eos_id,
        )

    generated = output_ids[0][len(input_ids):].tolist()
    text = tokenizer.ids_to_text(generated)

    # Stop at turn markers
    for stop in ["<|endofturn|>", "<|user|>", "<|system|>"]:
        if stop in text:
            text = text[:text.index(stop)]

    return text.strip()


def load_eval_prompts(n_pairs: int, seed: int = 42) -> list[str]:
    """
    Load evaluation prompts for win rate testing.
    Uses the DPO val set if available, otherwise falls back to
    the fixed generation eval prompts.
    """
    random.seed(seed)
    dpo_val = Path("/data/dpo/val.jsonl")

    if dpo_val.exists():
        prompts = []
        with open(dpo_val) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompts.append(item.get("prompt", ""))
        prompts = [p for p in prompts if p]
        random.shuffle(prompts)
        logger.info(f"Loaded {len(prompts)} prompts from DPO val set")
        return prompts[:n_pairs]

    # Fallback to generation eval prompts
    from generation import EVAL_PROMPTS
    prompts = [p["prompt"] for p in EVAL_PROMPTS]
    logger.info(f"Using {len(prompts)} fixed eval prompts (DPO val set not found)")
    return prompts[:n_pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_win_rate(
    policy_checkpoint: str,
    ref_checkpoint: str,
    n_pairs: int = 200,
    max_new_tokens: int = 256,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """
    Main win rate evaluation entry point.

    Loads both models, generates paired responses, judges each pair.
    """
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

    logger.info("Loading DPO (policy) model...")
    policy = MegatronGPTModel.restore_from(
        restore_path=policy_checkpoint,
        map_location=torch.device(device),
    )
    policy.eval().to(device)

    logger.info("Loading SFT (reference) model...")
    reference = MegatronGPTModel.restore_from(
        restore_path=ref_checkpoint,
        map_location=torch.device(device),
    )
    reference.eval().to(device)

    tokenizer = policy.tokenizer
    prompts = load_eval_prompts(n_pairs, seed=seed)

    if not prompts:
        logger.error("No evaluation prompts available")
        return {"error": "No prompts available"}

    logger.info(f"Evaluating {len(prompts)} prompt pairs...")
    logger.info(f"Judge: heuristic (length, refusal, code quality)")

    results = []
    outcomes = {"policy_wins": 0, "ref_wins": 0, "tie": 0}

    for i, prompt in enumerate(prompts):
        policy_response = generate(policy, tokenizer, prompt, max_new_tokens, device)
        ref_response = generate(reference, tokenizer, prompt, max_new_tokens, device)

        outcome = judge_pair(prompt, policy_response, ref_response)
        outcomes[outcome] += 1

        results.append({
            "prompt":          prompt[:200],
            "policy_response": policy_response[:500],
            "ref_response":    ref_response[:500],
            "outcome":         outcome,
        })

        if (i + 1) % 20 == 0:
            n = i + 1
            wr = outcomes["policy_wins"] / n
            logger.info(
                f"  [{n}/{len(prompts)}] Win rate so far: {wr:.1%} "
                f"(W:{outcomes['policy_wins']} T:{outcomes['tie']} L:{outcomes['ref_wins']})"
            )

    n_total = len(prompts)
    win_rate = outcomes["policy_wins"] / n_total
    tie_rate = outcomes["tie"] / n_total
    loss_rate = outcomes["ref_wins"] / n_total

    logger.info(
        f"Win rate: {win_rate:.1%} | Tie: {tie_rate:.1%} | Loss: {loss_rate:.1%}"
    )

    return {
        "win_rate":   round(win_rate, 4),
        "tie_rate":   round(tie_rate, 4),
        "loss_rate":  round(loss_rate, 4),
        "n_pairs":    n_total,
        "judge":      "heuristic (length + refusal + code quality)",
        "note":       "Heuristic judge — upgrade to LLM-as-judge for production eval",
        "outcomes":   outcomes,
        "samples":    results[:20],   # save first 20 pairs for inspection
    }
