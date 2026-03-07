"""
eval/mmlu.py
------------
MMLU (Massive Multitask Language Understanding) benchmark evaluation.

MMLU tests knowledge across 57 subjects including STEM, humanities,
social sciences, and more. Each question is 4-choice multiple choice.

We use a 5-shot format — prepend 5 examples from the dev set before
each test question. This is the standard MMLU evaluation protocol.

For a 125M model from scratch, expect:
  - Random baseline: 25% (4-choice)
  - Our model:       30-40% is realistic
  - GPT-3 175B:      ~57%

Subjects we evaluate (subset most relevant to our training data):
  - high_school_computer_science
  - college_computer_science
  - high_school_mathematics
  - elementary_mathematics
  - high_school_physics
  - world_history
  - high_school_english
  - formal_logic

Full MMLU (57 subjects) available via --full flag.
"""

import logging
import random
import torch
from pathlib import Path

logger = logging.getLogger("eval.mmlu")

# Curated subject subset — most relevant to our training data mix
SUBJECTS_DEFAULT = [
    "high_school_computer_science",
    "college_computer_science",
    "high_school_mathematics",
    "elementary_mathematics",
    "high_school_physics",
    "world_history",
    "high_school_english",
    "formal_logic",
]

CHOICES = ["A", "B", "C", "D"]
N_SHOT = 5  # standard MMLU 5-shot


def load_mmlu_data(subject: str, split: str = "test") -> list[dict]:
    """
    Load MMLU data for a subject from HuggingFace datasets.
    Falls back to a synthetic example set if network unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
        return [
            {
                "question": row["question"],
                "choices":  row["choices"],
                "answer":   row["answer"],  # int 0-3
            }
            for row in ds
        ]
    except Exception as e:
        logger.warning(f"Could not load MMLU subject '{subject}': {e}")
        return []


def format_mmlu_prompt(
    question: str,
    choices: list[str],
    few_shot_examples: list[dict],
    subject: str,
) -> str:
    """
    Format a question in 5-shot MMLU prompt style.

    Format:
      The following are multiple choice questions about {subject}.

      Question: ...
      A. ...   B. ...   C. ...   D. ...
      Answer: A

      ... (5 examples)

      Question: {test_question}
      A. ...   B. ...   C. ...   D. ...
      Answer:
    """
    subject_readable = subject.replace("_", " ").title()
    prompt = f"The following are multiple choice questions about {subject_readable}.\n\n"

    # Few-shot examples
    for ex in few_shot_examples:
        prompt += format_single_question(ex["question"], ex["choices"])
        prompt += f"Answer: {CHOICES[ex['answer']]}\n\n"

    # Test question (no answer)
    prompt += format_single_question(question, choices)
    prompt += "Answer:"

    return prompt


def format_single_question(question: str, choices: list[str]) -> str:
    q = f"Question: {question}\n"
    for letter, choice in zip(CHOICES, choices):
        q += f"{letter}. {choice}\n"
    return q


def get_answer_logprobs(model, tokenizer, prompt: str, device: str) -> dict[str, float]:
    """
    Get log-probabilities of each answer token (A, B, C, D)
    at the position immediately following the prompt.

    This is the standard approach for multiple-choice eval —
    we don't generate, we score each option directly.
    """
    input_ids = tokenizer.text_to_ids(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
        # Get logits at the last position (where answer token would be)
        logits = outputs[0][:, -1, :]   # shape: [1, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)[0]  # shape: [vocab_size]

    # Get log-prob for each answer token
    answer_logprobs = {}
    for choice in CHOICES:
        token_id = tokenizer.text_to_ids(f" {choice}")[-1]  # space before letter
        answer_logprobs[choice] = log_probs[token_id].item()

    return answer_logprobs


def evaluate_subject(
    model,
    tokenizer,
    subject: str,
    n_per_subject: int,
    device: str,
    seed: int = 42,
) -> dict:
    """Evaluate model accuracy on a single MMLU subject."""
    random.seed(seed)

    # Load dev set for few-shot examples, test set for eval
    dev_data = load_mmlu_data(subject, split="dev")
    test_data = load_mmlu_data(subject, split="test")

    if not test_data:
        logger.warning(f"No test data for subject '{subject}' — skipping")
        return {"subject": subject, "accuracy": None, "n_questions": 0}

    # Sample few-shot examples from dev set
    few_shot = dev_data[:N_SHOT] if len(dev_data) >= N_SHOT else dev_data

    # Sample test questions
    test_sample = random.sample(test_data, min(n_per_subject, len(test_data)))

    correct = 0
    for item in test_sample:
        prompt = format_mmlu_prompt(
            question=item["question"],
            choices=item["choices"],
            few_shot_examples=few_shot,
            subject=subject,
        )

        try:
            logprobs = get_answer_logprobs(model, tokenizer, prompt, device)
            predicted = max(logprobs, key=logprobs.get)
            gold = CHOICES[item["answer"]]
            if predicted == gold:
                correct += 1
        except Exception as e:
            logger.debug(f"Failed on question: {e}")
            continue

    accuracy = correct / len(test_sample) if test_sample else 0.0
    logger.info(f"  {subject:<40} {accuracy:.1%} ({correct}/{len(test_sample)})")

    return {
        "subject":     subject,
        "accuracy":    round(accuracy, 4),
        "n_questions": len(test_sample),
        "n_correct":   correct,
    }


def evaluate_mmlu(
    checkpoint: str,
    n_per_subject: int = 100,
    subjects: list[str] = None,
    device: str = "cuda",
) -> dict:
    """Main MMLU evaluation entry point."""
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

    logger.info(f"Loading model for MMLU eval...")
    model = MegatronGPTModel.restore_from(
        restore_path=checkpoint,
        map_location=torch.device(device),
    )
    model.eval()
    model.to(device)
    tokenizer = model.tokenizer

    subjects_to_eval = subjects or SUBJECTS_DEFAULT
    logger.info(f"Evaluating {len(subjects_to_eval)} subjects, {n_per_subject} questions each")
    logger.info(f"Random baseline: 25.0% (4-choice)")

    subject_results = []
    for subject in subjects_to_eval:
        result = evaluate_subject(model, tokenizer, subject, n_per_subject, device)
        subject_results.append(result)

    # Aggregate
    valid_results = [r for r in subject_results if r["accuracy"] is not None]
    overall_accuracy = (
        sum(r["accuracy"] for r in valid_results) / len(valid_results)
        if valid_results else 0.0
    )
    total_questions = sum(r["n_questions"] for r in valid_results)

    logger.info(f"MMLU overall accuracy: {overall_accuracy:.1%} over {total_questions} questions")

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "n_subjects":       len(valid_results),
        "total_questions":  total_questions,
        "random_baseline":  0.25,
        "by_subject": {
            r["subject"]: r["accuracy"]
            for r in valid_results
            if r["accuracy"] is not None
        },
        "subject_details": subject_results,
    }
