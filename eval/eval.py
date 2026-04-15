"""
eval/eval.py
-------------
Benchmark evaluation using lm-evaluation-harness.

Evaluates a trained SLM checkpoint on standard LLM benchmarks:
    - HellaSwag    — commonsense reasoning
    - ARC-Easy     — science QA (easy)
    - ARC-Challenge — science QA (hard)
    - MMLU         — broad knowledge (57 subjects)
    - TruthfulQA   — factual accuracy
    - HumanEval    — code generation (pass@1)

Results are written to eval/results/<model_name>/ as JSON
and printed as a formatted table.

Usage:
    python eval/eval.py --model results/slm-125m-dpo/final
    python eval/eval.py --model results/slm-125m-dpo/final --tasks hellaswag,arc_easy
    python eval/eval.py --model results/slm-125m/final --tasks all --num-fewshot 0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

EVAL_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results")) / "eval"

# ── Benchmark definitions ──────────────────────────────────────────────────────

BENCHMARKS = {
    "hellaswag": {
        "task": "hellaswag",
        "metric": "acc_norm",
        "num_fewshot": 10,
        "description": "Commonsense reasoning",
    },
    "arc_easy": {
        "task": "arc_easy",
        "metric": "acc_norm",
        "num_fewshot": 25,
        "description": "Science QA (easy)",
    },
    "arc_challenge": {
        "task": "arc_challenge",
        "metric": "acc_norm",
        "num_fewshot": 25,
        "description": "Science QA (challenge)",
    },
    "mmlu": {
        "task": "mmlu",
        "metric": "acc",
        "num_fewshot": 5,
        "description": "Broad knowledge (57 subjects)",
    },
    "truthfulqa": {
        "task": "truthfulqa_mc2",
        "metric": "acc",
        "num_fewshot": 0,
        "description": "Factual accuracy",
    },
    "humaneval": {
        "task": "humaneval",
        "metric": "pass@1,create_test",
        "num_fewshot": 0,
        "description": "Code generation",
    },
}

ALL_TASKS   = list(BENCHMARKS.keys())
QUICK_TASKS = ["hellaswag", "arc_easy", "arc_challenge"]


def run_evaluation(
    model_path: Path,
    tasks: list[str],
    num_fewshot_override: int | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    limit: int | None = None,
) -> dict:
    """
    Run lm-evaluation-harness on the given model and tasks.

    Registers SLMConfig and SLMForCausalLM with the AutoModel classes
    before creating the HFLM wrapper so lm-eval can load the checkpoint.
    """
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError("lm-eval not installed. Install with: pip install lm-eval")

    from transformers import AutoConfig, AutoModelForCausalLM
    from model import SLMConfig, SLMForCausalLM

    # Register custom model so AutoModelForCausalLM (used internally by HFLM)
    # can load the checkpoint without raising "Unrecognized configuration class".
    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    log.info(f"Loading model from {model_path}...")
    lm = HFLM(
        pretrained=str(model_path),
        device=device,
        batch_size=batch_size,
    )

    task_names  = []
    fewshot_map = {}
    for task_key in tasks:
        if task_key not in BENCHMARKS:
            log.warning(f"Unknown task: {task_key} — skipping")
            continue
        benchmark = BENCHMARKS[task_key]
        task_names.append(benchmark["task"])
        fewshot_map[benchmark["task"]] = (
            num_fewshot_override
            if num_fewshot_override is not None
            else benchmark["num_fewshot"]
        )

    log.info(f"Evaluating on: {task_names}")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=list(fewshot_map.values())[0] if len(set(fewshot_map.values())) == 1 else None,
        batch_size=batch_size,
        device=device,
        limit=limit,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )

    return results


def format_results(results: dict, tasks: list[str], model_name: str) -> str:
    """Format evaluation results as a readable table."""
    lines = [
        f"\n{'='*65}",
        f"Evaluation Results — {model_name}",
        f"{'='*65}",
        f"{'Benchmark':<20}  {'Metric':<12}  {'Score':>8}  {'Description'}",
        f"{'-'*65}",
    ]

    task_results = results.get("results", {})
    for task_key in tasks:
        if task_key not in BENCHMARKS:
            continue
        benchmark = BENCHMARKS[task_key]
        task_name = benchmark["task"]
        metric    = benchmark["metric"]

        if task_name in task_results:
            score = task_results[task_name].get(
                metric,
                task_results[task_name].get(f"{metric},none", "N/A")
            )
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        else:
            score_str = "N/A"

        lines.append(
            f"  {task_key:<18}  {metric:<12}  {score_str:>8}  {benchmark['description']}"
        )

    lines.append(f"{'='*65}\n")
    return "\n".join(lines)


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts non-serializable objects to strings."""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def save_results(results: dict, model_path: Path, tasks: list[str]) -> Path:
    """Save evaluation results to JSON."""
    model_name = model_path.parent.name if model_path.name == "final" else model_path.name
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = EVAL_RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"eval_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model":      str(model_path),
            "model_name": model_name,
            "tasks":      tasks,
            "timestamp":  timestamp,
            "results":    results.get("results", {}),
            "config":     results.get("config", {}),
        }, f, indent=2, cls=_SafeEncoder)

    log.info(f"Results saved to {out_path}")
    return out_path


def main():
    # HumanEval executes model-generated code — required by the code_eval metric.
    # Set before any lm-eval imports to avoid the ValueError at task load time.
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    parser = argparse.ArgumentParser(description="SLM Benchmark Evaluation")
    parser.add_argument("--model",      type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--tasks",      type=str,  default="all",
                        help=f"Comma-separated tasks or 'all' or 'quick'. Available: {', '.join(ALL_TASKS)}")
    parser.add_argument("--batch-size", type=int,  default=8)
    parser.add_argument("--num-fewshot",type=int,  default=None)
    parser.add_argument("--device",     type=str,  default="cuda")
    parser.add_argument("--limit",      type=int,  default=None,
                        help="Limit examples per task (for quick testing)")
    args = parser.parse_args()

    if not args.model.exists():
        log.error(f"Model not found: {args.model}")
        sys.exit(1)

    if args.tasks == "all":
        tasks = ALL_TASKS
    elif args.tasks == "quick":
        tasks = QUICK_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    model_name = args.model.parent.name if args.model.name == "final" else args.model.name
    log.info(f"=== SLM Evaluation ===")
    log.info(f"Model:  {args.model}")
    log.info(f"Tasks:  {tasks}")

    results = run_evaluation(
        model_path=args.model,
        tasks=tasks,
        num_fewshot_override=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    print(format_results(results, tasks, model_name))
    save_results(results, args.model, tasks)


if __name__ == "__main__":
    main()