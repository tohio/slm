"""
eval/eval.py
-------------
Benchmark evaluation using lm-evaluation-harness.

Evaluates a trained SLM checkpoint on standard LLM benchmarks:
    - HellaSwag     — commonsense reasoning
    - ARC-Easy      — science QA (easy)
    - ARC-Challenge — science QA (hard)
    - MMLU          — broad knowledge (57 subjects)
    - TruthfulQA    — factual accuracy
    - HumanEval     — code generation (pass@1)

Results are written to results/eval/<model_name>/ as JSON
and printed as a formatted table.

Tokenizer:
    The tokenizer is expected at <model_path>/tokenizer/ — the subdirectory
    written by train.py, train_sft.py, and train_dpo.py. The path is passed
    explicitly to HFLM so AutoTokenizer does not need to find it at the
    model root.

Precision:
    Eval loads the model in bfloat16 by default (matching training precision).
    Override with --dtype float16 or --dtype float32 if needed.

Result key format (lm-eval 0.4.x):
    Results are keyed by "{metric_name},{filter}" (e.g. "acc,none",
    "acc_norm,none", "pass@1,create_test"). The code-gen task HumanEval uses
    the "create_test" filter; all others use "none". metric_score() handles
    all current filter variants so adding a new task does not require
    touching result-parsing logic.

lm-eval compatibility:
    lm-eval 0.4.x requires num_fewshot to be an int, not a dict.
    We run each benchmark separately so each uses its canonical few-shot count.

Usage:
    python eval/eval.py --model results/slm-125m-dpo/final
    python eval/eval.py --model results/slm-125m-dpo/final --tasks hellaswag,arc_easy
    python eval/eval.py --model results/slm-125m/final --tasks all --num-fewshot 0
    python eval/eval.py --model results/slm-125m-dpo/final --dtype float16
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _eval_results_dir() -> Path:
    """Resolved at call time, not import, so env changes after import are honoured."""
    return Path(os.environ.get("RESULTS_DIR", "results")) / "eval"


# ── Benchmark definitions ──────────────────────────────────────────────────────

# `metric` is the metric name WITHOUT filter suffix. metric_score() handles
# filter resolution against lm-eval's "{metric},{filter}" key format.
BENCHMARKS = {
    "hellaswag": {
        "task":        "hellaswag",
        "metric":      "acc_norm",
        "num_fewshot": 10,
        "description": "Commonsense reasoning",
    },
    "arc_easy": {
        "task":        "arc_easy",
        "metric":      "acc_norm",
        "num_fewshot": 25,
        "description": "Science QA (easy)",
    },
    "arc_challenge": {
        "task":        "arc_challenge",
        "metric":      "acc_norm",
        "num_fewshot": 25,
        "description": "Science QA (challenge)",
    },
    "mmlu": {
        "task":        "mmlu",
        "metric":      "acc",
        "num_fewshot": 5,
        "description": "Broad knowledge (57 subjects)",
    },
    "truthfulqa": {
        "task":        "truthfulqa_mc2",
        "metric":      "acc",
        "num_fewshot": 0,
        "description": "Factual accuracy",
    },
    "humaneval": {
        "task":        "humaneval",
        "metric":      "pass@1",
        "num_fewshot": 0,
        "description": "Code generation",
    },
}

ALL_TASKS   = list(BENCHMARKS.keys())
QUICK_TASKS = ["hellaswag", "arc_easy", "arc_challenge"]

# Tasks that execute model-generated code. These need HF_ALLOW_CODE_EVAL=1
# in the environment and confirm_run_unsafe_code=True passed to
# simple_evaluate. Keep this list explicit so the "unsafe code" flag only
# applies where it's relevant.
CODE_EXECUTING_TASKS = {"humaneval"}


def model_display_name(model_path: Path) -> str:
    """
    Compact name for a checkpoint dir. `results/slm-125m-dpo/final` → `slm-125m-dpo`.
    """
    return model_path.parent.name if model_path.name == "final" else model_path.name


def resolve_tokenizer_path(model_path: Path) -> Path:
    """
    Resolve the tokenizer directory for a given model checkpoint.

    Checks for a tokenizer/ subdirectory first — this is where train.py,
    train_sft.py, and train_dpo.py copy the tokenizer alongside weights.
    Falls back to the model directory root for Hub-style checkpoints where
    the tokenizer files live alongside the model weights.
    """
    candidates = [
        model_path / "tokenizer",
        model_path,
    ]
    for candidate in candidates:
        if (candidate / "tokenizer_config.json").exists():
            return candidate

    raise FileNotFoundError(
        f"tokenizer_config.json not found in {model_path} or {model_path / 'tokenizer'}.\n"
        f"Ensure the tokenizer was copied during training: make tokenizer-download"
    )


def make_lm(model_path: Path, batch_size: int, device: str, dtype: str):
    """
    Create an HFLM wrapper for the given model checkpoint.
    Registers SLMConfig and SLMForCausalLM with AutoModel before loading.

    dtype matches training precision by default ("bfloat16"). Passing through
    HFLM's dtype argument is what actually loads the model at the requested
    precision — AutoModelForCausalLM alone would default to float32.
    """
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoConfig, AutoModelForCausalLM
    from model import SLMConfig, SLMForCausalLM

    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    tokenizer_path = resolve_tokenizer_path(model_path)
    log.info(f"Loading model from {model_path} (dtype={dtype})...")
    log.info(f"Tokenizer from {tokenizer_path}")

    return HFLM(
        pretrained=str(model_path),
        tokenizer=str(tokenizer_path),
        device=device,
        batch_size=batch_size,
        dtype=dtype,
    )


def metric_score(task_result: dict, metric: str):
    """
    Extract a metric value from an lm-eval task result dict.

    lm-eval 0.4.x keys results as "{metric},{filter}" (e.g. "acc,none",
    "pass@1,create_test"). We try, in order:
      1. Exact metric name (rare — mostly for older output formats)
      2. "{metric},none"           — default filter for multiple-choice tasks
      3. "{metric},create_test"    — code-generation tasks (HumanEval, MBPP)
      4. "{metric},strict-match"   — BBH cot, GSM8K with strict matching
      5. "{metric},flexible-extract" — GSM8K with lenient extraction
      6. Any key starting with "{metric},"   — catch-all for unknown filters

    Returns None if no variant is present.
    """
    if metric in task_result:
        return task_result[metric]

    for suffix in ("none", "create_test", "strict-match", "flexible-extract"):
        key = f"{metric},{suffix}"
        if key in task_result:
            return task_result[key]

    # Catch-all: any filter variant of this metric
    prefix = f"{metric},"
    for key, value in task_result.items():
        if key.startswith(prefix) and not key.endswith("_stderr"):
            return value

    return None


def run_evaluation(
    model_path: Path,
    tasks: list[str],
    num_fewshot_override: int | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: str = "bfloat16",
    limit: int | None = None,
    log_samples: bool = False,
) -> tuple[dict, list[str]]:
    """
    Run lm-evaluation-harness on the given model and tasks.

    Runs each benchmark separately so each uses its canonical num_fewshot
    count as an int — lm-eval 0.4.x does not accept num_fewshot as a dict.

    Returns (merged_results, failed_tasks). failed_tasks is the list of task
    keys that raised during evaluation — caller can decide whether to warn,
    abort, or continue.
    """
    try:
        from lm_eval import evaluator
    except ImportError:
        raise ImportError("lm-eval not installed. Install with: pip install lm-eval")

    lm = make_lm(model_path, batch_size, device, dtype)

    merged_results: dict = {"results": {}, "groups": {}, "config": {}}
    failed_tasks: list[str] = []

    for task_key in tasks:
        if task_key not in BENCHMARKS:
            log.warning(f"Unknown task: {task_key} — skipping")
            failed_tasks.append(task_key)
            continue

        benchmark   = BENCHMARKS[task_key]
        task_name   = benchmark["task"]
        num_fewshot = num_fewshot_override if num_fewshot_override is not None \
                      else benchmark["num_fewshot"]

        # confirm_run_unsafe_code only needed for tasks that execute model output
        confirm_unsafe = task_key in CODE_EXECUTING_TASKS
        if confirm_unsafe:
            log.info(
                f"Evaluating {task_key} ({num_fewshot}-shot) "
                f"— WILL EXECUTE MODEL-GENERATED CODE"
            )
        else:
            log.info(f"Evaluating {task_key} ({num_fewshot}-shot)...")

        try:
            # When model is an HFLM instance, simple_evaluate ignores
            # device/batch_size kwargs — they're already set on the LM.
            # Keep only the args that actually apply.
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=[task_name],
                num_fewshot=num_fewshot,   # int — required by lm-eval 0.4.x
                limit=limit,
                log_samples=log_samples,
                confirm_run_unsafe_code=confirm_unsafe,
            )
            merged_results["results"].update(results.get("results", {}))
            # MMLU and other group tasks report per-subtask in "results"
            # and aggregate in "groups"; merge both so metric_score can
            # find group-level scores.
            merged_results["groups"].update(results.get("groups", {}))
            merged_results["config"] = results.get("config", {})
        except Exception:
            log.exception(f"Failed to evaluate {task_key}")
            failed_tasks.append(task_key)
            continue

    return merged_results, failed_tasks


def format_results(
    results: dict,
    tasks: list[str],
    model_name: str,
    failed_tasks: list[str],
) -> str:
    """Format evaluation results as a readable table."""
    lines = [
        f"\n{'='*72}",
        f"Evaluation Results — {model_name}",
        f"{'='*72}",
        f"{'Benchmark':<20}  {'Metric':<10}  {'Score':>8}  {'Description'}",
        f"{'-'*72}",
    ]

    task_results  = results.get("results", {})
    group_results = results.get("groups", {})

    for task_key in tasks:
        if task_key not in BENCHMARKS:
            continue
        benchmark = BENCHMARKS[task_key]
        task_name = benchmark["task"]
        metric    = benchmark["metric"]

        if task_key in failed_tasks:
            score_str = "FAILED"
        else:
            # Prefer task-level result; fall back to group-level for tasks
            # like MMLU that may report only at the group level in some
            # lm-eval versions.
            task_result = task_results.get(task_name) or group_results.get(task_name)
            if task_result is None:
                score_str = "N/A"
            else:
                score = metric_score(task_result, metric)
                if isinstance(score, float):
                    score_str = f"{score:.4f}"
                elif score is None:
                    score_str = "N/A"
                else:
                    score_str = str(score)

        lines.append(
            f"  {task_key:<18}  {metric:<10}  {score_str:>8}  {benchmark['description']}"
        )

    if failed_tasks:
        lines.append(f"{'-'*72}")
        lines.append(f"  Failed tasks: {', '.join(failed_tasks)}")

    lines.append(f"{'='*72}\n")
    return "\n".join(lines)


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts non-serializable objects to strings."""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def save_results(
    results: dict,
    model_path: Path,
    tasks: list[str],
    failed_tasks: list[str],
    dtype: str,
) -> Path:
    """Save evaluation results to JSON."""
    model_name = model_display_name(model_path)
    # UTC so filenames sort correctly across machines in different timezones.
    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir    = _eval_results_dir() / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"eval_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model":        str(model_path),
            "model_name":   model_name,
            "tasks":        tasks,
            "failed_tasks": failed_tasks,
            "dtype":        dtype,
            "timestamp":    timestamp,
            "results":      results.get("results", {}),
            "groups":       results.get("groups", {}),
            "config":       results.get("config", {}),
        }, f, indent=2, cls=_SafeEncoder)

    log.info(f"Results saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="SLM Benchmark Evaluation")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=f"Comma-separated tasks, 'all', or 'quick'. Available: {', '.join(ALL_TASKS)}",
    )
    parser.add_argument("--batch-size",  type=int,  default=8)
    parser.add_argument("--num-fewshot", type=int,  default=None,
                        help="Override few-shot count for all tasks")
    parser.add_argument("--device",      type=str,  default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision (default: bfloat16 — matches training)",
    )
    parser.add_argument("--limit",       type=int,  default=None,
                        help="Limit examples per task (for quick testing)")
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Log per-example inputs/outputs/scores (for debugging)",
    )
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

    # Enable code execution ONLY if a code-executing task is being run.
    # Some organisations disable unsafe-code envs by policy; opting in only
    # when needed avoids setting this for eval runs that don't require it.
    if any(t in CODE_EXECUTING_TASKS for t in tasks):
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        log.warning(
            "Code-executing task selected — setting HF_ALLOW_CODE_EVAL=1. "
            "Model-generated code will be run in this process."
        )

    model_name = model_display_name(args.model)
    log.info(f"=== SLM Evaluation ===")
    log.info(f"Model:  {args.model}")
    log.info(f"Tasks:  {tasks}")
    log.info(f"Dtype:  {args.dtype}")

    results, failed_tasks = run_evaluation(
        model_path=args.model,
        tasks=tasks,
        num_fewshot_override=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        limit=args.limit,
        log_samples=args.log_samples,
    )

    print(format_results(results, tasks, model_name, failed_tasks))
    save_results(results, args.model, tasks, failed_tasks, args.dtype)

    # Non-zero exit if any task failed, so CI / Makefile chains catch it.
    if failed_tasks:
        log.error(f"{len(failed_tasks)} task(s) failed: {failed_tasks}")
        sys.exit(1)


if __name__ == "__main__":
    main()