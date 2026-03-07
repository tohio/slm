"""
eval/run_eval.py
----------------
Unified evaluation script for all SLM pipeline stages.

Runs the appropriate eval suite based on the stage being evaluated:

  pretrain  → perplexity on held-out text
  sft       → perplexity + generation samples
  dpo       → perplexity + generation samples + win rate vs SFT reference
  full      → all of the above + MMLU benchmark

Results written to:
  /results/eval/<stage>/<timestamp>/
    summary.json      ← all metrics, machine-readable
    summary.txt       ← printed human-readable summary

Usage:
  python eval/run_eval.py --stage pretrain --checkpoint /results/slm_gpt_125m/checkpoints/last.nemo
  python eval/run_eval.py --stage sft      --checkpoint /results/slm_sft_code/checkpoints/last.nemo
  python eval/run_eval.py --stage dpo      --checkpoint /results/slm_dpo/checkpoints/last.nemo \\
                                           --ref-checkpoint /results/slm_sft_code/checkpoints/last.nemo
  python eval/run_eval.py --stage full     --checkpoint /results/slm_dpo/checkpoints/last.nemo
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eval")

from perplexity import evaluate_perplexity
from generation import evaluate_generation
from mmlu import evaluate_mmlu
from win_rate import evaluate_win_rate


STAGE_SUITES = {
    "pretrain": ["perplexity"],
    "sft":      ["perplexity", "generation"],
    "dpo":      ["perplexity", "generation", "win_rate"],
    "full":     ["perplexity", "generation", "win_rate", "mmlu"],
}


def print_summary(results: dict, stage: str):
    """Print a clean human-readable summary to stdout."""
    width = 60
    print("\n" + "=" * width)
    print(f"  SLM EVALUATION SUMMARY — Stage: {stage.upper()}")
    print("=" * width)

    if "perplexity" in results:
        r = results["perplexity"]
        print(f"\n{'PERPLEXITY':}")
        print(f"  Val perplexity:     {r.get('val_perplexity', 'N/A'):.3f}")
        print(f"  Val loss:           {r.get('val_loss', 'N/A'):.4f}")
        print(f"  Tokens evaluated:   {r.get('tokens_evaluated', 0):,}")

    if "generation" in results:
        r = results["generation"]
        print(f"\n{'GENERATION SAMPLES':}")
        for i, sample in enumerate(r.get("samples", []), 1):
            print(f"\n  [{i}] Prompt: {sample['prompt'][:80]}{'...' if len(sample['prompt']) > 80 else ''}")
            print(f"      Response: {sample['response'][:200]}{'...' if len(sample['response']) > 200 else ''}")

    if "mmlu" in results:
        r = results["mmlu"]
        print(f"\n{'MMLU BENCHMARK':}")
        print(f"  Overall accuracy:   {r.get('overall_accuracy', 0):.1%}")
        for subject, acc in r.get("by_subject", {}).items():
            print(f"  {subject:<30} {acc:.1%}")

    if "win_rate" in results:
        r = results["win_rate"]
        print(f"\n{'WIN RATE vs SFT REFERENCE':}")
        print(f"  Win rate:           {r.get('win_rate', 0):.1%}")
        print(f"  Tie rate:           {r.get('tie_rate', 0):.1%}")
        print(f"  Loss rate:          {r.get('loss_rate', 0):.1%}")
        print(f"  Pairs evaluated:    {r.get('n_pairs', 0)}")
        print(f"  Judge:              {r.get('judge', 'N/A')}")

    print("\n" + "=" * width + "\n")


def main():
    parser = argparse.ArgumentParser(description="SLM Evaluation Suite")
    parser.add_argument("--stage",          required=True, choices=list(STAGE_SUITES.keys()))
    parser.add_argument("--checkpoint",     required=True, help="Path to .nemo checkpoint to evaluate")
    parser.add_argument("--ref-checkpoint", help="Reference checkpoint for win rate eval (SFT checkpoint)")
    parser.add_argument("--output-dir",     default="/results/eval")
    parser.add_argument("--val-data",       default="/data/pretrain", help="Validation data for perplexity")
    parser.add_argument("--n-samples",      type=int, default=10, help="Number of generation samples")
    parser.add_argument("--n-mmlu",         type=int, default=100, help="MMLU questions per subject")
    parser.add_argument("--n-pairs",        type=int, default=200, help="Preference pairs for win rate")
    parser.add_argument("--device",         default="cuda")
    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Win rate requires ref checkpoint
    suites = STAGE_SUITES[args.stage]
    if "win_rate" in suites and not args.ref_checkpoint:
        logger.error("--ref-checkpoint required for win_rate eval (pass the SFT checkpoint)")
        sys.exit(1)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / args.stage / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating stage: {args.stage}")
    logger.info(f"Checkpoint:       {args.checkpoint}")
    logger.info(f"Suites:           {suites}")
    logger.info(f"Output:           {out_dir}")

    results = {
        "stage":      args.stage,
        "checkpoint": args.checkpoint,
        "timestamp":  timestamp,
        "metrics":    {},
    }

    # ── Run each eval suite ───────────────────────────────────────────────────
    if "perplexity" in suites:
        logger.info("Running perplexity evaluation...")
        results["metrics"]["perplexity"] = evaluate_perplexity(
            checkpoint=args.checkpoint,
            val_data_dir=args.val_data,
            device=args.device,
        )

    if "generation" in suites:
        logger.info("Running generation evaluation...")
        results["metrics"]["generation"] = evaluate_generation(
            checkpoint=args.checkpoint,
            n_samples=args.n_samples,
            device=args.device,
        )

    if "mmlu" in suites:
        logger.info("Running MMLU evaluation...")
        results["metrics"]["mmlu"] = evaluate_mmlu(
            checkpoint=args.checkpoint,
            n_per_subject=args.n_mmlu,
            device=args.device,
        )

    if "win_rate" in suites:
        logger.info("Running win rate evaluation...")
        results["metrics"]["win_rate"] = evaluate_win_rate(
            policy_checkpoint=args.checkpoint,
            ref_checkpoint=args.ref_checkpoint,
            n_pairs=args.n_pairs,
            device=args.device,
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {json_path}")

    # Print summary and also capture to txt
    import io
    from contextlib import redirect_stdout
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_summary(results["metrics"], args.stage)
    summary_text = buffer.getvalue()

    print(summary_text)

    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Timestamp:  {timestamp}\n")
        f.write(summary_text)

    logger.info(f"Summary written to {txt_path}")


if __name__ == "__main__":
    main()
