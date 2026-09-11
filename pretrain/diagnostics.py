"""Pretraining diagnostics shared by Trainer callbacks and the existing eval CLI.

Generation reuses inference.generate; diagnostics never select checkpoints.
Only the explicitly final evaluation path opens test.bin.
"""
from __future__ import annotations

import json
import logging
import math
import random
from contextlib import contextmanager
from pathlib import Path

from config.holdout import load_contract, sha256_file, verify_jsonl_contract
from curator.state import atomic_write_json, stable_digest

log = logging.getLogger(__name__)
FIXED_PROMPTS = (
    "The history of computing began",
    "In mathematics, a prime number is",
    "Python functions are defined using",
    "The water cycle describes",
    "A neural network learns by",
)


def probe_settings(config: dict | None = None) -> dict:
    config = config or {}
    settings = {"do_sample": config.get("do_sample", False),
                "max_new_tokens": int(config.get("max_new_tokens", 64))}
    if not isinstance(settings["do_sample"], bool) or settings["max_new_tokens"] < 1:
        raise ValueError("Invalid qualitative probe decoding settings")
    return settings


def unwrap_model(model):
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


@contextmanager
def diagnostic_mode(model):
    """Do not advance training RNGs or leave a training model in eval mode."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    training = model.training
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            model.eval()
            yield
    finally:
        model.train(training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def generate_rows(model, tokenizer, cases: list[dict], *, step: int, settings: dict) -> list[dict]:
    import torch
    from inference.generate import generate
    from inference.utils import resolve_special_token_ids
    model = unwrap_model(model)
    ids = resolve_special_token_ids(tokenizer)
    rows = []
    # Trainer AMP does not cover callbacks. Keep FA3 probes in BF16 while
    # retaining FP32 master weights and restoring the model's training mode.
    use_bf16 = model.device.type == "cuda" and model.config._attn_implementation == "flash_attention_3"
    with diagnostic_mode(model), torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        for case in cases:
            # One prompt at a time bounds diagnostic memory and avoids changing
            # effective training batch size or generation padding semantics.
            completion = generate(model, tokenizer, ids, [case["prompt"]],
                                  chat=False, add_bos=True, **settings)[0]
            rows.append({**case, "checkpoint_step": step,
                         "continuation": completion, "decoding": settings})
    return rows


def generic_cases() -> list[dict]:
    return [{"category": "generic_generalization", "prompt": prompt} for prompt in FIXED_PROMPTS]


def make_probe_callback(tokenizer_dir: Path, output_dir: Path, config: dict):
    from transformers import AutoTokenizer, TrainerCallback
    import torch
    settings = probe_settings(config)
    every = int(config.get("every_steps", 5000))
    selected = {int(step) for step in config.get("steps", [])}
    if every < 0 or any(step <= 0 for step in selected):
        raise ValueError("Probe cadence must use nonnegative every_steps / positive explicit steps")

    class PretrainingProbes(TrainerCallback):
        tokenizer = None
        last_step = None

        def emit(self, state, model):
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            if distributed:
                torch.distributed.barrier()
            try:
                if state.is_world_process_zero and self.last_step != state.global_step:
                    if self.tokenizer is None:
                        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
                    log.info(
                        "Qualitative probe starting: step=%s prompts=%s decoding=%s",
                        state.global_step, len(FIXED_PROMPTS), settings,
                    )
                    rows = generate_rows(model, self.tokenizer, generic_cases(),
                                         step=state.global_step, settings=settings)
                    probe_path = output_dir / "probes" / f"step-{state.global_step:09d}.json"
                    atomic_write_json(probe_path, {"qualitative_only": True, "results": rows})
                    for row in rows:
                        log.info(
                            "Qualitative probe: step=%s prompt=%r continuation=%r decoding=%s",
                            state.global_step, row["prompt"], row["continuation"], row["decoding"],
                        )
                    log.info("Qualitative probe saved: %s", probe_path)
                    self.last_step = state.global_step
            except Exception:
                # A failed qualitative diagnostic is recorded, never a training
                # pass/fail decision. Core training/evaluation failures still raise.
                log.exception("Qualitative probe failed at step %s", state.global_step)
            finally:
                if distributed:
                    torch.distributed.barrier()

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step in selected or (every and state.global_step % every == 0):
                self.emit(state, model)

        def on_train_end(self, args, state, control, model=None, **kwargs):
            if config.get("at_final", True):
                self.emit(state, model)

    return PretrainingProbes()


def validate_final_binding(checkpoint: Path, tokenized_dir: Path, tokenizer_dir: Path) -> dict:
    """An old checkpoint already saw a test set carved later from its train pool."""
    audit_path = checkpoint / "pretrain_run_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Final test evaluation requires the checkpoint's holdout-split training audit")
    audit = json.loads(audit_path.read_text())
    contract = audit.get("contract", {})
    if audit.get("contract_sha256") != stable_digest(contract) or contract.get("contract_version") != 2:
        raise RuntimeError("Checkpoint lacks a valid holdout-split training audit; do not retrofit test provenance")
    from pretrain.train import tokenized_data_identity, tokenizer_fingerprint
    if contract.get("tokenized_data") != tokenized_data_identity(tokenized_dir):
        raise RuntimeError("Final evaluation data differs from this checkpoint's recorded training/holdout contract")
    if contract.get("tokenizer", {}).get("sha256") != tokenizer_fingerprint(tokenizer_dir / "slm_tokenizer.json"):
        raise RuntimeError("Final evaluation tokenizer differs from the trained tokenizer")
    return contract


def final_loss_metrics(trainer, tokenized_dir: Path, seq_len: int) -> dict:
    """Called only after optimization has ended, or by explicit final eval."""
    from pretrain.data.dataset import load_test
    from pretrain.data.tokenize_data import verify_dataset
    verify_dataset(tokenized_dir / "test.bin", tokenized_dir / "test.json")
    val_metrics = trainer.evaluate(metric_key_prefix="final_validation")
    test = load_test(tokenized_dir, seq_len)
    test_metrics = trainer.evaluate(eval_dataset=test, metric_key_prefix="final_test")
    result = {**val_metrics, **test_metrics}
    for category in ("final_validation", "final_test"):
        loss = float(result[f"{category}_loss"])
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite {category} loss: {loss}")
        result[f"{category}_perplexity"] = math.exp(loss) if loss < 709 else None
    result["test_token_accounting"] = test.token_budget()
    result["test_role"] = "final_only"
    return result


def final_cases(validated_dir: Path, tokenized_dir: Path, tokenizer, *, prefix_count: int = 5,
                prefix_tokens: int = 64, qa_path: Path | None = None) -> list[dict]:
    import hashlib
    if prefix_count < 0 or prefix_tokens < 1:
        raise ValueError("Invalid held-out prefix settings")
    holdout = verify_jsonl_contract(validated_dir, include_train=False, stage="validated")
    if holdout["sha256"] != load_contract(tokenized_dir, stage="validated")["sha256"]:
        raise RuntimeError("Validated text does not match the tokenized holdouts")
    cases, qa = [], []
    if qa_path:
        for line in qa_path.read_text().splitlines():
            if line.strip():
                qa.append(json.loads(line))
        if len(qa) > 32:
            raise ValueError("At most 32 qualitative corpus-supported QA probes are supported")
        for row in qa:
            for key in ("prompt", "document_sha256", "evidence", "reference_answer"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Corpus QA needs nonempty {key}")
    matched = set()
    with (validated_dir / "test.jsonl").open() as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            text = row["text"]
            text_sha = hashlib.sha256(text.encode()).hexdigest()
            provenance = {"split": "test", "line": line_no, "source": row["source"],
                          "document_sha256": text_sha, "test_split_sha256": holdout["sha256"]}
            if len(cases) < prefix_count:
                # A character cap bounds even book-length individual documents.
                ids = tokenizer.encode(text[:32768], add_special_tokens=False)
                if len(ids) > 1:
                    length = min(prefix_tokens, max(1, len(ids) // 2))
                    cases.append({"category": "held_out_test_prefix", "prompt": tokenizer.decode(ids[:length]),
                                  "reference_continuation": tokenizer.decode(ids[length:length + prefix_tokens]),
                                  "provenance": provenance})
            for index, question in enumerate(qa):
                if question["document_sha256"] == text_sha:
                    if question["evidence"] not in text:
                        raise RuntimeError("Corpus QA evidence is not present in the referenced held-out document")
                    matched.add(index)
                    question["provenance"] = provenance
            if len(cases) >= prefix_count and len(matched) == len(qa):
                break
    if len(matched) != len(qa):
        raise RuntimeError("Corpus QA references are missing from the holdout test corpus")
    cases.extend({**row, "category": "corpus_supported_qa"} for row in qa)
    return cases


def checkpoint_probe_inputs(checkpoint: Path, *, size=None, dataset_size=None,
                            data_root: Path | None = None, tokenizer_override: Path | None = None):
    """Resolve an intermediate checkpoint's tokenizer without a global fallback.

    Trainer checkpoints may not bundle tokenizers; their parent run audit is
    used to identify the source and validate the tokenizer before generation.
    """
    from config.paths import resolve_dataset_paths
    checkpoint = Path(checkpoint)
    audit_path = checkpoint / "pretrain_run_audit.json"
    if not audit_path.is_file() and checkpoint.name.startswith("checkpoint-"):
        audit_path = checkpoint.parent / "pretrain_run_audit.json"
    saved = {}
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        saved = audit.get("contract", {})
        if not isinstance(saved, dict) or audit.get("contract_sha256") != stable_digest(saved):
            raise RuntimeError("Checkpoint probe audit checksum mismatch")
    candidates = [Path(tokenizer_override)] if tokenizer_override else [checkpoint / "tokenizer", checkpoint]
    if not tokenizer_override and checkpoint.name.startswith("checkpoint-") and saved:
        candidates.append(checkpoint.parent / "final" / "tokenizer")
        source_size = dataset_size or saved.get("dataset_size") or size or saved.get("run_size")
        model_size = size or saved.get("run_size")
        if model_size and source_size:
            candidates.append(resolve_dataset_paths(model_size, source_size, data_root=data_root)["tokenizer"])
    tokenizer_dir = next((path for path in candidates if (path / "tokenizer_config.json").is_file()), None)
    if tokenizer_dir is None:
        raise RuntimeError("No matched checkpoint tokenizer; restore it or supply --tokenizer-dir with the run audit")
    if tokenizer_override and not saved:
        raise RuntimeError("An external tokenizer requires the checkpoint's original training audit")
    return tokenizer_dir, saved


def standalone(args):
    """Extra modes of eval/eval.py, not a second evaluation entry point."""
    import torch
    from config.paths import resolve_dataset_paths
    from transformers import AutoTokenizer, TrainingArguments
    from model import SLMForCausalLM
    tokenizer_dir, saved = checkpoint_probe_inputs(args.model, size=args.size,
        dataset_size=args.dataset_size, data_root=args.data_dir, tokenizer_override=args.tokenizer_dir)
    if saved:
        from pretrain.train import tokenizer_fingerprint
        if tokenizer_fingerprint(tokenizer_dir / "slm_tokenizer.json") != saved.get("tokenizer", {}).get("sha256"):
            raise RuntimeError("Probe tokenizer fingerprint differs from the original training audit")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    model = SLMForCausalLM.from_pretrained(str(args.model), dtype=getattr(torch, args.dtype)).to(args.device)
    step = None
    trainer_state = args.model / "trainer_state.json"
    if args.model.name == "final":
        trainer_state = args.model.parent / "trainer_state.json"
    if trainer_state.is_file():
        step = json.loads(trainer_state.read_text()).get("global_step")
    settings = probe_settings()
    payload = {"model": str(args.model), "mode": args.mode, "qualitative_only": args.mode == "pretrain-probes"}
    cases = generic_cases()
    if args.mode == "pretraining-validation":
        from pretrain.train import SLMTrainer, tokenizer_fingerprint
        from pretrain.data.dataset import PretrainingDataset
        from pretrain.data.tokenize_data import verify_dataset
        size = args.size or saved.get("run_size")
        paths = resolve_dataset_paths(size, args.dataset_size or saved.get("dataset_size") or size,
                                      data_root=args.data_dir)
        meta_path = paths["tokenized"] / "val.json"
        metadata = json.loads(meta_path.read_text())
        if metadata.get("tokenizer_sha256") != tokenizer_fingerprint(tokenizer_dir / "slm_tokenizer.json"):
            raise RuntimeError("Validation binary does not match this checkpoint's tokenizer")
        verify_dataset(paths["tokenized"] / "val.bin", meta_path)
        val = PretrainingDataset(paths["tokenized"] / "val.bin", seq_len=model.config.max_position_embeddings, split="val")
        trainer = SLMTrainer(model=model, args=TrainingArguments(
            output_dir=str(args.model.parent / "validation-runtime"), report_to=[],
            per_device_eval_batch_size=args.batch_size, use_cpu=args.device == "cpu",
            bf16=args.dtype == "bfloat16", fp16=args.dtype == "float16",
            dataloader_num_workers=0), eval_dataset=val)
        metrics = trainer.evaluate(metric_key_prefix="validation")
        loss = float(metrics["validation_loss"])
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite validation loss after checkpoint loading")
        metrics["validation_perplexity"] = math.exp(loss) if loss < 709 else None
        if args.expected_validation_loss is not None:
            if args.validation_loss_tolerance < 0 or abs(loss - args.expected_validation_loss) > args.validation_loss_tolerance:
                raise RuntimeError(f"Validation loss {loss} does not reproduce {args.expected_validation_loss} within tolerance")
        payload["metrics"] = metrics
        payload["test_evaluated"] = False
    elif args.mode == "pretraining-final":
        from pretrain.train import SLMTrainer, resolve_dataset_run_id
        from pretrain.data.dataset import load_train_val
        size = args.size or saved.get("run_size")
        dataset_size = args.dataset_size or saved.get("dataset_size")
        paths = resolve_dataset_paths(size, dataset_size, data_root=args.data_dir)
        contract = validate_final_binding(args.model, paths["tokenized"], tokenizer_dir)
        resolve_dataset_run_id(paths, args.dataset_run_id or contract.get("dataset_run_id"), required=dataset_size != size)
        seq_len = model.config.max_position_embeddings
        # No optimizer or training loop is run. load_train_val preserves the
        # existing dataset validation entry path; the train object is not read.
        _, val = load_train_val(paths["tokenized"], seq_len)
        trainer = SLMTrainer(model=model, args=TrainingArguments(
            output_dir=str(args.model.parent / "final-eval-runtime"), report_to=[],
            per_device_eval_batch_size=args.batch_size, use_cpu=args.device == "cpu",
            bf16=args.dtype == "bfloat16", fp16=args.dtype == "float16",
            dataloader_num_workers=0), eval_dataset=val)
        payload["metrics"] = final_loss_metrics(trainer, paths["tokenized"], seq_len)
        train_metrics = args.model.parent / "train_results.json"
        payload["training_metrics"] = json.loads(train_metrics.read_text()) if train_metrics.is_file() else None
        payload["training_contract_sha256"] = stable_digest(contract)
        cases = final_cases(paths["validated"], paths["tokenized"], tokenizer, qa_path=args.corpus_qa) + cases
        payload["corpus_supported_qa_status"] = "provided" if args.corpus_qa else "not_provided"
    try:
        payload["results"] = generate_rows(model, tokenizer, cases, step=step, settings=settings)
        payload["qualitative_error"] = None
    except Exception as exc:
        log.exception("Qualitative diagnostics failed; preserving completed loss metrics")
        payload["results"], payload["qualitative_error"] = [], str(exc)
    output = args.json_out or args.model.parent / f"{args.mode}.json"
    atomic_write_json(output, payload)
    log.info("Pretraining diagnostics written to %s", output)
