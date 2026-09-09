#!/usr/bin/env python3
"""Model-learning control on pinned HF data or existing read-only token batches.

--stage check compares local SLM with a native Transformers Llama initialized
from exactly the same learned tensors. --stage train uses the production
SLMTrainer on the selected inputs. This is evidence, not an automatic diagnosis
of data quality. See scripts/README.md; all writes stay in --run-dir.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrain_hf_125m import (
    atomic_write_json, stable_digest, model_config, state_digest,
    implementation_identity, versions, select_device,
)

log = logging.getLogger(__name__)


def compare_implementations(args, cfg, directory, tokenizer, data_identity):
    """Bounded forward/backward/update comparison, not a production export.

    FP32 without dropout/compile isolates arithmetic from stochastic masks and
    precision. The chosen full model dimensions are preserved. Short unpadded
    contexts come from the actual tokenized training stream. Both optimizers
    use the production Trainer's parameter grouping and AdamW configuration.
    """
    from dataclasses import replace
    import torch
    import torch.nn.functional as F
    from transformers import set_seed
    from model import SLMForCausalLM
    from export.export import _convert_to_native_llama
    from pretrain.data.dataset import PretrainingDataset
    from pretrain.train import SLMTrainer, build_training_args

    report_dir = args.run_dir / "checks"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / "implementation_check.json"
    if report_file.exists():
        raise RuntimeError(f"Comparison report already exists: {report_file}. Preserve it and use a new --run-dir for another check.")
    device = select_device(args.device)
    old_precision = torch.get_float32_matmul_precision()
    old_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = model_config(cfg, tokenizer)
    if config.attention_dropout != 0:
        log.info("Numerical comparison uses eval mode to disable attention dropout")
    length = min(args.check_seq_len, config.max_position_embeddings)
    dataset = PretrainingDataset(directory / "train.bin", seq_len=length, split="train",
                                 max_tokens=data_identity["selected_train_tokens"])
    if len(dataset) < args.check_batches:
        raise ValueError("Too few complete contexts for the requested check batches")
    set_seed(args.seed)
    source = SLMForCausalLM(config).float()
    initial_sha = state_digest(source)
    native = _convert_to_native_llama(source, tokenizer, torch.float32)
    if state_digest(native) != initial_sha:
        raise RuntimeError("Reference model did not receive identical initial tensors")
    source, native = source.to(device), native.to(device)
    source.eval()
    native.eval()
    models = {"slm": source, "llama": native}
    parameters = {kind: dict(model.named_parameters()) for kind, model in models.items()}
    if parameters["slm"].keys() != parameters["llama"].keys():
        raise RuntimeError("Unique learned parameter names differ (including tied-weight semantics)")

    # Use real Trainer instances for production parameter grouping/AdamW setup,
    # but do not call train(), initialize W&B, or publish models in this check.
    check_cfg = copy.deepcopy(cfg)
    check_cfg["training"].update({"precision": "fp32", "torch_compile": False, "gradient_checkpointing": False,
                                  "report_to": [], "max_steps": 1, "warmup_steps": 0,
                                  "torch_compile_backend": None, "torch_compile_mode": None})
    optimizers = {}
    trainers = {}
    for kind, model in models.items():
        training_args = replace(
            build_training_args(check_cfg, report_dir / kind, resume=False),
            use_cpu=device == "cpu", bf16=False, fp16=False, tf32=False,
            torch_compile=False, torch_compile_backend=None, torch_compile_mode=None,
            eval_strategy="no", save_strategy="no",
            optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        )
        if training_args.device.type != device:
            raise RuntimeError("Trainer device differs from the requested control device; use CUDA_VISIBLE_DEVICES='' for a CPU check")
        trainers[kind] = SLMTrainer(model=model, args=training_args)
        model.eval()
        optimizers[kind] = trainers[kind].create_optimizer()
    groups = lambda opt, params: [
        {"parameters": sorted(name for name, p in params.items() if any(p is q for q in group["params"])),
         "weight_decay": group["weight_decay"], "lr": group["lr"]}
        for group in opt.param_groups
    ]
    if groups(optimizers["slm"], parameters["slm"]) != groups(optimizers["llama"], parameters["llama"]):
        raise RuntimeError("SLM/native optimizer parameter grouping differs")

    checks = []
    failures = []

    def compare(label, actual, expected):
        if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            failures.append(label)
            checks.append({"check": label, "passed": False, "reason": "shape or non-finite values"})
            return
        diff = (actual.detach().float() - expected.detach().float()).abs()
        passed = torch.allclose(actual, expected, rtol=args.rtol, atol=args.atol)
        checks.append({"check": label, "passed": passed, "max_abs_difference": diff.max().item() if diff.numel() else 0.0})
        if not passed:
            failures.append(label)

    try:
        for iteration in range(args.check_batches):
            # Spread bounded samples over the selected training stream rather
            # than checking only its first document.
            index = iteration * (len(dataset) - 1) // max(args.check_batches - 1, 1)
            inputs = dataset[index]["input_ids"].unsqueeze(0).to(device)
            labels = inputs.clone()  # production contract: shift inside loss
            mask = torch.ones_like(inputs)
            boundary = length // 2
            changed = inputs.clone()
            changed[:, boundary:] = (changed[:, boundary:] + 17) % config.vocab_size
            outputs = {}
            for kind, model in models.items():
                optimizers[kind].zero_grad(set_to_none=True)
                batch = {"input_ids": inputs, "attention_mask": mask, "labels": labels, "use_cache": False}
                item_count = trainers[kind]._get_num_items_in_batch([batch], torch.device(device))
                if item_count is None:
                    raise RuntimeError(f"{kind}: Trainer did not count causal targets")
                compare(f"batch{iteration}/{kind}/trainer_target_count", torch.as_tensor(item_count, device=device), labels[:, 1:].ne(-100).sum())
                _, output = trainers[kind].compute_loss(model, batch, return_outputs=True, num_items_in_batch=item_count)
                manual = F.cross_entropy(output.logits[:, :-1].float().reshape(-1, config.vocab_size), labels[:, 1:].reshape(-1))
                compare(f"batch{iteration}/{kind}/next_token_loss", output.loss, manual)
                with torch.no_grad():
                    altered = model(input_ids=changed, attention_mask=mask, use_cache=False).logits
                    compare(f"batch{iteration}/{kind}/causality", output.logits[:, :boundary], altered[:, :boundary])
                    counted = model(input_ids=inputs, attention_mask=mask, labels=labels, use_cache=False,
                                    num_items_in_batch=labels[:, 1:].ne(-100).sum()).loss
                    compare(f"batch{iteration}/{kind}/counted_next_token_loss", counted, manual)
                outputs[kind] = output
                output.loss.backward()
            compare(f"batch{iteration}/logits", outputs["slm"].logits, outputs["llama"].logits)
            compare(f"batch{iteration}/loss", outputs["slm"].loss, outputs["llama"].loss)
            for name, parameter in parameters["slm"].items():
                other = parameters["llama"][name]
                if parameter.grad is None or other.grad is None:
                    failures.append(f"batch{iteration}/missing_gradient/{name}")
                else:
                    compare(f"batch{iteration}/gradient/{name}", parameter.grad, other.grad)
            before = {name: value.detach().clone() for name, value in parameters["slm"].items()}
            for kind, model in models.items():
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("gradient_clip_val", 1.0)))
                optimizers[kind].step()
            changed_parameters = 0
            for name, parameter in parameters["slm"].items():
                compare(f"batch{iteration}/updated_weight/{name}", parameter, parameters["llama"][name])
                changed_parameters += not torch.equal(parameter, before[name])
            if not changed_parameters:
                failures.append(f"batch{iteration}/no_parameter_updates")
            del outputs, before
        report = {"status": "failed" if failures else "passed", "failures": failures, "checks": checks,
            "model_config": config.to_dict(), "initial_state_sha256": initial_sha, "data": data_identity,
            "config": cfg, "seed": args.seed, "dtype": "float32", "device": device,
            "context_length_tested": length, "batches_tested": args.check_batches,
            "rtol": args.rtol, "atol": args.atol, "versions": versions(), "implementation": implementation_identity(),
            "limits": "Bounded unpadded FP32 eval-mode forward/backward/AdamW agreement. Not full-context, compiled, BF16, distributed, or capability acceptance."}
        atomic_write_json(report_file, report)
        if failures:
            raise RuntimeError(f"Implementation comparison failed ({len(failures)} checks). Inspect {report_file}; do not tune tolerances blindly.")
        log.info("SLM/native implementation check PASSED; report: %s", report_file)
        return report
    finally:
        # No models/checkpoints from the numerical check are retained or published.
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_cuda_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


if __name__ == "__main__":
    from scripts.pretrain_hf_125m import main
    main(sanity=True)
