"""Compatibility checks for the pinned Transformers/TRL training stack."""

from pathlib import Path

from alignment.train_dpo import build_dpo_args
from finetune.train_sft import build_sft_args


def _training() -> dict:
    return {
        "epochs": 1,
        "max_steps": 2,
        "micro_batch_size": 2,
        "eval_micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_ratio_recipe": 0.5,
        "eval_steps": 1,
        "save_steps": 1,
        "report_to": "none",
        "num_workers": 0,
        "precision": "bf16",
        "group_by_length": True,
    }


def test_sft_config_uses_current_length_sampler(tmp_path: Path):
    cfg = {
        "name": "sft-smoke",
        "training": _training(),
        "optimizer": {"lr": "1.0e-5"},
        "data": {"packing": False},
        "model": {"max_seq_length": 64},
    }

    args = build_sft_args(cfg, tmp_path, num_train_examples=8)

    assert args.train_sampling_strategy == "group_by_length"
    assert args.length_column_name == "length"
    assert args.optim.value == "adamw_torch"
    assert args.learning_rate == 1.0e-5
    assert args.assistant_only_loss is True
    assert args.loss_type == "chunked_nll"
    assert args.warmup_steps == 1


def test_dpo_config_coerces_yaml_numeric_values(tmp_path: Path):
    cfg = {
        "name": "dpo-smoke",
        "training": _training(),
        "optimizer": {
            "lr": "5.0e-7",
            "weight_decay": "0.01",
            "beta1": "0.9",
            "beta2": "0.98",
        },
        "dpo": {
            "beta": 0.1,
            "loss_type": "sigmoid",
            "precompute_ref_log_probs": True,
        },
        "model": {"max_seq_length": 64},
    }

    args = build_dpo_args(
        cfg,
        tmp_path,
        beta=0.1,
        num_train_examples=8,
    )

    assert args.optim.value == "adamw_torch"
    assert args.learning_rate == 5.0e-7
    assert args.weight_decay == 0.01
    assert args.adam_beta1 == 0.9
    assert args.adam_beta2 == 0.98
    assert args.max_length == 64
    assert args.loss_type == ["sigmoid"]
    assert args.precompute_ref_log_probs is True
    assert args.remove_unused_columns is True
    assert args.warmup_steps == 1
