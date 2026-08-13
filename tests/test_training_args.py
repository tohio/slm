"""Compatibility checks for the pinned Transformers/TRL training stack."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from alignment.train_dpo import build_dpo_args
from config.data_mix import ALL_SOURCES
from finetune.train_sft import build_sft_args
from pretrain.data.mixture import build_realized_mixture_report
from pretrain.schedule import resolve_realized_token_schedule
from pretrain.train import (
    resolve_distributed_strategy,
    resolve_pretrain_checkpoint,
    tokenized_data_identity,
    validate_model_tokenizer_contract,
    validate_or_write_pretrain_audit,
    validate_preflight_gpu,
    validate_tokenizer,
)


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


def test_realized_token_schedule_replaces_planning_steps():
    cfg = {
        "training": {
            "schedule_from_realized_tokens": True,
            "max_steps": 1000,
            "warmup_steps": 100,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
        }
    }

    resolved, schedule = resolve_realized_token_schedule(
        cfg,
        run_size="125m",
        realized_train_tokens=10_001,
        seq_len=100,
        world_size=2,
    )

    # 10,000 usable tokens × 2 epochs / (2 × 4 × 2 × 100) = 12.5 steps.
    assert resolved["training"]["max_steps"] == 13
    assert resolved["training"]["warmup_steps"] == 1
    assert schedule["target_consumed_tokens"] == 20_000
    assert schedule["scheduled_tokens"] == 20_800
    assert schedule["rounding_excess_tokens"] == 800
    assert schedule["tokens_discarded_by_sequence_packing"] == 1
    assert cfg["training"]["max_steps"] == 1000


def test_realized_token_schedule_leaves_bounded_recipe_unchanged():
    cfg = {
        "training": {
            "max_steps": 8,
            "warmup_steps": 2,
            "micro_batch_size": 1,
        }
    }

    resolved, schedule = resolve_realized_token_schedule(
        cfg,
        run_size="mini",
        realized_train_tokens=1_000_000,
        seq_len=1024,
        world_size=1,
    )

    assert resolved == cfg
    assert resolved is not cfg
    assert schedule is None


def test_realized_token_schedule_rejects_unusable_corpus():
    cfg = {
        "training": {
            "schedule_from_realized_tokens": True,
            "max_steps": 8,
            "warmup_steps": 1,
            "micro_batch_size": 1,
        }
    }

    with pytest.raises(ValueError, match="cannot form one"):
        resolve_realized_token_schedule(
            cfg,
            run_size="125m",
            realized_train_tokens=100,
            seq_len=1024,
            world_size=1,
        )


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


def test_pretrain_tokenizer_validation_uses_explicit_tokenized_dir(
    tmp_path: Path,
):
    tokenizer_dir = tmp_path / "active-tokenizer"
    tokenized_dir = tmp_path / "runs" / "125m" / "tokenized"
    tokenizer_dir.mkdir()
    tokenized_dir.mkdir(parents=True)

    tokenizer_path = tokenizer_dir / "slm_tokenizer.json"
    tokenizer = Tokenizer(
        WordLevel({"<UNK>": 0, "hello": 1}, unk_token="<UNK>")
    )
    tokenizer.save(str(tokenizer_path))
    (tokenizer_dir / "tokenizer_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    canonical = Tokenizer.from_file(str(tokenizer_path)).to_str()
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (tokenized_dir / "train.json").write_text(
        json.dumps({"tokenizer_sha256": fingerprint}) + "\n",
        encoding="utf-8",
    )

    validate_tokenizer(tokenizer_dir, tokenized_dir)


def test_pretrain_resume_requires_a_checkpoint(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Refusing to start from scratch"):
        resolve_pretrain_checkpoint(tmp_path, resume=True)


def test_pretrain_resume_selects_latest_checkpoint(tmp_path: Path):
    (tmp_path / "checkpoint-20").mkdir()
    expected = tmp_path / "checkpoint-120"
    expected.mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert resolve_pretrain_checkpoint(tmp_path, resume=True) == expected


@pytest.mark.parametrize("artifact", ["checkpoint-20", "final", "events.log"])
def test_new_pretrain_run_rejects_existing_training_artifacts(
    tmp_path: Path,
    artifact: str,
):
    path = tmp_path / artifact
    if "." in artifact:
        path.write_text("partial run\n", encoding="utf-8")
    else:
        path.mkdir()

    with pytest.raises(RuntimeError, match="already contains training artifacts"):
        resolve_pretrain_checkpoint(tmp_path, resume=False)


def test_pretrain_model_tokenizer_contract_rejects_special_token_mismatch(
    tmp_path: Path,
):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<PAD>": 0,
                "<UNK>": 1,
                "<EOS>": 2,
                "<BOS>": 3,
            },
            unk_token="<UNK>",
        )
    )
    tokenizer.save(str(tokenizer_dir / "slm_tokenizer.json"))
    model_config = SimpleNamespace(
        vocab_size=4,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )

    with pytest.raises(RuntimeError, match="special-token mismatch"):
        validate_model_tokenizer_contract(model_config, tokenizer_dir)


def test_tokenized_data_identity_captures_manifest_and_splits(tmp_path: Path):
    completion = {
        "manifest_version": 2,
        "contract_sha256": "contract",
        "input_signature": "input",
        "output_signature": "output",
    }
    (tmp_path / "_SUCCESS.json").write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    split_metadata = {}
    for split, multiplier in (("train", 2), ("val", 1)):
        source_counts = {
            source: {"documents": multiplier, "tokens": multiplier}
            for source in ALL_SOURCES
        }
        tokens = sum(row["tokens"] for row in source_counts.values())
        metadata = {
            "n_tokens": tokens,
            "n_docs": sum(row["documents"] for row in source_counts.values()),
            "bos_id": 2,
            "eos_id": 3,
            "dtype": "uint16",
            "format_version": "test",
            "input_sha256": f"{split}-input",
            "tokenizer_sha256": "tokenizer",
            "implementation_sha256": "implementation",
            "source_counts": source_counts,
        }
        split_metadata[split] = metadata
        (tmp_path / f"{split}.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        (tmp_path / f"{split}.bin").write_bytes(b"\0\0" * tokens)

    mixture = build_realized_mixture_report(
        split_metadata["train"], split_metadata["val"]
    )
    (tmp_path / "token_mixture.json").write_text(
        json.dumps(mixture),
        encoding="utf-8",
    )

    identity = tokenized_data_identity(tmp_path)

    assert identity["manifest"] == completion
    assert identity["splits"]["train"]["binary_bytes"] == 4 * len(ALL_SOURCES)
    assert identity["splits"]["val"]["n_tokens"] == len(ALL_SOURCES)
    assert identity["realized_mixture"]["status"] == (
        "passed_structural_checks_report_only"
    )


def test_pretrain_audit_rejects_changed_resume_contract(tmp_path: Path):
    contract = {
        "contract_version": 1,
        "resolved_config_sha256": "config-a",
        "distributed": {"world_size": 1, "strategy": "single"},
    }
    validate_or_write_pretrain_audit(
        tmp_path,
        contract,
        resume=False,
        write=True,
    )
    validate_or_write_pretrain_audit(
        tmp_path,
        contract,
        resume=True,
        write=False,
    )

    changed = {
        **contract,
        "distributed": {"world_size": 2, "strategy": "ddp"},
    }
    with pytest.raises(RuntimeError, match="distributed.strategy"):
        validate_or_write_pretrain_audit(
            tmp_path,
            changed,
            resume=True,
            write=False,
        )


def test_pretrain_resume_requires_provenance_audit(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Cannot resume without"):
        validate_or_write_pretrain_audit(
            tmp_path,
            {"contract_version": 1},
            resume=True,
            write=False,
        )


@pytest.mark.parametrize(
    ("requested", "world_size", "expected"),
    [
        (None, 1, "single"),
        (None, 4, "ddp"),
        ("ddp", 2, "ddp"),
        ("fsdp", 8, "fsdp"),
    ],
)
def test_distributed_strategy_resolution(
    monkeypatch,
    requested: str | None,
    world_size: int,
    expected: str,
):
    monkeypatch.delenv("SLM_DISTRIBUTED_STRATEGY", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    assert resolve_distributed_strategy(requested, world_size) == expected


def test_pretrain_gpu_preflight_rejects_missing_cuda(monkeypatch):
    monkeypatch.setattr(
        "pretrain.train.torch.cuda.is_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="requires CUDA"):
        validate_preflight_gpu(expected_gpus=1, precision="bf16")


def test_pretrain_gpu_preflight_accepts_requested_bf16_devices(monkeypatch):
    monkeypatch.setattr(
        "pretrain.train.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "pretrain.train.torch.cuda.device_count",
        lambda: 4,
    )
    monkeypatch.setattr(
        "pretrain.train.torch.cuda.is_bf16_supported",
        lambda: True,
    )

    validate_preflight_gpu(expected_gpus=4, precision="bf16")
