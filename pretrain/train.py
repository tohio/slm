"""
pretrain/train.py
-----------------
Pretraining entry point using HuggingFace Trainer.
"""

import argparse
import json
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path
from tokenizers import Tokenizer
from transformers import Trainer, TrainerCallback

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

def _is_rank_zero() -> bool:
    """True only on global rank 0 under torchrun/accelerate DDP."""
    return int(os.environ.get("RANK", "0")) == 0



DATA_DIR    = Path(os.environ.get("DATA_DIR", "data"))
from config.paths import pretrain_dir, tokenized_dir as run_tokenized_dir, BASE_RESULTS_DIR
from config.runtime import configure_torch_runtime
from curator.state import atomic_write_json, stable_digest
from pretrain.data.mixture import validate_realized_mixture_report
from pretrain.schedule import resolve_realized_token_schedule

RESULTS_DIR = BASE_RESULTS_DIR
PRETRAIN_AUDIT_FILENAME = "pretrain_run_audit.json"
PRETRAIN_AUDIT_VERSION = 1


_NUMERIC_CONFIG_KEYS = {
    "lr", "eps", "weight_decay", "beta1", "beta2",
    "gradient_clip_val", "warmup_ratio",
    "rms_norm_eps", "rope_theta", "initializer_range",
    "dpo_beta", "beta",
}


def _coerce_numeric(node):
    if isinstance(node, dict):
        return {
            k: (float(v) if k in _NUMERIC_CONFIG_KEYS and isinstance(v, str) else _coerce_numeric(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_coerce_numeric(x) for x in node]
    return node


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return _coerce_numeric(cfg)



def tokenizer_fingerprint(tokenizer_path: Path) -> str:
    """Return the same tokenizer fingerprint used by tokenization metadata."""
    tok = Tokenizer.from_file(str(tokenizer_path))
    return hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()


def validate_active_tokenizer_matches_tokenized_data(tokenized_dir: Path, tokenizer_dir: Path) -> None:
    """Fail fast if tokenized train.bin was produced with a different tokenizer."""
    train_meta_path = tokenized_dir / "train.json"
    active_tokenizer_path = tokenizer_dir / "slm_tokenizer.json"

    if not train_meta_path.exists():
        raise FileNotFoundError(f"Missing tokenized metadata: {train_meta_path}")

    if not active_tokenizer_path.exists():
        raise FileNotFoundError(f"Missing active tokenizer: {active_tokenizer_path}")

    with train_meta_path.open("r", encoding="utf-8") as f:
        train_meta = json.load(f)

    expected = train_meta.get("tokenizer_sha256")
    if not expected:
        raise RuntimeError(
            f"{train_meta_path} does not contain tokenizer_sha256; refusing to train."
        )

    actual = tokenizer_fingerprint(active_tokenizer_path)

    if actual != expected:
        raise RuntimeError(
            "Tokenizer/tokenized artifact mismatch.\n"
            f"Tokenized metadata: {train_meta_path}\n"
            f"Expected tokenizer fingerprint: {expected}\n"
            f"Active tokenizer: {active_tokenizer_path}\n"
            f"Actual tokenizer fingerprint:   {actual}\n"
            "Refusing to train. Restore the size-specific tokenizer that produced train.bin."
        )

    log.info("Tokenizer fingerprint matches tokenized training metadata: %s", actual)


def validate_tokenizer(tokenizer_dir: Path, tokenized_dir: Path) -> None:
    if not tokenizer_dir.exists() or not any(tokenizer_dir.iterdir()):
        raise RuntimeError(
            f"Tokenizer directory missing or empty: {tokenizer_dir}\n"
            f"Retrain with: make tokenizer SIZE=<size>\n"
            f"Or restore with: make artifacts-download SIZE=<size> "
            f"RUN_ID=<run-id> ARTIFACT_STAGES=tokenizer"
        )

    required_files = {
        "tokenizer_config.json": (
            "Contains the baked-in chat_template required by train_sft.py. "
            "Retrain the tokenizer: make tokenizer"
        ),
        "slm_tokenizer.json": (
            "Raw BPE tokenizer required by tokenize_data.py. "
            "Retrain the tokenizer: make tokenizer"
        ),
    }

    for filename, hint in required_files.items():
        path = tokenizer_dir / filename
        if not path.exists():
            raise RuntimeError(f"Missing tokenizer file: {path}\n{hint}")
        try:
            with open(path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Tokenizer file is not valid JSON: {path}\n"
                f"  Error: {e}\n"
                f"  Hint: {hint}"
            ) from e

    log.info(f"Tokenizer validated at {tokenizer_dir}")
    validate_active_tokenizer_matches_tokenized_data(tokenized_dir, tokenizer_dir)


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not candidates:
        return None
    numbered = []
    for p in candidates:
        try:
            step = int(p.name.split("-", 1)[1])
            numbered.append((step, p))
        except (IndexError, ValueError):
            continue
    if not numbered:
        return None
    numbered.sort()
    return numbered[-1][1]


def resolve_pretrain_checkpoint(
    output_dir: Path,
    *,
    resume: bool,
) -> Path | None:
    """Resolve a safe start state without silently mixing training runs."""
    latest_checkpoint = _find_latest_checkpoint(output_dir)

    if resume:
        if latest_checkpoint is None:
            raise RuntimeError(
                f"--resume was requested but no checkpoint-* directory exists "
                f"in {output_dir}. Refusing to start from scratch. Use the "
                f"normal pretrain target for a new run or restore the expected "
                f"checkpoint."
            )
        return latest_checkpoint

    if output_dir.exists():
        existing_artifacts = sorted(
            path
            for path in output_dir.iterdir()
            if path.name != PRETRAIN_AUDIT_FILENAME
        )
        if existing_artifacts:
            rendered = "\n  ".join(str(path) for path in existing_artifacts)
            raise RuntimeError(
                "A new pretraining run was requested in an output directory "
                "that already contains training artifacts:\n  "
                f"{rendered}\n"
                "Use --resume for the interrupted run or choose a different "
                "RESULTS_DIR for a new run."
            )

    return None


def validate_model_tokenizer_contract(model_config, tokenizer_dir: Path) -> None:
    """Validate vocabulary and special-token IDs without allocating a model."""
    tokenizer_path = tokenizer_dir / "slm_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

    if tokenizer_vocab_size != model_config.vocab_size:
        raise RuntimeError(
            f"Model/tokenizer vocabulary mismatch: model config expects "
            f"{model_config.vocab_size:,} tokens but {tokenizer_path} contains "
            f"{tokenizer_vocab_size:,}."
        )

    required_tokens = {
        "pad_token_id": "<PAD>",
        "bos_token_id": "<BOS>",
        "eos_token_id": "<EOS>",
    }
    for name, token in required_tokens.items():
        token_id = getattr(model_config, name)
        if token_id is None or not 0 <= token_id < tokenizer_vocab_size:
            raise RuntimeError(
                f"Model config {name}={token_id!r} is outside tokenizer "
                f"vocabulary size {tokenizer_vocab_size:,}."
            )
        tokenizer_token_id = tokenizer.token_to_id(token)
        if tokenizer_token_id != token_id:
            raise RuntimeError(
                f"Model/tokenizer special-token mismatch: model config "
                f"{name}={token_id}, but tokenizer token {token!r} has ID "
                f"{tokenizer_token_id!r}."
            )


def validate_preflight_gpu(expected_gpus: int, precision: str) -> None:
    """Fail before optimization when the requested GPU contract is unavailable."""
    if expected_gpus < 1:
        raise ValueError(f"expected_gpus must be >= 1, got {expected_gpus}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Pretraining preflight requires CUDA for GPUS={expected_gpus}, "
            "but torch.cuda.is_available() is false."
        )

    visible_gpus = torch.cuda.device_count()
    if visible_gpus < expected_gpus:
        raise RuntimeError(
            f"Pretraining requested {expected_gpus} GPU(s), but only "
            f"{visible_gpus} are visible."
        )
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The generated pretraining config requires BF16 support.")

    log.info(
        "GPU preflight passed: requested=%d, visible=%d, precision=%s",
        expected_gpus,
        visible_gpus,
        precision,
    )


def resolve_distributed_strategy(
    requested: str | None,
    world_size: int,
) -> str:
    """Resolve the topology without changing Accelerate's launch behavior."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    strategy = requested or os.environ.get("SLM_DISTRIBUTED_STRATEGY")
    if not strategy:
        if world_size == 1:
            strategy = "single"
        elif os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true":
            strategy = "fsdp"
        else:
            strategy = "ddp"

    if strategy not in {"single", "ddp", "fsdp"}:
        raise ValueError(
            f"distributed strategy must be single, ddp, or fsdp; got {strategy!r}"
        )
    if strategy == "single" and world_size != 1:
        raise RuntimeError(
            f"single-process topology cannot use world_size={world_size}"
        )
    if strategy in {"ddp", "fsdp"} and world_size < 2:
        raise RuntimeError(
            f"{strategy} topology requires at least two processes"
        )
    return strategy


def _read_required_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def tokenized_data_identity(tokenized_dir: Path) -> dict:
    """Return a portable identity for the manifest-complete tokenized corpus."""
    completion = _read_required_json(
        tokenized_dir / "_SUCCESS.json",
        "tokenized completion manifest",
    )
    required_manifest_fields = (
        "manifest_version",
        "contract_sha256",
        "input_signature",
        "output_signature",
    )
    missing = [
        field for field in required_manifest_fields
        if completion.get(field) in (None, "")
    ]
    if missing:
        raise RuntimeError(
            f"{tokenized_dir / '_SUCCESS.json'} is missing required fields: "
            f"{missing}"
        )

    splits = {}
    split_metadata = {}
    for split in ("train", "val"):
        metadata_path = tokenized_dir / f"{split}.json"
        binary_path = tokenized_dir / f"{split}.bin"
        metadata = _read_required_json(
            metadata_path,
            f"tokenized {split} metadata",
        )
        required_metadata_fields = (
            "n_tokens",
            "n_docs",
            "bos_id",
            "eos_id",
            "dtype",
            "format_version",
            "input_sha256",
            "tokenizer_sha256",
            "implementation_sha256",
        )
        missing_metadata = [
            field for field in required_metadata_fields
            if metadata.get(field) in (None, "")
        ]
        if missing_metadata:
            raise RuntimeError(
                f"{metadata_path} is missing required fields: "
                f"{missing_metadata}"
            )
        if not binary_path.is_file():
            raise FileNotFoundError(f"Missing tokenized {split} binary: {binary_path}")
        split_metadata[split] = metadata
        splits[split] = {
            "metadata_sha256": stable_digest(metadata),
            "binary_bytes": binary_path.stat().st_size,
            "n_tokens": metadata["n_tokens"],
            "n_docs": metadata["n_docs"],
            "bos_id": metadata["bos_id"],
            "eos_id": metadata["eos_id"],
            "dtype": metadata["dtype"],
            "format_version": metadata["format_version"],
            "input_sha256": metadata["input_sha256"],
            "tokenizer_sha256": metadata["tokenizer_sha256"],
            "implementation_sha256": metadata["implementation_sha256"],
        }

    if (
        splits["train"]["tokenizer_sha256"]
        != splits["val"]["tokenizer_sha256"]
    ):
        raise RuntimeError(
            "Train and validation binaries were created with different tokenizers"
        )

    mixture_path = tokenized_dir / "token_mixture.json"
    mixture = _read_required_json(mixture_path, "realized token mixture report")
    validate_realized_mixture_report(
        mixture,
        split_metadata["train"],
        split_metadata["val"],
    )

    return {
        "manifest": {
            field: completion[field]
            for field in required_manifest_fields
        },
        "splits": splits,
        "realized_mixture": {
            "report_sha256": stable_digest(mixture),
            "contract_sha256": mixture["contract_sha256"],
            "status": mixture["status"],
        },
    }


def build_pretrain_run_contract(
    *,
    cfg: dict,
    run_size: str,
    tokenizer_dir: Path,
    tokenized_dir: Path,
    world_size: int,
    distributed_strategy: str,
) -> dict:
    """Build the immutable inputs required to resume a pretraining run."""
    return {
        "contract_version": PRETRAIN_AUDIT_VERSION,
        "run_size": run_size,
        "resolved_config": cfg,
        "resolved_config_sha256": stable_digest(cfg),
        "tokenizer": {
            "sha256": tokenizer_fingerprint(
                tokenizer_dir / "slm_tokenizer.json"
            ),
        },
        "tokenized_data": tokenized_data_identity(tokenized_dir),
        "distributed": {
            "world_size": world_size,
            "strategy": distributed_strategy,
        },
    }


def _contract_differences(expected, actual, prefix: str = "") -> list[str]:
    """Return concise key paths that differ between two run contracts."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                differences.append(path)
            else:
                differences.extend(
                    _contract_differences(expected[key], actual[key], path)
                )
        return differences
    return [] if expected == actual else [prefix or "<root>"]


def validate_or_write_pretrain_audit(
    output_dir: Path,
    contract: dict,
    *,
    resume: bool,
    write: bool,
) -> Path:
    """Persist a new-run contract or reject an incompatible resume."""
    audit_path = output_dir / PRETRAIN_AUDIT_FILENAME
    current_payload = {
        "schema_version": PRETRAIN_AUDIT_VERSION,
        "contract_sha256": stable_digest(contract),
        "contract": contract,
    }

    if audit_path.exists():
        saved_payload = _read_required_json(audit_path, "pretraining run audit")
        saved_contract = saved_payload.get("contract")
        if not isinstance(saved_contract, dict):
            raise RuntimeError(
                f"{audit_path} does not contain a valid run contract"
            )
        if saved_payload.get("contract_sha256") != stable_digest(saved_contract):
            raise RuntimeError(
                f"{audit_path} failed its own contract checksum"
            )
        if saved_contract != contract:
            differences = _contract_differences(saved_contract, contract)
            rendered = "\n  ".join(differences[:20])
            raise RuntimeError(
                "Pretraining run contract mismatch. Refusing to "
                f"{'resume' if resume else 'reuse the output directory'}.\n"
                f"Changed fields:\n  {rendered}"
            )
    elif resume:
        raise RuntimeError(
            f"Cannot resume without {audit_path}. The checkpoint predates the "
            "fail-closed provenance contract or its audit was removed."
        )

    if write and not audit_path.exists():
        atomic_write_json(audit_path, current_payload)
        log.info("Pretraining run audit written to %s", audit_path)
    elif audit_path.exists():
        log.info("Pretraining run audit matches: %s", audit_path)
    return audit_path


class VRAMProbe(TrainerCallback):
    """Log peak VRAM at step 200 so the analytical profile can be calibrated."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 200 and torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            log.info(f"[VRAMProbe step 200] allocated peak: {alloc:.2f} GB, "
                     f"reserved peak: {reserved:.2f} GB")


class SLMTrainer(Trainer):
    """
    Trainer subclass for pretraining.

    Handles dict / ModelOutput / tuple return types, including nested
    compiled-output cases like ({'loss': ..., 'logits': ...},).
    """

    @staticmethod
    def _extract_loss(outputs, context: str):
        """
        Extract loss from common model output forms:

        - {"loss": tensor, "logits": tensor}
        - {"loss": {"loss": tensor, "logits": tensor}}
        - ModelOutput with .loss
        - tuple/list where first item is loss
        - tuple/list where first item is dict/ModelOutput containing loss
        """
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            if isinstance(loss, (dict, tuple, list)):
                return SLMTrainer._extract_loss(loss, context)

        elif hasattr(outputs, "loss"):
            loss = outputs.loss
            if isinstance(loss, (dict, tuple, list)):
                return SLMTrainer._extract_loss(loss, context)

        elif isinstance(outputs, (tuple, list)):
            if len(outputs) == 0:
                raise TypeError(f"Empty output tuple/list during {context}")
            return SLMTrainer._extract_loss(outputs[0], context)

        else:
            raise TypeError(
                f"Unsupported model output type during {context}: {type(outputs)}"
            )

        if loss is None:
            raise TypeError(
                f"Output did not contain loss during {context}. "
                f"Output type: {type(outputs)}"
            )

        if not torch.is_tensor(loss):
            raise TypeError(
                f"Expected {context} loss to be a torch.Tensor, got {type(loss)}"
            )

        return loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(
            **inputs,
            num_items_in_batch=num_items_in_batch,
        )
        loss = self._extract_loss(outputs, context="training")
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Evaluation step for causal LM pretraining.

        Always returns loss only. Logits are intentionally dropped because:
        1. Pretraining eval only needs loss/perplexity tracking.
        2. Logits are [B, T, vocab]. At B=64, T=2048, vocab=32000,
           that is enormous and can cause eval OOMs.

        If logits are needed later for compute_metrics, do not return full
        logits from this method. Instead compute the metric in chunks, move
        reduced values to CPU, or compute the metric inside the model/trainer
        and log only scalar summaries.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        loss = self._extract_loss(outputs, context="eval")
        return (loss.detach().mean(), None, None)
    

def build_training_args(cfg: dict, output_dir: Path, resume: bool):
    from transformers import TrainingArguments

    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

    has_cuda = torch.cuda.is_available()
    precision = train_cfg.get("precision", "bf16")
    use_bf16  = has_cuda and precision == "bf16"
    use_fp16  = has_cuda and precision == "fp16"

    # torch.compile — defaults ON because graph compilation is essentially
    # free performance for this workload: static shapes (fixed seq_len, fixed
    # micro_batch), no dynamic control flow, no custom kernels. The 1-2 min
    # compilation cost on the first step amortizes to <1% overhead on a
    # multi-hour run and buys ~1.3-1.5x throughput on H100/H200/B200.
    #
    # Opt out by setting `torch_compile: false` in the training config if
    # you hit a kernel issue or are debugging the model.
    torch_compile = bool(train_cfg.get("torch_compile", True)) and has_cuda

    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=train_cfg["max_steps"],
        warmup_steps=train_cfg.get("warmup_steps", 2000),
        per_device_train_batch_size=train_cfg["micro_batch_size"],
        per_device_eval_batch_size=train_cfg["micro_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),

        learning_rate=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.1)),
        adam_beta1=float(optim_cfg.get("beta1", 0.9)),
        adam_beta2=float(optim_cfg.get("beta2", 0.95)),
        adam_epsilon=float(optim_cfg.get("eps", 1e-8)),
        max_grad_norm=float(train_cfg.get("gradient_clip_val", 1.0)),
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",

        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=has_cuda,

        torch_compile=torch_compile,
        torch_compile_backend=train_cfg.get("torch_compile_backend", "inductor"),
        torch_compile_mode=train_cfg.get("torch_compile_mode", "default"),

        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 1000),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 1000),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=False,

        logging_strategy="steps",
        logging_steps=train_cfg.get("log_steps", 10),
        report_to=train_cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("name", "slm-pretrain"),

        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),

        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
    )


def _size_from_model_name(model_name: str) -> str:
    name = model_name.removeprefix("slm-")
    return name.split("-")[0]


def main():
    parser = argparse.ArgumentParser(description="SLM Pretraining")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate run inputs, GPU visibility, and resume state without optimization",
    )
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        help="Required visible GPU count for --preflight-only",
    )
    parser.add_argument(
        "--distributed-strategy",
        choices=["single", "ddp", "fsdp"],
        default=None,
        help="Optional topology identity; otherwise inferred from the launched world",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    model_name = cfg["name"]
    run_size = (
        cfg.get("size")
        or cfg.get("data", {}).get("size")
        or model_name.removeprefix("slm-")
    )
    resolved_tokenized_dir = run_tokenized_dir(run_size)
    if cfg.get("output_dir"):
        output_dir = Path(cfg["output_dir"])
        if not output_dir.is_absolute():
            output_dir = args.results_dir.parent / output_dir
    else:
        output_dir = args.results_dir / "runs" / _size_from_model_name(model_name) / "pretrain"

    log.info(f"=== SLM Pretraining ===")
    log.info(f"Config:     {args.config}")
    log.info(f"Model:      {model_name}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    configure_torch_runtime(log)
    if torch.cuda.is_available():
        log.info(f"GPU:        {torch.cuda.get_device_name(0)} "
                 f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")

    resume_checkpoint = resolve_pretrain_checkpoint(
        output_dir,
        resume=args.resume,
    )
    if resume_checkpoint is not None:
        log.info(f"Resuming from checkpoint: {resume_checkpoint}")

    tokenizer_dir = args.data_dir / "tokenizer"
    validate_tokenizer(tokenizer_dir, resolved_tokenized_dir)

    from model import SLMConfig, SLMForCausalLM

    model_cfg_dict = cfg["model"]
    model_config = SLMConfig(
        vocab_size=model_cfg_dict["vocab_size"],
        hidden_size=model_cfg_dict["hidden_size"],
        intermediate_size=model_cfg_dict.get("intermediate_size"),
        num_hidden_layers=model_cfg_dict["num_hidden_layers"],
        num_attention_heads=model_cfg_dict["num_attention_heads"],
        num_key_value_heads=model_cfg_dict["num_key_value_heads"],
        max_position_embeddings=model_cfg_dict["max_position_embeddings"],
        rope_theta=model_cfg_dict.get("rope_theta", 10000.0),
        rms_norm_eps=model_cfg_dict.get("rms_norm_eps", 1e-5),
        initializer_range=model_cfg_dict.get("initializer_range", 0.02),
        tie_word_embeddings=model_cfg_dict.get("tie_word_embeddings", True),
    )
    validate_model_tokenizer_contract(model_config, tokenizer_dir)

    from pretrain.data.dataset import load_train_val

    seq_len = model_cfg_dict["max_position_embeddings"]

    log.info(f"Loading datasets from {resolved_tokenized_dir}")
    train_ds, val_ds = load_train_val(tokenized_dir=resolved_tokenized_dir, seq_len=seq_len)

    log.info(f"Train examples: {len(train_ds):,}")
    log.info(f"Val examples:   {len(val_ds):,}")

    budget = train_ds.token_budget()
    log.info(f"Training tokens: {budget['total_training_tokens'] / 1e9:.2f}B")

    world_size = (
        args.expected_gpus
        if args.preflight_only and args.expected_gpus is not None
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    cfg, realized_schedule = resolve_realized_token_schedule(
        cfg,
        run_size=run_size,
        realized_train_tokens=budget["n_tokens"],
        seq_len=seq_len,
        world_size=world_size,
    )
    training_args = build_training_args(cfg, output_dir, resume=args.resume)
    distributed_strategy = resolve_distributed_strategy(
        args.distributed_strategy,
        world_size,
    )
    global_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size
    )
    tokens_per_step = global_batch * seq_len
    scheduled_tokens = tokens_per_step * training_args.max_steps

    if realized_schedule is not None:
        log.info(
            "Resolved schedule from tokenized train corpus: usable_tokens=%s, "
            "epochs=%d, rounding_excess_tokens=%s",
            f'{realized_schedule["usable_train_tokens_per_epoch"]:,}',
            realized_schedule["epochs"],
            f'{realized_schedule["rounding_excess_tokens"]:,}',
        )

    log.info(
        "Training plan: strategy=%s, processes=%d, global_batch=%d sequences, "
        "tokens_per_step=%s, max_steps=%s, scheduled_tokens=%.2fB",
        distributed_strategy,
        world_size,
        global_batch,
        f"{tokens_per_step:,}",
        f"{training_args.max_steps:,}",
        scheduled_tokens / 1e9,
    )

    run_contract = build_pretrain_run_contract(
        cfg=cfg,
        run_size=run_size,
        tokenizer_dir=tokenizer_dir,
        tokenized_dir=resolved_tokenized_dir,
        world_size=world_size,
        distributed_strategy=distributed_strategy,
    )

    if args.preflight_only:
        validate_or_write_pretrain_audit(
            output_dir,
            run_contract,
            resume=args.resume,
            write=False,
        )
        validate_preflight_gpu(
            world_size,
            str(cfg["training"].get("precision", "bf16")),
        )
        log.info("Pretraining preflight passed; no model weights were allocated.")
        return

    if _is_rank_zero():
        validate_or_write_pretrain_audit(
            output_dir,
            run_contract,
            resume=args.resume,
            write=True,
        )

    log.info("Initializing model from scratch...")
    model = SLMForCausalLM(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    log.info(
        f"Throughput knobs: "
        f"micro_batch={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, "
        f"bf16={training_args.bf16}, "
        f"optim={training_args.optim}, "
        f"compile={training_args.torch_compile}, "
        f"grad_ckpt={training_args.gradient_checkpointing}"
    )

    if "wandb" in training_args.report_to and _is_rank_zero():
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "slm"),
            name=model_name,
            config={
                "model":           model_cfg_dict,
                "training":        cfg["training"],
                "optimizer":       cfg["optimizer"],
                "n_params":        n_params,
                "n_train_tokens":  budget["total_training_tokens"],
                "n_val_examples":  len(val_ds),
            },
        )

    trainer = SLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[VRAMProbe()],
    )

    run_baseline_eval = cfg.get(
        "run_baseline_eval",
        cfg.get("training", {}).get("run_baseline_eval", True),
    )
    if not args.resume and run_baseline_eval:
        log.info("Running baseline eval before training (step 0)...")
        baseline = trainer.evaluate()
        log.info(f"Baseline eval: {baseline}")
    elif not args.resume:
        log.info("Skipping baseline eval before training (run_baseline_eval=false)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint if args.resume else None)

    final_dir = output_dir / "final"
    if _is_rank_zero():
        log.info("Saving final model...")
        trainer.save_model(str(final_dir))
        model_config.save_pretrained(str(final_dir))

        shutil.copytree(tokenizer_dir, final_dir / "tokenizer", dirs_exist_ok=True)
        log.info(f"Tokenizer copied to {final_dir / 'tokenizer'}")
        shutil.copy2(
            output_dir / PRETRAIN_AUDIT_FILENAME,
            final_dir / PRETRAIN_AUDIT_FILENAME,
        )
        log.info(
            "Pretraining run audit copied to %s",
            final_dir / PRETRAIN_AUDIT_FILENAME,
        )

        log.info(f"Model saved to {final_dir}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if _is_rank_zero():
        log.info("Pretraining complete.")
        log.info("Next step: make sft")


if __name__ == "__main__":
    main()
