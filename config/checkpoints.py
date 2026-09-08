"""Checkpoint identity and immutable start/resume checks shared by training stages.

These checks read local files only. A tokenizer belongs to its checkpoint, not
its model-size label. They never repair tokenizers or reinitialize model weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from config.chat import tokenizer_fingerprint
from config.holdout import sha256_file
from curator.state import atomic_write_json, stable_digest


ARCHITECTURE_FIELDS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
    "rope_theta", "rms_norm_eps", "tie_word_embeddings",
    "pad_token_id", "bos_token_id", "eos_token_id",
)


def architecture_identity(config) -> dict:
    return {name: getattr(config, name) for name in ARCHITECTURE_FIELDS}


def bundled_tokenizer_dir(checkpoint: Path) -> Path:
    """Require the checkpoint's own HF tokenizer; never fall back by SIZE."""
    checkpoint = Path(checkpoint)
    for directory in (checkpoint / "tokenizer", checkpoint):
        if all((directory / name).is_file() for name in (
            "tokenizer.json", "tokenizer_config.json",
        )):
            return directory
    raise FileNotFoundError(
        f"Missing bundled tokenizer at {checkpoint}. Restore the tokenizer saved "
        "with this exact checkpoint; a same-size tokenizer is not a replacement."
    )


def checkpoint_identity(checkpoint: Path, *, include_provenance: bool = True) -> dict:
    """Hash config and complete weights, independent of the checkpoint's location."""
    checkpoint = Path(checkpoint)
    config = checkpoint / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config}")
    weights = sorted({
        p for pattern in ("model*.safetensors", "pytorch_model*.bin")
        for p in checkpoint.glob(pattern) if p.is_file()
    })
    if not weights:
        raise FileNotFoundError(f"No model weights in {checkpoint}")
    files = [config, *weights]
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = checkpoint / name
        if not index.is_file():
            continue
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid checkpoint weight index: {index}")
        for shard in set(weight_map.values()):
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise RuntimeError(f"Invalid weight shard name in {index}")
            if checkpoint / shard not in weights:
                raise RuntimeError(f"Missing indexed weight shard: {checkpoint / shard}")
        files.append(index)
    identity = {
        "files": {p.name: sha256_file(p) for p in files},
        "tokenizer_sha256": tokenizer_fingerprint(bundled_tokenizer_dir(checkpoint)),
    }
    provenance = checkpoint / "training_provenance.json"
    if include_provenance and provenance.is_file():
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        digest = payload.pop("sha256", None)
        if digest != stable_digest(payload):
            raise RuntimeError(f"Invalid training-provenance checksum: {provenance}")
        identity["provenance_sha256"] = digest
    return identity


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                suffix = path.name.removeprefix("checkpoint-")
                if suffix.isdigit():
                    candidates.append((int(suffix), path))
    return max(candidates, default=(0, None))[1]


def validate_recovery_checkpoint(checkpoint: Path, *, world_size: int = 1,
                                 require_scaler: bool = False) -> dict:
    """Require Trainer recovery files, not just weights or a numbered folder.

    Supported backends are ordinary PyTorch and DDP. Native loading still checks
    serialization/content; this gate prevents the Trainer's missing-file fallback
    from silently resetting optimizer, scheduler or RNG state. No state is loaded
    with unrestricted pickle here.
    """
    checkpoint = Path(checkpoint)
    if world_size < 1:
        raise ValueError("world_size must be positive")
    required = ["config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt"]
    required += (["rng_state.pth"] if world_size == 1 else
                 [f"rng_state_{rank}.pth" for rank in range(world_size)])
    if require_scaler:
        required.append("scaler.pt")
    missing = [name for name in required
               if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
    weight_files = []
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (checkpoint / name).is_file():
            weight_files.append(name)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = checkpoint / name
        if not index.is_file():
            continue
        try:
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("empty weight map")
            for shard in set(mapping.values()):
                if not isinstance(shard, str) or Path(shard).name != shard:
                    raise ValueError("invalid shard path")
                weight_files.append(shard)
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid recovery weight index: {index}") from exc
    if not weight_files:
        missing.append("model weights (single file or complete shard index)")
    missing.extend(name for name in weight_files
                   if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0)
    if missing:
        raise RuntimeError(
            f"Incomplete recovery checkpoint {checkpoint}: missing/empty {', '.join(missing)}. "
            "Restore the complete checkpoint or explicitly select a verified earlier "
            "checkpoint from this run with --resume PATH; no automatic fallback is performed."
        )
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        step = state["global_step"]
        suffix = checkpoint.name.removeprefix("checkpoint-")
        if (type(step) is not int or step < 0 or not suffix.isdigit()
                or step != int(suffix)):
            raise ValueError("global_step does not match checkpoint suffix")
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid Trainer recovery state in {checkpoint}") from exc
    return state


def resolve_training_checkpoint(output_dir: Path, *, resume: bool | str | Path | None,
                                audit_filename: str, world_size: int | None = None,
                                require_scaler: bool = False) -> Path | None:
    """Resolve a complete recovery checkpoint; reject occupied new-run outputs.

    A path explicitly selects an earlier checkpoint in the same run. Keep the
    run audit and checkpoints together when restoring to another host.
    """
    output_dir = Path(output_dir)
    if resume:
        if resume is True or resume == "latest":
            checkpoint = find_latest_checkpoint(output_dir)
        else:
            checkpoint = Path(os.path.expandvars(str(resume))).expanduser()
            if (checkpoint.resolve().parent != output_dir.resolve()
                    or not checkpoint.name.startswith("checkpoint-")
                    or not checkpoint.name.removeprefix("checkpoint-").isdigit()):
                raise RuntimeError("An explicit --resume checkpoint must be a numbered "
                                   f"checkpoint directly under this run's output directory: {output_dir}")
        if checkpoint is None:
            raise RuntimeError(
                f"--resume was requested but no checkpoint-* directory exists in "
                f"{output_dir}. Refusing to start from scratch. Restore the expected "
                "checkpoint or start a new run explicitly."
            )
        validate_recovery_checkpoint(
            checkpoint, world_size=(int(os.environ.get("WORLD_SIZE", "1"))
                                    if world_size is None else world_size),
            require_scaler=require_scaler,
        )
        return checkpoint
    if output_dir.exists():
        artifacts = sorted(p for p in output_dir.iterdir() if p.name != audit_filename)
        if artifacts:
            raise RuntimeError(
                "A new run was requested in an output directory that already contains "
                "training artifacts:\n  " + "\n  ".join(map(str, artifacts)) +
                "\nUse --resume for the interrupted run or a different RESULTS_DIR."
            )
    return None


def contract_differences(expected, actual, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                differences.append(path)
            else:
                differences.extend(contract_differences(expected[key], actual[key], path))
        return differences
    return [] if expected == actual else [prefix or "<root>"]


def validate_or_write_run_audit(output_dir: Path, contract: dict, *,
                                audit_filename: str, schema_version: int,
                                resume: bool, write: bool,
                                details: dict | None = None) -> Path:
    """Compare before writing. Existing audits, including their details, are immutable."""
    path = output_dir / audit_filename
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        previous = saved.get("contract") if isinstance(saved, dict) else None
        if not isinstance(previous, dict):
            raise RuntimeError(f"{path} does not contain a valid run contract; "
                               "legacy runs cannot be resumed without verified provenance")
        if saved.get("contract_sha256") != stable_digest(previous):
            raise RuntimeError(f"{path} failed its own contract checksum")
        if previous != contract:
            changed = "\n  ".join(contract_differences(previous, contract)[:20])
            raise RuntimeError(f"Training run contract mismatch. Refusing to "
                               f"{'resume' if resume else 'reuse the output directory'}.\n"
                               f"Changed fields:\n  {changed}")
    elif resume:
        raise RuntimeError(f"Cannot resume without {path}. Restore the original audit "
                           "and checkpoint together; do not manufacture a replacement.")
    elif write:
        atomic_write_json(path, {
            **(details or {}), "schema_version": schema_version,
            "contract_sha256": stable_digest(contract), "contract": contract,
        })
    return path


def posttraining_contract(*, cfg: dict, size: str, stage: str, model_config,
                          base_identity: dict, tokenizer_path: Path,
                          chat_template, data_manifest: dict,
                          world_size: int) -> dict:
    """Bind post-training to the same inputs for new-run and resume invocations."""
    return {
        "contract_version": 1, "run_size": size, "stage": stage,
        "resolved_config": cfg, "architecture": architecture_identity(model_config),
        "base_checkpoint": base_identity,
        # Both explicit-reference and precomputed-reference DPO use the original
        # instruct policy, never the policy being resumed from an optimizer step.
        "reference_checkpoint": base_identity if stage == "dpo_chat" else None,
        "tokenizer": {"files_sha256": tokenizer_fingerprint(tokenizer_path),
                      "chat_template_sha256": stable_digest(chat_template)},
        "data_manifest_sha256": stable_digest(data_manifest),
        "distributed": {"world_size": world_size},
    }
