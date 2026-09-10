#!/usr/bin/env python3
"""Isolated HF reference-data control using the production encoder/dataset/Trainer.

The historical filename is retained. --config selects the actual model recipe,
including Mini. This diagnostic deliberately bypasses corpus curation, not the
next-token training components. It does not manufacture a production mixture or
publishable pretraining audit. See scripts/README.md for the experiment boundary.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths import BASE_DATA_DIR, BASE_RESULTS_DIR, BASE_EXPORTS_DIR
from config.holdout import SPLITS, sha256_file, jsonl_identity, write_contract, verify_jsonl_contract
from curator.state import atomic_write_json, stable_digest, write_manifest, manifest_outputs_match

log = logging.getLogger(__name__)
SCHEMA = 1
AUDIT = "reference_run_audit.json"
MANIFEST = "reference_data.json"
DEFAULT_DATASET = "HuggingFaceTB/dclm-edu"
FINEWEB = "HuggingFaceFW/fineweb-edu"
PROMPTS = (
    "The history of the internet began", "A prime number is",
    "Photosynthesis is the process by which", "In Python, a function is defined using",
    "The capital of France is", "Machine learning is",
)


def parse_args(argv=None, *, sanity=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["prepare", "check", "train", "pretrain", "all"],
                   default="check" if sanity else "prepare")
    p.add_argument("--size", "--arch", dest="size", choices=["smoke", "mini", "125m", "350m", "1b"])
    p.add_argument("--config", type=Path, help="Actual generated pretraining YAML; never an independent architecture table")
    p.add_argument("--run-dir", "--scratch-dir", dest="run_dir", type=Path, required=True,
                   help="Dedicated diagnostic directory; no overwrite or deletion of existing runs")
    p.add_argument("--dataset", default=FINEWEB if sanity else DEFAULT_DATASET)
    p.add_argument("--dataset-config", "--subset", dest="dataset_config")
    p.add_argument("--dataset-revision", default="main", help="Resolved once to an immutable Hub commit and recorded")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    tk = p.add_mutually_exclusive_group()
    tk.add_argument("--tokenizer-dir", type=Path, help="Existing local tokenizer, copied read-only; default: selected size's tokenizer")
    tk.add_argument("--reference-tokenizer", help="Explicit separate tokenizer experiment, e.g. mistralai/Mistral-7B-v0.1")
    p.add_argument("--tokenizer-revision", default="main")
    p.add_argument("--tokenized-dir", type=Path, help="Use an existing verified train/val/test bundle read-only instead of HF preparation")
    p.add_argument("--eval-tokenized-dir", type=Path,
                   help="Optional shared validation bundle for corpus comparisons; HF preparation excludes its adjacent validated val/test documents")
    p.add_argument("--target-tokens", type=int, help="Usable unique TRAIN tokens, excluding holdouts; defaults to all local tokens or 50M HF tokens")
    p.add_argument("--max-docs", type=int, help="Maximum streamed records examined; an unmet token target fails closed")
    p.add_argument("--val-fraction", type=float, default=0.005)
    p.add_argument("--test-fraction", type=float, default=0.005)
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    p.add_argument("--seed", type=int, help="Defaults to recipe training.seed; applies before model initialization")
    p.add_argument("--exclude-jsonl", type=Path, action="append", default=[],
                   help="Exclude normalized exact document matches to these holdouts before selection (repeatable)")
    p.add_argument("--reuse-tokens", action="store_true", help="Reuse only a complete, hash-verified matching diagnostic bundle")
    p.add_argument("--backend", choices=["slm", "llama"], default="slm", help="Identical SLM-initialized tensors; same production Trainer")
    p.add_argument("--max-steps", type=int, help="Explicit bounded experiment override; logged, not a production token-floor waiver")
    p.add_argument("--probe-every-steps", type=int, help="Override qualitative generation-probe cadence for bounded diagnostic training")
    p.add_argument("--batch-size", type=int, help="Explicit micro-batch override; default: recipe")
    p.add_argument("--gradient-accumulation", type=int, help="Explicit accumulation override; default: recipe")
    p.add_argument("--resume", nargs="?", const="latest")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--check-seq-len", type=int, default=128, help="Bounded numerical comparison context; does not change the model config")
    p.add_argument("--check-batches", type=int, default=2)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--save", action="store_true", help="Compatibility flag; diagnostic training now always preserves its checkpoint")
    args = p.parse_args(argv)
    for name in ("target_tokens", "max_docs", "shuffle_buffer", "max_steps", "batch_size", "gradient_accumulation", "check_seq_len", "check_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.check_seq_len < 4:
        p.error("--check-seq-len must be at least 4")
    if (not 0 < args.val_fraction < 1 or not 0 < args.test_fraction < 1
            or args.val_fraction + args.test_fraction >= 1):
        p.error("Validation/test fractions must be positive with sum below one")
    if any(not math.isfinite(v) or v < 0 for v in (args.rtol, args.atol)):
        p.error("Comparison tolerances must be finite and nonnegative")
    if args.tokenized_dir and args.reference_tokenizer:
        p.error("Existing tokens require their original local --tokenizer-dir, not a replacement tokenizer")
    return args


def read_recipe(args):
    from pretrain.train import load_config
    path = (args.config or ROOT / "pretrain" / "configs" / f"gpt_{args.size or '125m'}.yaml").resolve()
    cfg = load_config(path)
    size = cfg.get("size") or cfg.get("data", {}).get("size") or cfg["name"].removeprefix("slm-")
    if size not in {"smoke", "mini", "125m", "350m", "1b"} or (args.size and args.size != size):
        raise ValueError("--size and the selected recipe's model identity disagree")
    args.size = size
    args.config = path
    if args.seed is None:
        args.seed = int(cfg["training"].get("seed", 42))
    cfg = copy.deepcopy(cfg)
    cfg["training"]["seed"] = args.seed
    if args.batch_size is not None:
        cfg["training"]["micro_batch_size"] = args.batch_size
    if args.gradient_accumulation is not None:
        cfg["training"]["gradient_accumulation_steps"] = args.gradient_accumulation
    if cfg["model"]["max_position_embeddings"] < 4:
        raise ValueError("Recipe context length is too short")
    # Do not write this copy back to the generated recipe.
    log.info("Recipe %s: size=%s, layers=%s, hidden=%s", path, size,
             cfg["model"]["num_hidden_layers"], cfg["model"]["hidden_size"])
    return cfg


def _overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def validate_run_dir(path: Path, inputs=()):
    """Never repurpose a production root/checkpoint or an input directory."""
    path = path.expanduser().resolve()
    for base in (ROOT, Path.home(), BASE_DATA_DIR.resolve(), BASE_RESULTS_DIR.resolve(), BASE_EXPORTS_DIR.resolve()):
        if path == base or path in base.parents:
            raise ValueError(f"Unsafe diagnostic root: {path}")
    for protected in (BASE_DATA_DIR / "runs", BASE_RESULTS_DIR / "runs", BASE_EXPORTS_DIR):
        if _overlaps(path, protected):
            raise ValueError(f"Diagnostic root overlaps production artifacts: {protected}")
    for source in inputs:
        if source is not None and _overlaps(path, Path(source)):
            raise ValueError(f"Diagnostic root overlaps a read-only input: {source}")
    marker = path / ".hf-control.json"
    if path.exists() and any(path.iterdir()) and not marker.is_file():
        raise RuntimeError(f"Refusing unowned nonempty directory: {path}. Choose a new --run-dir.")
    if marker.exists() and json.loads(marker.read_text()) != {"kind": "slm-hf-control", "version": SCHEMA}:
        raise RuntimeError("Unrecognized diagnostic-directory marker")
    path.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        atomic_write_json(marker, {"kind": "slm-hf-control", "version": SCHEMA})
    # Refuse redirected writes, including dangling symlinks, within an owned run.
    if any(p.is_symlink() for p in path.rglob("*")):
        raise RuntimeError("Symlinks are not allowed inside a diagnostic output directory")
    return path


@contextmanager
def run_lock(path):
    import fcntl
    with (path / ".control.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another diagnostic is already using this run directory") from exc
        yield


def resolve_revision(repo_id: str, revision: str, *, tokenizer=False) -> str:
    from huggingface_hub import HfApi
    info = (HfApi().model_info if tokenizer else HfApi().dataset_info)(repo_id, revision=revision)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise RuntimeError(f"Hub did not return an immutable commit for {repo_id}@{revision}")
    return info.sha


def file_hashes(directory):
    return {str(p.relative_to(directory)): sha256_file(p)
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def load_tokenizer(args, *, snapshot=None):
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    if snapshot is not None:
        path = Path(snapshot)
        identity = None
    elif args.reference_tokenizer:
        revision = resolve_revision(args.reference_tokenizer, args.tokenizer_revision, tokenizer=True)
        tok = AutoTokenizer.from_pretrained(args.reference_tokenizer, revision=revision, trust_remote_code=False, use_fast=True)
        identity = {"kind": "reference_tokenizer", "repo": args.reference_tokenizer,
                    "requested_revision": args.tokenizer_revision, "revision": revision}
        path = None
    else:
        path = (args.tokenizer_dir or BASE_DATA_DIR / "runs" / args.size / "tokenizer").resolve()
        identity = {"kind": "same_local_tokenizer", "path": str(path), "files": file_hashes(path)}
    if path is not None:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True, use_fast=True)
    if not tok.is_fast or not 0 < len(tok) < 65536:
        raise ValueError("A fast tokenizer with fewer than 65,536 entries is required; vocabulary is never resized")
    # PAD is not inserted in packed training. For a reference tokenizer lacking it,
    # use UNK rather than aliasing BOS/EOS (which could freeze their embedding row).
    if tok.pad_token_id is None:
        if tok.unk_token_id is None or tok.unk_token_id in (tok.bos_token_id, tok.eos_token_id):
            raise ValueError("Reference tokenizer needs a distinct existing PAD or UNK token")
        tok.pad_token = tok.unk_token
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tok, key)
        if value is None or not 0 <= value < len(tok):
            raise ValueError(f"Missing or invalid {key}")
    if len({tok.bos_token_id, tok.eos_token_id, tok.pad_token_id}) != 3:
        raise ValueError("BOS, EOS, PAD must have distinct existing IDs")
    raw_path = path / "slm_tokenizer.json" if path is not None else None
    raw = Tokenizer.from_file(str(raw_path)) if raw_path is not None and raw_path.is_file() else Tokenizer.from_str(tok.backend_tokenizer.to_str())
    if raw.get_vocab_size(with_added_tokens=True) != len(tok) or raw.get_vocab() != tok.get_vocab():
        raise ValueError("Raw/HF tokenizer vocabulary or ID mapping mismatch")
    raw_config, hf_config = json.loads(raw.to_str()), json.loads(tok.backend_tokenizer.to_str())
    if any(raw_config.get(k) != hf_config.get(k) for k in ("model", "normalizer", "pre_tokenizer", "decoder")):
        raise ValueError("Raw/HF tokenizer processing differs; restore one matched bundle")
    for special in (tok.bos_token, tok.eos_token, tok.pad_token):
        if raw.token_to_id(special) != tok.convert_tokens_to_ids(special):
            raise ValueError("Raw/HF special-token IDs disagree")
    raw_sha = hashlib.sha256(raw.to_str().encode()).hexdigest()
    return tok, raw, identity, raw_sha


def document_key(text):
    """Group normalized exact copies; not a near-duplicate or semantic filter."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def document_split(key, seed, val_fraction, test_fraction):
    value = int.from_bytes(hashlib.sha256(str(seed).encode() + b"\0" + key).digest()[:8], "big") / 2**64
    return "test" if value < test_fraction else "val" if value < test_fraction + val_fraction else "train"


def preparation_spec(args, raw_sha, seq_len):
    from pretrain.data import tokenize_data
    return {
        "dataset": args.dataset, "dataset_config": args.dataset_config or ("sample-10BT" if args.dataset == FINEWEB else None),
        "requested_revision": args.dataset_revision, "upstream_split": args.split, "text_field": args.text_field,
        "seed": args.seed, "shuffle_buffer": args.shuffle_buffer, "max_docs": args.max_docs,
        "val_fraction": args.val_fraction, "test_fraction": args.test_fraction,
        "target_train_tokens": args.target_tokens or 50_000_000, "seq_len": seq_len,
        "tokenizer_sha256": raw_sha, "tokenizer_request": args.reference_tokenizer,
        "tokenizer_requested_revision": args.tokenizer_revision if args.reference_tokenizer else None,
        "excluded_files": {str(p.resolve()): sha256_file(p) for p in args.exclude_jsonl},
        "implementation": {"driver": sha256_file(Path(__file__)), "encoder": sha256_file(Path(tokenize_data.__file__))},
        "packages": {name: importlib.metadata.version(name) for name in ("datasets", "tokenizers", "transformers", "huggingface_hub")},
        "split_unit": "NFKC-whitespace-normalized-document-sha256", "near_dedup": "not_performed",
        "quality_filter": "upstream_only; local empty-text and exact-duplicate rejection",
    }


def verify_reference_bundle(data_root, *, expected_spec=None):
    data_root = Path(data_root)
    path = data_root / MANIFEST
    if not path.is_file():
        raise RuntimeError(f"No completed reference-data manifest: {path}")
    bundle = json.loads(path.read_text())
    digest = bundle.pop("sha256", None)
    if digest != stable_digest(bundle) or bundle.get("schema_version") != SCHEMA or bundle.get("status") != "complete":
        raise RuntimeError("Reference-data manifest is incomplete or corrupt")
    if expected_spec is not None:
        # The driver also contains training/check orchestration. Training-only edits
        # (for example probe cadence wiring) must not invalidate immutable prepared
        # reference data. Keep the full driver hash in the manifest for provenance,
        # but exclude it from cache-compatibility comparison; the production encoder
        # hash and all data-selection/tokenization request fields remain fail-closed.
        cached_spec = copy.deepcopy(bundle["spec"])
        requested_spec = copy.deepcopy(expected_spec)
        cached_spec.get("implementation", {}).pop("driver", None)
        requested_spec.get("implementation", {}).pop("driver", None)
        if cached_spec != requested_spec:
            raise RuntimeError("Reference-data request changed. Use a new diagnostic run directory.")
    for name, expected in bundle["files"].items():
        path = data_root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise RuntimeError(f"Reference-data file changed or missing: {name}")
    holdout = verify_jsonl_contract(data_root / "reference", stage="reference")
    if not manifest_outputs_match(data_root / "tokenized", output_pattern="[tv]*"):
        raise RuntimeError("Reference tokenized bundle is not manifest-complete")
    from pretrain.data.tokenize_data import verify_dataset
    for split in SPLITS:
        meta = json.loads((data_root / "tokenized" / f"{split}.json").read_text())
        if (meta["input_sha256"] != holdout["contract"]["splits"][split]["sha256"]
                or meta["test_split_sha256"] != holdout["sha256"]
                or meta["tokenizer_sha256"] != bundle["spec"]["tokenizer_sha256"]):
            raise RuntimeError("Reference binary/text/tokenizer identities disagree")
        verify_dataset(data_root / "tokenized" / f"{split}.bin", data_root / "tokenized" / f"{split}.json")
    bundle["sha256"] = digest
    return bundle


def prepare_data(args, cfg):
    """Stage complete documents and append only written tokens; publish once."""
    from datasets import load_dataset
    from pretrain.data import tokenize_data as encoder
    seq_len = cfg["model"]["max_position_embeddings"]
    destination = args.run_dir / "data" / "runs" / args.size
    if destination.exists():
        if not args.reuse_tokens and args.stage in ("prepare", "all"):
            raise RuntimeError("Reference data already exists. Use --reuse-tokens to verify it, not overwrite it.")
        bundle = verify_reference_bundle(destination)
        tok, raw, _, raw_sha = load_tokenizer(args, snapshot=destination / "tokenizer")
        if not args.reference_tokenizer:
            _, _, _, requested_sha = load_tokenizer(args)
            if requested_sha != raw_sha:
                raise RuntimeError("Current local tokenizer differs from the cached control tokenizer")
        if args.target_tokens is None:
            args.target_tokens = bundle["spec"]["target_train_tokens"]
        if args.stage not in ("prepare", "all") and not args.exclude_jsonl:
            args.exclude_jsonl = [Path(p) for p in bundle["spec"]["excluded_files"]]
        model_config(cfg, tok)  # reject mismatched vocabulary before model allocation
        spec = preparation_spec(args, raw_sha, seq_len)
        verify_reference_bundle(destination, expected_spec=spec)
        return destination / "tokenized", tok, bundle
    if args.stage not in ("prepare", "all"):
        raise RuntimeError("Prepare reference data first with --stage prepare (or supply --tokenized-dir).")
    tok, raw, tokenizer_identity, raw_sha = load_tokenizer(args)
    model_config(cfg, tok)
    spec = preparation_spec(args, raw_sha, seq_len)
    revision = resolve_revision(args.dataset, args.dataset_revision)
    log.info("Reference dataset pinned to %s@%s", args.dataset, revision)
    ds = load_dataset(args.dataset, name=spec["dataset_config"], split=args.split, revision=revision, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    target = math.ceil(spec["target_train_tokens"] / seq_len) * seq_len
    counts = {split: {"n_docs": 0, "n_tokens": 0} for split in SPLITS}
    seen = skipped = duplicates = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    if list(destination.parent.glob(".reference-partial-*")):
        raise RuntimeError("Interrupted reference preparation remains. Inspect its .reference-partial-* directory; it is not a reusable token cache.")
    with tempfile.TemporaryDirectory(prefix=".reference-partial-", dir=destination.parent) as temp:
        stage = Path(temp)
        for name in ("reference", "tokenized", "tokenizer"):
            (stage / name).mkdir()
        tok.save_pretrained(stage / "tokenizer")
        raw.save(str(stage / "tokenizer" / "slm_tokenizer.json"))
        # Use the production literal-safe encoding and BOS/doc/EOS insertion.
        encoder._configure_pretraining_tokenizer(raw)
        encoder._worker_tokenizer, encoder._worker_bos_id, encoder._worker_eos_id = raw, tok.bos_token_id, tok.eos_token_id
        with ExitStack() as stack:
            db = sqlite3.connect(stage / "selection.sqlite")
            stack.callback(db.close)
            db.execute("CREATE TABLE seen (hash BLOB PRIMARY KEY) WITHOUT ROWID")
            for path in args.exclude_jsonl:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        text = json.loads(line)["text"]
                        db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (document_key(text),))
            text_files = {s: stack.enter_context((stage / "reference" / f"{s}.jsonl").open("w", encoding="utf-8")) for s in SPLITS}
            bins = {s: stack.enter_context((stage / "tokenized" / f"{s}.bin").open("wb")) for s in SPLITS}
            for example in itertools.islice(ds, args.max_docs):
                seen += 1
                text = example.get(args.text_field)
                if not isinstance(text, str) or not text.strip():
                    skipped += 1
                    continue
                key = document_key(text)
                if not db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (key,)).rowcount:
                    duplicates += 1
                    continue
                split = document_split(key, args.seed, args.val_fraction, args.test_fraction)
                tokens, _, _ = encoder._tokenize_chunk([(text, args.dataset)])
                encoder._write_tokens(tokens, bins[split])
                text_files[split].write(json.dumps({"text": text, "source": args.dataset,
                    "upstream_revision": revision, "document_sha256": key.hex()}, ensure_ascii=False) + "\n")
                counts[split]["n_docs"] += 1
                counts[split]["n_tokens"] += len(tokens)
                if seen % 10000 == 0:
                    db.commit()
                    log.info("Reference selection: %s examined; train=%s tokens", f"{seen:,}", f"{counts['train']['n_tokens']:,}")
                if counts["train"]["n_tokens"] >= target and all(counts[s]["n_tokens"] >= 4 * seq_len for s in ("val", "test")):
                    break
            for handle in [*bins.values(), *text_files.values()]:
                handle.flush()
                os.fsync(handle.fileno())
        (stage / "selection.sqlite").unlink()
        if counts["train"]["n_tokens"] < target or any(counts[s]["n_tokens"] < 4 * seq_len for s in ("val", "test")):
            raise RuntimeError(f"Reference source/cap exhausted before train and document-heldout targets: {counts}. No cache published.")
        identities = {s: jsonl_identity(stage / "reference" / f"{s}.jsonl") for s in SPLITS}
        holdout = write_contract(stage / "reference", {"schema_version": 1, "status": "established", "stage": "reference",
            "size": args.size, "splits": identities, "origin": {"dataset": args.dataset, "revision": revision},
            "split_policy": {k: spec[k] for k in ("seed", "split_unit", "val_fraction", "test_fraction", "near_dedup", "quality_filter")}})
        for s in SPLITS:
            meta = {**counts[s], "dtype": "uint16", "split": s, "vocab_size": len(tok),
                "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id,
                "format_version": encoder.TOKENIZED_FORMAT_VERSION, "input_sha256": identities[s]["sha256"],
                "binary_sha256": sha256_file(stage / "tokenized" / f"{s}.bin"), "tokenizer_sha256": raw_sha,
                "test_split_sha256": holdout["sha256"], "source_counts": {args.dataset: {"documents": counts[s]["n_docs"], "tokens": counts[s]["n_tokens"]}}}
            atomic_write_json(stage / "tokenized" / f"{s}.json", meta)
            encoder.verify_dataset(stage / "tokenized" / f"{s}.bin", stage / "tokenized" / f"{s}.json")
        shutil.copy2(stage / "reference" / "test_contract.json", stage / "tokenized" / "test_contract.json")
        write_manifest(stage / "tokenized", stage="hf-reference-tokenize", contract=spec,
                       input_signature=holdout["sha256"], output_pattern="[tv]*")
        bundle = {"schema_version": SCHEMA, "status": "complete", "kind": "diagnostic_not_production_curation",
            "spec": spec, "dataset_revision": revision, "tokenizer_identity": tokenizer_identity,
            "selected_train_tokens": target, "counts": counts, "examined": seen, "empty_skipped": skipped,
            "duplicates_or_excluded": duplicates, "files": file_hashes(stage)}
        atomic_write_json(stage / MANIFEST, {**bundle, "sha256": stable_digest(bundle)})
        verify_reference_bundle(stage, expected_spec=spec)
        # Destination was checked absent under the run lock. Never replace a prior bundle.
        if destination.exists():
            raise RuntimeError("Reference destination appeared during preparation")
        stage.rename(destination)
    return destination / "tokenized", tok, {**bundle, "sha256": stable_digest(bundle)}


def verify_existing_tokens(directory, tok, raw_sha):
    from pretrain.train import tokenized_data_identity
    from pretrain.data.tokenize_data import verify_dataset
    directory = Path(directory).resolve()
    identity = tokenized_data_identity(directory)
    for split in SPLITS:
        meta = json.loads((directory / f"{split}.json").read_text())
        if meta["tokenizer_sha256"] != raw_sha or (meta["bos_id"], meta["eos_id"]) != (tok.bos_token_id, tok.eos_token_id):
            raise RuntimeError("Supplied tokenizer did not produce the existing tokenized bundle")
        verify_dataset(directory / f"{split}.bin", directory / f"{split}.json")
    return identity


def get_inputs(args, cfg):
    if args.tokenized_dir is None:
        return prepare_data(args, cfg)
    if args.exclude_jsonl:
        raise ValueError("--exclude-jsonl applies to HF selection, not immutable existing tokens")
    tok, _, tokenizer_identity, raw_sha = load_tokenizer(args)
    directory = args.tokenized_dir.resolve()

    # A bundle created by this diagnostic has a different sidecar contract from
    # production tokenized data. Reuse it through its own immutable manifest
    # rather than forcing it through pretrain.train.tokenized_data_identity().
    reference_root = directory.parent
    reference_manifest = reference_root / MANIFEST
    if directory.name == "tokenized" and reference_manifest.is_file():
        bundle = verify_reference_bundle(reference_root)
        if bundle.get("kind") != "diagnostic_not_production_curation":
            raise RuntimeError(f"Unsupported reference-data bundle kind in {reference_manifest}")
        if bundle["spec"]["tokenizer_sha256"] != raw_sha:
            raise RuntimeError("Supplied tokenizer differs from the prepared reference-data tokenizer")
        model_config(cfg, tok)
        selected = int(bundle["selected_train_tokens"])
        if args.target_tokens is not None:
            length = cfg["model"]["max_position_embeddings"]
            requested = math.ceil(args.target_tokens / length) * length
            if requested != selected:
                raise ValueError(
                    "Prepared reference data is immutable and was selected for "
                    f"{selected} usable train tokens; requested {requested}. "
                    "Use the prepared budget or create a separate reference bundle."
                )
        log.info("Reusing verified reference-data bundle: %s", reference_root)
        return directory, tok, bundle

    identity = verify_existing_tokens(directory, tok, raw_sha)
    available = identity["splits"]["train"]["n_tokens"]
    length = cfg["model"]["max_position_embeddings"]
    target = args.target_tokens if args.target_tokens is not None else available // length * length
    target = math.ceil(target / length) * length
    if target > available:
        raise ValueError("Requested training prefix exceeds the existing corpus")
    return directory, tok, {"kind": "existing_tokenized_read_only", "path": str(directory), "identity": identity,
                           "tokenizer_identity": tokenizer_identity, "selected_train_tokens": target}


def model_config(cfg, tokenizer):
    from model import SLMConfig
    if cfg["model"]["vocab_size"] != len(tokenizer):
        raise ValueError("Recipe/tokenizer vocabulary mismatch. Choose a matched tokenizer; diagnostic never resizes embeddings.")
    config = copy.deepcopy(cfg["model"])
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        config[name] = getattr(tokenizer, name)
    return SLMConfig(**config)


def versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "datasets", "accelerate")}


def implementation_identity():
    paths = ["scripts/sanity_train.py", "scripts/pretrain_hf_125m.py", "pretrain/train.py", "pretrain/data/dataset.py", "pretrain/schedule.py",
             "model/model.py", "model/attention.py", "model/config.py", "model/block.py",
             "model/mlp.py", "model/norm.py", "export/export.py"]
    return {p: sha256_file(ROOT / p) for p in paths}


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + b"\0")
        array = value.detach().cpu().contiguous()
        digest.update(str((tuple(array.shape), array.dtype)).encode())
        digest.update(array.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def generate_probes(model, tokenizer):
    import torch
    was_training = model.training
    model.eval()
    rows = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
            inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
            budget = min(100, model.config.max_position_embeddings - len(ids))
            if budget < 1:
                continue
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=budget,
                do_sample=False, num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0, use_cache=True,
                bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            rows.append({"prompt": prompt, "continuation": tokenizer.decode(output[0, len(ids):], skip_special_tokens=True),
                         "do_sample": False, "max_new_tokens": budget, "explicit_bos": True})
    model.train(was_training)
    return rows


def select_device(request):
    import torch
    device = "cuda" if request == "auto" and torch.cuda.is_available() else "cpu" if request == "auto" else request
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This diagnostic is single-process; do not launch it through DDP")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=0; implicit DataParallel is not a controlled comparison")
    return device


def run_training(args, cfg, directory, tokenizer, data_identity):
    from dataclasses import replace
    # Import runtime policy before torch/transformers so diagnostic training
    # uses the same persistent Inductor-cache bootstrap as production.
    from config.runtime import configure_torch_runtime
    import torch
    from transformers import set_seed, TrainerCallback
    from model import SLMForCausalLM
    from pretrain.train import SLMTrainer, build_training_args
    from pretrain.data.dataset import load_train_val, load_test
    from pretrain.schedule import resolve_realized_token_schedule
    from config.checkpoints import resolve_training_checkpoint, validate_or_write_run_audit
    import wandb
    # Training is an explicit W&B-tracked experiment. Pure numerical checks do not log in.
    for key in ("WANDB_API_KEY", "WANDB_PROJECT"):
        if not os.environ.get(key, "").strip() or os.environ[key].strip() == "...":
            raise RuntimeError(f"{key} is required for reference training")
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        raise RuntimeError("W&B tracking must not be disabled for reference training")
    device = select_device(args.device)
    config = model_config(cfg, tokenizer)
    train_ds, val_ds = load_train_val(directory, seq_len=config.max_position_embeddings,
                                      max_train_tokens=data_identity["selected_train_tokens"])
    cfg = copy.deepcopy(cfg)
    # Resolve using the same production horizon function, on selected train tokens
    # only. This diagnostic's smaller corpus is never presented as a production run.
    cfg["training"]["schedule_from_realized_tokens"] = True
    cfg, schedule = resolve_realized_token_schedule(cfg, run_size=args.size,
        realized_train_tokens=train_ds.n_tokens, seq_len=config.max_position_embeddings, world_size=1)
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
        cfg["training"]["warmup_steps"] = min(args.max_steps, round(args.max_steps * schedule["warmup_ratio_from_planning_config"]))
    if args.probe_every_steps is not None:
        if args.probe_every_steps < 0:
            raise ValueError("--probe-every-steps must be >= 0")
        cfg.setdefault("generation_probes", {})["every_steps"] = args.probe_every_steps
        cfg["generation_probes"]["enabled"] = True
    common_val = None
    common_identity = None
    if args.eval_tokenized_dir is not None:
        from pretrain.data.dataset import PretrainingDataset
        from pretrain.data.tokenize_data import tokenizer_fingerprint
        token_snapshot = (args.run_dir / "data" / "runs" / args.size / "tokenizer" / "slm_tokenizer.json"
                          if args.tokenized_dir is None else
                          (args.tokenizer_dir or BASE_DATA_DIR / "runs" / args.size / "tokenizer") / "slm_tokenizer.json")
        common_identity = verify_existing_tokens(args.eval_tokenized_dir, tokenizer, tokenizer_fingerprint(token_snapshot))
        common_val = PretrainingDataset(args.eval_tokenized_dir / "val.bin", seq_len=config.max_position_embeddings, split="val")
    output = args.run_dir / "results" / "runs" / args.size / args.backend / "pretrain"
    training_args = build_training_args(cfg, output, resume=bool(args.resume))
    if device == "cpu":
        training_args = replace(training_args, use_cpu=True, bf16=False, fp16=False,
                                tf32=False, torch_compile=False, torch_compile_backend=None,
                                torch_compile_mode=None, optim="adamw_torch")
    if training_args.device.type != device:
        raise RuntimeError("Trainer device differs from the requested control device; run CPU controls with CUDA_VISIBLE_DEVICES='' in a fresh process")
    training_args.report_to = ["wandb"]
    training_args.run_name = f"hf-control-{args.size}-{args.backend}"
    training_args.logging_nan_inf_filter = False
    checkpoint = resolve_training_checkpoint(output, resume=args.resume, audit_filename=AUDIT,
        world_size=1, require_scaler=training_args.fp16)
    contract = {"version": SCHEMA, "kind": "diagnostic_not_production_pretraining", "backend": args.backend,
        "config": cfg, "effective_model_config": config.to_dict(), "data": data_identity, "common_validation_identity": common_identity,
        "seed": args.seed, "device_request": args.device, "versions": versions(),
        "implementation": implementation_identity(), "training_args": training_args.to_dict()}
    validate_or_write_run_audit(output, contract, audit_filename=AUDIT, schema_version=SCHEMA,
                               resume=bool(args.resume), write=True)
    configure_torch_runtime(log)
    if device == "cpu":
        torch.set_float32_matmul_precision("highest")
    set_seed(args.seed)
    model = SLMForCausalLM(config)
    initial_sha = state_digest(model)
    if args.backend == "llama":
        from export.export import _convert_to_native_llama
        native = _convert_to_native_llama(model, tokenizer, torch.float32)
        if state_digest(native) != initial_sha:
            raise RuntimeError("Reference conversion did not preserve all initial tensors")
        del model
        model = native
    # Native construction consumes RNG; reset so dropout/data ordering start alike.
    set_seed(args.seed)
    experiment = wandb.init(project=os.environ["WANDB_PROJECT"], name=training_args.run_name,
                            config={"control_contract_sha256": stable_digest(contract), "backend": args.backend,
                                    "seed": args.seed, "initial_state_sha256": initial_sha})
    try:
        class FiniteLoss(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key, value in (logs or {}).items():
                    if (key == "loss" or key.endswith("_loss") or key == "grad_norm") and not math.isfinite(float(value)):
                        raise FloatingPointError(f"Non-finite {key} at step {state.global_step}")

        callbacks = [FiniteLoss()]
        probe_cfg = cfg.get("generation_probes", {})
        if probe_cfg.get("enabled", True):
            from pretrain.diagnostics import make_probe_callback
            tokenizer_dir = directory.parent / "tokenizer"
            if not tokenizer_dir.is_dir():
                tokenizer_dir = args.tokenizer_dir or BASE_DATA_DIR / "runs" / args.size / "tokenizer"
            callbacks.append(make_probe_callback(tokenizer_dir, output, probe_cfg))
            log.info(
                "Training-time qualitative probes enabled: every_steps=%s explicit_steps=%s at_final=%s",
                probe_cfg.get("every_steps", 5000), probe_cfg.get("steps", []), probe_cfg.get("at_final", True),
            )
        trainer = SLMTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                             callbacks=callbacks)
        if not checkpoint:
            baseline = trainer.evaluate(metric_key_prefix="baseline_validation")
            if not math.isfinite(baseline["baseline_validation_loss"]):
                raise RuntimeError("Non-finite baseline validation loss")
            if common_val is not None:
                baseline.update(trainer.evaluate(eval_dataset=common_val, metric_key_prefix="baseline_common_validation"))
            atomic_write_json(output / "baseline.json", {"metrics": baseline, "initial_state_sha256": initial_sha})
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        final = output / "final"
        trainer.save_model(str(final))
        tokenizer.save_pretrained(final / "tokenizer")
        shutil.copy2(output / AUDIT, final / AUDIT)
        validation = trainer.evaluate(metric_key_prefix="final_validation")
        test = trainer.evaluate(eval_dataset=load_test(directory, config.max_position_embeddings), metric_key_prefix="final_test")
        metrics = {**validation, **test}
        if common_val is not None:
            metrics.update(trainer.evaluate(eval_dataset=common_val, metric_key_prefix="final_common_validation"))
        if any(not math.isfinite(value) for key, value in metrics.items() if key.endswith("_loss")):
            raise RuntimeError("Non-finite final evaluation loss")
        try:
            probes = generate_probes(trainer.accelerator.unwrap_model(trainer.model), tokenizer)
            probe_error = None
        except Exception as exc:
            log.exception("Generation failed; preserving metrics and the diagnostic checkpoint")
            probes, probe_error = [], str(exc)
        report = {"contract_sha256": stable_digest(contract), "initial_state_sha256": initial_sha,
            "training_metrics": result.metrics, "metrics": metrics, "selected_training_tokens": train_ds.token_budget(),
            "global_step": trainer.state.global_step, "consumed_input_tokens": trainer.state.num_input_tokens_seen,
            "probes": probes, "probe_error": probe_error,
            "interpretation": "Control measurements, not a model-vs-data verdict. Curated and HF validation losses use different distributions."}
        atomic_write_json(output / "learning_report.json", report)
        log.info("Reference training report: %s", output / "learning_report.json")
        if probe_error is not None:
            raise RuntimeError("Reference generation failed; inspect learning_report.json")
        return report
    finally:
        experiment.finish()


def main(argv=None, *, sanity=False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv, sanity=sanity)
    if args.stage == "all":
        # Accelerate's process-global precision state must not leak from the FP32
        # numerical control into BF16 training. Each phase is a fresh invocation,
        # with the same immutable inputs and its own run-directory lock.
        original = list(sys.argv[1:] if argv is None else argv)
        forwarded = []
        position = 0
        while position < len(original):
            value = original[position]
            if value == "--stage":
                position += 2
                continue
            if not value.startswith("--stage="):
                forwarded.append(value)
            position += 1
        entrypoint = ROOT / "scripts" / ("sanity_train.py" if sanity else "pretrain_hf_125m.py")
        for phase in ("prepare", "check", "train"):
            subprocess.run([sys.executable, str(entrypoint), *forwarded, "--stage", phase], check=True)
        return
    cfg = read_recipe(args)
    local_tokenizer = None if args.reference_tokenizer else (args.tokenizer_dir or BASE_DATA_DIR / "runs" / args.size / "tokenizer")
    if args.eval_tokenized_dir is not None and args.tokenized_dir is None:
        # A common heldout must not be knowingly reintroduced by HF selection.
        # This excludes normalized exact documents only, not arbitrary near matches.
        for split in ("val", "test"):
            text_path = args.eval_tokenized_dir.parent / "validated" / f"{split}.jsonl"
            meta = json.loads((args.eval_tokenized_dir / f"{split}.json").read_text())
            if not text_path.is_file() or sha256_file(text_path) != meta["input_sha256"]:
                raise RuntimeError("Shared evaluation requires its original adjacent validated val/test JSONL for exclusion")
            if text_path not in args.exclude_jsonl:
                args.exclude_jsonl.append(text_path)
    args.run_dir = validate_run_dir(args.run_dir, [args.config, local_tokenizer, args.tokenized_dir,
                                                  args.eval_tokenized_dir, *args.exclude_jsonl])
    with run_lock(args.run_dir):
        directory, tokenizer, identity = get_inputs(args, cfg)
        if args.stage in ("check", "all"):
            from scripts.sanity_train import compare_implementations
            compare_implementations(args, cfg, directory, tokenizer, identity)
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if args.stage in ("train", "pretrain", "all"):
            run_training(args, cfg, directory, tokenizer, identity)
        if args.stage == "prepare":
            log.info("Reference inputs ready: %s. No model trained.", directory)


if __name__ == "__main__":
    main()
