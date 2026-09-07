"""Small real artifact bundles for integrity tests; no ML tokenizer dependency."""
import json
import shutil
from pathlib import Path

import numpy as np

from config.data_mix import ALL_SOURCES
from config.holdout import SPLITS, jsonl_identity, sha256_file, write_contract
from curator.state import atomic_write_json, write_manifest
from pretrain.data.mixture import build_realized_mixture_report


def make_bundle(data_root: Path, size: str = "350m") -> Path:
    root = data_root / "runs" / size
    for stage in ("validated", "tokenized", "tokenizer"):
        (root / stage).mkdir(parents=True)
    identities = {}
    for split in SPLITS:
        path = root / "validated" / f"{split}.jsonl"
        path.write_text("".join(json.dumps({"text": f"{split} document for {source}",
            "source": source}) + "\n" for source in ALL_SOURCES))
        identities[split] = jsonl_identity(path)
    payload = write_contract(root / "validated", {"schema_version": 1, "status": "established",
        "stage": "validated", "size": size, "splits": identities,
        "curated_contract_sha256": "c" * 64})
    shutil.copy2(root / "validated" / "test_contract.json", root / "tokenized" / "test_contract.json")
    # Only the artifact/checksum tests use this minimal tokenizer JSON.
    # Native model/tokenizer tests construct a real tokenizer separately.
    tokenizer_path = root / "tokenizer" / "slm_tokenizer.json"
    tokenizer_path.write_text('{}\n')
    metadata = {}
    for split in SPLITS:
        binary = root / "tokenized" / f"{split}.bin"
        values = np.array([2, 4, 5, 3] * len(ALL_SOURCES), dtype=np.uint16)
        values.tofile(binary)
        meta = {"n_tokens": len(values), "n_docs": len(ALL_SOURCES),
            "dtype": "uint16", "bos_id": 2, "eos_id": 3, "vocab_size": 8,
            "input_sha256": identities[split]["sha256"],
            "binary_sha256": sha256_file(binary), "format_version": "test",
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "tokenizer_file_sha256": sha256_file(tokenizer_path),
            "implementation_sha256": "a" * 64, "test_split_sha256": payload["sha256"],
            "source_counts": {source: {"documents": 1, "tokens": 4} for source in ALL_SOURCES}}
        atomic_write_json(binary.with_suffix(".json"), meta)
        metadata[split] = meta
    atomic_write_json(root / "tokenized" / "token_mixture.json",
        build_realized_mixture_report(*(metadata[s] for s in SPLITS)))
    for stage, pattern in (("validated", "*.json*"), ("tokenized", "[tv]*"), ("tokenizer", "*")):
        write_manifest(root / stage, stage=stage, contract={"fixture": 1},
                       input_signature="fixture-input", output_pattern=pattern)
    return root
