"""CPU tests of frozen identities, token integrity, and cross-size resolution."""
import json
from pathlib import Path

import numpy as np
import pytest

from config.holdout import SPLITS, jsonl_identity, load_contract, verify_jsonl_contract
from config.paths import resolve_dataset_paths
from pretrain.data.dataset import PretrainingDataset, load_train_val
from pretrain.data.tokenize_data import verify_dataset
from pretrain.schedule import resolve_train_token_limit
from tests.frozen_helpers import make_bundle


@pytest.mark.parametrize("size", ["smoke", "mini", "125m", "350m", "1b"])
def test_cross_size_paths_keep_matched_artifact_set(tmp_path, size):
    paths = resolve_dataset_paths(size, "350m", data_root=tmp_path)
    assert paths["size"] == size
    for stage in ("tokenizer", "tokenized", "validated", "metadata"):
        assert paths[stage] == tmp_path / "runs" / "350m" / stage
    assert resolve_dataset_paths(size, data_root=tmp_path)["dataset_size"] == size


def test_interrupted_restore_is_not_usable(tmp_path):
    root = tmp_path / "runs" / "350m"
    root.mkdir(parents=True)
    (root / "_RESTORE_PENDING.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Interrupted"):
        resolve_dataset_paths("mini", "350m", data_root=tmp_path)


@pytest.mark.parametrize("split", SPLITS)
def test_frozen_full_byte_identity_rejects_tamper(tmp_path, split):
    root = make_bundle(tmp_path)
    verify_jsonl_contract(root / "validated", stage="validated")
    path = root / "validated" / f"{split}.jsonl"
    path.write_bytes(path.read_bytes().replace(b"document", b"DOCUMENT", 1))
    with pytest.raises(RuntimeError, match=f"Frozen {split}"):
        verify_jsonl_contract(root / "validated")


def test_frozen_envelope_checksum_and_membership(tmp_path):
    root = make_bundle(tmp_path) / "validated"
    path = root / "test_contract.json"
    data = json.loads(path.read_text())
    data["contract"]["splits"]["test"]["documents"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="checksum"):
        load_contract(root)


@pytest.mark.parametrize("split", SPLITS)
def test_binary_integrity_counts_range_and_full_checksum(tmp_path, split):
    root = make_bundle(tmp_path) / "tokenized"
    binary = root / f"{split}.bin"
    verify_dataset(binary, binary.with_suffix(".json"))
    tokens = np.fromfile(binary, dtype=np.uint16)
    tokens[len(tokens) // 2] = 300
    tokens.tofile(binary)
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        verify_dataset(binary, binary.with_suffix(".json"))


def test_training_budget_does_not_slice_or_open_test(tmp_path, monkeypatch):
    root = make_bundle(tmp_path) / "tokenized"
    opened = []
    original = np.memmap
    def tracked(filename, *args, **kwargs):
        opened.append(Path(filename).name)
        return original(filename, *args, **kwargs)
    monkeypatch.setattr(np, "memmap", tracked)
    train, val = load_train_val(root, seq_len=4, max_train_tokens=9)
    assert len(train) == 2 and train.token_budget()["selected_unique_tokens"] == 8
    assert val.n_tokens == train.available_tokens and val.n_tokens > train.n_tokens
    assert opened == ["train.bin", "val.bin"]
    assert np.array_equal(train[0]["labels"], train[0]["input_ids"])
    with pytest.raises(ValueError, match="Only training"):
        PretrainingDataset(root / "test.bin", seq_len=4, split="test", max_tokens=4)


def test_selected_unique_tokens_are_not_double_counted_with_overlap(tmp_path):
    root = make_bundle(tmp_path) / "tokenized"
    data = PretrainingDataset(root / "train.bin", seq_len=4, stride=2, max_tokens=10)
    assert data.token_budget()["selected_unique_tokens"] == 10
    assert data.token_budget()["total_training_tokens"] == 16


def test_cross_size_budget_is_model_config_controlled():
    cfg = {"training": {"cross_size_max_train_tokens": 1400001792}}
    assert resolve_train_token_limit(cfg, run_size="mini", dataset_size="350m") == 1400001792
    assert resolve_train_token_limit(cfg, run_size="mini", dataset_size="mini") is None
    cfg["training"]["max_train_tokens"] = 9999
    assert resolve_train_token_limit(cfg, run_size="mini", dataset_size="350m") == 9999
    with pytest.raises(ValueError):
        resolve_train_token_limit({"training": {}}, run_size="mini", dataset_size="350m")


def test_jsonl_accounting_rejects_blank_or_missing_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"x"}\n')
    with pytest.raises(RuntimeError, match="text/source"):
        jsonl_identity(path)
