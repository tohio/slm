"""CPU-only checks; actual HF generation is tested with the native model stack."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pretrain.benchmark import candidate_configs
from pretrain.diagnostics import diagnostic_mode, generic_cases, probe_settings, validate_final_binding


def test_fixed_probes_are_separate_and_deterministic_by_default():
    rows = generic_cases()
    assert len(rows) == 5
    assert {r["category"] for r in rows} == {"generic_generalization"}
    assert rows[0]["prompt"] == "The history of computing began"
    assert probe_settings() == {"do_sample": False, "max_new_tokens": 64}
    with pytest.raises(ValueError):
        probe_settings({"max_new_tokens": 0})


def test_diagnostics_preserve_training_mode_and_random_states():
    model = torch.nn.Linear(2, 2)
    model.device = torch.device("cpu")
    model.train()
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with diagnostic_mode(model):
        assert not model.training
        random.random(); np.random.random(); torch.rand(2)
    assert model.training
    assert py == random.getstate()
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert torch.equal(cpu, torch.get_rng_state())


def test_benchmark_only_changes_microbatch_and_accumulation():
    cfg = {"model": {"max_position_embeddings": 2048}, "training": {
        "micro_batch_size": 2, "gradient_accumulation_steps": 8, "max_steps": 100,
        "cross_size_max_train_tokens": 1400001792}, "optimizer": {"lr": .0003}}
    original = copy.deepcopy(cfg)
    for candidate in candidate_configs(cfg, [2, 4, 8, 16], 3):
        training = candidate["training"]
        assert training["micro_batch_size"] * training["gradient_accumulation_steps"] * 3 == 48
        normalized = copy.deepcopy(candidate)
        normalized["training"]["micro_batch_size"] = 2
        normalized["training"]["gradient_accumulation_steps"] = 8
        assert normalized == original
    assert cfg == original
    with pytest.raises(ValueError, match="preserve"):
        candidate_configs(cfg, [3], 3)


def test_final_binding_rejects_checkpoint_without_frozen_training_audit(tmp_path):
    with pytest.raises(RuntimeError, match="audit|frozen|trained|provenance"):
        validate_final_binding(tmp_path, tmp_path / "tokenized", "tokenizer")


def test_saved_checkpoint_resolves_original_dataset_with_parent_audit(tmp_path):
    from pretrain.diagnostics import checkpoint_probe_inputs
    from curator.state import atomic_write_json, stable_digest
    checkpoint = tmp_path / "results" / "runs" / "mini" / "pretrain" / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    tokenizer = tmp_path / "data" / "runs" / "350m" / "tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text("{}")
    contract = {"run_size": "mini", "dataset_size": "350m", "tokenizer": {"sha256": "test"}}
    atomic_write_json(checkpoint.parent / "pretrain_run_audit.json",
                      {"contract": contract, "contract_sha256": stable_digest(contract)})
    actual, saved = checkpoint_probe_inputs(checkpoint, data_root=tmp_path / "data")
    assert actual == tokenizer and saved == contract
    with pytest.raises(RuntimeError, match="No matched"):
        checkpoint_probe_inputs(checkpoint, data_root=tmp_path / "other-data")
