"""Real existing DataTrove/MinHash integration; run on the curation stack."""
import hashlib
import json
import random
import shutil

import pytest

pytest.importorskip("datatrove", reason="Run with requirements-curation.txt")
from config.holdout import jsonl_identity, verify_jsonl_contract
from curator.filters.dedup import MINHASH_CONTRACT
from curator.frozen_split import freeze_test_split
from curator.state import write_manifest

pytestmark = pytest.mark.slow


def old_blend(path):
    path.mkdir()
    rng = random.Random(19)
    for split, count in (("train", 60), ("val", 4)):
        rows = []
        for number in range(count):
            # Distinct five-grams avoid accidental lexical LSH clusters in a tiny fixture.
            words = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=9)) for _ in range(70)]
            rows.append({"id": f"{split}-{number}", "source": "fineweb", "text": " ".join(words)})
        (path / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_manifest(path, stage="blend", contract={"fixture": 1}, input_signature="fixture",
                   output_pattern="*.json*")


def test_real_three_pair_policy_preserves_validation_and_freezes_selection(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    old_blend(first)
    shutil.copytree(first, second)
    val_bytes = (first / "val.jsonl").read_bytes()
    result = freeze_test_split(first, size="smoke", test_fraction=.1, workers=1)
    other = freeze_test_split(second, size="smoke", test_fraction=.1, workers=1)
    assert (first / "val.jsonl").read_bytes() == val_bytes
    assert (first / "test.jsonl").read_bytes() == (second / "test.jsonl").read_bytes()
    assert result["contract"]["splits"] == other["contract"]["splits"]
    assert result["contract"]["minhash"] == MINHASH_CONTRACT
    assert set(result["contract"]["pair_audits"]) == {"train_val", "train_test", "test_val"}
    for report in result["contract"]["pair_audits"].values():
        assert report["near"]["passed"] and report["final_exact"]["passed"]
    assert freeze_test_split(first, size="smoke", test_fraction=.1, workers=1) == result
    with pytest.raises(RuntimeError, match="cannot change"):
        freeze_test_split(first, size="smoke", seed=99, test_fraction=.1, workers=1)
    verify_jsonl_contract(first, stage="curated")
    (first / "test_membership.jsonl").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="membership"):
        verify_jsonl_contract(first)


def test_incorrect_minhash_removal_accounting_rolls_back(tmp_path, monkeypatch):
    import curator.filters.near_overlap as near
    source = tmp_path / "curated"
    old_blend(source)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    def incorrect(candidate, reference, scratch, **kwargs):
        scratch.mkdir(parents=True)
        return {"passed": True, "removed_train_documents": 1,
                "removed_train_characters": 0, "train_documents": jsonl_identity(candidate)["documents"]}
    monkeypatch.setattr(near, "audit_minhash_split_overlap", incorrect)
    with pytest.raises(RuntimeError, match="physical removals"):
        freeze_test_split(source, size="smoke", test_fraction=.1, workers=1)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == originals
    assert not (tmp_path / ".curated.freeze-pending").exists()
