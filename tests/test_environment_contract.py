"""Keep installation pins and runtime verification on one version contract."""

from pathlib import Path
import re

from infra.verify_environment import CURATION_EXPECTED, EXPECTED


ROOT = Path(__file__).resolve().parents[1]


def _exact_pins(path: Path) -> dict[str, str]:
    pins = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^;\s]+)",
            line,
        )
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_runtime_verifier_matches_training_requirements():
    pins = _exact_pins(ROOT / "requirements-training.txt")

    missing = sorted(set(EXPECTED) - set(pins))
    assert not missing, f"Runtime verifier packages are not pinned: {missing}"

    mismatches = {
        package: {"verifier": expected, "requirements": pins[package]}
        for package, expected in EXPECTED.items()
        if pins[package].split("+", 1)[0] != expected
    }
    assert not mismatches, f"Version contract drift: {mismatches}"


def test_runtime_verifier_matches_curation_requirements():
    pins = _exact_pins(ROOT / "requirements-curation.txt")

    missing = sorted(set(CURATION_EXPECTED) - set(pins))
    assert not missing, f"Curation verifier packages are not pinned: {missing}"

    mismatches = {
        package: {"verifier": expected, "requirements": pins[package]}
        for package, expected in CURATION_EXPECTED.items()
        if pins[package].split("+", 1)[0] != expected
    }
    assert not mismatches, f"Curation version contract drift: {mismatches}"


def test_curation_requirements_do_not_import_training_stack():
    requirements = (ROOT / "requirements-curation.txt").read_text(encoding="utf-8")
    assert "-r requirements-training.txt" not in requirements


def test_training_requirements_select_the_cuda_build():
    training = _exact_pins(ROOT / "requirements-training.txt")
    assert training["torch"] == f"{EXPECTED['torch']}+cu130"


def test_model_facing_make_targets_require_training_environment():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in (
        "test-model",
        "test-export",
        "test-export-acceptance",
        "test-training-args",
        "test-training",
        "test-sft-instruct",
        "test-sft-code",
        "test-dpo-chat",
        "test-comparison",
        "test-misc",
        "test-gpu-gate",
    ):
        pattern = rf"(?m)^{re.escape(target)}:[^\n]*\bcheck-training-env\b"
        assert re.search(pattern, makefile), (
            f"{target} must fail closed unless the pinned training stack is active"
        )


def test_role_setup_uses_the_matching_requirements():
    for role, filename in (("curate", "curation"), ("train", "training")):
        script = (ROOT / "infra" / f"setup_{role}.sh").read_text()
        assert f"requirements-{filename}.txt" in script
        assert "install_environment" in script


def test_hf_control_groups_duplicate_documents_and_refuses_unowned_outputs(tmp_path):
    import pytest
    from scripts.pretrain_hf_125m import document_key, document_split, validate_run_dir

    a = document_key("A\u00a0document\nwith  whitespace")
    b = document_key("A document with whitespace")
    assert a == b
    assert document_split(a, 42, .005, .005) == document_split(b, 42, .005, .005)
    assert {document_split(document_key(str(i)), 42, .2, .2) for i in range(100)} == {"train", "val", "test"}
    root = tmp_path / "unowned"
    root.mkdir()
    sentinel = root / "checkpoint"
    sentinel.write_bytes(b"must remain unchanged")
    with pytest.raises(RuntimeError, match="unowned"):
        validate_run_dir(root)
    assert sentinel.read_bytes() == b"must remain unchanged"
    with pytest.raises(ValueError, match="overlaps a read-only input"):
        validate_run_dir(tmp_path / "new", [tmp_path])


def test_hf_control_rejects_legacy_unmanifested_token_cache(tmp_path):
    import pytest
    from scripts.pretrain_hf_125m import verify_reference_bundle

    # Old preallocation must never be mistaken for valid written tokens.
    (tmp_path / "fineweb_mistral.bin").write_bytes(b"\x00" * 1024)
    with pytest.raises(RuntimeError, match="No completed reference-data manifest"):
        verify_reference_bundle(tmp_path)
