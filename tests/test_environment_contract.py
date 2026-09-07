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
