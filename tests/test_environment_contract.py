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
    pins = _exact_pins(ROOT / "requirements.txt")

    missing = sorted(set(CURATION_EXPECTED) - set(pins))
    assert not missing, f"Curation verifier packages are not pinned: {missing}"

    mismatches = {
        package: {"verifier": expected, "requirements": pins[package]}
        for package, expected in CURATION_EXPECTED.items()
        if pins[package].split("+", 1)[0] != expected
    }
    assert not mismatches, f"Curation version contract drift: {mismatches}"


def test_curation_requirements_do_not_import_training_stack():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "-r requirements-training.txt" not in requirements


def test_gpu_requirements_select_the_matching_cuda_build():
    training = _exact_pins(ROOT / "requirements-training.txt")
    gpu = _exact_pins(ROOT / "requirements-gpu.txt")

    assert gpu["torch"] == f"{training['torch']}+cu130"


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


def test_cpu_training_install_uses_training_requirements():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^install-training:\n(?P<body>(?:\t.*\n|\n)+?)(?=^[A-Za-z0-9_.-]+:)",
        makefile,
    )
    assert match is not None
    body = match.group("body")
    assert "requirements-training.txt" in body
    assert "verify_environment.py --profile training" in body
