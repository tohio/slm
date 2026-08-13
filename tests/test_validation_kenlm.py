import json
from pathlib import Path

from validation.scripts.validate import (
    PerplexityAudit,
    validate_manual_split,
)


class FakeKenLM:
    def perplexity(self, text: str) -> float:
        return 900.0 if "unusual" in text else 40.0


def _write_records(path: Path) -> None:
    records = [
        {"source": "wikipedia", "text": "A conventional sentence."},
        {"source": "wikipedia", "text": "An unusual technical sentence."},
        {"source": "synthetic_arithmetic", "text": "2 + 2 = 4"},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_kenlm_is_report_only_without_explicit_threshold(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_records(input_path)

    stats = validate_manual_split(
        input_path,
        output_path,
        FakeKenLM(),
        None,
        "train",
        perplexity_sample_size=10,
    )

    assert stats["kept"] == 3
    assert stats["rejected_perplexity"] == 0
    assert stats["perplexity_audit"]["policy"] == "report_only"
    assert stats["perplexity_audit"]["documents_scored"] == 2
    wikipedia = stats["perplexity_audit"]["by_source"]["wikipedia"]
    assert wikipedia["documents_scored"] == 2
    assert wikipedia["min"] == 40.0
    assert wikipedia["max"] == 900.0


def test_explicit_kenlm_threshold_filters_eligible_prose(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_records(input_path)

    stats = validate_manual_split(
        input_path,
        output_path,
        FakeKenLM(),
        500.0,
        "train",
        perplexity_sample_size=10,
    )

    assert stats["kept"] == 2
    assert stats["rejected_perplexity"] == 1
    assert stats["perplexity_audit"]["policy"] == "explicit_threshold"
    assert stats["perplexity_audit"]["threshold"] == 500.0


def test_perplexity_distribution_sample_is_bounded_and_deterministic():
    first = PerplexityAudit(sample_size=3)
    second = PerplexityAudit(sample_size=3)
    rows = [(f"document {index}", float(index + 1)) for index in range(20)]

    for text, score in rows:
        first.observe("wikipedia", text, score)
    for text, score in reversed(rows):
        second.observe("wikipedia", text, score)

    first_report = first.report(policy="report_only", threshold=None)
    second_report = second.report(policy="report_only", threshold=None)

    assert first_report == second_report
    assert first_report["by_source"]["wikipedia"]["documents_scored"] == 20
    assert first_report["by_source"]["wikipedia"]["sampled_documents"] == 3
