"""Focused tests for the finalized-corpus sensitive-content audit."""

import json

import pytest

from curator.filters.sensitive_content import (
    DETECTORS,
    SENSITIVE_CONTENT_CONTRACT,
    SensitiveContentAuditor,
)


@pytest.mark.parametrize(
    ("detector_name", "value"),
    [
        ("pem_private_key", "-----BEGIN PRIVATE KEY-----"),
        ("aws_access_key_id", "AKIA" + "A" * 16),
        ("github_token", "ghp_" + "A" * 36),
        ("huggingface_token", "hf_" + "A" * 34),
        ("google_api_key", "AIza" + "A" * 35),
        ("slack_token", "xoxb-" + "A" * 20),
        ("stripe_live_secret_key", "sk_live_" + "A" * 20),
        ("sk_prefixed_api_key", "sk-proj-" + "A" * 30),
    ],
)
def test_credential_detector_shapes(detector_name, value):
    detector = next(item for item in DETECTORS if item.name == detector_name)

    assert detector.pattern.search(value)


def test_credential_material_requires_review_without_retaining_matched_value():
    token = "ghp_" + "A" * 36
    auditor = SensitiveContentAuditor()
    auditor.observe(
        "train",
        4,
        {
            "id": "doc-4",
            "source": "stack_v1",
            "text": f"export GITHUB_TOKEN={token}",
        },
    )

    report = auditor.report({"train": 1, "validation": 0})
    serialized = json.dumps(report)

    assert report["passed"] is True
    assert report["credential_review_required"] is True
    assert report["credential_findings"] == 1
    assert report["identifier_findings"] == 0
    assert report["documents_with_credential_findings"] == 1
    assert report["credential_findings_by_detector"] == {"github_token": 1}
    assert report["samples"][0]["findings"][0]["match_length"] == len(token)
    assert token not in serialized
    assert "export GITHUB_TOKEN" not in serialized


def test_personal_identifiers_are_review_only():
    auditor = SensitiveContentAuditor()
    auditor.observe(
        "validation",
        2,
        {
            "id": "doc-2",
            "source": "common_crawl",
            "text": (
                "Contact person@example.org or +1 (212) 555-0123. "
                "Identifier: 123-45-6789."
            ),
        },
    )

    report = auditor.report({"train": 0, "validation": 1})

    assert report["passed"] is True
    assert report["credential_findings"] == 0
    assert report["identifier_findings"] == 3
    assert report["findings_by_detector"] == {
        "email_address": 1,
        "international_phone_number": 1,
        "us_ssn": 1,
    }


def test_all_matches_are_counted_when_sample_details_are_bounded():
    auditor = SensitiveContentAuditor(sample_matches_per_document=1)
    auditor.observe(
        "train",
        1,
        {
            "source": "fineweb",
            "text": "first@example.org second@example.org third@example.org",
        },
    )

    report = auditor.report({"train": 1})

    assert report["findings"] == 3
    assert report["identifier_findings"] == 3
    assert len(report["samples"][0]["findings"]) == 1
    assert report["sample_findings_truncated_documents"] == 1


def test_contract_has_explicit_review_only_policy():
    severities = {detector.severity for detector in DETECTORS}

    assert severities == {"credential_review", "identifier_review"}
    assert SENSITIVE_CONTENT_CONTRACT["enforcement_policy"] == (
        "report only pending corpus incidence review"
    )


def test_incomplete_scan_fails_closed():
    auditor = SensitiveContentAuditor()
    auditor.observe(
        "train",
        1,
        {"source": "fineweb", "text": "ordinary document"},
    )

    report = auditor.report({"train": 2, "validation": 0})

    assert report["split_counts_match"] is False
    assert report["passed"] is False
