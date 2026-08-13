"""Full-corpus sensitive-content audit without retaining matched values."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:
    name: str
    category: str
    severity: str
    pattern: re.Pattern[str]


DETECTORS = (
    Detector(
        "pem_private_key",
        "credential_material",
        "credential_review",
        re.compile(
            r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"
        ),
    ),
    Detector(
        "aws_access_key_id",
        "credential_material",
        "credential_review",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    Detector(
        "github_token",
        "credential_material",
        "credential_review",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,255}|"
            r"github_pat_[A-Za-z0-9_]{22,255})(?![A-Za-z0-9_])"
        ),
    ),
    Detector(
        "huggingface_token",
        "credential_material",
        "credential_review",
        re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{34,255}(?![A-Za-z0-9_])"),
    ),
    Detector(
        "google_api_key",
        "credential_material",
        "credential_review",
        re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])"),
    ),
    Detector(
        "slack_token",
        "credential_material",
        "credential_review",
        re.compile(
            r"(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{10,200}"
            r"(?![A-Za-z0-9-])"
        ),
    ),
    Detector(
        "stripe_live_secret_key",
        "credential_material",
        "credential_review",
        re.compile(
            r"(?<![A-Za-z0-9_])sk_live_[A-Za-z0-9]{16,255}"
            r"(?![A-Za-z0-9_])"
        ),
    ),
    Detector(
        "sk_prefixed_api_key",
        "credential_material",
        "credential_review",
        re.compile(
            r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,200}"
            r"(?![A-Za-z0-9_-])"
        ),
    ),
    Detector(
        "email_address",
        "personal_identifier",
        "identifier_review",
        re.compile(
            r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
            r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
            r"(?![A-Za-z0-9-])"
        ),
    ),
    Detector(
        "international_phone_number",
        "personal_identifier",
        "identifier_review",
        re.compile(
            r"(?<!\w)\+(?=(?:[ .()-]*\d){8,15}(?!\d))"
            r"(?:[0-9][ .()-]*){7,14}[0-9](?!\w)"
        ),
    ),
    Detector(
        "us_ssn",
        "personal_identifier",
        "identifier_review",
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    ),
)

SENSITIVE_CONTENT_CONTRACT = {
    "schema_version": 1,
    "detectors": [
        {
            "name": detector.name,
            "category": detector.category,
            "severity": detector.severity,
            "pattern_sha256": hashlib.sha256(
                detector.pattern.pattern.encode("utf-8")
            ).hexdigest(),
        }
        for detector in DETECTORS
    ],
    "enforcement_policy": "report only pending corpus incidence review",
    "limitations": [
        "pattern matches do not establish that a credential is active",
        (
            "generic passwords, connection strings, and unprefixed secrets "
            "are not detected"
        ),
        "names and street addresses are not detected",
        "email, phone, and SSN-shaped matches require policy review",
    ],
}


class SensitiveContentAuditor:
    """Incremental detector that can share the finalized-corpus audit pass."""

    def __init__(
        self,
        *,
        sample_limit: int = 20,
        sample_matches_per_document: int = 20,
    ):
        if sample_limit < 0:
            raise ValueError("sample_limit must be non-negative")
        if sample_matches_per_document < 1:
            raise ValueError("sample_matches_per_document must be positive")
        self.sample_limit = sample_limit
        self.sample_matches_per_document = sample_matches_per_document
        self.scanned_documents = 0
        self.findings = 0
        self.credential_findings = 0
        self.identifier_findings = 0
        self.documents_with_findings = 0
        self.documents_with_credential_findings = 0
        self.findings_by_detector: Counter[str] = Counter()
        self.findings_by_source: Counter[str] = Counter()
        self.findings_by_split: Counter[str] = Counter()
        self.credential_findings_by_detector: Counter[str] = Counter()
        self.samples: list[dict] = []
        self.sample_findings_truncated_documents = 0

    def observe(self, split: str, line_number: int, record: dict) -> None:
        """Inspect one already-validated finalized-corpus record."""
        self.scanned_documents += 1
        text = record["text"]
        source = record.get("source")
        record_id = record.get("id")
        document_findings: list[dict] = []
        document_finding_count = 0
        credential_in_document = False
        sample_findings_truncated = False

        for detector in DETECTORS:
            for match in detector.pattern.finditer(text):
                matched_value = match.group(0)
                finding = {
                    "detector": detector.name,
                    "category": detector.category,
                    "severity": detector.severity,
                    "match_sha256": hashlib.sha256(
                        matched_value.encode("utf-8")
                    ).hexdigest(),
                    "match_length": len(matched_value),
                }
                document_finding_count += 1
                if len(document_findings) < self.sample_matches_per_document:
                    document_findings.append(finding)
                else:
                    sample_findings_truncated = True
                self.findings += 1
                self.findings_by_detector[detector.name] += 1
                self.findings_by_source[str(source)] += 1
                self.findings_by_split[split] += 1
                if detector.severity == "credential_review":
                    credential_in_document = True
                    self.credential_findings += 1
                    self.credential_findings_by_detector[detector.name] += 1
                else:
                    self.identifier_findings += 1
        if not document_finding_count:
            return
        self.sample_findings_truncated_documents += int(
            sample_findings_truncated
        )
        self.documents_with_findings += 1
        self.documents_with_credential_findings += int(credential_in_document)
        if len(self.samples) < self.sample_limit:
            self.samples.append({
                "split": split,
                "line": line_number,
                "record_id": str(record_id) if record_id is not None else None,
                "source": source,
                "findings": document_findings,
            })

    def report(self, split_documents: dict[str, int]) -> dict:
        """Return the durable, value-free audit report."""
        expected_documents = sum(int(value) for value in split_documents.values())
        split_counts_match = self.scanned_documents == expected_documents
        return {
            "schema_version": 1,
            "algorithm": "sensitive_content_regex_audit",
            "scope": "full_corpus",
            "contract": SENSITIVE_CONTENT_CONTRACT,
            "split_documents": dict(sorted(split_documents.items())),
            "scanned_documents": self.scanned_documents,
            "split_counts_match": split_counts_match,
            "documents_with_findings": self.documents_with_findings,
            "documents_with_credential_findings": (
                self.documents_with_credential_findings
            ),
            "findings": self.findings,
            "credential_findings": self.credential_findings,
            "identifier_findings": self.identifier_findings,
            "findings_by_detector": dict(sorted(self.findings_by_detector.items())),
            "credential_findings_by_detector": dict(
                sorted(self.credential_findings_by_detector.items())
            ),
            "findings_by_source": dict(sorted(self.findings_by_source.items())),
            "findings_by_split": dict(sorted(self.findings_by_split.items())),
            "sample_findings_truncated_documents": (
                self.sample_findings_truncated_documents
            ),
            "review_required": self.findings > 0,
            "credential_review_required": self.credential_findings > 0,
            "passed": split_counts_match,
            "samples": self.samples,
        }
