"""Build and validate the tokenizer-measured pretraining source mix."""

from __future__ import annotations

from typing import Any

from config.data_mix import ALL_SOURCES, CODE_SUBMIX, DATA_MIX
from curator.state import stable_digest


REALIZED_MIXTURE_SCHEMA_VERSION = 1


def configured_source_shares() -> dict[str, float]:
    """Expand the logical code bucket into concrete, corpus-wide shares."""
    code_share = float(DATA_MIX["code"]["pct"]) / 100.0
    shares = {
        source: float(DATA_MIX[source]["pct"]) / 100.0
        for source in DATA_MIX
        if source != "code"
    }
    shares.update(
        {
            source: code_share * (float(entry["pct"]) / 100.0)
            for source, entry in CODE_SUBMIX.items()
        }
    )
    if set(shares) != set(ALL_SOURCES):
        raise RuntimeError("Concrete source-share expansion is incomplete")
    return shares


def mixture_contract() -> dict[str, Any]:
    """Return the policy input against which realized shares are reported."""
    return {
        "schema_version": REALIZED_MIXTURE_SCHEMA_VERSION,
        "configured_source_shares": configured_source_shares(),
        "deviation_policy": {
            "threshold": None,
            "enforcement": "report_only",
            "reason": "requires review of tokenizer-measured corpus incidence",
        },
    }


def _validated_split_counts(split: str, metadata: dict) -> dict[str, dict[str, int]]:
    source_counts = metadata.get("source_counts")
    if not isinstance(source_counts, dict):
        raise RuntimeError(f"{split} metadata is missing source_counts")

    unknown = sorted(set(source_counts) - set(ALL_SOURCES))
    if unknown:
        raise RuntimeError(
            f"{split} source_counts violates the configured source set: "
            f"unknown={unknown}"
        )

    for source, counts in source_counts.items():
        if not isinstance(counts, dict):
            raise RuntimeError(f"{split} source_counts[{source!r}] must be an object")
        for field in ("documents", "tokens"):
            value = counts.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(
                    f"{split} source_counts[{source!r}][{field!r}] "
                    "must be a non-negative integer"
                )

    total_docs = sum(row["documents"] for row in source_counts.values())
    total_tokens = sum(row["tokens"] for row in source_counts.values())
    if total_docs != metadata.get("n_docs"):
        raise RuntimeError(
            f"{split} source document counts sum to {total_docs}, "
            f"metadata records {metadata.get('n_docs')!r}"
        )
    if total_tokens != metadata.get("n_tokens"):
        raise RuntimeError(
            f"{split} source token counts sum to {total_tokens}, "
            f"metadata records {metadata.get('n_tokens')!r}"
        )
    if total_docs <= 0 or total_tokens <= 0:
        raise RuntimeError(f"{split} tokenized corpus must be non-empty")
    return {
        source: dict(source_counts.get(source, {"documents": 0, "tokens": 0}))
        for source in ALL_SOURCES
    }


def _build_realized_mixture_report(train_metadata: dict, val_metadata: dict) -> dict:
    split_metadata = {"train": train_metadata, "val": val_metadata}
    validated = {
        split: _validated_split_counts(split, metadata)
        for split, metadata in split_metadata.items()
    }
    contract = mixture_contract()
    intended = contract["configured_source_shares"]
    total_tokens = sum(metadata["n_tokens"] for metadata in split_metadata.values())
    total_docs = sum(metadata["n_docs"] for metadata in split_metadata.values())

    sources: dict[str, dict[str, Any]] = {}
    for source in ALL_SOURCES:
        tokens = sum(validated[split][source]["tokens"] for split in validated)
        documents = sum(
            validated[split][source]["documents"] for split in validated
        )
        realized_share = tokens / total_tokens
        sources[source] = {
            "configured_share": intended[source],
            "realized_token_share": realized_share,
            "deviation_percentage_points": 100.0 * (
                realized_share - intended[source]
            ),
            "tokens": tokens,
            "documents": documents,
            "splits": {
                split: dict(validated[split][source])
                for split in ("train", "val")
            },
        }

    absent_sources = sorted(
        source
        for source, row in sources.items()
        if row["documents"] <= 0 or row["tokens"] <= 0
    )
    if absent_sources:
        raise RuntimeError(
            "Configured sources absent from the combined tokenized corpus: "
            f"{absent_sources}"
        )

    code_tokens = sum(sources[source]["tokens"] for source in CODE_SUBMIX)
    top_level = {
        source: {
            "configured_share": float(entry["pct"]) / 100.0,
            "realized_token_share": (
                code_tokens / total_tokens
                if source == "code"
                else sources[source]["realized_token_share"]
            ),
        }
        for source, entry in DATA_MIX.items()
    }
    for row in top_level.values():
        row["deviation_percentage_points"] = 100.0 * (
            row["realized_token_share"] - row["configured_share"]
        )

    return {
        "schema_version": REALIZED_MIXTURE_SCHEMA_VERSION,
        "status": "passed_structural_checks_report_only",
        "contract": contract,
        "contract_sha256": stable_digest(contract),
        "split_metadata_sha256": {
            split: stable_digest(metadata)
            for split, metadata in split_metadata.items()
        },
        "total_tokens": total_tokens,
        "total_documents": total_docs,
        "sources": sources,
        "top_level": top_level,
    }


def build_realized_mixture_report(train_metadata: dict, val_metadata: dict) -> dict:
    """Compare configured shares with authoritative tokenizer token counts."""
    report = _build_realized_mixture_report(train_metadata, val_metadata)
    validate_realized_mixture_report(report, train_metadata, val_metadata)
    return report


def validate_realized_mixture_report(
    report: dict,
    train_metadata: dict,
    val_metadata: dict,
) -> None:
    """Reject a missing, stale, or structurally incomplete mixture report."""
    expected = _build_realized_mixture_report(train_metadata, val_metadata)
    if report.get("schema_version") != REALIZED_MIXTURE_SCHEMA_VERSION:
        raise RuntimeError("Realized-mixture report has an unsupported schema")
    contract = report.get("contract")
    if contract != mixture_contract():
        raise RuntimeError("Realized-mixture report does not match the configured mix")
    if report.get("contract_sha256") != stable_digest(contract):
        raise RuntimeError("Realized-mixture report contract checksum mismatch")
    expected_metadata = {
        "train": stable_digest(train_metadata),
        "val": stable_digest(val_metadata),
    }
    if report.get("split_metadata_sha256") != expected_metadata:
        raise RuntimeError("Realized-mixture report is stale for tokenized metadata")
    if set(report.get("sources", {})) != set(ALL_SOURCES):
        raise RuntimeError("Realized-mixture report has an incomplete source set")
    if report.get("total_tokens") != (
        train_metadata.get("n_tokens", 0) + val_metadata.get("n_tokens", 0)
    ):
        raise RuntimeError("Realized-mixture report total token count mismatch")
    if report != expected:
        raise RuntimeError("Realized-mixture report contents do not match token counts")
