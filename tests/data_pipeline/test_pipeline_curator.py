"""
tests/data_pipeline/test_pipeline_curator.py
---------------------------------------------
Validates real outputs from 'make curate-mini'.

Run after: make curate-mini
Command:   make test-curator

Checks:
    - Raw source directories exist and have shards for all configured sources
    - Filtered shards exist and contain no documents failing quality checks
    - Deduped shards exist and sampled deduped output has no exact duplicates
    - data/curated/train.jsonl exists, is non-empty, contains most sources
    - data/curated/blend_stats.json is correct and complete
    - Synthetic sources are present and non-empty in curated outputs
    - Cap-and-redistribute: unresolved deficits are not allowed

This file is intentionally fast. Full-corpus exact dedup coverage belongs to
the curation stage itself; pytest spot-checks representative output so normal
developer validation does not hang on large intermediate shards.
"""

import json

import pytest

from tests.data_pipeline.helpers import requires_stage, read_jsonl, pipeline_path
from curator.filters.quality import QualityFilter
from curator.filters.dedup import exact_hash

# Import source lists from config — the single source of truth.
from config import (
    ALL_SOURCES,
    NON_CODE_SOURCES,
    PROSE_HEURISTIC_SKIP_SOURCES,
    SYNTHETIC_SOURCES,
)

# Generated/template-like sources should bypass fuzzy MinHash only. Nemotron
# sources bypass prose filters, but they should not bypass fuzzy dedup.
FUZZY_DEDUP_SKIP_SOURCES = set(SYNTHETIC_SOURCES)

QUALITY_SKIP_SOURCES = PROSE_HEURISTIC_SKIP_SOURCES


pytestmark = requires_stage("curate-mini")

REQUIRED_IN_TRAIN = list(ALL_SOURCES)

# Keep pytest fast. Full exact-dedup coverage is performed by the curation
# stage; this test samples enough rows to catch obvious regressions.
DEDUP_EXACT_SAMPLE_PER_SOURCE = 2_000
DEDUP_EXACT_MAX_SHARDS_PER_SOURCE = 2


# ── Configuration drift guard ──────────────────────────────────────────────────

class TestConfigurationDrift:
    """Catch drift between quality-filter skip routing and config source lists."""

    def test_quality_filter_skip_sources_match_expected_sources(self):
        """
        Curation and validation share one source-routing contract.
        """
        assert set(QualityFilter().config.skip_language_sources) == set(
            PROSE_HEURISTIC_SKIP_SOURCES
        )

    def test_prose_skip_sources_are_real_sources(self):
        missing = set(PROSE_HEURISTIC_SKIP_SOURCES) - set(ALL_SOURCES)
        assert not missing, (
            f"Quality prose-skip sources are not in ALL_SOURCES: {missing}"
        )

    def test_generated_sources_skip_fuzzy_dedup_only(self):
        """
        Generated/template-like sources should bypass fuzzy MinHash dedup, but
        still run exact dedup. MinHash collapses useful near-duplicate template
        signal; exact dedup only removes true duplicate rows.
        """
        from curator.scripts.curate import SKIP_FUZZY_DEDUP_SOURCES

        assert SKIP_FUZZY_DEDUP_SOURCES == FUZZY_DEDUP_SKIP_SOURCES


# ── Raw data ───────────────────────────────────────────────────────────────────

class TestRawData:
    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_raw_completion_manifest_exists(self, source):
        assert pipeline_path("raw", source, "_SUCCESS.json").exists()

    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_raw_shards_exist(self, source):
        shards = list(pipeline_path("raw", source).glob("*.jsonl"))
        assert len(shards) > 0, f"No raw shards found for {source}"

    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_raw_shards_are_valid_jsonl(self, source):
        shards = list(pipeline_path("raw", source).glob("*.jsonl"))
        if not shards:
            pytest.skip(f"No raw shards for {source} — covered by presence test")

        shard = sorted(shards)[0]
        docs = read_jsonl(shard)
        assert len(docs) > 0, f"Empty shard: {shard}"
        for doc in docs[:10]:
            assert "text" in doc, f"Missing 'text' field in {shard}"
            assert "source" in doc, f"Missing 'source' field in {shard}"
            assert len(doc["text"]) > 0, f"Empty text in {shard}"

    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_raw_source_tag_matches_directory(self, source):
        """
        Each shard's `source` field should match the directory name.
        Catches bugs where a source writes to the wrong output directory
        or uses the wrong SOURCE_TAG constant.
        """
        shards = list(pipeline_path("raw", source).glob("*.jsonl"))
        if not shards:
            pytest.skip(f"No raw shards for {source}")
        shard = sorted(shards)[0]
        docs = read_jsonl(shard)
        for doc in docs[:5]:
            assert doc["source"] == source, (
                f"Shard in {source}/ has source='{doc['source']}' "
                f"(expected '{source}')"
            )


# ── Filtered data ──────────────────────────────────────────────────────────────

class TestFilteredData:
    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_filtered_completion_manifest_exists(self, source):
        assert pipeline_path("filtered", source, "_SUCCESS.json").exists()

    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_filtered_shards_exist(self, source):
        shards = list(pipeline_path("filtered", source).glob("*.jsonl"))
        assert len(shards) > 0, f"No filtered shards for {source}"

    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_filtered_manifest_accounts_for_segmentation(self, source):
        manifest_path = pipeline_path("filtered", source, "_SUCCESS.json")
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        audit = manifest["metadata"]["audit"]

        assert audit["schema_version"] == 2
        assert audit["input_documents"] >= 0
        assert audit["segmented_input_documents"] >= 0
        assert audit["produced_segments"] == audit["total"]
        assert audit["segmented_input_documents"] <= audit["input_documents"]

    def test_filtered_docs_pass_quality_checks(self):
        """Every sampled document in filtered output should pass quality filters."""
        qf = QualityFilter()
        failures = []

        for source in ALL_SOURCES:
            shards = sorted(pipeline_path("filtered", source).glob("*.jsonl"))
            if not shards:
                continue
            docs = read_jsonl(shards[0])
            for doc in docs[:50]:
                passed, reason = qf.check(doc)
                if not passed:
                    failures.append(
                        f"{source}: rejected '{reason}' — {doc['text'][:80]}"
                    )

        assert len(failures) == 0, (
            f"{len(failures)} documents in filtered output fail quality checks:\n"
            + "\n".join(failures[:5])
        )

    @pytest.mark.parametrize(
        "source",
        [s for s in NON_CODE_SOURCES if s not in QUALITY_SKIP_SOURCES],
    )
    def test_filtered_non_code_has_minimum_length(self, source):
        """
        Prose-like non-code sources go through the full minimum-length filter.
        Generated/template-like and symbol-heavy sources are excluded because
        they intentionally produce short or non-prose examples.
        """
        MIN_CHARS = 500
        shards = sorted(pipeline_path("filtered", source).glob("*.jsonl"))
        if not shards:
            pytest.skip(f"No filtered shards for {source}")
        docs = read_jsonl(shards[0])
        short = [d for d in docs if len(d["text"]) < MIN_CHARS]
        assert len(short) == 0, (
            f"{len(short)} docs in {source} filtered shard are below "
            f"{MIN_CHARS} chars — quality filter may not have run"
        )


# ── Deduped data ───────────────────────────────────────────────────────────────

class TestDedupedData:
    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_dedup_completion_manifest_exists(self, source):
        assert pipeline_path(
            "filtered", f"{source}_deduped", "_SUCCESS.json"
        ).exists()

    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_deduped_shards_exist(self, source):
        shards = list(pipeline_path("filtered", f"{source}_deduped").glob("*.jsonl"))
        assert len(shards) > 0, f"No deduped shards for {source}"

    def test_no_exact_duplicates_in_deduped_output_sample(self):
        """
        Spot-check exact duplicate removal without scanning the whole corpus.

        The curation stage already performs full exact dedup and logs the full
        hash index size. This pytest check is intentionally bounded so
        make test-curator stays quick even when mini/raw sources contain many
        upstream records.
        """
        seen_hashes: set[bytes] = set()
        duplicates = []

        for source in ALL_SOURCES:
            checked = 0
            shards = sorted(
                pipeline_path("filtered", f"{source}_deduped").glob("*.jsonl")
            )[:DEDUP_EXACT_MAX_SHARDS_PER_SOURCE]

            for shard in shards:
                with open(shard, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue

                        doc = json.loads(line)
                        h = exact_hash(doc.get("text", ""))
                        if h in seen_hashes:
                            duplicates.append(f"{source}: {doc.get('text', '')[:60]}")
                        seen_hashes.add(h)

                        checked += 1
                        if checked >= DEDUP_EXACT_SAMPLE_PER_SOURCE:
                            break

                if checked >= DEDUP_EXACT_SAMPLE_PER_SOURCE:
                    break

        assert len(duplicates) == 0, (
            f"{len(duplicates)} exact duplicates found in sampled deduped output:\n"
            + "\n".join(duplicates[:5])
        )


# ── Curated train.jsonl ────────────────────────────────────────────────────────

class TestCuratedOutput:
    def test_train_jsonl_exists(self):
        assert pipeline_path("curated", "train.jsonl").exists()

    def test_val_jsonl_exists(self):
        assert pipeline_path("curated", "val.jsonl").exists()

    def test_train_jsonl_is_non_empty(self):
        path = pipeline_path("curated", "train.jsonl")
        with open(path, encoding="utf-8") as f:
            assert any(line.strip() for line in f), f"{path} is empty"

    def test_train_jsonl_contains_required_sources(self):
        """
        train.jsonl should contain most sources in the mix.

        The 1% conala share at mini scale can plausibly round to zero docs
        after blend cap trimming — this is why we allow up to one source to be
        absent rather than requiring strict presence of every one.
        """
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        sources = {d["source"] for d in docs}
        present_required = sources & set(REQUIRED_IN_TRAIN)
        missing = set(REQUIRED_IN_TRAIN) - sources

        assert len(present_required) >= len(REQUIRED_IN_TRAIN) - 1, (
            f"train.jsonl missing too many sources. "
            f"Expected at least {len(REQUIRED_IN_TRAIN) - 1} of "
            f"{len(REQUIRED_IN_TRAIN)} required sources; "
            f"got {len(present_required)}. Missing: {missing}"
        )

    def test_curated_splits_contain_all_synthetic_sources(self):
        """
        All synthetic sources should survive into the curated split.

        Check train + val together because very small mini shares can place
        one of a source's few examples into val via reservoir sampling.
        """
        train_docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        val_docs = read_jsonl(pipeline_path("curated", "val.jsonl"))

        sources = {d["source"] for d in train_docs + val_docs}
        missing = SYNTHETIC_SOURCES - sources

        assert not missing, f"Synthetic sources missing from curated split: {missing}"

    def test_train_jsonl_has_no_unknown_sources(self):
        """Every source tag in train.jsonl should be one we recognize."""
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        sources = {d["source"] for d in docs}
        unknown = sources - set(ALL_SOURCES)
        assert not unknown, (
            f"train.jsonl contains unknown source tags: {unknown}. "
            f"Expected only: {sorted(ALL_SOURCES)}"
        )

    def test_train_jsonl_has_no_short_documents(self):
        """
        No prose-like document in train.jsonl should be below the quality
        filter threshold. Code/generated/symbol-heavy sources bypass this
        filter, so they may legitimately have short documents.
        """
        MIN_CHARS = 500
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        short_non_code = [
            d for d in docs
            if d.get("source") not in QUALITY_SKIP_SOURCES
            and len(d.get("text", "")) < MIN_CHARS
        ]
        assert len(short_non_code) == 0, (
            f"{len(short_non_code)} prose-like documents in train.jsonl "
            f"are below {MIN_CHARS} chars"
        )

    def test_train_jsonl_has_required_fields(self):
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        for doc in docs[:20]:
            assert "text" in doc
            assert "source" in doc
            assert len(doc["text"]) > 0

    def test_val_jsonl_has_required_fields(self):
        docs = read_jsonl(pipeline_path("curated", "val.jsonl"))
        for doc in docs[:20]:
            assert "text" in doc
            assert "source" in doc
            assert len(doc["text"]) > 0

    def test_train_jsonl_has_no_exact_duplicates_sample(self):
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        seen: set[bytes] = set()
        duplicates = 0
        for doc in docs[:10_000]:
            h = exact_hash(doc.get("text", ""))
            if h in seen:
                duplicates += 1
            seen.add(h)
        assert duplicates == 0, (
            f"{duplicates} exact duplicates found in sampled train.jsonl"
        )


# ── blend_stats.json ───────────────────────────────────────────────────────────

class TestBlendStats:
    def _load_stats(self) -> dict:
        path = pipeline_path("curated", "blend_stats.json")
        assert path.exists(), "blend_stats.json not found"
        with open(path) as f:
            return json.load(f)

    def test_blend_stats_exists(self):
        assert pipeline_path("curated", "blend_stats.json").exists()

    def test_blend_stats_has_required_fields(self):
        stats = self._load_stats()
        assert "target" in stats
        assert "target_tokens" in stats
        assert "total_documents" in stats
        assert "train_documents" in stats
        assert "val_documents" in stats
        assert "val_fraction" in stats
        assert "estimated_tokens_from_chars" in stats
        assert stats["token_count_status"].startswith("estimate_only")
        assert "chars_per_token" in stats
        assert "source_mix" in stats

    def test_blend_stats_total_documents_positive(self):
        stats = self._load_stats()
        assert stats["total_documents"] > 0
        assert stats["train_documents"] > 0
        assert stats["val_documents"] >= 0

    def test_blend_stats_sources_recorded(self):
        """Every configured source must appear in fail-closed blend stats."""
        stats = self._load_stats()
        mix = stats["source_mix"]
        assert set(mix) == set(REQUIRED_IN_TRAIN), (
            f"blend_stats.json source_mix missing sources. "
            f"Got: {sorted(mix.keys())}"
        )

    def test_blend_stats_includes_all_synthetic_sources(self):
        """Synthetic sources should be present and non-empty in mini curation."""
        stats = self._load_stats()
        mix = stats["source_mix"]

        missing = SYNTHETIC_SOURCES - set(mix)
        assert not missing, f"Missing synthetic sources in blend_stats: {missing}"

        for source in SYNTHETIC_SOURCES:
            row = mix[source]
            assert row["docs"] > 0, f"{source} has no blended docs: {row}"
            assert row["chars"] > 0, f"{source} has no blended chars: {row}"
            assert row["unresolved_deficit"] == 0, (
                f"{source} has non-zero unresolved deficit: {row}"
            )

    def test_blend_stats_per_source_schema(self):
        """Each source entry must expose intended and unresolved capacity."""
        stats = self._load_stats()
        mix = stats["source_mix"]

        for source, row in mix.items():
            assert "docs" in row, f"{source} missing docs"
            assert "chars" in row, f"{source} missing chars"
            assert "target_chars" in row, f"{source} missing target_chars"
            assert "initial_deficit" in row, f"{source} missing initial_deficit"
            assert "unresolved_deficit" in row, (
                f"{source} missing unresolved_deficit"
            )
            assert "val_docs" in row, f"{source} missing val_docs"

            assert isinstance(row["docs"], int), f"{source} docs is not int"
            assert isinstance(row["chars"], int), f"{source} chars is not int"
            assert isinstance(row["target_chars"], int), (
                f"{source} target_chars is not int"
            )
            assert isinstance(row["initial_deficit"], int)
            assert isinstance(row["unresolved_deficit"], int)
            assert isinstance(row["val_docs"], int), f"{source} val_docs is not int"

            assert row["docs"] >= 0, f"{source} docs is negative"
            assert row["chars"] >= 0, f"{source} chars is negative"
            assert row["target_chars"] >= 0, f"{source} target_chars is negative"
            assert row["initial_deficit"] >= 0
            assert row["unresolved_deficit"] >= 0
            assert row["val_docs"] >= 0, f"{source} val_docs is negative"

    def test_blend_stats_document_counts_match_curated_files(self):
        stats = self._load_stats()
        train_docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        val_docs = read_jsonl(pipeline_path("curated", "val.jsonl"))

        assert stats["train_documents"] == len(train_docs)
        assert stats["val_documents"] == len(val_docs)
        assert stats["total_documents"] == len(train_docs) + len(val_docs)

    def test_blend_stats_deficits_are_closed(self):
        """
        Mini curation should not leave unresolved deficits after overflow.

        If this fails, a source ran short and overflow routing did not make up
        the missing characters.
        """
        stats = self._load_stats()
        deficits = {
            source: row
            for source, row in stats["source_mix"].items()
            if row.get("unresolved_deficit", 0) != 0
        }

        assert not deficits, f"Unresolved source deficits in blend_stats: {deficits}"
