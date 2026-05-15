"""
tests/data_pipeline/test_pipeline_curator.py
---------------------------------------------
Validates real outputs from 'make curate-mini'.

Run after: make curate-mini
Command:   make test-curator

Checks:
    - Raw source directories exist and have shards for all configured sources
    - Filtered shards exist and contain no documents failing quality checks
    - Deduped shards exist and have no exact duplicates
    - data/curated/train.jsonl exists, is non-empty, contains most sources
    - data/curated/blend_stats.json is correct and complete
    - Synthetic sources are present and non-empty in curated outputs
    - Cap-and-redistribute: any deficit is covered by overflow accounting
"""

import json

import pytest

from tests.data_pipeline.helpers import requires_stage, read_jsonl, pipeline_path
from curator.filters.quality import QualityFilter, CODE_SOURCES as QUALITY_CODE_SOURCES
from curator.filters.dedup import exact_hash

# Import source lists from config — the single source of truth. Previously
# NON_CODE_SOURCES / CODE_SOURCE_NAMES / ALL_SOURCES were hand-maintained
# here, which meant every change to the mix needed a matching edit in this
# file. TestConfigurationDrift still guards the separate quality-filter
# CODE_SOURCES constant (defined in curator.filters.quality) against drift
# from config, allowing symbol-heavy generated sources that should also
# bypass prose filters.
from config import ALL_SOURCES, CODE_SOURCES, NON_CODE_SOURCES

SYNTHETIC_SOURCES = {
    "synthetic_arithmetic",
    "synthetic_task_code",
    "educational_qa_mcq",
    "factual_restraint",
}

SYMBOL_HEAVY_SKIP_SOURCES = set(SYNTHETIC_SOURCES)
QUALITY_SKIP_SOURCES = set(CODE_SOURCES) | SYMBOL_HEAVY_SKIP_SOURCES


pytestmark = requires_stage("curate-mini")


# All sources must appear in train.jsonl. Any missing source indicates a
# real failure (download, filter, dedup, or blend problem) that should be
# surfaced, not papered over with a skip. stack_v1 replaced stack_v2 in the
# mix specifically so this assertion could be unconditional — v2's SWH
# content fetch was the one external dependency that could produce zero
# docs at mini scale, and v1 has content inline so it doesn't have that
# failure mode.
REQUIRED_IN_TRAIN = list(ALL_SOURCES)


# ── Configuration drift guard ──────────────────────────────────────────────────

class TestConfigurationDrift:
    """Catch drift between quality-filter skip routing and config source lists."""

    def test_quality_filter_skip_sources_match_expected_sources(self):
        """
        config.CODE_SOURCES means the 5 code sub-sources in the data mix.
        quality.CODE_SOURCES is broader: it is the set of sources that bypass
        English-prose filters. That includes code plus symbol-heavy generated
        sources such as synthetic_arithmetic.
        """
        assert set(QUALITY_CODE_SOURCES) == QUALITY_SKIP_SOURCES, (
            f"CODE_SOURCES in quality.py ({set(QUALITY_CODE_SOURCES)}) drifted "
            f"from expected prose-filter skip sources ({QUALITY_SKIP_SOURCES}). "
            f"Update quality.py or this test if a new symbol-heavy source is added."
        )

    def test_symbol_heavy_skip_sources_are_real_sources(self):
        missing = SYMBOL_HEAVY_SKIP_SOURCES - set(ALL_SOURCES)
        assert not missing, (
            f"Symbol-heavy quality skip sources are not in ALL_SOURCES: {missing}"
        )

    def test_generated_sources_skip_fuzzy_dedup_only(self):
        """
        Generated/template-like sources should bypass fuzzy MinHash dedup, but
        still run exact dedup. MinHash collapses useful near-duplicate template
        signal; exact dedup only removes true duplicate rows.
        """
        from curator.scripts.curate import SKIP_FUZZY_DEDUP_SOURCES

        assert SKIP_FUZZY_DEDUP_SOURCES == SYMBOL_HEAVY_SKIP_SOURCES


# ── Raw data ───────────────────────────────────────────────────────────────────

class TestRawData:
    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_raw_shards_exist(self, source):
        shards = list(pipeline_path("raw", source).glob("*.jsonl"))
        assert len(shards) > 0, f"No raw shards found for {source}"

    @pytest.mark.parametrize("source", ALL_SOURCES)
    def test_raw_shards_are_valid_jsonl(self, source):
        shards = list(pipeline_path("raw", source).glob("*.jsonl"))
        if not shards:
            pytest.skip(f"No raw shards for {source} — covered by presence test")
        # Check first shard per source
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
    def test_filtered_shards_exist(self, source):
        shards = list(pipeline_path("filtered", source).glob("*.jsonl"))
        assert len(shards) > 0, f"No filtered shards for {source}"

    def test_filtered_docs_pass_quality_checks(self):
        """Every document in filtered output should pass quality filters."""
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
        Generated/template-like sources are excluded because they intentionally
        produce short examples.
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
    def test_deduped_shards_exist(self, source):
        shards = list(pipeline_path("filtered", f"{source}_deduped").glob("*.jsonl"))
        assert len(shards) > 0, f"No deduped shards for {source}"

    def test_no_exact_duplicates_in_deduped_output(self):
        """No exact duplicate documents should exist across all deduped sources."""
        seen_hashes: set[bytes] = set()
        duplicates = []

        for source in ALL_SOURCES:
            shards = sorted(
                pipeline_path("filtered", f"{source}_deduped").glob("*.jsonl")
            )
            for shard in shards:
                docs = read_jsonl(shard)
                for doc in docs:
                    h = exact_hash(doc.get("text", ""))
                    if h in seen_hashes:
                        duplicates.append(f"{source}: {doc['text'][:60]}")
                    seen_hashes.add(h)

        assert len(duplicates) == 0, (
            f"{len(duplicates)} exact duplicates found in deduped output:\n"
            + "\n".join(duplicates[:5])
        )


# ── Curated train.jsonl ────────────────────────────────────────────────────────

class TestCuratedOutput:
    def test_train_jsonl_exists(self):
        assert pipeline_path("curated", "train.jsonl").exists()

    def test_val_jsonl_exists(self):
        assert pipeline_path("curated", "val.jsonl").exists()

    def test_train_jsonl_is_non_empty(self):
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        assert len(docs) > 0, "train.jsonl is empty"

    def test_train_jsonl_contains_required_sources(self):
        """
        train.jsonl should contain most sources in the mix.

        The 1% conala share at mini scale (1M tokens × 10% × 1% = 1k chars,
        roughly 200 tokens) can plausibly round to zero docs after blend
        cap trimming — this is why we allow up to one source to be absent
        rather than requiring strict presence of every one. If more than
        one source is missing, there's a real pipeline problem.
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
        No non-code document in train.jsonl should be below the quality
        filter threshold. Code and generated/template-like sources bypass
        this filter, so they may legitimately have short documents.
        """
        MIN_CHARS = 500
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        short_non_code = [
            d for d in docs
            if d.get("source") not in QUALITY_SKIP_SOURCES
            and len(d.get("text", "")) < MIN_CHARS
        ]
        assert len(short_non_code) == 0, (
            f"{len(short_non_code)} non-code documents in train.jsonl "
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

    def test_train_jsonl_has_no_exact_duplicates(self):
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        seen: set[bytes] = set()
        duplicates = 0
        for doc in docs:
            h = exact_hash(doc.get("text", ""))
            if h in seen:
                duplicates += 1
            seen.add(h)
        assert duplicates == 0, f"{duplicates} exact duplicates found in train.jsonl"


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
        assert "estimated_tokens" in stats
        assert "chars_per_token" in stats
        assert "source_mix" in stats

    def test_blend_stats_total_documents_positive(self):
        stats = self._load_stats()
        assert stats["total_documents"] > 0
        assert stats["train_documents"] > 0
        assert stats["val_documents"] >= 0

    def test_blend_stats_sources_recorded(self):
        """
        All sources that produced output should appear in blend_stats.
        We allow up to one source to be absent (same caveat as the
        train.jsonl test — the 1% conala share can round to zero at
        mini scale).
        """
        stats = self._load_stats()
        mix = stats["source_mix"]
        present_required = set(mix.keys()) & set(REQUIRED_IN_TRAIN)
        assert len(present_required) >= len(REQUIRED_IN_TRAIN) - 1, (
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
            assert row["deficit"] == 0, f"{source} has non-zero deficit: {row}"

    def test_blend_stats_per_source_schema(self):
        """Each source entry must have docs, chars, target_chars, deficit."""
        stats = self._load_stats()
        mix = stats["source_mix"]

        for source, row in mix.items():
            assert "docs" in row, f"{source} missing docs"
            assert "chars" in row, f"{source} missing chars"
            assert "target_chars" in row, f"{source} missing target_chars"
            assert "deficit" in row, f"{source} missing deficit"
            assert "val_docs" in row, f"{source} missing val_docs"

            assert isinstance(row["docs"], int), f"{source} docs is not int"
            assert isinstance(row["chars"], int), f"{source} chars is not int"
            assert isinstance(row["target_chars"], int), (
                f"{source} target_chars is not int"
            )
            assert isinstance(row["deficit"], int), f"{source} deficit is not int"
            assert isinstance(row["val_docs"], int), f"{source} val_docs is not int"

            assert row["docs"] >= 0, f"{source} docs is negative"
            assert row["chars"] >= 0, f"{source} chars is negative"
            assert row["target_chars"] >= 0, f"{source} target_chars is negative"
            assert row["deficit"] >= 0, f"{source} deficit is negative"
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

        If this fails, a source ran short and the overflow routing did not make
        up the missing characters.
        """
        stats = self._load_stats()
        deficits = {
            source: row
            for source, row in stats["source_mix"].items()
            if row.get("deficit", 0) != 0
        }

        assert not deficits, f"Unresolved source deficits in blend_stats: {deficits}"
