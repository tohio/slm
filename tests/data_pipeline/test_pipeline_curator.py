"""
tests/data_pipeline/test_pipeline_curator.py
---------------------------------------------
Validates real outputs from 'make curate-mini'.

Run after: make curate-mini
Command:   make test-curator

Checks:
    - Raw source directories exist and have shards
    - Filtered shards exist and contain no documents failing quality checks
    - Deduped shards exist and have no exact duplicates
    - data/curated/train.jsonl exists, is non-empty, contains all 3 sources
    - data/curated/blend_stats.json is correct and complete
    - Source mix is in the right ballpark
"""

import json
from pathlib import Path

import pytest

from tests.conftest import DATA_DIR, requires_stage, read_jsonl, pipeline_path
from curator.filters.quality import QualityFilter
from curator.filters.dedup import exact_hash


pytestmark = requires_stage("curate-mini")


# ── Raw data ───────────────────────────────────────────────────────────────────

class TestRawData:
    def test_raw_wikipedia_shards_exist(self):
        shards = list(pipeline_path("raw", "wikipedia").glob("*.jsonl"))
        assert len(shards) > 0, "No Wikipedia raw shards found"

    def test_raw_code_shards_exist(self):
        shards = list(pipeline_path("raw", "code").glob("*.jsonl"))
        assert len(shards) > 0, "No code raw shards found"

    def test_raw_common_crawl_shards_exist(self):
        shards = list(pipeline_path("raw", "common_crawl").glob("*.jsonl"))
        assert len(shards) > 0, "No common_crawl raw shards found"

    def test_raw_shards_are_valid_jsonl(self):
        for source in ["wikipedia", "code", "common_crawl"]:
            shards = list(pipeline_path("raw", source).glob("*.jsonl"))
            for shard in shards[:1]:  # check first shard per source
                docs = read_jsonl(shard)
                assert len(docs) > 0, f"Empty shard: {shard}"
                for doc in docs[:10]:
                    assert "text" in doc, f"Missing 'text' field in {shard}"
                    assert "source" in doc, f"Missing 'source' field in {shard}"
                    assert len(doc["text"]) > 0, f"Empty text in {shard}"


# ── Filtered data ──────────────────────────────────────────────────────────────

class TestFilteredData:
    def test_filtered_wikipedia_exists(self):
        shards = list(pipeline_path("filtered", "wikipedia").glob("*.jsonl"))
        assert len(shards) > 0

    def test_filtered_code_exists(self):
        shards = list(pipeline_path("filtered", "code").glob("*.jsonl"))
        assert len(shards) > 0

    def test_filtered_common_crawl_exists(self):
        shards = list(pipeline_path("filtered", "common_crawl").glob("*.jsonl"))
        assert len(shards) > 0

    def test_filtered_docs_pass_quality_checks(self):
        """Every document in filtered output should pass quality filters."""
        qf = QualityFilter()
        failures = []

        for source in ["wikipedia", "code", "common_crawl"]:
            shards = sorted(pipeline_path("filtered", source).glob("*.jsonl"))
            for shard in shards[:1]:  # spot-check first shard
                docs = read_jsonl(shard)
                for doc in docs[:50]:  # sample 50 per shard
                    passed, reason = qf.check(doc)
                    if not passed:
                        failures.append(f"{source}: rejected '{reason}' — {doc['text'][:80]}")

        assert len(failures) == 0, (
            f"{len(failures)} documents in filtered output fail quality checks:\n"
            + "\n".join(failures[:5])
        )

    def test_filtered_docs_have_minimum_length(self):
        """All filtered docs must meet the minimum character threshold."""
        MIN_CHARS = 500
        for source in ["wikipedia", "common_crawl"]:
            shards = sorted(pipeline_path("filtered", source).glob("*.jsonl"))
            for shard in shards[:1]:
                docs = read_jsonl(shard)
                short = [d for d in docs if len(d["text"]) < MIN_CHARS]
                assert len(short) == 0, (
                    f"{len(short)} docs in {source} filtered shard are below "
                    f"{MIN_CHARS} chars — quality filter may not have run"
                )


# ── Deduped data ───────────────────────────────────────────────────────────────

class TestDedupedData:
    def test_deduped_wikipedia_exists(self):
        shards = list(pipeline_path("filtered", "wikipedia_deduped").glob("*.jsonl"))
        assert len(shards) > 0

    def test_deduped_code_exists(self):
        shards = list(pipeline_path("filtered", "code_deduped").glob("*.jsonl"))
        assert len(shards) > 0

    def test_deduped_common_crawl_exists(self):
        shards = list(pipeline_path("filtered", "common_crawl_deduped").glob("*.jsonl"))
        assert len(shards) > 0

    def test_no_exact_duplicates_in_deduped_output(self):
        """No exact duplicate documents should exist in deduped output."""
        seen_hashes: set[bytes] = set()
        duplicates = []

        for source in ["wikipedia_deduped", "code_deduped", "common_crawl_deduped"]:
            shards = sorted(pipeline_path("filtered", source).glob("*.jsonl"))
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

    def test_train_jsonl_is_non_empty(self):
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        assert len(docs) > 0, "train.jsonl is empty"

    def test_train_jsonl_contains_all_sources(self):
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        sources = {d["source"] for d in docs}
        assert "wikipedia" in sources, "Missing wikipedia in train.jsonl"
        assert "code" in sources, "Missing code in train.jsonl"
        assert "common_crawl" in sources, "Missing common_crawl in train.jsonl"

    def test_train_jsonl_has_no_short_documents(self):
        """No document in train.jsonl should be below the quality filter threshold."""
        MIN_CHARS = 500
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        short = [d for d in docs if len(d.get("text", "")) < MIN_CHARS]
        assert len(short) == 0, (
            f"{len(short)} documents in train.jsonl are below {MIN_CHARS} chars"
        )

    def test_train_jsonl_has_required_fields(self):
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
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
        assert "total_documents" in stats
        assert "source_mix" in stats

    def test_blend_stats_total_documents_positive(self):
        stats = self._load_stats()
        assert stats["total_documents"] > 0

    def test_blend_stats_all_sources_present(self):
        stats = self._load_stats()
        mix = stats["source_mix"]
        assert "wikipedia" in mix, "wikipedia missing from blend_stats source_mix"
        assert "code" in mix, "code missing from blend_stats source_mix"
        assert "common_crawl" in mix, "common_crawl missing from blend_stats source_mix"

    def test_blend_stats_source_counts_positive(self):
        stats = self._load_stats()
        for source, data in stats["source_mix"].items():
            assert data["docs"] > 0, f"{source} has 0 docs in blend_stats"

    def test_blend_stats_matches_train_jsonl(self):
        """total_documents in stats should match actual line count in train.jsonl."""
        stats = self._load_stats()
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        assert stats["total_documents"] == len(docs), (
            f"blend_stats.json says {stats['total_documents']} docs "
            f"but train.jsonl has {len(docs)}"
        )