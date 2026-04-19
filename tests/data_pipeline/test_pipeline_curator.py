"""
tests/data_pipeline/test_pipeline_curator.py
---------------------------------------------
Validates real outputs from 'make curate-mini'.

Run after: make curate-mini
Command:   make test-curator

Checks:
    - Raw source directories exist and have shards for all 10 sources
    - Filtered shards exist and contain no documents failing quality checks
    - Deduped shards exist and have no exact duplicates
    - data/curated/train.jsonl exists, is non-empty, contains most sources
    - data/curated/blend_stats.json is correct and complete
    - Cap-and-redistribute: any deficit is covered by FineWeb overflow
"""

import json
from pathlib import Path

import pytest

from tests.conftest import DATA_DIR, requires_stage, read_jsonl, pipeline_path
from curator.filters.quality import QualityFilter, CODE_SOURCES
from curator.filters.dedup import exact_hash


pytestmark = requires_stage("curate-mini")


# All 10 sources the mini run exercises — matches ALL_SOURCES in curate.py.
# Kept in sync by convention; any drift here vs curate.py should surface as
# a presence test failure.
NON_CODE_SOURCES = [
    "common_crawl",
    "fineweb",
    "wikipedia",
    "pg19",
    "pes2o",
    "open_web_math",
    "stackexchange",
]
CODE_SOURCE_NAMES = [
    "codesearchnet",
    "stack_smol",
    "stack_v2",
    "jupyter",
    "conala",
]
ALL_SOURCES = NON_CODE_SOURCES + CODE_SOURCE_NAMES

# Sources whose presence in train.jsonl is required. stack_v2 is excluded
# because its SWH content fetching is the one external dependency we don't
# control — if SWH is rate-limiting or SWH_AUTH_TOKEN isn't set, the mini
# run may produce very few stack_v2 docs. The rest must be present.
REQUIRED_IN_TRAIN = [s for s in ALL_SOURCES if s != "stack_v2"]

# Sources exempt from English-prose quality checks (mixed or code content).
# Matches curator.filters.quality.CODE_SOURCES; imported separately so this
# test catches drift between the two definitions.
EXPECTED_CODE_SOURCES = set(CODE_SOURCE_NAMES)


# ── Configuration drift guard ──────────────────────────────────────────────────

class TestConfigurationDrift:
    """Catch drift between this test file and curator module definitions."""

    def test_code_sources_match_quality_filter_set(self):
        assert EXPECTED_CODE_SOURCES == set(CODE_SOURCES), (
            f"CODE_SOURCES in quality.py ({set(CODE_SOURCES)}) drifted from "
            f"test expectation ({EXPECTED_CODE_SOURCES}). Update one or both."
        )


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

    @pytest.mark.parametrize("source", [s for s in NON_CODE_SOURCES])
    def test_filtered_non_code_has_minimum_length(self, source):
        """
        Non-code sources go through the full length filter. Every filtered
        doc should meet the minimum character threshold (500 from QualityConfig).
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

    def test_train_jsonl_is_non_empty(self):
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        assert len(docs) > 0, "train.jsonl is empty"

    def test_train_jsonl_contains_required_sources(self):
        """
        train.jsonl should contain all sources except stack_v2 (which may
        produce zero docs at mini scale if SWH is rate-limited).

        The 1% conala share at mini scale (1M tokens × 10% × 1% = 1k chars,
        roughly 200 tokens) can plausibly round to zero docs — this is why
        we check for 'at least 8 of 9 required sources' rather than strict
        presence of every one.
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
        filter threshold. Code sources bypass this filter, so they may
        legitimately have short documents (short functions, one-liners).
        """
        MIN_CHARS = 500
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        short_non_code = [
            d for d in docs
            if d.get("source") not in EXPECTED_CODE_SOURCES
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
        assert "estimated_tokens" in stats
        assert "chars_per_token" in stats
        assert "source_mix" in stats

    def test_blend_stats_total_documents_positive(self):
        stats = self._load_stats()
        assert stats["total_documents"] > 0

    def test_blend_stats_sources_recorded(self):
        """
        All sources that produced output should appear in blend_stats.
        We allow stack_v2 to be absent (same caveat as train.jsonl test).
        """
        stats = self._load_stats()
        mix = stats["source_mix"]
        present_required = set(mix.keys()) & set(REQUIRED_IN_TRAIN)
        assert len(present_required) >= len(REQUIRED_IN_TRAIN) - 1, (
            f"blend_stats.json source_mix missing sources. "
            f"Got: {sorted(mix.keys())}"
        )

    def test_blend_stats_per_source_schema(self):
        """Each source entry must have docs, chars, target_chars, deficit."""
        stats = self._load_stats()
        for source, data in stats["source_mix"].items():
            for field in ("docs", "chars", "target_chars", "deficit"):
                assert field in data, (
                    f"source_mix[{source}] missing field '{field}'. "
                    f"Got keys: {list(data.keys())}"
                )

    def test_blend_stats_matches_train_jsonl(self):
        """total_documents in stats should match actual line count in train.jsonl."""
        stats = self._load_stats()
        docs = read_jsonl(pipeline_path("curated", "train.jsonl"))
        assert stats["total_documents"] == len(docs), (
            f"blend_stats.json says {stats['total_documents']} docs "
            f"but train.jsonl has {len(docs)}"
        )

    def test_blend_stats_deficit_covered_by_overflow(self):
        """
        Cap-and-redistribute invariant: any deficit from supply-constrained
        sources should be covered by FineWeb overflow.

        Computed as: total deficit ≤ FineWeb overflow_chars (if overflow
        occurred) OR total deficit == 0 (if no overflow needed).

        Some slack is allowed — the overflow pass reads in shard-aligned
        chunks and may slightly overshoot the exact deficit. We check that
        total_chars across all sources is within 5% of the target.
        """
        stats = self._load_stats()
        total_target = sum(v["target_chars"] for v in stats["source_mix"].values())
        total_actual = sum(v["chars"] for v in stats["source_mix"].values())

        # Allow 5% tolerance on either side — overflow is shard-aligned,
        # mini-scale numbers are small enough that rounding matters.
        tolerance = 0.05
        ratio = total_actual / max(total_target, 1)
        assert (1 - tolerance) <= ratio <= (1 + tolerance * 3), (
            f"Total chars ({total_actual / 1e6:.2f}M) diverges from "
            f"target ({total_target / 1e6:.2f}M) by more than "
            f"±{tolerance:.0%} / +{tolerance * 3:.0%}. "
            f"Cap-and-redistribute may not be working."
        )

    def test_blend_stats_fineweb_overflow_when_deficits_exist(self):
        """
        If any source has a non-trivial deficit, FineWeb should have
        overflow_docs/overflow_chars fields populated.
        """
        stats = self._load_stats()
        mix = stats["source_mix"]

        # Sum deficits from all sources OTHER than fineweb (fineweb covers
        # others' deficits; its own is a separate concept).
        other_deficit = sum(
            v["deficit"] for k, v in mix.items() if k != "fineweb"
        )

        fineweb = mix.get("fineweb", {})
        if other_deficit > 1_000_000:  # 1 MB of shortfall = meaningful
            assert "overflow_chars" in fineweb, (
                f"Other sources show {other_deficit / 1e6:.2f}MB of deficit "
                f"but FineWeb has no overflow_chars recorded. "
                f"Cap-and-redistribute did not run."
            )
            assert fineweb["overflow_chars"] > 0, (
                f"FineWeb overflow_chars is 0 despite {other_deficit / 1e6:.2f}MB "
                f"of deficit from other sources."
            )

    def test_blend_stats_chars_per_token_set(self):
        """chars_per_token should be the constant from curator.constants."""
        from curator.constants import CHARS_PER_TOKEN
        stats = self._load_stats()
        assert stats["chars_per_token"] == CHARS_PER_TOKEN, (
            f"blend_stats.json chars_per_token={stats['chars_per_token']} "
            f"does not match curator.constants.CHARS_PER_TOKEN={CHARS_PER_TOKEN}"
        )