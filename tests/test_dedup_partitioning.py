"""Focused tests for FineWeb-compatible Common Crawl MinHash boundaries."""

import orjson
import pytest

from curator.filters.dedup import (
    MINHASH_CONFIG,
    MINHASH_CONTRACT,
    partition_jsonl_by_field,
)
from curator.scripts.curate import FUZZY_DEDUP_PARTITION_FIELDS


def _write_records(path, records):
    with open(path, "wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")


def _read_records(path):
    with open(path, "rb") as handle:
        return [orjson.loads(line) for line in handle]


def test_minhash_contract_matches_official_fineweb_configuration():
    assert MINHASH_CONTRACT == {
        "hash_fc": "sha1",
        "precision": 64,
        "num_buckets": 14,
        "hashes_per_bucket": 8,
        "n_grams": 5,
    }
    assert MINHASH_CONFIG.hash_config.hash_fc == "sha1"
    assert MINHASH_CONFIG.hash_config.precision == 64


def test_common_crawl_is_the_only_partitioned_fuzzy_dedup_source():
    assert FUZZY_DEDUP_PARTITION_FIELDS == {"common_crawl": "crawl"}


def test_partition_jsonl_by_crawl_splits_mixed_shard(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "partitioned"
    input_dir.mkdir()
    _write_records(
        input_dir / "cc_0000.jsonl",
        [
            {"text": "first", "crawl": "CC-MAIN-2024-10"},
            {"text": "second", "crawl": "CC-MAIN-2023-50"},
            {"text": "third", "crawl": "CC-MAIN-2024-10"},
        ],
    )

    partitions = partition_jsonl_by_field(input_dir, output_dir, "crawl")

    assert set(partitions) == {"CC-MAIN-2023-50", "CC-MAIN-2024-10"}
    assert _read_records(partitions["CC-MAIN-2023-50"] / "cc_0000.jsonl") == [
        {"text": "second", "crawl": "CC-MAIN-2023-50"}
    ]
    assert _read_records(partitions["CC-MAIN-2024-10"] / "cc_0000.jsonl") == [
        {"text": "first", "crawl": "CC-MAIN-2024-10"},
        {"text": "third", "crawl": "CC-MAIN-2024-10"},
    ]


def test_partition_jsonl_by_crawl_fails_closed_without_metadata(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_records(input_dir / "cc_0000.jsonl", [{"text": "missing crawl"}])

    with pytest.raises(RuntimeError, match="Missing required string field 'crawl'"):
        partition_jsonl_by_field(input_dir, tmp_path / "partitioned", "crawl")
