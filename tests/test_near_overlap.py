"""Focused tests for disk-backed train/validation MinHash overlap helpers."""

import struct

import orjson

from curator.filters.near_overlap import (
    JsonlByteRangeReader,
    build_cross_index_removals,
)


def test_byte_range_reader_covers_each_jsonl_record_once(tmp_path):
    dataset = tmp_path / "records.jsonl"
    records = [
        {"id": f"doc-{i}", "text": f"document number {i}", "source": "test"}
        for i in range(17)
    ]
    with open(dataset, "wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record) + b"\n")

    observed = []
    for rank in range(5):
        observed.extend(
            document.id
            for document in JsonlByteRangeReader(dataset).run(
                rank=rank, world_size=5
            )
        )

    assert sorted(observed) == sorted(record["id"] for record in records)
    assert len(observed) == len(set(observed)) == len(records)


def test_byte_range_reader_keeps_line_start_on_exact_boundary(tmp_path):
    dataset = tmp_path / "equal-lines.jsonl"
    record_1 = {"id": "first", "text": "one two three", "source": "test"}
    record_2 = {"id": "other", "text": "one two three", "source": "test"}
    line_1 = orjson.dumps(record_1) + b"\n"
    line_2 = orjson.dumps(record_2) + b"\n"
    assert len(line_1) == len(line_2)
    dataset.write_bytes(line_1 + line_2)

    observed = []
    for rank in range(2):
        observed.extend(
            document.id
            for document in JsonlByteRangeReader(dataset).run(
                rank=rank, world_size=2
            )
        )

    assert observed == ["first", "other"]


def test_cross_index_cluster_excludes_train_only_components(tmp_path):
    duplicate_dir = tmp_path / "duplicates"
    removal_dir = tmp_path / "removals"
    duplicate_dir.mkdir()
    sentinel = (2**32 - 1, 2**32 - 1)
    pairs = [
        (*sentinel, 0, 1),
        (0, 1, 0, 2),
        (1, 4, 1, 5),
    ]
    with open(duplicate_dir / "00000_00.dups", "wb") as handle:
        for pair in pairs:
            handle.write(struct.pack("<4I", *pair))

    report = build_cross_index_removals(duplicate_dir, removal_dir)

    with open(removal_dir / "000000.remove", "rb") as handle:
        removals = [item[0] for item in struct.iter_unpack("<I", handle.read())]
    assert removals == [1, 2]
    assert not (removal_dir / "000001.remove").exists()
    assert report["candidate_pairs"] == 3
    assert report["validation_index_pairs"] == 1
    assert report["matched_documents"] == 2
