from pathlib import Path

from curator.state import (
    manifest_matches,
    manifest_outputs_match,
    tree_signature,
    write_manifest,
)


def test_manifest_requires_matching_contract_input_and_outputs(tmp_path: Path):
    output_dir = tmp_path / "stage"
    output_dir.mkdir()
    shard = output_dir / "part-000.jsonl"
    shard.write_text('{"text":"one"}\n')

    contract = {"setting": 1}
    write_manifest(
        output_dir,
        stage="test",
        contract=contract,
        input_signature="input-a",
    )

    assert manifest_matches(
        output_dir,
        stage="test",
        contract=contract,
        input_signature="input-a",
    )
    assert manifest_outputs_match(output_dir)
    assert not manifest_matches(
        output_dir,
        stage="test",
        contract={"setting": 2},
        input_signature="input-a",
    )
    assert not manifest_matches(
        output_dir,
        stage="test",
        contract=contract,
        input_signature="input-b",
    )

    shard.write_text('{"text":"changed"}\n')
    assert not manifest_matches(
        output_dir,
        stage="test",
        contract=contract,
        input_signature="input-a",
    )


def test_manifest_survives_staging_directory_rename(tmp_path: Path):
    staging = tmp_path / ".partial"
    final = tmp_path / "final"
    staging.mkdir()
    (staging / "part.jsonl").write_text('{"text":"one"}\n')
    write_manifest(
        staging,
        stage="download",
        contract={"source": "example"},
        input_signature=None,
    )

    staging.replace(final)
    assert manifest_matches(
        final,
        stage="download",
        contract={"source": "example"},
        input_signature=None,
    )


def test_tree_signature_is_root_relative(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "part.jsonl").write_text('{"text":"same"}\n')
    (second / "part.jsonl").write_text('{"text":"same"}\n')

    assert tree_signature(first) == tree_signature(second)


def test_tree_signature_ignores_mtime_for_restored_artifacts(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "part.jsonl").write_text('{"text":"same"}\n')
    (second / "part.jsonl").write_text('{"text":"same"}\n')

    import os

    stat = (second / "part.jsonl").stat()
    os.utime(
        second / "part.jsonl",
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )
    assert tree_signature(first) == tree_signature(second)
