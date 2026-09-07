"""Real bundle/restore code with explicitly fake network transports.

These are not live S3/HF integration tests. Every manifest, disk staging,
projection, SHA check and directory promotion uses the production code.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest

from config.holdout import sha256_file
from curator import artifact_hf, artifacts
from curator.scripts import upload_s3 as api
from curator.state import manifest_outputs_match
from tests.frozen_helpers import make_bundle

RUN = "350m-20260907-fixture"


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.fail_upload = None

    def upload_file(self, filename, bucket, key, **kwargs):
        if self.fail_upload and self.fail_upload in key:
            raise OSError("injected upload interruption")
        self.objects[key] = Path(filename).read_bytes()
        self.writes.append(key)

    def download_file(self, bucket, key, filename, **kwargs):
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "GetObject")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self.objects[key])


@pytest.fixture
def remote(tmp_path, monkeypatch):
    store = MemoryS3()
    source = tmp_path / "source"
    root = make_bundle(source)
    monkeypatch.setattr(api, "DATA_DIR", source)
    monkeypatch.setattr(api, "get_s3_client", lambda workers=16: store)
    return store, source, root


def upload():
    return artifacts.upload(api, "350m", RUN, None, "bucket", "prefix", 2, "s3", "training-ready", False, None)


def restore(destination, monkeypatch, **kwargs):
    monkeypatch.setattr(api, "DATA_DIR", destination)
    return artifacts.download(api, "350m", RUN, None, "bucket", "prefix", 2, "s3", "training-ready",
        kwargs.get("overwrite", False), kwargs.get("restore_results", False), kwargs.get("model_size"))


def test_training_ready_roundtrip_keeps_all_holdouts_and_tokenizer(remote, tmp_path, monkeypatch):
    store, source, root = remote
    source_hash = sha256_file(root / "validated" / "_SUCCESS.json")
    upload()
    assert store.writes[-1].endswith("/metadata/bundle_manifest.json")
    assert not any(key.endswith("validated/train.jsonl") for key in store.objects)
    assert (root / "validated" / "train.jsonl").exists()
    assert sha256_file(root / "validated" / "_SUCCESS.json") == source_hash
    restore(tmp_path / "restored", monkeypatch)
    restored = tmp_path / "restored" / "runs" / "350m"
    assert not (restored / "validated" / "train.jsonl").exists()
    assert (restored / "validated" / "_RETENTION.json").exists()
    assert manifest_outputs_match(restored / "validated", output_pattern="*.json*")
    assert (restored / "metadata" / "provenance" / "tokenized" / "test_contract.json").is_file()
    for split in ("train", "val", "test"):
        assert sha256_file(restored / "tokenized" / f"{split}.bin") == sha256_file(root / "tokenized" / f"{split}.bin")
    assert (restored / "RUN_ID").exists()
    # Re-uploading a restored training-ready bundle must not re-project or lose identities.
    projection = sha256_file(restored / "validated" / "_SUCCESS.json")
    upload()
    assert sha256_file(restored / "validated" / "_SUCCESS.json") == projection
    with pytest.raises(RuntimeError, match="train.jsonl"):
        artifacts._copy_selected(restored / "validated", tmp_path / "full", "validated", "full")


def test_corrupt_remote_data_is_rejected_before_replacement(remote, tmp_path, monkeypatch):
    store, _, _ = remote
    upload()
    dest = tmp_path / "restored"
    restore(dest, monkeypatch)
    before = sha256_file(dest / "runs" / "350m" / "tokenized" / "test.bin")
    key = f"prefix/350m/{RUN}/tokenized/test.bin"
    store.objects[key] = b"X" + store.objects[key][1:]
    with pytest.raises(RuntimeError, match="checksum"):
        restore(dest, monkeypatch)
    assert sha256_file(dest / "runs" / "350m" / "tokenized" / "test.bin") == before
    assert not (dest / "runs" / "350m" / "_RESTORE_PENDING.json").exists()


def test_failed_upload_does_not_publish_completion_descriptor(remote):
    store, _, _ = remote
    store.fail_upload = "test.bin"
    with pytest.raises(RuntimeError, match="upload failed"):
        upload()
    assert not any(key.endswith("bundle_manifest.json") for key in store.objects)


def test_frozen_identity_change_never_overwrites_same_run(remote):
    _, _, root = remote
    upload()
    path = root / "tokenized" / "test.json"
    meta = json.loads(path.read_text()); meta["binary_sha256"] = "0" * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="Frozen artifact identity changed"):
        upload()


def test_restore_wrong_local_run_requires_explicit_overwrite(remote, tmp_path, monkeypatch):
    upload()
    destination = tmp_path / "new"
    monkeypatch.setattr(api, "DATA_DIR", destination)
    api._write_run_id_record("350m", "350m-20260906-other")
    with pytest.raises(RuntimeError, match="another dataset RUN_ID"):
        restore(destination, monkeypatch)


def test_run_id_is_stable_across_days(remote, monkeypatch):
    api._write_run_id_record("350m", RUN)
    monkeypatch.setattr(api, "_today_iso", lambda: "2026-10-01")
    assert api.resolve_upload_run_id("350m", None) == RUN


def test_s3_pool_matches_file_times_multipart_concurrency(monkeypatch):
    captured = {}
    def client(service, **kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr(api.boto3, "client", client)
    api.get_s3_client(16)
    assert captured["config"].max_pool_connections == 16 * api._transfer_config(16).max_concurrency


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a/../../b", "a\\b", ""])
def test_bundle_and_hf_reject_unsafe_paths(name):
    with pytest.raises((ValueError, RuntimeError)):
        artifacts.safe_path(name)
    with pytest.raises((ValueError, RuntimeError)):
        artifact_hf.safe_relative(name)


def test_hf_transport_uses_bucket_apis_and_exact_downloads(tmp_path):
    calls = []
    fake = SimpleNamespace(batch_bucket_files=lambda **kw: calls.append(("upload", kw)),
                           download_bucket_files=lambda **kw: calls.append(("download", kw)))
    job = {"operation": "upload", "root": str(tmp_path), "files": ["test.bin"],
           "bucket": "owner/operational", "prefix": f"350m/{RUN}/tokenized"}
    artifact_hf.run(job, fake)
    artifact_hf.run({**job, "operation": "download"}, fake)
    assert calls[0][1]["bucket_id"] == "owner/operational"
    assert calls[0][1]["add"] == [(tmp_path / "test.bin", f"350m/{RUN}/tokenized/test.bin")]
    assert calls[1][1]["raise_on_missing_files"] is True
    assert calls[1][1]["files"] == [(f"350m/{RUN}/tokenized/test.bin", tmp_path / "test.bin")]


def test_hf_backend_does_not_upload_to_s3(remote, tmp_path, monkeypatch):
    store, _, _ = remote
    def fake_hf(operation, root, names, bucket, prefix):
        root = Path(root)
        for name in names:
            key = api.build_key(prefix, name)
            if operation == "upload":
                store.upload_file(root / name, bucket, key)
            elif operation == "read" and key not in store.objects:
                return {"missing": True}
            else:
                store.download_file(bucket, key, root / name)
        return {"missing": False, "files": len(names)}
    monkeypatch.setattr(artifacts, "_hf", fake_hf)
    monkeypatch.setattr(api, "get_s3_client", lambda *a, **k: pytest.fail("HF must not call S3"))
    artifacts.upload(api, "350m", RUN, None, "owner/bucket", "prefix", 2, "hf", "training-ready", False, None)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path / "hf-restore")
    artifacts.download(api, "350m", RUN, None, "owner/bucket", "prefix", 2, "hf", "training-ready", False, False, None)
    assert (tmp_path / "hf-restore" / "runs" / "350m" / "tokenized" / "test.bin").is_file()


def test_model_results_remain_under_model_not_dataset_size(remote, tmp_path, monkeypatch):
    results = tmp_path / "results"
    final = results / "runs" / "mini" / "pretrain" / "final"
    final.mkdir(parents=True)
    (final / "config.json").write_text('{"fixture": true}')
    monkeypatch.setenv("RESULTS_DIR", str(results))
    artifacts.upload(api, "350m", RUN, None, "bucket", "prefix", 2, "s3", "training-ready", True, "mini")
    restored_results = tmp_path / "restored-results"
    monkeypatch.setenv("RESULTS_DIR", str(restored_results))
    restore(tmp_path / "restored", monkeypatch, restore_results=True, model_size="mini")
    assert (restored_results / "runs" / "mini" / "pretrain" / "final" / "config.json").is_file()
    assert not (restored_results / "runs" / "350m").exists()


def test_mixed_tokenizer_is_rejected_even_with_rebuilt_stage_manifests(remote):
    from curator.state import write_manifest
    _, _, root = remote
    (root / "tokenizer" / "slm_tokenizer.json").write_text('{"different": true}\n')
    write_manifest(root / "tokenizer", stage="tokenizer", contract={"fixture": 2},
                   input_signature="fixture", output_pattern="*")
    with pytest.raises(RuntimeError, match="tokenizer file does not match"):
        upload()
