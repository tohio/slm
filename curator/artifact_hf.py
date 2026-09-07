"""Narrow HF Storage Buckets transport, callable from a separate Hub-1.x env.

The curation environment intentionally retains Hub 0.36.x / DataTrove 0.9.
ARTIFACT_HF_PYTHON selects the transfer interpreter without changing that stack.
The parent process supplies an allowlisted job over stdin, never credentials.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path, PurePosixPath


def safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe bucket object path: {name!r}")
    return str(path)


def run(job: dict, api=None) -> dict:
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    if not all(hasattr(api, name) for name in ("batch_bucket_files", "download_bucket_files")):
        raise RuntimeError("HF Storage Buckets requires the artifact Hub-1.x environment; set ARTIFACT_HF_PYTHON")
    bucket = job["bucket"]
    prefix = job.get("prefix", "").strip("/")
    if prefix:
        safe_relative(prefix)
    def key(relative):
        relative = safe_relative(relative)
        return f"{prefix}/{relative}" if prefix else relative
    root = Path(job["root"])
    files = [safe_relative(name) for name in job["files"]]
    if job["operation"] == "upload":
        # Bucket batches are not transactional. The parent publishes the
        # bundle descriptor separately, only after every stage succeeds.
        for offset in range(0, len(files), 128):
            api.batch_bucket_files(bucket_id=bucket, add=[(root / name, key(name)) for name in files[offset:offset + 128]])
    elif job["operation"] in {"download", "read"}:
        try:
            api.download_bucket_files(bucket_id=bucket,
                files=[(key(name), root / name) for name in files], raise_on_missing_files=True)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if job["operation"] == "read" and (isinstance(exc, FileNotFoundError) or status == 404):
                return {"missing": True}
            raise
    else:
        raise ValueError("Unsupported bucket transfer operation")
    return {"files": len(files), "missing": False}


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin))))
