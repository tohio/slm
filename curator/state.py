"""Small, dependency-free helpers for trustworthy pipeline stage state.

Pipeline outputs are reusable only when a completion manifest matches the
current stage contract and its inputs.  A directory merely containing files is
not evidence that the stage completed: it may be the residue of an interrupted
write or an older configuration.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Iterable


MANIFEST_NAME = "_SUCCESS.json"
MANIFEST_VERSION = 2
CONTENT_SAMPLE_BYTES = 64 * 1024


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def stable_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible stage state."""
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def code_fingerprint(*objects: Any) -> str:
    """Hash the Python source files that implement a stage contract."""
    files: set[Path] = set()
    for obj in objects:
        source = inspect.getsourcefile(inspect.unwrap(obj))
        if source is None:
            raise RuntimeError(f"Cannot resolve source file for {obj!r}")
        files.add(Path(source).resolve())

    digest = hashlib.sha256()
    for path in sorted(files, key=str):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_snapshot(
    paths: Iterable[Path],
    *,
    root: Path | None = None,
    exclude_manifest: bool = True,
) -> list[dict[str, Any]]:
    """Describe files using a portable, inexpensive content fingerprint.

    Object-store downloads do not preserve modification times, so mtime cannot
    be part of artifact identity. Small files are hashed in full. For large
    immutable corpus shards, hash the size plus the first and last 64 KiB so
    resume checks do not reread a multi-terabyte corpus.
    """
    snapshot: list[dict[str, Any]] = []
    root = Path(root) if root is not None else None
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        if exclude_manifest and path.name == MANIFEST_NAME:
            continue
        stat = path.stat()
        digest = hashlib.sha256()
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            if stat.st_size <= 2 * CONTENT_SAMPLE_BYTES:
                for chunk in iter(
                    lambda: handle.read(1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)
                hash_scope = "full"
            else:
                digest.update(handle.read(CONTENT_SAMPLE_BYTES))
                handle.seek(-CONTENT_SAMPLE_BYTES, os.SEEK_END)
                digest.update(handle.read(CONTENT_SAMPLE_BYTES))
                hash_scope = "head_tail_64k"
        snapshot.append(
            {
                "path": str(path.relative_to(root)) if root else str(path),
                "size": stat.st_size,
                "content_sha256": digest.hexdigest(),
                "hash_scope": hash_scope,
            }
        )
    return snapshot


def tree_signature(root: Path, pattern: str = "*.jsonl") -> str:
    root = Path(root)
    return stable_digest(file_snapshot(root.glob(pattern), root=root))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_manifest(
    output_dir: Path,
    *,
    stage: str,
    contract: dict[str, Any],
    input_signature: str | None,
    output_pattern: str = "*.jsonl",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    outputs = file_snapshot(output_dir.glob(output_pattern), root=output_dir)
    if not outputs:
        raise RuntimeError(
            f"{stage}: refusing to mark empty output directory complete: {output_dir}"
        )
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "stage": stage,
        "contract": _jsonable(contract),
        "contract_sha256": stable_digest(contract),
        "input_signature": input_signature,
        "outputs": outputs,
        "output_signature": stable_digest(outputs),
    }
    if metadata is not None:
        payload["metadata"] = _jsonable(metadata)
        payload["metadata_sha256"] = stable_digest(payload["metadata"])
    atomic_write_json(output_dir / MANIFEST_NAME, payload)
    return payload


def load_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def manifest_outputs_match(
    output_dir: Path,
    *,
    output_pattern: str = "*.jsonl",
) -> bool:
    """Verify a manifest's version and output files without its stage contract."""
    manifest = load_manifest(output_dir)
    if not manifest or manifest.get("manifest_version") != MANIFEST_VERSION:
        return False
    if "metadata_sha256" in manifest and manifest.get(
        "metadata_sha256"
    ) != stable_digest(manifest.get("metadata")):
        return False
    output_dir = Path(output_dir)
    current_outputs = file_snapshot(
        output_dir.glob(output_pattern),
        root=output_dir,
    )
    return bool(current_outputs) and (
        manifest.get("output_signature") == stable_digest(current_outputs)
    )


def manifest_matches(
    output_dir: Path,
    *,
    stage: str,
    contract: dict[str, Any],
    input_signature: str | None,
    output_pattern: str = "*.jsonl",
) -> bool:
    manifest = load_manifest(output_dir)
    if not manifest:
        return False
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        return False
    if manifest.get("stage") != stage:
        return False
    if manifest.get("contract_sha256") != stable_digest(contract):
        return False
    if manifest.get("input_signature") != input_signature:
        return False
    return manifest_outputs_match(
        output_dir,
        output_pattern=output_pattern,
    )
