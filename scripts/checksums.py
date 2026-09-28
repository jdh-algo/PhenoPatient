"""Write and verify SHA-256 checksums for the review bundle."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath


MANIFEST = Path("data/manifests/files.sha256")
IGNORED = {"__pycache__", ".pytest_cache", ".DS_Store"}
LOCAL_ROOT_DIRS = {".venv", ".git"}


def _files(root: Path) -> list[Path]:
    result = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (relative.parts[0] in LOCAL_ROOT_DIRS
                or any(part in IGNORED for part in relative.parts)
                or relative == MANIFEST):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink in bundle: {relative}")
        if path.is_file():
            result.append(relative)
    return sorted(result, key=lambda path: path.as_posix())


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_manifest(root: Path) -> int:
    root = Path(root)
    files = _files(root)
    manifest = root / MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(f"{_digest(root / path)}  {path.as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def verify_manifest(root: Path) -> int:
    root = Path(root)
    expected = {}
    for line in (root / MANIFEST).read_text(encoding="utf-8").splitlines():
        try:
            digest, raw_path = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("malformed manifest entry") from exc
        path = PurePosixPath(raw_path)
        if (len(digest) != 64 or not all(char in "0123456789abcdef" for char in digest)
                or path.is_absolute() or ".." in path.parts or raw_path in expected):
            raise ValueError(f"malformed manifest entry: {raw_path}")
        expected[raw_path] = digest
    actual = _files(root)
    if set(expected) != {path.as_posix() for path in actual}:
        raise ValueError("file list differs from manifest")
    for path in actual:
        if _digest(root / path) != expected[path.as_posix()]:
            raise ValueError(f"checksum mismatch: {path.as_posix()}")
    return len(actual)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true", help="regenerate the manifest")
    args = parser.parse_args()
    count = write_manifest(args.root) if args.write else verify_manifest(args.root)
    print(f"CHECKSUMS_OK files={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
