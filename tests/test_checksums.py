from pathlib import Path

import pytest

from scripts.checksums import verify_manifest, write_manifest


def test_manifest_detects_changed_payload(tmp_path):
    (tmp_path / "README.md").write_text("review bundle\n", encoding="utf-8")
    data = tmp_path / "data" / "final_patients"
    data.mkdir(parents=True)
    (data / "patients.csv").write_text("case_id\ncase_1\n", encoding="utf-8")

    assert write_manifest(tmp_path) == 2
    assert verify_manifest(tmp_path) == 2
    (data / "patients.csv").write_text("case_id\ncase_2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch.*patients.csv"):
        verify_manifest(tmp_path)


def test_manifest_rejects_unlisted_file(tmp_path):
    (tmp_path / "README.md").write_text("review bundle\n", encoding="utf-8")
    write_manifest(tmp_path)
    (tmp_path / "requirements.txt").write_text("pandas\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file list differs"):
        verify_manifest(tmp_path)


def test_manifest_ignores_local_venv_and_git_metadata(tmp_path):
    (tmp_path / "README.md").write_text("review bundle\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").symlink_to("/usr/bin/python3")
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "objects" / "blob").write_bytes(b"local metadata")

    assert write_manifest(tmp_path) == 1
    assert verify_manifest(tmp_path) == 1
