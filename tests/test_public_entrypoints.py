import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m3_cli_help_loads_the_published_runner():
    result = subprocess.run(
        [
            sys.executable,
            "src/phenopatient/virtual_clinical_interaction.py",
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--csv_file" in result.stdout
    assert "--num_workers" in result.stdout


def test_generation_entrypoint_loads_all_published_modules_before_input_validation(
    tmp_path,
):
    input_path = tmp_path / "empty.json"
    input_path.write_text(json.dumps([]), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "src/phenopatient/run_phenopatient.py",
            "--input_file",
            str(input_path),
            "--output_root",
            str(tmp_path / "output"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert result.stderr.strip() == "输入为空"
