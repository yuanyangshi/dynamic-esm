"""
Unit and integration tests for Dynamic-ESM CLI scripts in scripts/.
"""

import os
import sys
import subprocess
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))

CLI_SCRIPTS = [
    "download_data.py",
    "run_preprocessing.py",
    "run_pretrain.py",
    "run_finetune.py",
    "run_benchmark.py",
    "run_interpretability.py",
    "run_screening.py",
]


@pytest.mark.parametrize("script_name", CLI_SCRIPTS)
def test_cli_help(script_name):
    """Test that all CLI scripts parse flags cleanly and return exit code 0 on --help."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    assert os.path.exists(script_path), f"Script {script_name} does not exist at {script_path}"

    res = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res.returncode == 0, f"Failed --help on {script_name}: {res.stderr}"
    assert "usage:" in res.stdout.lower() or "options:" in res.stdout.lower()


def test_download_data_cli_info():
    """Test that download_data.py --info prints official academic data sources."""
    script_path = os.path.join(SCRIPTS_DIR, "download_data.py")
    res = subprocess.run(
        [sys.executable, script_path, "--info"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res.returncode == 0
    assert "MISATO" in res.stdout
    assert "Zenodo" in res.stdout
    assert "CASF-2016" in res.stdout


def test_benchmark_cli_strict_missing_data():
    """Verify that run_benchmark.py raises FileNotFoundError rather than generating dummy data."""
    script_path = os.path.join(SCRIPTS_DIR, "run_benchmark.py")
    res = subprocess.run(
        [sys.executable, script_path, "--predictions-file", "./non_existent_predictions_file.npz"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # Must reject non-existent file under scientific authenticity rules
    assert res.returncode != 0
    assert "FileNotFoundError" in res.stderr or "FileNotFoundError" in res.stdout
