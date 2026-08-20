from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_released_checkpoint_hash() -> None:
    checkpoint = ROOT / "checkpoints" / "stage_a.pt"
    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        "759dabb7a5b20f9377a8d9ddaa2ae333d8e7166bc07c8d17a04f14d58acdd5a7"
    )


def test_reusable_release_excludes_private_and_paper_artifacts() -> None:
    for name in (
        "data",
        "docs",
        "external",
        "memory-bank",
        "paper",
        "results",
        ".venv",
    ):
        assert not (ROOT / name).exists()


def test_custom_adapter_example_is_present() -> None:
    assert (ROOT / "examples" / "custom_adapter.py").is_file()
