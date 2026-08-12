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


def test_private_working_material_is_absent() -> None:
    for name in ("data", "external", "memory-bank", "paper", ".venv"):
        assert not (ROOT / name).exists()
