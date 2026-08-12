#!/usr/bin/env python3
"""Run a data-free inference check with the released Stage-A checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from glucoflow.models import build_flow_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "stage_a.pt"
EXPECTED_SHA256 = "759dabb7a5b20f9377a8d9ddaa2ae333d8e7166bc07c8d17a04f14d58acdd5a7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    actual_hash = sha256(CHECKPOINT)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint hash mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
        )

    model = build_flow_model()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    generator = torch.Generator().manual_seed(7)
    history = torch.randn(2, 24, generator=generator)
    time_features = torch.zeros(2, 24, 4)

    for nfe in (1, 2, 4):
        torch.manual_seed(7)
        samples = model.sample(
            history,
            nfe=nfe,
            num_samples=3,
            time_features=time_features,
        )
        if samples.shape != (2, 3, 24) or not torch.isfinite(samples).all():
            raise RuntimeError(f"Invalid output for NFE={nfe}: {tuple(samples.shape)}")

    print(f"OK: checkpoint={actual_hash[:12]}..., NFE=1/2/4, shape=(2, 3, 24)")


if __name__ == "__main__":
    main()
