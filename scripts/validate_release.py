#!/usr/bin/env python3
"""Fail-closed checks for the reusable public release tree."""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_HASH = "759dabb7a5b20f9377a8d9ddaa2ae333d8e7166bc07c8d17a04f14d58acdd5a7"
REQUIRED = {
    ".gitignore",
    "CITATION.cff",
    "MODEL_CARD.md",
    "README.md",
    "THIRD_PARTY.md",
    "checkpoints/stage_a.pt",
    "examples/custom_adapter.py",
    "pyproject.toml",
    "scripts/finetune_cgmacros.py",
    "scripts/prepare_data.py",
    "scripts/pretrain.py",
    "scripts/smoke_test.py",
    "scripts/validate_release.py",
}
BANNED_PREFIXES = (
    ".agents/",
    ".claude/",
    ".codex/",
    ".venv/",
    "data/",
    "docs/",
    "external/",
    "memory-bank/",
    "paper/",
    "results/",
)
BANNED_SUFFIXES = (".log", ".npy", ".npz", ".pkl", ".pyc", ".zip")
MAX_FILE_BYTES = 10 * 1024 * 1024


def release_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return sorted(line for line in output.splitlines() if line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = release_files()
    failures: list[str] = []

    missing = sorted(REQUIRED.difference(files))
    failures.extend(f"missing required file: {path}" for path in missing)

    token_patterns = [
        re.compile("gh" + r"p_[A-Za-z0-9]{20,}"),
        re.compile("github" + r"_pat_[A-Za-z0-9_]{20,}"),
        re.compile("sk-" + r"[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile("BEGIN " + r"(?:RSA |OPENSSH )?PRIVATE KEY"),
    ]
    local_path_pattern = re.compile(r"/(?:home|mnt|Users)/")

    for rel in files:
        path = ROOT / rel
        if any(rel == prefix[:-1] or rel.startswith(prefix) for prefix in BANNED_PREFIXES):
            failures.append(f"banned path: {rel}")
        if rel.endswith(BANNED_SUFFIXES):
            failures.append(f"banned artifact type: {rel}")
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 10 MiB: {rel}")
        if rel.endswith(".pt") and rel != "checkpoints/stage_a.pt":
            failures.append(f"unexpected model checkpoint: {rel}")

        if rel.endswith(".py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeDecodeError) as exc:
                failures.append(f"invalid Python source {rel}: {exc}")

        if rel == "scripts/validate_release.py" or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in token_patterns):
            failures.append(f"possible credential in: {rel}")
        if local_path_pattern.search(text):
            failures.append(f"local absolute path in: {rel}")

    checkpoint = ROOT / "checkpoints" / "stage_a.pt"
    if checkpoint.exists() and sha256(checkpoint) != CHECKPOINT_HASH:
        failures.append("Stage-A checkpoint SHA-256 does not match the release manifest")

    if failures:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(failures))
    print(f"OK: {len(files)} files passed reusable-release integrity checks")


if __name__ == "__main__":
    main()
