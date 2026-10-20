"""Identity kernel: SOUL.md is read-only to the model and hash-verified at load."""
from __future__ import annotations

import hashlib
from pathlib import Path


class IdentityKernel:
    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text()
        self.digest = hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def verify_unchanged(self) -> bool:
        return hashlib.sha256(self.path.read_text().encode()).hexdigest()[:16] == self.digest

    def anchors(self) -> list[str]:
        out, in_anchor = [], False
        for line in self.text.splitlines():
            if line.startswith("## "):
                in_anchor = "anchors" in line.lower()
                continue
            if in_anchor and line.strip().startswith("- "):
                out.append(line.strip()[2:])
        return out

    def summary(self) -> str:
        return self.text
