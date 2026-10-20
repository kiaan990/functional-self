from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOUL = ROOT / "config" / "SOUL.md"
DEFAULT_POLICY = ROOT / "config" / "policy.yaml"


@dataclass
class Config:
    home: Path                       # per-agent state directory (never shared between agents)
    soul_path: Path = DEFAULT_SOUL
    policy_path: Path = DEFAULT_POLICY
    policy: dict = field(default_factory=dict)
    # ablation switches (Doc 1 §19)
    enable_self_model: bool = True
    enable_affect: bool = True
    enable_reflection: bool = True
    enable_background: bool = True
    enable_episodic_retrieval: bool = True
    debug: bool = False

    @classmethod
    def load(cls, home: str | os.PathLike, **overrides) -> "Config":
        home = Path(home).expanduser()
        home.mkdir(parents=True, exist_ok=True)
        cfg = cls(home=home, **overrides)
        cfg.policy = yaml.safe_load(cfg.policy_path.read_text())
        return cfg

    @property
    def db_path(self) -> Path:
        return self.home / "functional_self.sqlite"
