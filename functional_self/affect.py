"""Bounded affect dynamics. The LLM proposes; this code decides.

Every variable has [min, max, baseline, half_life_hours]. Per-event deltas are clamped to
max_delta_per_event, values are clamped to bounds, and state decays toward baseline over time.
Affect may influence retrieval and decision thresholds — never permissions.
"""
from __future__ import annotations

import math
from datetime import datetime

from .schemas import AFFECT_DIMS, AffectState, now


class AffectEngine:
    def __init__(self, policy: dict, enabled: bool = True):
        p = policy["affect"]
        self.bounds = {d: tuple(p[d]) for d in AFFECT_DIMS}
        self.max_delta = float(p["max_delta_per_event"])
        self.enabled = enabled

    def clamp_delta(self, proposed: dict[str, float]) -> dict[str, float]:
        return {d: max(-self.max_delta, min(self.max_delta, float(v)))
                for d, v in proposed.items() if d in AFFECT_DIMS}

    def decay(self, state: AffectState, at: datetime | None = None) -> AffectState:
        at = at or now()
        hours = max(0.0, (at - state.updated_at).total_seconds() / 3600)
        if hours == 0:
            return state
        for d in AFFECT_DIMS:
            lo, hi, base, hl = self.bounds[d]
            v = getattr(state, d)
            v = base + (v - base) * math.pow(0.5, hours / hl)
            setattr(state, d, max(lo, min(hi, v)))
        state.updated_at = at
        return state

    def apply(self, state: AffectState, proposed: dict[str, float], at: datetime | None = None
              ) -> tuple[AffectState, dict[str, float]]:
        """Returns (new_state, applied_delta)."""
        if not self.enabled:            # ablation: affect neutralised, state pinned to baseline
            return self.neutral(), {}
        state = self.decay(state, at)
        applied = {}
        for d, dv in self.clamp_delta(proposed).items():
            lo, hi, _, _ = self.bounds[d]
            before = getattr(state, d)
            after = max(lo, min(hi, before + dv))
            setattr(state, d, after)
            applied[d] = round(after - before, 4)
        state.updated_at = at or now()
        return state, applied

    def neutral(self) -> AffectState:
        s = AffectState()
        for d in AFFECT_DIMS:
            setattr(s, d, self.bounds[d][2])
        return s
