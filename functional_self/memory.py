"""Episodic memory retrieval: relevance + importance + recency + affect congruence.
No embeddings dependency by default (token overlap); swap `relevance()` for a vector
scorer if needed. Not a transcript dump — only structured episodes are stored."""
from __future__ import annotations

import math
import re
from datetime import datetime

from .schemas import Episode, AffectState, now
from .storage import Store

_tok = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> set[str]:
    return {t for t in _tok.findall(s.lower()) if len(t) > 2}


def relevance(query: str, ep: Episode) -> float:
    q = tokens(query)
    if not q:
        return 0.0
    e = tokens(ep.event + " " + ep.interpretation + " " + (ep.lesson_candidate or ""))
    return len(q & e) / math.sqrt(len(q) * max(1, len(e)))


def affect_congruence(state: AffectState, ep: Episode) -> float:
    """Episodes whose state change matches the current state direction surface more readily
    (mood-congruent retrieval, bounded)."""
    if not ep.state_delta:
        return 0.0
    score = 0.0
    for d, dv in ep.state_delta.items():
        cur = getattr(state, d, 0.0)
        score += math.copysign(1, dv) * (cur - 0.5 if d != "valence" else cur)
    return max(0.0, min(1.0, 0.5 + score / (2 * len(ep.state_delta))))


class EpisodicMemory:
    def __init__(self, store: Store, policy: dict, enabled: bool = True):
        self.store, self.enabled = store, enabled
        r = policy["retrieval"]
        self.k, self.w = int(r["k"]), r["weights"]

    def retrieve(self, query: str, state: AffectState, at: datetime | None = None,
                 k: int | None = None) -> list[tuple[Episode, float]]:
        if not self.enabled:   # ablation: no autobiographical retrieval
            return []
        at = at or now()
        scored = []
        for ep in self.store.episodes(limit=500):
            age_h = max(0.0, (at - ep.ts).total_seconds() / 3600)
            rec = math.exp(-age_h / 72)
            s = (self.w["relevance"] * relevance(query, ep) + self.w["importance"] * ep.importance
                 + self.w["recency"] * rec + self.w["affect_congruence"] * affect_congruence(state, ep))
            scored.append((ep, s))
        scored.sort(key=lambda x: -x[1])
        return scored[: (k or self.k)]
