"""Event-driven background cognition (Doc 1 §8). A cheap tick decides whether anything deserves
deeper reflection. Most ticks do nothing. Budgets prevent runaway cost."""
from __future__ import annotations

from datetime import datetime, timedelta

from .schemas import now
from .storage import Store


class BackgroundScheduler:
    def __init__(self, store: Store, policy: dict, reflector, consolidator, enabled: bool = True):
        self.store, self.reflector, self.consolidator, self.enabled = store, reflector, consolidator, enabled
        p = policy["scheduler"]
        self.t = p["triggers"]
        self.max_per_day = int(p["max_reflections_per_day"])
        self._last_affect: dict[str, float] = {}
        self._last_consolidation: datetime | None = None

    def evaluate_triggers(self, at: datetime | None = None) -> list[str]:
        """Return the list of reasons a reflection is warranted right now (may be empty)."""
        at = at or now()
        reasons = []
        affect = self.store.load_affect().as_dict()
        if self._last_affect:
            if affect["frustration"] - self._last_affect["frustration"] >= self.t["frustration_jump"]:
                reasons.append("frustration_jump")
            if abs(affect["confidence"] - self._last_affect["confidence"]) >= self.t["confidence_jump"]:
                reasons.append("confidence_jump")
        self._last_affect = affect
        for q in self.store.questions(status="open"):
            if q.curiosity >= self.t["curiosity_min"]:
                reasons.append(f"curiosity:{q.id}")
            if q.hypothesis_claim_id:
                new = self.store.episode_count() - len(q.evidence_seen)
                if new >= self.t["min_new_evidence_for_question"] and (at - q.last_visited) > timedelta(hours=1):
                    reasons.append(f"new_evidence:{q.id}")
        # contradiction: a held belief whose latest evidence is counter
        for c in self.store.claims():
            if c.status == "belief" and c.counterevidence and c.counterevidence[-1].ref not in \
                    {e.ref for e in c.supporting_evidence}:
                last_ids = [e.ref for e in (c.supporting_evidence + c.counterevidence)]
                if last_ids and last_ids[-1] == c.counterevidence[-1].ref:
                    reasons.append(f"contradiction:{c.id}")
        return reasons

    def tick(self, at: datetime | None = None, force_consolidation: bool = False) -> dict:
        at = at or now()
        if not self.enabled:
            return {"ran": False, "reasons": [], "note": "background disabled"}
        reasons = self.evaluate_triggers(at)
        day_key = at.strftime("%Y-%m-%d")
        used = self.store.counter(f"reflections:{day_key}")
        ran, results = False, []
        if reasons and used < self.max_per_day:
            self.store.counter(f"reflections:{day_key}", 1)
            results = [r.model_dump(mode="json", include={"claim_id", "verdict", "notes"})
                       for r in self.reflector.reflect(trigger=",".join(reasons)[:60])]
            ran = True
        consolidated = None
        if force_consolidation or (self._last_consolidation is None and self.store.episode_count() >= 20) \
                or (self._last_consolidation and at - self._last_consolidation >= timedelta(hours=24)):
            consolidated = self.consolidator.run()
            self._last_consolidation = at
        self.store.audit("tick", reasons=reasons, ran=ran, budget_used=used, consolidated=bool(consolidated))
        return {"ran": ran, "reasons": reasons, "reflections": results, "consolidation": consolidated}
