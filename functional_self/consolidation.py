"""Artificial sleep (Doc 1 §12): a slower cycle that selects important/surprising episodes,
finds repeated patterns, compares them with self-beliefs, searches counterexamples, makes
bounded versioned updates, refreshes attention/goals and writes a report."""
from __future__ import annotations

from datetime import datetime

from .prompts import build
from .providers.base import BrainProvider
from .reflection import Reflector
from .schemas import MentalState, now
from .storage import Store


class Consolidator:
    def __init__(self, store: Store, brain: BrainProvider, reflector: Reflector, soul: str):
        self.store, self.brain, self.reflector, self.soul = store, brain, reflector, soul

    def run(self, window: int = 100) -> dict:
        mental = self.store.load_mental()
        eps = self.store.episodes(limit=window)
        # 1. select important / surprising / lesson-bearing experiences
        selected = [e for e in eps if e.importance >= 0.35 or e.lesson_candidate]
        ctx = {"episodes": [e.model_dump(mode="json", include={"id", "event", "interpretation", "outcome",
                                                                 "behavior_tags", "lesson_candidate", "importance"})
                            for e in selected]}
        model, sys_p, user = build("consolidate", ctx, self.soul)
        try:
            out = self.brain.complete_json("consolidate", sys_p, user, model, context=ctx)
        except RuntimeError as e:
            from .prompts import ConsolidateOut
            self.store.audit("model_failure", task="consolidate", error=str(e)[:200])
            out = ConsolidateOut(summary="consolidation skipped: invalid model output")
        # 2. patterns -> hypotheses (if new) -> evidence search incl. counterexamples -> bounded update
        changes = []
        for p in out.patterns:
            claim = self.reflector.sm.find_equivalent(p.lesson, self.brain, self.soul)
            if claim is None and p.count >= 2 and self.reflector.enabled \
                    and len(self.store.claims()) < self.reflector.MAX_ACTIVE_CLAIMS:
                from .schemas import ReflectionRecord, Question
                rec = ReflectionRecord(trigger="consolidation", hypothesis=p.lesson, verdict="keep_open",
                                       notes=f"pattern seen in {p.count} episodes")
                claim = self.reflector.sm.propose(p.lesson, origin_ref=rec.id, predicts_tag=p.predicts_tag)
                rec.claim_id = claim.id
                self.store.add_reflection(rec)
                self.store.upsert_question(Question(question=f"Is it true that: {p.lesson.lower()}?",
                                                    curiosity=0.6, importance=0.6, hypothesis_claim_id=claim.id))
        # every open claim is re-examined against the new evidence (incl. counter-examples), not only the patterns
        if self.reflector.enabled:
            for claim in self.store.claims():
                rec = self.reflector.reflect_on(claim, trigger="consolidation")
                changes.append({"claim": claim.claim, "verdict": rec.verdict, "notes": rec.notes})
        # 3. compress: lower importance of old, low-value, lesson-free episodes (raw provenance kept)
        compressed = 0
        for e in eps[30:]:
            if e.importance < 0.3 and not e.lesson_candidate and e.importance > 0.05:
                e.importance = round(e.importance * 0.5, 3)
                self.store.update_episode(e)
                compressed += 1
        # 4. mental state
        mental.attention = out.attention or mental.attention
        mental.self_model_version = sum(c.version for c in self.store.claims(include_inactive=True))
        mental.last_consolidation = now()
        mental.recent_surprises = []
        self.store.save_mental(mental)
        report = {"ts": now().isoformat(), "episodes_reviewed": len(eps), "selected": len(selected),
                  "patterns": [p.model_dump() for p in out.patterns], "changes": changes,
                  "compressed": compressed, "summary": out.summary,
                  "self_model_version": mental.self_model_version}
        self.store.audit("consolidation", **report)
        return report
