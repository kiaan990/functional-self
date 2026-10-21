"""Thought-to-learning pipeline (Doc 1 §9):

THOUGHT -> hypothesis/question -> search REAL history -> support + counterevidence
-> validator -> reject | keep_open | bounded self-model update.

A reflection may *create* a hypothesis, but the reflection itself is not evidence for it.
"""
from __future__ import annotations

from typing import Optional

from .prompts import build
from .providers.base import BrainProvider
from .schemas import (Episode, EvidenceRef, Provenance, Question, ReflectionRecord, SelfClaim, now)
from .self_model import SelfModel
from .storage import Store


class Reflector:
    def __init__(self, store: Store, brain: BrainProvider, self_model: SelfModel, soul: str,
                 enabled: bool = True):
        self.store, self.brain, self.sm, self.soul, self.enabled = store, brain, self_model, soul, enabled

    # ---- step 1: thought -> hypothesis + question ----
    MAX_ACTIVE_CLAIMS = 6   # attention budget: no new hypotheses while this many are open

    def hypothesize(self, trigger: str = "manual", window: int = 40) -> Optional[tuple[SelfClaim, Question]]:
        if len(self.store.claims()) >= self.MAX_ACTIVE_CLAIMS:
            return None
        eps = self.store.episodes(limit=window)
        ctx = {"episodes": [e.model_dump(mode="json", include={"id", "event", "interpretation",
                                                                 "outcome", "behavior_tags", "lesson_candidate"})
                            for e in eps],
               "existing_claims": [{"claim": c.claim} for c in self.store.claims(include_inactive=True)]}
        model, sys_p, user = build("hypothesize", ctx, self.soul)
        try:
            out = self.brain.complete_json("hypothesize", sys_p, user, model, context=ctx)
        except RuntimeError as e:
            self.store.audit("model_failure", task="hypothesize", error=str(e)[:200]); return None
        if not out.hypothesis:
            return None
        existing = self.sm.find_equivalent(out.hypothesis, self.brain, self.soul)
        if existing:
            return None   # a paraphrase of an open claim is not a new claim
        rec = ReflectionRecord(trigger=trigger, hypothesis=out.hypothesis, verdict="keep_open",
                               notes="hypothesis generated; no evidence yet")
        claim = self.sm.propose(out.hypothesis, origin_ref=rec.id, predicts_tag=out.predicts_tag)
        rec.claim_id = claim.id
        self.store.add_reflection(rec)
        q = Question(question=out.question or f"Is it true that {out.hypothesis}?",
                     curiosity=0.7, importance=0.6, hypothesis_claim_id=claim.id)
        self.store.upsert_question(q)
        return claim, q

    # ---- step 2+3: search history for support AND counterevidence ----
    def gather_evidence(self, claim: SelfClaim, window: int = 200) -> list[EvidenceRef]:
        refs = []
        known = claim.evidence_ids()
        for ep in self.store.episodes(limit=window):
            if ep.id in known:
                continue
            cached = self.store.get_classification(claim.id, ep.id)
            if cached is not None:
                polarity, note = cached
            else:
                ctx = {"claim": claim.claim, "predicts_tag": claim.predicts_tag,
                       "episode": ep.model_dump(mode="json", include={"id", "event", "interpretation", "outcome",
                                                                      "behavior_tags", "state_delta", "certainty"})}
                model, sys_p, user = build("classify_evidence", ctx, self.soul)
                try:
                    out = self.brain.complete_json("classify_evidence", sys_p, user, model, context=ctx)
                    polarity, note = out.polarity, out.note
                except RuntimeError as e:
                    self.store.audit("model_failure", task="classify_evidence", error=str(e)[:200])
                    continue   # not cached: will be judged again next time
                # cached per (claim, episode); record_outcome() invalidates the episode's cache when its outcome changes
                self.store.set_classification(claim.id, ep.id, polarity, note)
            if polarity in ("support", "counter"):
                prov = Provenance.OUTCOME if ep.outcome else Provenance.EPISODE
                refs.append(EvidenceRef(ref=ep.id, provenance=prov, polarity=polarity, note=note))
        return refs

    # ---- step 4: validator + bounded update ----
    def reflect_on(self, claim: SelfClaim, trigger: str = "manual") -> ReflectionRecord:
        if not self.enabled:
            return ReflectionRecord(trigger=trigger, hypothesis=claim.claim, claim_id=claim.id,
                                    verdict="keep_open", notes="reflection disabled (ablation)")
        before = claim.confidence
        refs = self.gather_evidence(claim)
        sup = [r.ref for r in refs if r.polarity == "support"]
        con = [r.ref for r in refs if r.polarity == "counter"]
        if not refs:
            verdict, notes = "keep_open", "no new evidence; confidence unchanged (thoughts are not facts)"
        else:
            claim, report = self.sm.add_evidence(claim, refs, reason=f"reflection:{trigger}")
            verdict = "update"
            if claim.status == "rejected":
                verdict = "reject"
            notes = f"+{len(sup)} support, +{len(con)} counter; {before:.2f} -> {claim.confidence:.2f}"
        rec = ReflectionRecord(trigger=trigger, hypothesis=claim.claim, claim_id=claim.id,
                               support=sup, counter=con, verdict=verdict,
                               confidence_before=before, confidence_after=claim.confidence, notes=notes)
        self.store.add_reflection(rec)
        # bookkeeping on the linked question
        for q in self.store.questions(status="open"):
            if q.hypothesis_claim_id == claim.id:
                q.evidence_seen = sorted(set(q.evidence_seen) | set(sup) | set(con))
                q.last_visited = now()
                q.curiosity = max(0.1, q.curiosity - 0.15)      # curiosity is spent by visiting
                if claim.status in ("belief", "rejected"):
                    q.status = "answered"
                self.store.upsert_question(q)
        return rec

    def reflect(self, trigger: str = "manual") -> list[ReflectionRecord]:
        """One full reflection: maybe generate a hypothesis, then test all open ones."""
        out = []
        if not self.enabled:
            return out
        self.hypothesize(trigger=trigger)
        for c in self.store.claims():
            if c.status in ("hypothesis", "belief"):
                out.append(self.reflect_on(c, trigger=trigger))
        return out
