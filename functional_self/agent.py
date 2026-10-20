"""The core experience loop:

perceive -> decay state -> retrieve -> appraise (bounded) -> reason/respond -> metacognitive
check -> act -> store episode -> (outcome) -> background tick -> reflection/consolidation.

The application owns identity, state, memory and updates. The model is the reasoning engine.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from .affect import AffectEngine
from .config import Config
from .consolidation import Consolidator
from .identity import IdentityKernel
from .memory import EpisodicMemory
from .prompts import build
from .providers.base import BrainProvider
from .reflection import Reflector
from .scheduler import BackgroundScheduler
from .schemas import AppraisalProposal, Episode, MetacognitiveCheck, now
from .self_model import SelfModel
from .storage import Store


class FunctionalSelfAgent:
    def __init__(self, config: Config, brain: BrainProvider):
        self.cfg, self.brain = config, brain
        self.identity = IdentityKernel(config.soul_path)
        self.store = Store(config.db_path)
        self.affect = AffectEngine(config.policy, enabled=config.enable_affect)
        self.memory = EpisodicMemory(self.store, config.policy, enabled=config.enable_episodic_retrieval)
        self.self_model = SelfModel(self.store, config.policy, enabled=config.enable_self_model)
        self.reflector = Reflector(self.store, brain, self.self_model, self.identity.summary(),
                                  enabled=config.enable_reflection)
        self.consolidator = Consolidator(self.store, brain, self.reflector, self.identity.summary())
        self.scheduler = BackgroundScheduler(self.store, config.policy, self.reflector, self.consolidator,
                                             enabled=config.enable_background)
        self.state = self.store.load_affect()
        if not config.enable_affect:
            self.state = self.affect.neutral()
        self.trace: list[dict] = []
        self.store.audit("boot", identity_digest=self.identity.digest, provider=brain.name,
                         ablations={k: getattr(config, k) for k in vars(config) if k.startswith("enable_")})

    # ---- relationships (minimal): familiarity grows with interactions ----
    def familiarity(self, person: str) -> float:
        n = self.store.counter(f"person:{person}")
        return n / (n + 4.0)

    # ---- main loop ----
    def interact(self, event: str, person: str = "user", at: Optional[datetime] = None,
                 condition: Optional[str] = None, consequential: bool = True) -> dict:
        at = at or now()
        fam = self.familiarity(person)
        state_before = self.affect.decay(self.state, at).as_dict() if self.cfg.enable_affect else self.state.as_dict()

        # retrieve
        memories = self.memory.retrieve(event, self.state, at=at)
        mem_ctx = [{"id": e.id, "event": e.event, "interpretation": e.interpretation,
                    "outcome": e.outcome, "lesson": e.lesson_candidate} for e, _ in memories]

        # appraise (LLM proposes) -> bounded apply (code decides)
        ctx = {"event": event, "person": person, "familiarity": round(fam, 2), "state": state_before,
               "relevant_memories": mem_ctx, "identity_anchors": self.identity.anchors()}
        model, sys_p, user = build("appraise", ctx, self.identity.summary())
        try:
            proposal: AppraisalProposal = self.brain.complete_json("appraise", sys_p, user, model, context=ctx)
        except RuntimeError as e:   # fail-soft: an invalid model output never kills the agent; it is logged
            self.store.audit("model_failure", task="appraise", error=str(e)[:200])
            proposal = AppraisalProposal(goal_relevance=0.2, goal_congruence=0.0, novelty=0.2, agency=0.2, certainty=0.5,
                                         social_meaning=0.0, identity_relevance=0.0, importance=0.2,
                                         interpretation="appraisal unavailable (invalid model output)", proposed_deltas={})
        self.state, applied = self.affect.apply(self.state, proposal.proposed_deltas, at=at)
        self.store.save_affect(self.state)

        # reason / respond
        beliefs = self.self_model.held_beliefs()
        pol = self.cfg.policy["behavior"]
        rctx = {"event": event, "person": person, "state": self.state.as_dict(), "beliefs": beliefs,
                "relevant_memories": mem_ctx,
                "policy": {"verify_below": pol["verify_before_commit_when_confidence_below"],
                           "ask_above": pol["ask_clarifying_when_arousal_above"],
                           "rule": "verify_first when a held belief says you commit early and the request "
                                   "is ambiguous; ask_clarification when arousal is above ask_above"}}
        model, sys_p, user = build("respond", rctx, self.identity.summary())
        try:
            resp = self.brain.complete_json("respond", sys_p, user, model, context=rctx)
        except RuntimeError as e:
            from .prompts import RespondOut
            self.store.audit("model_failure", task="respond", error=str(e)[:200])
            resp = RespondOut(reply="(no reply: invalid model output)", action="answer_directly",
                              reason="FALLBACK: invalid model output")
        action = resp.action

        # metacognitive check before consequential actions
        meta: Optional[MetacognitiveCheck] = None
        if consequential:
            mctx = {"state": self.state.as_dict(), "recent_causes": [proposal.interpretation],
                    "memories": mem_ctx, "intended_action": action, "identity_anchors": self.identity.anchors()}
            model, sys_p, user = build("metacognition", mctx, self.identity.summary())
            try:
                meta = self.brain.complete_json("metacognition", sys_p, user, model, context=mctx)
            except RuntimeError as e:
                self.store.audit("model_failure", task="metacognition", error=str(e)[:200]); meta = None
            if meta and meta.conflicts_with_identity and action == "answer_directly":
                action = "verify_first"

        # store structured episode (never a transcript dump)
        ep = Episode(ts=at, event=event, people=[person], interpretation=proposal.interpretation,
                     state_before=state_before, state_delta=applied, action=resp.reply,
                     behavior_tags=[action], importance=proposal.importance, certainty=proposal.certainty,
                     lesson_candidate=proposal.lesson_candidate, condition=condition)
        self.store.add_episode(ep)
        self.store.counter(f"person:{person}", 1)

        rec = {"episode_id": ep.id, "reply": resp.reply, "action": action, "reason": resp.reason,
               "interpretation": proposal.interpretation, "state_before": state_before,
               "state_after": self.state.as_dict(), "applied_delta": applied,
               "beliefs_in_play": [b["claim"] for b in beliefs if b["status"] == "belief"],
               "metacognition": meta.model_dump() if meta else None,
               "memories_used": [m["id"] for m in mem_ctx]}
        if self.cfg.debug:
            self.trace.append(rec)
        return rec

    def record_outcome(self, episode_id: str, outcome: str, at: Optional[datetime] = None) -> dict:
        """Outcome feedback ('positive'/'negative'/text) becomes high-weight evidence and a second appraisal."""
        ep = self.store.get_episode(episode_id)
        if not ep:
            raise KeyError(episode_id)
        ep.outcome = outcome
        ep.importance = min(1.0, ep.importance + (0.25 if outcome == "negative" else 0.1))
        if outcome == "negative" and "answer_directly" in ep.behavior_tags and ep.certainty < 0.6:
            ep.lesson_candidate = ep.lesson_candidate or "When uncertainty is high, I commit before checking assumptions"
        self.store.update_episode(ep)
        self.store.invalidate_classifications(episode_id)   # the outcome changed what this episode is evidence of
        deltas = {"negative": {"valence": -0.12, "frustration": 0.12, "confidence": -0.1},
                  "positive": {"valence": 0.12, "confidence": 0.08}}.get(outcome, {})
        self.state, applied = self.affect.apply(self.state, deltas, at=at)
        self.store.save_affect(self.state)
        self.store.audit("outcome", episode=episode_id, outcome=outcome, applied=applied)
        return {"episode": episode_id, "outcome": outcome, "applied": applied}

    # ---- background / manual cognition ----
    def tick(self, at: Optional[datetime] = None, force_consolidation: bool = False) -> dict:
        return self.scheduler.tick(at=at, force_consolidation=force_consolidation)

    def reflect(self) -> list[dict]:
        return [r.model_dump(mode="json") for r in self.reflector.reflect(trigger="manual")]

    def consolidate(self) -> dict:
        return self.consolidator.run()

    # ---- self-prediction (for evaluation) ----
    def self_predict(self, event: str, person: str = "user") -> dict:
        pol = self.cfg.policy["behavior"]
        ctx = {"event": event, "person": person, "state": self.state.as_dict(),
               "beliefs": self.self_model.held_beliefs(),
               "policy": {"verify_below": pol["verify_before_commit_when_confidence_below"],
                          "ask_above": pol["ask_clarifying_when_arousal_above"]}}
        model, sys_p, user = build("self_predict", ctx, self.identity.summary())
        try:
            return self.brain.complete_json("self_predict", sys_p, user, model, context=ctx).model_dump()
        except RuntimeError as e:
            self.store.audit("model_failure", task="self_predict", error=str(e)[:200])
            return {"predicted_action": "answer_directly", "confidence": 0.5, "fallback": True}

    # ---- inspection ----
    def self_state(self) -> dict:
        m = self.store.load_mental()
        return {"affect": self.state.as_dict(), "attention": m.attention, "goals": m.active_goals,
                "self_model_version": m.self_model_version, "episodes": self.store.episode_count(),
                "identity_digest": self.identity.digest, "identity_intact": self.identity.verify_unchanged()}

    def snapshot(self, name: str) -> dict:
        return self.store.snapshot(name)

    def set_state(self, at: Optional[datetime] = None, **overrides: float) -> None:
        """Experimental intervention: set affect directly (used by causal tests; audited)."""
        for k, v in overrides.items():
            setattr(self.state, k, float(v))
        self.state.updated_at = at or now()
        self.store.save_affect(self.state)
        self.store.audit("intervention", overrides=overrides)
