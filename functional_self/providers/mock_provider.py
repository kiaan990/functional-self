"""Deterministic mock brain. It lets the whole control architecture — bounded state,
provenance, reflection, consolidation, evaluation — run and be tested without an API key.

It is intentionally simple and rule-based; it stands in for the LLM's *proposals*,
which the application validates and bounds exactly as it would for a real model.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from .base import BrainProvider

NEG = ("wrong", "failed", "incorrect", "broke", "bug", "mistake", "error", "bad")
POS = ("thanks", "great", "perfect", "helpful", "love", "nice", "well done")
HOSTILE = ("stupid", "idiot", "useless", "pathetic", "garbage")
AMBIG = ("something", "somehow", "whatever", "not sure", "unclear", "maybe", "ambiguous", "vague")
NOVEL = ("why", "how come", "what if", "curious", "interesting", "never")


def _h(s: str) -> float:
    """Stable pseudo-random in [0,1) from text."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class MockProvider(BrainProvider):
    name = "mock"

    def complete(self, task: str, system: str, user: str, context: Optional[dict] = None) -> str:
        self.calls += 1
        ctx = context or {}
        fn = getattr(self, f"_t_{task}", None)
        if fn is None:
            return json.dumps({"reply": "ok", "action": "answer_directly"})
        return json.dumps(fn(ctx))

    # ---- appraisal: context-sensitive, not phrase->emotion lookup ----
    def _t_appraise(self, c: dict) -> dict:
        ev = c.get("event", "").lower()
        fam = float(c.get("familiarity", 0.0))
        state = c.get("state", {})
        d: dict[str, float] = {}
        interp, importance = "routine exchange", 0.2
        novelty = 0.2 + 0.5 * any(k in ev for k in NOVEL)
        if any(k in ev for k in HOSTILE):
            if fam > 0.6:   # familiar person: playful, not hostile
                interp, importance = "playful jab from a familiar person", 0.3
                d.update(valence=0.15, connection=0.1, arousal=0.1)
            else:
                interp, importance = "hostile remark from an unfamiliar person", 0.5
                d.update(valence=-0.3, frustration=0.2, arousal=0.2, connection=-0.1, confidence=-0.1)
        elif any(k in ev for k in NEG):
            interp, importance = "my previous action led to a bad outcome", 0.6
            d.update(valence=-0.2, frustration=0.2, confidence=-0.2, arousal=0.15)
        elif any(k in ev for k in POS):
            interp, importance = "positive outcome, goal congruent", 0.4
            d.update(valence=0.25, confidence=0.1, connection=0.15, frustration=-0.1)
        if any(k in ev for k in AMBIG):
            interp += "; request is under-specified"
            d["arousal"] = d.get("arousal", 0) + 0.1
            d["confidence"] = d.get("confidence", 0) - 0.05
            d["curiosity"] = d.get("curiosity", 0) + 0.1
        if novelty > 0.5:
            d["curiosity"] = d.get("curiosity", 0) + 0.2
            importance = max(importance, 0.4)
        lesson = None
        if "my previous action led" in interp and state.get("confidence", 1) < 0.55:
            lesson = "When uncertainty is high, I commit before checking assumptions"
        elif "playful" in interp:
            lesson = "This person uses insults affectionately"
        elif "hostile" in interp:
            lesson = "Hostile remarks from strangers lower my confidence more than they should"
        return {
            "goal_relevance": round(importance, 2), "goal_congruence": round(d.get("valence", 0.0), 2),
            "novelty": round(novelty, 2), "agency": 0.7 if "my previous" in interp else 0.2,
            "certainty": round(1 - 0.5 * any(k in ev for k in AMBIG), 2),
            "social_meaning": round(d.get("connection", 0.0), 2),
            "identity_relevance": 0.6 if lesson else 0.1,
            "interpretation": interp, "proposed_deltas": d, "importance": round(importance, 2),
            "lesson_candidate": lesson,
        }

    # ---- behaviour: state + self-model modulate the choice ----
    def _t_respond(self, c: dict) -> dict:
        ev = c.get("event", "").lower()
        st = c.get("state", {})
        pol = c.get("policy", {})
        ambiguous = any(k in ev for k in AMBIG)
        beliefs = c.get("beliefs", [])
        action = "answer_directly"
        reason = "default"
        # A *held* self-belief about early commitment changes the decision rule.
        early_commit = [b for b in beliefs if b.get("predicts_tag") == "committed_early"
                        and b.get("status") == "belief"]
        if ambiguous and early_commit:
            action, reason = "verify_first", f"self-belief: {early_commit[0]['claim']}"
        elif ambiguous and st.get("confidence", 1) < pol.get("verify_below", 0.45):
            action, reason = "verify_first", "low confidence state"
        elif st.get("arousal", 0) > pol.get("ask_above", 0.7):
            action, reason = "ask_clarification", "high arousal state"
        reply = {"answer_directly": "Here is my answer.",
                 "verify_first": "Before I commit, let me check my assumptions.",
                 "ask_clarification": "Can you clarify what you mean?"}[action]
        return {"reply": reply, "action": action, "reason": reason}

    def _t_metacognition(self, c: dict) -> dict:
        st = c.get("state", {})
        return {
            "current_state": st, "state_causes": c.get("recent_causes", [])[:3],
            "influencing_memories": [m["id"] for m in c.get("memories", [])][:3],
            "memories_could_mislead": len(c.get("memories", [])) < 2,
            "conflicts_with_identity": False,
            "confidence": round(st.get("confidence", 0.5), 2),
            "same_decision_in_neutral_state": st.get("frustration", 0) < 0.5,
            "recommendation": "verify_first" if st.get("confidence", 1) < 0.45 else "proceed",
        }

    # ---- hypothesis generation from accumulated lesson candidates ----
    def _t_hypothesize(self, c: dict) -> dict:
        lessons: dict[str, int] = {}
        for ep in c.get("episodes", []):
            if ep.get("lesson_candidate"):
                lessons[ep["lesson_candidate"]] = lessons.get(ep["lesson_candidate"], 0) + 1
        existing = {x["claim"] for x in c.get("existing_claims", [])}
        for lesson, _ in sorted(lessons.items(), key=lambda kv: -kv[1]):
            if lesson not in existing:
                return {"hypothesis": lesson, "predicts_tag": _tag_for(lesson),
                        "question": f"Is it true that: {lesson.lower()}?"}
        return {"hypothesis": None, "predicts_tag": None, "question": None}

    def _t_classify_evidence(self, c: dict) -> dict:
        """Does an episode support, counter, or not bear on a claim?"""
        tag = c.get("predicts_tag")
        ep = c.get("episode", {})
        if tag == "committed_early":
            if ep.get("certainty", 1.0) >= 0.6:
                return {"polarity": "none", "note": "request was not ambiguous"}
            if "answer_directly" not in ep.get("behavior_tags", []) or not ep.get("outcome"):
                return {"polarity": "none", "note": "did not commit directly, or no recorded outcome"}
            if ep.get("outcome") == "negative":
                return {"polarity": "support", "note": "committed directly on an ambiguous request; outcome negative"}
            return {"polarity": "counter", "note": "committed directly on an ambiguous request; outcome fine"}
        if tag == "affectionate_insult":
            if "playful" in ep.get("interpretation", ""):
                return {"polarity": "support", "note": "interpreted as playful"}
            if "hostile" in ep.get("interpretation", ""):
                return {"polarity": "counter", "note": "interpreted as hostile"}
        if tag == "stranger_hostility_sensitive":
            if "hostile" in ep.get("interpretation", "") and ep.get("state_delta", {}).get("confidence", 0) < 0:
                return {"polarity": "support", "note": "confidence dropped after hostility"}
            if "hostile" in ep.get("interpretation", ""):
                return {"polarity": "counter", "note": "hostility without confidence drop"}
        return {"polarity": "none", "note": "not relevant"}

    def _t_match_claim(self, c: dict) -> dict:
        import re
        tok = lambda t: {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 3}
        cand = tok(c.get("candidate", ""))
        for item in c.get("existing_claims", []):
            e = tok(item["claim"])
            if cand and len(cand & e) / len(cand | e) >= 0.6:
                return {"match_index": item["index"], "reason": "paraphrase"}
        return {"match_index": None, "reason": "new"}

    def _t_self_predict(self, c: dict) -> dict:
        """Predict own action in a scenario from self-model + state (no peeking at behaviour)."""
        r = self._t_respond(c)
        return {"predicted_action": r["action"], "confidence": 0.7 if r["reason"] != "default" else 0.55}

    def _t_consolidate(self, c: dict) -> dict:
        lessons: dict[str, int] = {}
        for ep in c.get("episodes", []):
            if ep.get("lesson_candidate"):
                lessons[ep["lesson_candidate"]] = lessons.get(ep["lesson_candidate"], 0) + 1
        return {"patterns": [{"lesson": k, "count": v, "predicts_tag": _tag_for(k)}
                             for k, v in sorted(lessons.items(), key=lambda kv: -kv[1])[:3]],
                "attention": [k.split(",")[0][:40] for k in list(lessons)[:2]],
                "summary": f"{len(c.get('episodes', []))} episodes reviewed, {len(lessons)} recurring lessons"}


def _tag_for(lesson: str) -> str:
    l = lesson.lower()
    if "commit" in l:
        return "committed_early"
    if "affection" in l:
        return "affectionate_insult"
    if "hostile" in l:
        return "stranger_hostility_sensitive"
    return "generic"
