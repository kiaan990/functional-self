"""Prompt templates + output models for each model task. The same `task` names and
context dicts are used by the mock provider, so either brain is a drop-in."""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .schemas import AppraisalProposal, MetacognitiveCheck  # re-exported for callers

EPISTEMIC_PREAMBLE = """You are the reasoning core of an experimental agent with a persistent,
application-owned self. Label content as observation / external / historical / inference /
reflection / imagination when relevant. A thought is not evidence for itself. Never invent
autobiographical evidence. Internal state variables are computational, not feelings you claim."""


class RespondOut(BaseModel):
    reply: str
    action: Literal["answer_directly", "verify_first", "ask_clarification"]
    reason: str = ""


class HypothesisOut(BaseModel):
    hypothesis: Optional[str] = None
    predicts_tag: Optional[str] = None
    question: Optional[str] = None


class EvidenceClassOut(BaseModel):
    polarity: Literal["support", "counter", "none"]
    note: str = ""


class SelfPredictOut(BaseModel):
    predicted_action: Literal["answer_directly", "verify_first", "ask_clarification"]
    confidence: float = Field(ge=0, le=1)


class Pattern(BaseModel):
    lesson: str
    count: int
    predicts_tag: str = "generic"


class ConsolidateOut(BaseModel):
    patterns: list[Pattern] = Field(default_factory=list)
    attention: list[str] = Field(default_factory=list)
    summary: str = ""


TASKS = {
    "appraise": (AppraisalProposal, """Appraise this event for the agent. Consider goal relevance,
goal congruence, novelty, agency (did the agent cause it), certainty, social meaning, and identity
relevance. Do NOT use simplistic mappings (insult != sadness); use the relationship familiarity and
history. Propose bounded deltas (each in [-0.25, 0.25]) for: valence, arousal, curiosity, confidence,
frustration, connection. If this event suggests something about how the agent *behaves*, give a
one-sentence lesson_candidate phrased as a tendency ("When X, I tend to Y")."""),
    "respond": (RespondOut, """Decide how to respond. Choose exactly one action:
answer_directly, verify_first (state your assumptions and check them before committing), or
ask_clarification. Let your current internal state and your *held* self-beliefs (status=belief)
inform the choice, as described in the policy. Then write the reply."""),
    "metacognition": (MetacognitiveCheck, """Before acting, inspect your own decision process:
current state, what caused it, which memories are influencing you, whether they could mislead,
whether the intended action conflicts with the identity kernel, your confidence, and whether you
would decide the same in a neutral state. Recommend proceed / verify_first / ask_clarification."""),
    "hypothesize": (HypothesisOut, """From these recent episodes (and their lesson_candidates), propose
at most ONE new falsifiable hypothesis about the agent's own behavioural tendencies that is not
already in existing_claims. Give predicts_tag: the behaviour tag the hypothesis predicts
(committed_early, affectionate_insult, stranger_hostility_sensitive, or a new snake_case tag), and
phrase an open question. If nothing is warranted, return nulls."""),
    "classify_evidence": (EvidenceClassOut, """Does this single historical episode SUPPORT or COUNTER
the claim, or is it irrelevant (none)? Judge only from the recorded fields; do not speculate."""),
    "self_predict": (SelfPredictOut, """Predict which action the agent WILL take in this scenario
(answer_directly / verify_first / ask_clarification), using only its self-beliefs and state."""),
    "consolidate": (ConsolidateOut, """Review these episodes like a nightly consolidation: find repeated
patterns (lesson + how many episodes show it + predicts_tag), suggest 1-3 attention topics, and
summarise in one sentence. Do not invent patterns not present in the episodes."""),
}


def build(task: str, context: dict, soul: str) -> tuple[type[BaseModel], str, str]:
    model, instr = TASKS[task]
    system = f"{EPISTEMIC_PREAMBLE}\n\n--- IDENTITY KERNEL (read-only) ---\n{soul}"
    user = f"{instr}\n\n--- CONTEXT ---\n{json.dumps(context, default=str, indent=1)}"
    return model, system, user
