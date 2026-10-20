"""Structured schemas. The LLM only ever produces *proposals* in these shapes;
application code validates, clamps and commits them."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex[:12]}"


AFFECT_DIMS = ("valence", "arousal", "curiosity", "confidence", "frustration", "connection")


class Provenance(str, Enum):
    """Evidence classes, ordered by evidential weight (Doc 1 §10)."""
    OUTCOME = "outcome"          # recorded outcome of an action
    EPISODE = "episode"          # historical experience
    EXTERNAL = "external"        # external source
    INFERENCE = "inference"
    REFLECTION = "reflection"    # generated hypothesis
    IMAGINATION = "imagination"  # zero weight


class AffectState(BaseModel):
    valence: float = 0.0
    arousal: float = 0.3
    curiosity: float = 0.4
    confidence: float = 0.6
    frustration: float = 0.1
    connection: float = 0.3
    updated_at: datetime = Field(default_factory=now)

    def as_dict(self) -> dict[str, float]:
        return {d: round(getattr(self, d), 3) for d in AFFECT_DIMS}


class AppraisalProposal(BaseModel):
    """What the LLM proposes after an event. Deltas are *requests*; code clamps them."""
    goal_relevance: float = Field(ge=0, le=1)
    goal_congruence: float = Field(ge=-1, le=1)
    novelty: float = Field(ge=0, le=1)
    agency: float = Field(ge=0, le=1, description="how much the agent caused this")
    certainty: float = Field(ge=0, le=1)
    social_meaning: float = Field(ge=-1, le=1)
    identity_relevance: float = Field(ge=0, le=1)
    interpretation: str
    proposed_deltas: dict[str, float] = Field(default_factory=dict)
    importance: float = Field(ge=0, le=1)
    lesson_candidate: Optional[str] = None

    @field_validator("proposed_deltas")
    @classmethod
    def only_known_dims(cls, v: dict[str, float]) -> dict[str, float]:
        return {k: float(x) for k, x in v.items() if k in AFFECT_DIMS}


class Episode(BaseModel):
    id: str = Field(default_factory=lambda: new_id("episode"))
    ts: datetime = Field(default_factory=now)
    event: str
    people: list[str] = Field(default_factory=list)
    interpretation: str = ""
    state_before: dict[str, float] = Field(default_factory=dict)
    state_delta: dict[str, float] = Field(default_factory=dict)
    action: str = ""
    outcome: Optional[str] = None
    behavior_tags: list[str] = Field(default_factory=list)   # e.g. ["asked_clarification"]
    importance: float = 0.3
    certainty: float = 1.0          # appraisal certainty about what the event meant / what was asked
    lesson_candidate: Optional[str] = None
    provenance: Provenance = Provenance.EPISODE
    condition: Optional[str] = None   # experiment condition label


class EvidenceRef(BaseModel):
    ref: str                       # episode:xxx / reflection:xxx / external:xxx
    provenance: Provenance
    polarity: Literal["support", "counter"]
    note: str = ""


class SelfClaim(BaseModel):
    id: str = Field(default_factory=lambda: new_id("claim"))
    claim: str
    confidence: float = 0.5
    status: Literal["hypothesis", "belief", "rejected", "retired"] = "hypothesis"
    supporting_evidence: list[EvidenceRef] = Field(default_factory=list)
    counterevidence: list[EvidenceRef] = Field(default_factory=list)
    predicts_tag: Optional[str] = None  # behavior tag this claim predicts, for calibration
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    version: int = 1
    origin: str = "reflection"

    def evidence_ids(self) -> set[str]:
        return {e.ref for e in self.supporting_evidence} | {e.ref for e in self.counterevidence}


class Question(BaseModel):
    id: str = Field(default_factory=lambda: new_id("q"))
    question: str
    curiosity: float = 0.5
    importance: float = 0.5
    status: Literal["open", "answered", "parked"] = "open"
    hypothesis_claim_id: Optional[str] = None
    evidence_seen: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)
    last_visited: datetime = Field(default_factory=now)


class ReflectionRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("reflection"))
    ts: datetime = Field(default_factory=now)
    trigger: str
    question_id: Optional[str] = None
    hypothesis: str
    claim_id: Optional[str] = None
    support: list[str] = Field(default_factory=list)
    counter: list[str] = Field(default_factory=list)
    verdict: Literal["reject", "keep_open", "update"]
    confidence_before: Optional[float] = None
    confidence_after: Optional[float] = None
    notes: str = ""


class MentalState(BaseModel):
    attention: list[str] = Field(default_factory=list)
    active_goals: list[str] = Field(default_factory=list)
    recent_surprises: list[str] = Field(default_factory=list)
    self_model_version: int = 0
    last_consolidation: Optional[datetime] = None


class MetacognitiveCheck(BaseModel):
    current_state: dict[str, float]
    state_causes: list[str]
    influencing_memories: list[str]
    memories_could_mislead: bool
    conflicts_with_identity: bool
    confidence: float
    same_decision_in_neutral_state: bool
    recommendation: Literal["proceed", "verify_first", "ask_clarification"]
