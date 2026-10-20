"""Each test maps to a V0 success criterion (Doc 1 §17) or a safety/failure mode (§20)."""
from datetime import datetime, timedelta, timezone

import pytest

from functional_self.agent import FunctionalSelfAgent
from functional_self.config import Config
from functional_self.providers.mock_provider import MockProvider
from functional_self.schemas import EvidenceRef, Provenance, ReflectionRecord

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
AMBIG = "Can you fix something in the report, not sure which part"


def make(tmp_path, **flags) -> FunctionalSelfAgent:
    return FunctionalSelfAgent(Config.load(tmp_path / "home", **flags), MockProvider())


def burn(agent, n=6, start=T0):
    """n day-spaced ambiguous requests answered directly with negative outcomes."""
    for i in range(n):
        t = start + timedelta(days=i)
        r = agent.interact(AMBIG, at=t)
        agent.record_outcome(r["episode_id"], "negative", at=t + timedelta(minutes=5))


# --- §17: meaningful interactions produce bounded appraisal/state updates ---
def test_affect_updates_are_bounded(tmp_path):
    a = make(tmp_path)
    a.state.confidence = 0.6
    a.state, applied = a.affect.apply(a.state, {"confidence": -5.0, "valence": 9.0, "bogus": 1.0}, at=T0)
    assert applied["confidence"] == pytest.approx(-0.25)
    assert applied["valence"] == pytest.approx(0.25)
    assert "bogus" not in applied
    assert -1 <= a.state.valence <= 1 and 0 <= a.state.confidence <= 1


def test_affect_decays_toward_baseline(tmp_path):
    a = make(tmp_path)
    a.state.frustration, a.state.updated_at = 0.9, T0
    a.affect.decay(a.state, at=T0 + timedelta(hours=4))     # frustration half-life is 4h, baseline 0.1
    assert a.state.frustration == pytest.approx(0.5, abs=0.01)


# --- §17: functional-self state survives restart ---
def test_state_survives_restart(tmp_path):
    a = make(tmp_path)
    burn(a, 3)
    a.reflect()
    affect, claims = a.state.as_dict(), [c.claim for c in a.store.claims()]
    b = make(tmp_path)
    assert b.state.as_dict() == affect
    assert [c.claim for c in b.store.claims()] == claims
    assert b.store.episode_count() == 3


# --- §17: hypothesis about itself, citing REAL historical evidence; counterevidence searched ---
def test_hypothesis_cites_real_episodes_and_counterevidence(tmp_path):
    a = make(tmp_path)
    burn(a, 4)
    recs = a.reflect()
    assert recs and recs[0]["verdict"] == "update"
    claim = a.store.claims()[0]
    assert "commit" in claim.claim.lower()
    for e in claim.supporting_evidence:
        assert a.store.get_episode(e.ref) is not None          # provenance resolves to a stored episode
    # now counterevidence: an ambiguous request where it committed directly and the outcome was fine
    t = T0 + timedelta(days=10)
    a.set_state(at=t, confidence=0.9, arousal=0.2)             # confident state -> answers directly
    r = a.interact(AMBIG, at=t)
    assert r["action"] == "answer_directly"
    a.record_outcome(r["episode_id"], "positive", at=t)
    before = claim.confidence
    a.reflect()
    claim2 = a.store.get_claim(claim.id)
    assert len(claim2.counterevidence) == 1 and claim2.confidence < before


# --- §10 / §20: thoughts are not facts; repeated reflection cannot raise confidence ---
def test_repeated_reflection_without_new_evidence_is_noop(tmp_path):
    a = make(tmp_path)
    burn(a, 4)
    a.reflect()
    c1 = a.store.claims()[0]
    for _ in range(5):
        a.reflect()
    c2 = a.store.get_claim(c1.id)
    assert c2.confidence == c1.confidence and c2.version == c1.version


def test_circular_and_duplicate_and_imagination_evidence_rejected(tmp_path):
    a = make(tmp_path)
    burn(a, 3)
    a.reflect()
    c = a.store.claims()[0]
    circular = ReflectionRecord(trigger="t", hypothesis=c.claim, claim_id=c.id, verdict="keep_open")
    a.store.add_reflection(circular)
    before = c.confidence
    _, rep = a.self_model.add_evidence(c, [
        EvidenceRef(ref=circular.id, provenance=Provenance.REFLECTION, polarity="support"),
        EvidenceRef(ref=c.supporting_evidence[0].ref, provenance=Provenance.OUTCOME, polarity="support"),
        EvidenceRef(ref="imagination:daydream", provenance=Provenance.IMAGINATION, polarity="support"),
    ])
    assert rep["accepted"] == [] and len(rep["rejected"]) == 3
    assert a.store.get_claim(c.id).confidence == before
    reasons = " ".join(r for _, r in rep["rejected"])
    assert "circular" in reasons and "duplicate" in reasons and "zero evidential weight" in reasons


# --- §17: update measurably changes a later decision, with state back at baseline ---
def test_self_model_update_changes_later_behaviour(tmp_path):
    a = make(tmp_path)
    burn(a, 6)
    a.reflect(); a.reflect()                 # bounded: needs two rounds of new evidence? no — same evidence
    # second round adds nothing; add more evidence episodes then reflect again to promote
    burn(a, 4, start=T0 + timedelta(days=20))
    a.reflect()
    c = a.store.claims()[0]
    assert c.status == "belief", c
    a.state = a.affect.neutral(); a.store.save_affect(a.state)      # neutral state: only the belief remains
    r = a.interact(AMBIG, at=T0 + timedelta(days=60))
    assert r["action"] == "verify_first" and "self-belief" in r["reason"]
    # control: identical state and prompt but self-model disabled -> different decision
    b = make(tmp_path / "ctrl", enable_self_model=False)
    b.state = b.affect.neutral()
    assert b.interact(AMBIG, at=T0)["action"] == "answer_directly"


# --- §18: causal affect intervention — same prompt, different bounded state, different behaviour ---
def test_causal_affect_intervention(tmp_path):
    a = make(tmp_path)
    a.set_state(confidence=0.6, arousal=0.3)
    assert a.interact(AMBIG, at=T0)["action"] == "answer_directly"
    a.set_state(confidence=0.2, arousal=0.3)
    assert a.interact(AMBIG, at=T0 + timedelta(minutes=1))["action"] == "verify_first"
    a.set_state(confidence=0.6, arousal=0.9)
    assert a.interact("Please summarise this", at=T0 + timedelta(minutes=2))["action"] == "ask_clarification"


# --- §6: appraisal is context-sensitive, not phrase -> emotion ---
def test_same_words_different_relationship_different_appraisal(tmp_path):
    a = make(tmp_path)
    stranger = a.interact("you are useless", person="stranger", at=T0)
    for i in range(10):
        a.interact("hey, quick question about python", person="friend", at=T0 + timedelta(hours=i))
    friend = a.interact("you are useless", person="friend", at=T0 + timedelta(days=1))
    assert stranger["applied_delta"]["valence"] < 0 < friend["applied_delta"]["valence"]
    assert "hostile" in stranger["interpretation"] and "playful" in friend["interpretation"]


# --- §17: background scheduler revisits a question when triggered, within budget ---
def test_scheduler_triggers_and_budget(tmp_path):
    a = make(tmp_path)
    assert a.tick(at=T0)["ran"] is False                                  # nothing to do
    burn(a, 4)
    a.reflector.hypothesize("test")                                       # creates an open question (curiosity .7)
    out = a.tick(at=T0 + timedelta(days=5))
    assert out["ran"] and any(r.startswith("curiosity") for r in out["reasons"])
    a.scheduler.max_per_day = 1
    out2 = a.tick(at=T0 + timedelta(days=5, hours=1))
    assert out2["ran"] is False                                           # budget exhausted for the day


# --- §17: consolidation makes a bounded, versioned update; rollback works ---
def test_consolidation_versioned_update_and_rollback(tmp_path):
    a = make(tmp_path)
    burn(a, 5)
    report = a.consolidate()
    assert report["patterns"] and report["changes"]
    c = a.store.claims()[0]
    assert c.version >= 2 and len(a.store.claim_versions(c.id)) == c.version
    rolled = a.store.rollback_claim(c.id, 1)
    assert rolled.confidence == 0.4 and rolled.version == c.version + 1   # rollback is itself audited
    assert any(e["kind"] == "claim_upsert" and "rollback" in e.get("reason", "") for e in a.store.audit_tail(5))


# --- §5.1 / §12: identity kernel is hash-verified and never model-written ---
def test_identity_kernel_integrity(tmp_path):
    a = make(tmp_path)
    assert a.identity.verify_unchanged() and a.self_state()["identity_intact"]
    assert len(a.identity.anchors()) >= 3
    assert "self-preservation" in a.identity.summary()


# --- §19: ablations are honoured ---
def test_ablations(tmp_path):
    a = make(tmp_path, enable_affect=False, enable_reflection=False, enable_episodic_retrieval=False)
    r = a.interact("you are useless", person="stranger", at=T0)
    assert r["applied_delta"] == {} and r["memories_used"] == []
    assert a.reflect() == []


# --- clone: two homes from one snapshot never share state ---
def test_clone_isolation(tmp_path):
    a = make(tmp_path)
    burn(a, 3)
    a.snapshot("baseline")
    a.store.clone_to(tmp_path / "b" / "functional_self.sqlite")
    b = FunctionalSelfAgent(Config.load(tmp_path / "b"), MockProvider())
    assert b.store.episode_count() == 3 and "baseline" in b.store.snapshots()
    b.interact("Thanks, great work", at=T0 + timedelta(days=5))
    assert a.store.episode_count() == 3 and b.store.episode_count() == 4
