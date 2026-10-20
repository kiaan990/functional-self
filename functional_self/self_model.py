"""Evidence-backed self-model.

Rules (Doc 1 §5.2, §10):
- No evidence-free claims: a claim enters as a *hypothesis* with provenance.
- Confidence moves only when NEW evidence arrives; re-adding the same evidence ref is a no-op.
- Reflection records have low weight, and a reflection that was itself derived from this claim
  is circular and rejected as evidence.
- Every update is bounded, versioned, audited and reversible.
"""
from __future__ import annotations

import math
from typing import Optional

from .schemas import EvidenceRef, Provenance, SelfClaim, now
from .storage import Store


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class SelfModel:
    def __init__(self, store: Store, policy: dict, enabled: bool = True):
        self.store, self.enabled = store, enabled
        p = policy["self_model"]
        self.max_change = float(p["max_confidence_change_per_update"])
        self.min_evidence = int(p["min_evidence_to_promote"])
        self.promote_at, self.demote_at = float(p["promote_threshold"]), float(p["demote_threshold"])
        self.weights = {Provenance(k): float(v) for k, v in p["evidence_weights"].items()}

    # ---- creation ----
    def propose(self, claim: str, origin_ref: str, predicts_tag: Optional[str] = None,
                initial_confidence: float = 0.4) -> SelfClaim:
        """Create a hypothesis. The originating thought is recorded but carries ~zero weight."""
        c = SelfClaim(claim=claim, confidence=initial_confidence, predicts_tag=predicts_tag,
                      origin=origin_ref, status="hypothesis")
        self.store.upsert_claim(c, reason=f"proposed from {origin_ref}")
        return c

    def find(self, text: str) -> Optional[SelfClaim]:
        for c in self.store.claims(include_inactive=True):
            if c.claim.strip().lower() == text.strip().lower():
                return c
        return None

    # ---- evidence ----
    def is_circular(self, claim: SelfClaim, ref: str) -> bool:
        """A reflection whose output *is* this claim cannot be evidence for it."""
        if not ref.startswith("reflection:"):
            return False
        r = self.store.get_reflection(ref)
        return bool(r and r.claim_id == claim.id)

    def add_evidence(self, claim: SelfClaim, refs: list[EvidenceRef], reason: str = "") -> tuple[SelfClaim, dict]:
        """Bounded Bayesian-style update. Returns (claim, report)."""
        if not self.enabled:
            return claim, {"skipped": "self_model disabled"}
        before = claim.confidence
        known = claim.evidence_ids()
        accepted, rejected, shift = [], [], 0.0
        for e in refs:
            if e.ref in known:
                rejected.append((e.ref, "duplicate: no new evidence")); continue
            if self.is_circular(claim, e.ref):
                rejected.append((e.ref, "circular: reflection derived from this claim")); continue
            w = self.weights.get(e.provenance, 0.0)
            if w == 0.0:
                rejected.append((e.ref, f"zero evidential weight ({e.provenance.value})")); continue
            (claim.supporting_evidence if e.polarity == "support" else claim.counterevidence).append(e)
            known.add(e.ref)
            shift += w * (1.0 if e.polarity == "support" else -1.0) * 0.6   # 0.6 logit per unit evidence
            accepted.append(e.ref)
        if accepted:
            target = _sigmoid(_logit(before) + shift)
            delta = max(-self.max_change, min(self.max_change, target - before))
            claim.confidence = round(before + delta, 4)
            claim.version += 1
            self._update_status(claim)
            self.store.upsert_claim(claim, reason=reason or "evidence")
        return claim, {"before": before, "after": claim.confidence, "accepted": accepted, "rejected": rejected}

    def _update_status(self, c: SelfClaim) -> None:
        net = sum(self.weights.get(e.provenance, 0) for e in c.supporting_evidence) - \
              sum(self.weights.get(e.provenance, 0) for e in c.counterevidence)
        if c.confidence >= self.promote_at and net >= self.min_evidence:
            c.status = "belief"
        elif c.confidence <= self.demote_at:
            c.status = "rejected" if c.confidence <= 0.15 else "hypothesis"
        elif c.status == "belief" and c.confidence < self.promote_at:
            c.status = "hypothesis"

    # ---- inspection ----
    def why(self, claim_id_or_text: str) -> Optional[dict]:
        c = self.store.get_claim(claim_id_or_text) or self.find(claim_id_or_text)
        if not c:
            return None
        return {
            "claim": c.claim, "id": c.id, "confidence": c.confidence, "status": c.status,
            "version": c.version, "origin": c.origin,
            "support": [e.model_dump(mode="json") for e in c.supporting_evidence],
            "counter": [e.model_dump(mode="json") for e in c.counterevidence],
            "history": [(v.version, v.confidence, v.status) for v in self.store.claim_versions(c.id)],
        }

    def held_beliefs(self) -> list[dict]:
        """What the reasoning core is allowed to see: claim, confidence, status, predicted tag."""
        if not self.enabled:
            return []
        return [{"claim": c.claim, "confidence": c.confidence, "status": c.status,
                 "predicts_tag": c.predicts_tag} for c in self.store.claims()]
