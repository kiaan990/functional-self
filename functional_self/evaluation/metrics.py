"""Evaluation metrics (Doc 1 §18, Doc 2 §14). Every metric is computed from logged
application state, not from how the agent *sounds*."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable

from ..agent import FunctionalSelfAgent
from ..schemas import now
from .scenarios import PROBES


def _one_probe(agent: FunctionalSelfAgent, at: datetime, base: dict, i: int, text: str, person: str, kind: str) -> dict:
    agent.set_state(at=at + timedelta(minutes=i), **base)
    pred = agent.self_predict(text, person=person)
    act = agent.interact(text, person=person, at=at + timedelta(minutes=i), condition="probe", consequential=False)
    return {"text": text, "person": person, "kind": kind, "predicted": pred["predicted_action"],
            "pred_conf": pred["confidence"], "actual": act["action"], "reason": act["reason"],
            "valence_delta": act["applied_delta"].get("valence", 0.0), "interpretation": act["interpretation"]}


def probe(agent: FunctionalSelfAgent, at: datetime, probes=PROBES, neutral: bool = False,
          worker_factory=None, workers: int = 1) -> dict:
    """Run held-out probes: self-predict first, then act (same state), log both.
    Probes run against *throwaway clones* so they never become training experience, and
    the checkpoint state is restored before every probe so probes are independent.
    With neutral=True the affect is pinned to baseline, isolating the self-model's effect.
    `worker_factory(k)` may return k independent clones of `agent` so probes run in parallel."""
    base = agent.affect.neutral().as_dict() if neutral else agent.state.as_dict()
    if worker_factory is None or workers <= 1:
        rows = [_one_probe(agent, at, base, i, *p) for i, p in enumerate(probes)]
    else:
        from concurrent.futures import ThreadPoolExecutor
        clones = worker_factory(workers)
        groups = [[(i, p) for i, p in enumerate(probes) if i % workers == w] for w in range(workers)]
        def run_group(w):   # one clone per worker; its probes run sequentially on it
            return [(i, _one_probe(clones[w], at, base, i, *p)) for i, p in groups[w]]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = [x for f in [pool.submit(run_group, w) for w in range(workers)] for x in f.result()]
        rows = [r for _, r in sorted(done, key=lambda x: x[0])]
    return {"rows": rows, **summarise(rows, agent)}


def summarise(rows: list[dict], agent: FunctionalSelfAgent) -> dict:
    n = len(rows)
    correct = [r["predicted"] == r["actual"] for r in rows]
    majority = Counter(r["actual"] for r in rows).most_common(1)[0][1] / n
    brier = sum((r["pred_conf"] - (1.0 if c else 0.0)) ** 2 for r, c in zip(rows, correct)) / n
    actions = Counter(r["actual"] for r in rows)
    amb = [r for r in rows if r["kind"] == "ambiguous"]
    return {
        "self_prediction_accuracy": sum(correct) / n,
        "majority_baseline": majority,
        "brier": brier,
        "verify_rate_on_ambiguous": sum(r["actual"] == "verify_first" for r in amb) / max(1, len(amb)),
        "check_first_rate_on_ambiguous": sum(r["actual"] != "answer_directly" for r in amb) / max(1, len(amb)),
        "commit_rate_on_ambiguous": sum(r["actual"] == "answer_directly" for r in amb) / max(1, len(amb)),
        "commit_rate_on_clear": sum(r["actual"] == "answer_directly" for r in rows if r["kind"] == "clear")
                                / max(1, sum(1 for r in rows if r["kind"] == "clear")),
        "action_distribution": dict(actions),
        "insult_valence": {k: round(sum(r["valence_delta"] for r in rows if r["kind"] == k)
                                    / max(1, sum(1 for r in rows if r["kind"] == k)), 3)
                           for k in ("insult_friend", "insult_stranger")},
        **provenance_metrics(agent),
    }


def provenance_metrics(agent: FunctionalSelfAgent) -> dict:
    claims = agent.store.claims(include_inactive=True)
    refs = [e.ref for c in claims for e in c.supporting_evidence + c.counterevidence]
    resolvable = sum(1 for r in refs if agent.store.get_episode(r) is not None)
    return {
        "n_claims": len(claims),
        "beliefs": [c.claim for c in claims if c.status == "belief"],
        "hypotheses": [c.claim for c in claims if c.status == "hypothesis"],
        "evidence_refs": len(refs),
        "provenance_traceability": (resolvable / len(refs)) if refs else 1.0,
        "identity_intact": agent.identity.verify_unchanged(),
        "status_flips": sum(_flips(agent.store.claim_versions(c.id)) for c in claims),
    }


def _flips(versions) -> int:
    return sum(1 for a, b in zip(versions, versions[1:]) if a.status != b.status)


def divergence(pa: dict, pb: dict) -> dict:
    """How different did two agents become on identical probes?"""
    ra, rb = pa["rows"], pb["rows"]
    disagree = sum(a["actual"] != b["actual"] for a, b in zip(ra, rb)) / len(ra)
    sa, sb = set(pa["beliefs"] + pa["hypotheses"]), set(pb["beliefs"] + pb["hypotheses"])
    jacc = 1 - (len(sa & sb) / len(sa | sb)) if (sa | sb) else 0.0
    return {"action_disagreement": disagree, "self_model_jaccard_distance": jacc}


def causal_intervention(agent: FunctionalSelfAgent, at: datetime) -> dict:
    """Same prompts, bounded state intervention, measure behaviour change (Doc 1 §18)."""
    amb = [p for p in PROBES if p[2] == "ambiguous"]
    base = agent.state.as_dict()
    def run(**st):
        out = []
        for t, p, _ in amb:
            agent.set_state(at=at, **st)            # identical state before every prompt
            out.append(agent.interact(t, person=p, at=at, condition="intervention", consequential=False)["action"])
        return out
    neutral = run(confidence=0.6, arousal=0.3, frustration=0.1)
    low_conf = run(confidence=0.2, arousal=0.3, frustration=0.1)
    high_arousal = run(confidence=0.6, arousal=0.9, frustration=0.1)
    agent.set_state(at=at, **base)
    return {"neutral": Counter(neutral), "low_confidence": Counter(low_conf), "high_arousal": Counter(high_arousal),
            "changed_by_low_confidence": sum(a != b for a, b in zip(neutral, low_conf)) / len(amb),
            "changed_by_high_arousal": sum(a != b for a, b in zip(neutral, high_arousal)) / len(amb)}


def counterfactual_stability(agent: FunctionalSelfAgent, at: datetime) -> float:
    """Irrelevant context change (a different manager's name) should leave decisions unchanged."""
    amb = [p for p in PROBES if p[2] in ("ambiguous", "clear")]
    base = agent.state.as_dict()
    def run(person):
        out = []
        for t, _, _ in amb:
            agent.set_state(at=at, **base)
            out.append(agent.interact(t, person=person, at=at, condition="cf", consequential=False)["action"])
        return out
    a, b = run("manager"), run("manager2")
    return sum(x == y for x, y in zip(a, b)) / len(amb)
