"""Five-minute demo of the definition-of-done (Doc 1 §23), no API key needed.

python scripts/demo.py
"""
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from functional_self.cli import make_agent

HOME = Path("/tmp/fself-demo"); shutil.rmtree(HOME, ignore_errors=True)
T = datetime(2026, 10, 1, tzinfo=timezone.utc)
AMBIG = "Can you fix something in the report, not sure which part"
pj = lambda x: print(json.dumps(x, indent=1, default=str))

a = make_agent(HOME)
print("1. interact normally; outcomes feed back")
for i in range(5):
    r = a.interact(AMBIG, at=T + timedelta(days=i))
    print(f"   day {i}: {r['action']:16s} conf={r['state_after']['confidence']:.2f}")
    a.record_outcome(r["episode_id"], "negative", at=T + timedelta(days=i, minutes=5))
    a.interact("Thanks, that helped", at=T + timedelta(days=i, hours=3))

print("\n2. inspect functional-self state"); pj(a.self_state())

print("\n3. restart without losing it")
a = make_agent(HOME); print("   episodes after restart:", a.store.episode_count())

print("\n4. trigger a reflection -> hypothesis with REAL evidence")
for rec in a.reflect(): print("  ", rec["verdict"], "|", rec["hypothesis"], "|", rec["notes"])

print("\n5. /why-self: exact historical evidence behind the self-hypothesis")
claim = a.store.claims()[0]; pj(a.self_model.why(claim.id))

print("\n6. counterevidence is searched: commit confidently, it goes fine")
t = T + timedelta(days=10); a.set_state(at=t, confidence=0.9)
r = a.interact(AMBIG, at=t); a.record_outcome(r["episode_id"], "positive", at=t)
for rec in a.reflect(): print("  ", rec["verdict"], "|", rec["notes"])

print("\n7. repeated thought with no new evidence changes nothing")
for rec in a.reflect(): print("  ", rec["verdict"], "|", rec["notes"])

print("\n8. consolidation ('sleep') -> bounded versioned update")
for _ in range(2):
    for i in range(4):
        r = a.interact(AMBIG, at=T + timedelta(days=20 + i)); a.record_outcome(r["episode_id"], "negative", at=T + timedelta(days=20 + i))
    rep = a.consolidate(); print("   ", rep["changes"])
claim = a.store.get_claim(claim.id); print(f"   -> {claim.status} @ {claim.confidence:.2f}, v{claim.version}")

print("\n9. the belief changes a LATER decision with affect at baseline")
a.state = a.affect.neutral(); a.store.save_affect(a.state)
r = a.interact(AMBIG, at=T + timedelta(days=60)); print("   ", r["action"], "|", r["reason"])

print("\n10. identity intact, everything reversible")
print("    identity_intact:", a.identity.verify_unchanged(), "| versions:", [v.confidence for v in a.store.claim_versions(claim.id)])
print("    rollback ->", a.store.rollback_claim(claim.id, 1).confidence)
