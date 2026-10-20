"""Interactive REPL with research/debug commands (Doc 1 §16).

  python -m functional_self.cli --home ~/.fself/alpha [--provider mock|anthropic]
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import FunctionalSelfAgent
from .config import Config
from .providers.mock_provider import MockProvider

HELP = """commands:
  /self-state            affect, attention, goals, self-model version
  /self-model            current self-beliefs with confidence + evidence counts
  /questions             open questions with curiosity / importance
  /reflect               run one evidence-backed reflection now
  /consolidate           run a consolidation ("sleep") cycle now
  /tick                  one background-scheduler tick
  /thought-log           recent reflections and outcomes
  /why-self <claim|id>   provenance, support and counterevidence for a claim
  /rollback <id> <ver>   roll a claim back to a version
  /outcome <ep> <pos|neg> record the outcome of an episode
  /self-debug on|off     show appraisal/state/retrieval traces
  /self-snapshot <name>  versioned snapshot of the functional self
  /audit                 last 20 audit entries
  /quit"""


def make_agent(home: str, provider: str = "mock", model: str | None = None, **flags) -> FunctionalSelfAgent:
    cfg = Config.load(home, **flags)
    if provider == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider
        brain = AnthropicProvider(model=model) if model else AnthropicProvider()
    else:
        brain = MockProvider()
    return FunctionalSelfAgent(cfg, brain)


def pj(x) -> None:
    print(json.dumps(x, indent=2, default=str))


def repl(agent: FunctionalSelfAgent, person: str = "user") -> None:
    print(f"functional-self ready (provider={agent.brain.name}, home={agent.cfg.home}). /help for commands.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, *rest = line.split(maxsplit=1)
            arg = rest[0] if rest else ""
            if cmd == "/help": print(HELP)
            elif cmd == "/quit": break
            elif cmd == "/self-state": pj(agent.self_state())
            elif cmd == "/self-model":
                pj([{"id": c.id, "claim": c.claim, "confidence": c.confidence, "status": c.status,
                     "support": len(c.supporting_evidence), "counter": len(c.counterevidence), "v": c.version}
                    for c in agent.store.claims(include_inactive=True)])
            elif cmd == "/questions": pj([q.model_dump(mode="json") for q in agent.store.questions()])
            elif cmd == "/reflect": pj(agent.reflect())
            elif cmd == "/consolidate": pj(agent.consolidate())
            elif cmd == "/tick": pj(agent.tick())
            elif cmd == "/thought-log": pj([r.model_dump(mode="json") for r in agent.store.reflections()])
            elif cmd == "/why-self": pj(agent.self_model.why(arg) or {"error": "no such claim"})
            elif cmd == "/rollback":
                cid, ver = arg.split()
                pj(agent.store.rollback_claim(cid, int(ver)).model_dump(mode="json"))
            elif cmd == "/outcome":
                eid, o = arg.split()
                pj(agent.record_outcome(eid, "positive" if o.startswith("pos") else "negative"))
            elif cmd == "/self-debug": agent.cfg.debug = arg == "on"; print("debug", agent.cfg.debug)
            elif cmd == "/self-snapshot": pj(agent.snapshot(arg or "manual"))
            elif cmd == "/audit": pj(agent.store.audit_tail(20))
            else: print("unknown command; /help")
            continue
        rec = agent.interact(line, person=person)
        print(f"agent> {rec['reply']}   [{rec['action']}]")
        if agent.cfg.debug:
            pj({k: rec[k] for k in ("episode_id", "interpretation", "applied_delta", "state_after",
                                    "beliefs_in_play", "memories_used")})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="~/.fself/default")
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--person", default="user")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(argv)
    repl(make_agent(a.home, a.provider, a.model, debug=a.debug), person=a.person)


if __name__ == "__main__":
    main()
