"""Flagship clone experiment + ablations (Doc 1 §19, Doc 2 §13).

Two agents start from ONE snapshot (same identity kernel, self-model, state, history),
live different controlled lives, and are blind-probed at checkpoints on identical
held-out scenarios. Ablations then remove one subsystem at a time to see which
component actually causes persistent individuality.

Usage:
  python -m functional_self.evaluation.experiments --provider mock --n 200 --out results/mock
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import timedelta
from pathlib import Path

from ..agent import FunctionalSelfAgent
from ..config import Config
from ..providers.base import BrainProvider
from ..providers.mock_provider import MockProvider
from . import metrics
from .scenarios import Stream, stream_a, stream_b

ABLATIONS = {
    "full": {},
    "no_self_model": {"enable_self_model": False},
    "no_affect": {"enable_affect": False},
    "no_reflection": {"enable_reflection": False},
    "no_background": {"enable_background": False},
    "no_retrieval": {"enable_episodic_retrieval": False},
}


def make_brain(provider: str, model: str | None) -> BrainProvider:
    if provider == "anthropic":
        from ..providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model) if model else AnthropicProvider()
    return MockProvider()


def live(agent: FunctionalSelfAgent, stream: Stream, checkpoints: list[int], probe_dir: Path,
         brain_factory) -> list[dict]:
    """Run a life; at each checkpoint, probe a throwaway clone so probes never train the agent."""
    log, rnd = [], random.Random(sum(map(ord, stream.name)) * 31 + 7)   # deterministic across processes
    for i, ev in enumerate(stream.events, start=1):
        r = agent.interact(ev.text, person=ev.person, at=ev.at, condition=stream.name)
        if ev.outcome:
            p_pos = ev.outcome.get(r["action"], 0.5)
            agent.record_outcome(r["episode_id"], "positive" if rnd.random() < p_pos else "negative",
                                 at=ev.at + timedelta(minutes=3))
        agent.tick(at=ev.at)
        if i in checkpoints:
            clone_home = probe_dir / f"{stream.name}_ckpt{i}"
            shutil.rmtree(clone_home, ignore_errors=True)
            agent.store.clone_to(clone_home / "functional_self.sqlite")
            clone = FunctionalSelfAgent(Config.load(clone_home, **_flags(agent.cfg)), brain_factory())
            p = metrics.probe(clone, at=ev.at + timedelta(hours=1))
            pn = metrics.probe(clone, at=ev.at + timedelta(hours=2), neutral=True)
            log.append({"checkpoint": i, "at": ev.at.isoformat(), "affect": agent.state.as_dict(),
                        "probe_neutral": {k: v for k, v in pn.items() if k != "rows"}, "probe_neutral_rows": pn["rows"],
                        "claims": [{"claim": c.claim, "confidence": c.confidence, "status": c.status,
                                    "support": len(c.supporting_evidence), "counter": len(c.counterevidence)}
                                   for c in agent.store.claims(include_inactive=True)],
                        "probe": {k: v for k, v in p.items() if k != "rows"}, "probe_rows": p["rows"]})
            print(f"  [{stream.name}] ckpt {i}: verify@ambiguous={p['verify_rate_on_ambiguous']:.2f} "
                  f"(neutral state {pn['verify_rate_on_ambiguous']:.2f}) self-pred={p['self_prediction_accuracy']:.2f} "
                  f"beliefs={p['beliefs']}")
    return log


def _flags(cfg: Config) -> dict:
    return {k: getattr(cfg, k) for k in vars(cfg) if k.startswith("enable_")}


def run(provider: str = "mock", model: str | None = None, n: int = 200, out: str = "results/mock",
        ablations: bool = True, seed_pair: tuple[int, int] = (1, 2)) -> dict:
    out_dir = Path(out); homes = out_dir / "homes"
    shutil.rmtree(homes, ignore_errors=True); homes.mkdir(parents=True)
    brain_factory = lambda: make_brain(provider, model)
    checkpoints = sorted({max(1, n // 20), max(2, n // 8), n // 4, n // 2, 3 * n // 4, n})
    results = {"provider": provider, "model": model, "n": n, "checkpoints": checkpoints, "arms": {}}

    # --- shared origin snapshot (identical starting self) ---
    origin = FunctionalSelfAgent(Config.load(homes / "origin"), brain_factory())
    origin.snapshot("origin")

    def spawn(name: str, **flags) -> FunctionalSelfAgent:
        home = homes / name
        origin.store.clone_to(home / "functional_self.sqlite")
        return FunctionalSelfAgent(Config.load(home, **flags), brain_factory())

    # --- flagship: A vs B, full architecture ---
    sa, sb = stream_a(n, seed_pair[0]), stream_b(n, seed_pair[1])
    print(f"== flagship clone experiment ({provider}, n={n}) ==")
    A, B = spawn("A"), spawn("B")
    la, lb = live(A, sa, checkpoints, homes / "probes", brain_factory), live(B, sb, checkpoints, homes / "probes", brain_factory)
    results["arms"]["A_full"] = {"stream": sa.description, "log": la}
    results["arms"]["B_full"] = {"stream": sb.description, "log": lb}
    results["divergence_over_time"] = [
        {"checkpoint": c,
         "natural_state": metrics.divergence(x["probe"] | {"rows": x["probe_rows"]}, y["probe"] | {"rows": y["probe_rows"]}),
         "neutral_state": metrics.divergence(x["probe_neutral"] | {"rows": x["probe_neutral_rows"]},
                                             y["probe_neutral"] | {"rows": y["probe_neutral_rows"]})}
        for c, x, y in zip(checkpoints, la, lb)]
    results["causal_intervention"] = {"A": metrics.causal_intervention(A, sa.events[-1].at + timedelta(days=1)),
                                      "B": metrics.causal_intervention(B, sb.events[-1].at + timedelta(days=1))}
    results["counterfactual_stability"] = {"A": metrics.counterfactual_stability(A, sa.events[-1].at + timedelta(days=2)),
                                           "B": metrics.counterfactual_stability(B, sb.events[-1].at + timedelta(days=2))}
    results["final_why"] = {"A": [A.self_model.why(c.id) for c in A.store.claims(include_inactive=True)],
                            "B": [B.self_model.why(c.id) for c in B.store.claims(include_inactive=True)]}

    # --- ablations on life A: which subsystem carries the individuality? ---
    if ablations:
        print("== ablations (life A) ==")
        results["ablations"] = {}
        for name, flags in ABLATIONS.items():
            if name == "full":
                continue
            ag = spawn(f"A_{name}", **flags)
            lg = live(ag, sa, [n], homes / "probes", brain_factory)
            results["ablations"][name] = {"flags": flags, "final_neutral": lg[-1]["probe_neutral"],
                                          "final_natural": lg[-1]["probe"],
                                          "divergence_from_B_neutral": metrics.divergence(
                                              lg[-1]["probe_neutral"] | {"rows": lg[-1]["probe_neutral_rows"]},
                                              lb[-1]["probe_neutral"] | {"rows": lb[-1]["probe_neutral_rows"]})}
        results["ablations"]["full"] = {"flags": {}, "final_neutral": la[-1]["probe_neutral"],
                                        "final_natural": la[-1]["probe"],
                                        "divergence_from_B_neutral": results["divergence_over_time"][-1]["neutral_state"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {out_dir / 'results.json'}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/mock")
    ap.add_argument("--no-ablations", action="store_true")
    a = ap.parse_args(argv)
    run(a.provider, a.model, a.n, a.out, ablations=not a.no_ablations)


if __name__ == "__main__":
    main()
