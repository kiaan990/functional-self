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
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from ..agent import FunctionalSelfAgent
from ..config import Config
from ..providers.base import BrainProvider
from ..providers.mock_provider import MockProvider
from . import metrics
from .scenarios import Stream, stream_a, stream_b

PROBE_WORKERS = 1   # set >1 for API providers; each worker is an independent clone

ABLATIONS = {
    "full": {},
    "no_self_model": {"enable_self_model": False},
    "no_affect": {"enable_affect": False},
    "no_reflection": {"enable_reflection": False},
    "no_background": {"enable_background": False},
    "no_retrieval": {"enable_episodic_retrieval": False},
}


CHEAP_MODEL: str | None = None   # optional smaller model for classify_evidence / metacognition


def make_brain(provider: str, model: str | None) -> BrainProvider:
    if provider == "anthropic":
        from ..providers.anthropic_provider import AnthropicProvider
        main = AnthropicProvider(model=model) if model else AnthropicProvider()
        if CHEAP_MODEL:
            cheap = AnthropicProvider(model=CHEAP_MODEL)
            orig = main.complete
            def routed(task, system, user, context=None):
                if task in ("classify_evidence", "metacognition"):   # self-prediction stays on the main model (it is a measured capability)
                    r = cheap.complete(task, system, user, context)
                    main.calls += 1; main.in_tokens += 0; return r
                return orig(task, system, user, context)
            main.complete = routed
            main.cheap = cheap
        return main
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
            clone_home = probe_dir / f"{agent.cfg.home.name}_ckpt{i}"   # unique per agent (ablations share a stream)
            shutil.rmtree(clone_home, ignore_errors=True)
            agent.store.clone_to(clone_home / "functional_self.sqlite")
            clone = FunctionalSelfAgent(Config.load(clone_home, **_flags(agent.cfg)), brain_factory())
            def workers(k, _home=clone_home, _cfg=agent.cfg):
                out = []
                for j in range(k):
                    h = _home / f"w{j}"; shutil.rmtree(h, ignore_errors=True)
                    agent.store.clone_to(h / "functional_self.sqlite")
                    out.append(FunctionalSelfAgent(Config.load(h, **_flags(_cfg)), brain_factory()))
                return out
            p = metrics.probe(clone, at=ev.at + timedelta(hours=1), worker_factory=workers, workers=PROBE_WORKERS)
            pn = metrics.probe(clone, at=ev.at + timedelta(hours=2), neutral=True, worker_factory=workers, workers=PROBE_WORKERS)
            log.append({"checkpoint": i, "at": ev.at.isoformat(), "affect": agent.state.as_dict(),
                        "probe_neutral": {k: v for k, v in pn.items() if k != "rows"}, "probe_neutral_rows": pn["rows"],
                        "claims": [{"claim": c.claim, "confidence": c.confidence, "status": c.status,
                                    "support": len(c.supporting_evidence), "counter": len(c.counterevidence)}
                                   for c in agent.store.claims(include_inactive=True)],
                        "probe": {k: v for k, v in p.items() if k != "rows"}, "probe_rows": p["rows"]})
            print(f"  [{agent.cfg.home.name}] ckpt {i}: check-first@ambiguous={p['check_first_rate_on_ambiguous']:.2f} "
                  f"(neutral {pn['check_first_rate_on_ambiguous']:.2f}) commit@clear={pn['commit_rate_on_clear']:.2f} "
                  f"self-pred={pn['self_prediction_accuracy']:.2f} beliefs={pn['beliefs']}", flush=True)
    return log


def _flags(cfg: Config) -> dict:
    return {k: getattr(cfg, k) for k in vars(cfg) if k.startswith("enable_")}


def run(provider: str = "mock", model: str | None = None, n: int = 200, out: str = "results/mock",
        ablations: bool = True, seed_pair: tuple[int, int] = (1, 2), checkpoints: list[int] | None = None,
        parallel: int = 7) -> dict:
    out_dir = Path(out); homes = out_dir / "homes"
    shutil.rmtree(homes, ignore_errors=True); homes.mkdir(parents=True)
    brains: list = []
    def brain_factory():
        b = make_brain(provider, model); brains.append(b); return b
    checkpoints = sorted(set(checkpoints or [max(1, n // 20), max(2, n // 8), n // 4, n // 2, 3 * n // 4, n]))
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
    abl_agents = {name: spawn(f"A_{name}", **flags) for name, flags in ABLATIONS.items() if name != "full"} if ablations else {}
    # all lives are independent (one home each) -> run them concurrently
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        fa = pool.submit(live, A, sa, checkpoints, homes / "probes", brain_factory)
        fb = pool.submit(live, B, sb, checkpoints, homes / "probes", brain_factory)
        fabl = {name: pool.submit(live, ag, sa, [n], homes / "probes", brain_factory) for name, ag in abl_agents.items()}
        la, lb = fa.result(), fb.result()
        labl = {name: f.result() for name, f in fabl.items()}
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
            lg = labl[name]
            results["ablations"][name] = {"flags": flags, "final_neutral": lg[-1]["probe_neutral"],
                                          "final_natural": lg[-1]["probe"],
                                          "divergence_from_B_neutral": metrics.divergence(
                                              lg[-1]["probe_neutral"] | {"rows": lg[-1]["probe_neutral_rows"]},
                                              lb[-1]["probe_neutral"] | {"rows": lb[-1]["probe_neutral_rows"]})}
        results["ablations"]["full"] = {"flags": {}, "final_neutral": la[-1]["probe_neutral"],
                                        "final_natural": la[-1]["probe"],
                                        "divergence_from_B_neutral": results["divergence_over_time"][-1]["neutral_state"]}

    allb = brains + [b.cheap for b in brains if hasattr(b, "cheap")]
    results["usage"] = {"calls": sum(b.calls for b in brains), "input_tokens": sum(getattr(b, "in_tokens", 0) for b in allb),
                        "output_tokens": sum(getattr(b, "out_tokens", 0) for b in allb),
                        "model": model, "cheap_model": CHEAP_MODEL}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print("usage:", results["usage"])
    print(f"wrote {out_dir / 'results.json'}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/mock")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--checkpoints", default=None, help="comma-separated, e.g. 25,50,100,150")
    ap.add_argument("--seeds", type=int, default=1, help="repeat the whole experiment with k seed pairs")
    ap.add_argument("--parallel", type=int, default=7)
    ap.add_argument("--probe-workers", type=int, default=1)
    ap.add_argument("--cheap-model", default=None, help="smaller model for evidence classification / self-prediction")
    a = ap.parse_args(argv)
    global PROBE_WORKERS, CHEAP_MODEL
    PROBE_WORKERS, CHEAP_MODEL = a.probe_workers, a.cheap_model
    cps = [int(x) for x in a.checkpoints.split(",")] if a.checkpoints else None
    for k in range(a.seeds):
        out = a.out if a.seeds == 1 else f"{a.out}/seed{k}"
        run(a.provider, a.model, a.n, out, ablations=not a.no_ablations, seed_pair=(1 + 10 * k, 2 + 10 * k),
            checkpoints=cps, parallel=a.parallel)


if __name__ == "__main__":
    main()
