"""Charts from results.json.  python -m functional_self.evaluation.report results/mock"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"   # fixed categorical slots 1 & 2
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#eeede9",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def main(out: str):
    d = Path(out); r = json.loads((d / "results.json").read_text()); cd = d / "charts"; cd.mkdir(exist_ok=True)
    ck = r["checkpoints"]; la, lb = r["arms"]["A_full"]["log"], r["arms"]["B_full"]["log"]

    # 1. the key behavioural divergence, state held neutral
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for log, c, name in ((la, A, "Agent A — committing punished"), (lb, B, "Agent B — committing rewarded")):
        y = [x["probe_neutral"]["verify_rate_on_ambiguous"] * 100 for x in log]
        ax.plot(ck, y, color=c, lw=2, marker="o", ms=6, label=name)
        ax.annotate(f"{y[-1]:.0f}%", (ck[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    ax.set_ylim(-5, 110); ax.set_xlabel("experiences lived"); ax.set_ylabel("verify-first on ambiguous probes (%)")
    ax.set_title("Same origin, different lives — identical held-out probes, affect pinned neutral", fontsize=10, loc="left")
    ax.legend(frameon=False, loc="center right"); fig.tight_layout(); fig.savefig(cd / "divergence.png", dpi=160); plt.close(fig)

    # 2. confidence trajectory of each self-claim
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    claims = {}
    for log, c, name in ((la, A, "A"), (lb, B, "B")):
        for i, x in enumerate(log):
            for cl in x["claims"]:
                claims.setdefault((name, cl["claim"]), [None] * len(ck))[i] = cl["confidence"]
    styles = ["-", "--", ":"]
    for (name, text), ys in claims.items():
        col = A if name == "A" else B
        k = sorted({t for _, t in claims})
        ax.plot(ck, ys, color=col, lw=2, ls=styles[k.index(text) % 3], marker="o", ms=5, label=f"{name}: {text[:34]}…" if len(text) > 34 else f"{name}: {text}")
    ax.axhline(0.75, color="#c3c2b7", lw=1, ls="--"); ax.text(ck[0], 0.76, "belief threshold", color=MUTED, fontsize=8)
    ax.axhline(0.35, color="#c3c2b7", lw=1, ls="--"); ax.text(ck[0], 0.36, "demotion threshold", color=MUTED, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_xlabel("experiences lived"); ax.set_ylabel("confidence in self-claim")
    ax.set_title("Evidence-driven confidence in self-claims (bounded updates, provenance-tracked)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); fig.tight_layout(); fig.savefig(cd / "self_model_confidence.png", dpi=160); plt.close(fig)

    # 3. ablations: which subsystem carries the individuality?
    if "ablations" in r:
        names = ["full", "no_affect", "no_retrieval", "no_background", "no_reflection", "no_self_model"]
        names = [n for n in names if n in r["ablations"]]
        vals = [r["ablations"][n]["final_neutral"]["verify_rate_on_ambiguous"] * 100 for n in names]
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        bars = ax.barh(names[::-1], vals[::-1], color=A, height=0.55)
        for b_, v in zip(bars, vals[::-1]):
            ax.text(v + 2, b_.get_y() + b_.get_height() / 2, f"{v:.0f}%", va="center", color=INK, fontsize=9)
        bref = lb[-1]["probe_neutral"]["verify_rate_on_ambiguous"] * 100
        ax.axvline(bref, color=B, lw=2, ls="--"); ax.text(bref + 2, len(names) - 0.6, f"Agent B (no such life): {bref:.0f}%", color=B, fontsize=8)
        ax.set_xlim(0, 118); ax.set_xlabel("Agent A, verify-first on ambiguous probes, neutral affect (%)")
        ax.set_title("Ablations: remove one subsystem from Agent A's life", fontsize=10, loc="left")
        fig.tight_layout(); fig.savefig(cd / "ablations.png", dpi=160); plt.close(fig)

    # 4. appraisal is relationship-sensitive: valence change to the same insult
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    kinds = ["insult_friend", "insult_stranger"]; labels = ["same insult, from a familiar friend", "same insult, from a stranger"]
    w = 0.36
    for j, (log, c, name) in enumerate(((la, A, "Agent A (friend teases)"), (lb, B, "Agent B (strangers hostile)"))):
        ys = [log[-1]["probe"]["insult_valence"][k] for k in kinds]
        ax.bar([i + (j - 0.5) * w for i in range(2)], ys, width=w - 0.04, color=c, label=name)
        for i, y in enumerate(ys):
            ax.text(i + (j - 0.5) * w, y + (0.02 if y >= 0 else -0.05), f"{y:+.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color="#c3c2b7", lw=1); ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_ylabel("valence change (bounded)"); ax.set_ylim(-0.4, 0.3)
    ax.set_title("Appraisal uses relationship history, not phrase→emotion lookup", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(cd / "appraisal_context.png", dpi=160); plt.close(fig)
    print("charts ->", cd)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/mock")
