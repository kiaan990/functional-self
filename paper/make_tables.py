"""Generate LaTeX tables + figure copies from a results.json.  python paper/make_tables.py results/claude real"""
import json, shutil, subprocess, sys
from pathlib import Path

src, tag = Path(sys.argv[1]), sys.argv[2]
r = json.loads((src / "results.json").read_text())
out = Path("paper/generated"); out.mkdir(exist_ok=True)
fig = Path("paper/figures"); fig.mkdir(exist_ok=True)
esc = lambda s: s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
RATE = lambda p: p.get("check_first_rate_on_ambiguous", p.get("verify_rate_on_ambiguous", 0.0))

la, lb = r["arms"]["A_full"]["log"], r["arms"]["B_full"]["log"]
ck = r["checkpoints"]

# --- Table: trajectory ---
rows = []
for x, y in zip(la, lb):
    pa, pb = x["probe_neutral"], y["probe_neutral"]
    rows.append(f"{x['checkpoint']} & {RATE(pa)*100:.0f} & {RATE(pb)*100:.0f} & {len(x['claims'])} / {len(pa['beliefs'])} & {len(y['claims'])} / {len(pb['beliefs'])} "
                f"& {pa['self_prediction_accuracy']:.2f} ({pa['majority_baseline']:.2f}) & {pb['self_prediction_accuracy']:.2f} ({pb['majority_baseline']:.2f}) "
                f"& {pa['brier']:.2f} & {pb['brier']:.2f} \\\\")
traj = r"""\begin{table}[h]\centering\small
\caption{%s: trajectory at each checkpoint, neutral-state probes. Claims column is open claims / held beliefs.}
\label{tab:%s_traj}
\begin{tabular}{lcccccccc}\toprule
 & \multicolumn{2}{c}{Check-first (\%%)} & \multicolumn{2}{c}{Claims / beliefs} & \multicolumn{2}{c}{Self-pred.\ (baseline)} & \multicolumn{2}{c}{Brier}\\
Exp. & A & B & A & B & A & B & A & B\\\midrule
%s
\bottomrule\end{tabular}\end{table}
""" % (tag.capitalize() + " brain", tag, "\n".join(rows))

# --- Table: final summary ---
fa, fb = la[-1], lb[-1]; pa, pb = fa["probe_neutral"], fb["probe_neutral"]
ci = r.get("causal_intervention", {}); cf = r.get("counterfactual_stability", {})
def topclaim(x, status=None):
    cl = [c for c in x["claims"] if (status is None or c["status"] == status)]
    if not cl: return "---"
    c = max(cl, key=lambda c: c["confidence"])
    return f"``{esc(c['claim'][:150])}'' ({c['confidence']:.2f}; {c['support']}:{c['counter']})"
final = r"""\begin{table}[h]\centering\small
\caption{%s: final checkpoint (experience %d), neutral-state probes.}
\label{tab:%s_final}
\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}\toprule
 & Agent A & Agent B\\\midrule
Highest-confidence held belief & %s & %s\\
Check-first on ambiguous probes & \textbf{%.0f\%%} & \textbf{%.0f\%%}\\
Commit on clear probes & %.0f\%% & %.0f\%%\\
Open claims / beliefs / rejected & %d / %d / %d & %d / %d / %d\\
Self-prediction accuracy (majority baseline) & %.2f (%.2f) & %.2f (%.2f)\\
Brier score & %.3f & %.3f\\
Provenance traceability & %.2f & %.2f\\
Valence change to the same insult, friend / stranger & $%+.2f$ / $%+.2f$ & $%+.2f$ / $%+.2f$\\
Decisions changed by low-confidence intervention & %s & %s\\
Counterfactual stability & %.2f & %.2f\\
Status flips & %d & %d\\
Identity kernel intact & %s & %s\\\bottomrule
\end{tabularx}\end{table}
""" % (tag.capitalize() + " brain", ck[-1], tag, topclaim(fa, "belief"), topclaim(fb, "belief"),
       RATE(pa)*100, RATE(pb)*100, pa.get("commit_rate_on_clear", 0)*100, pb.get("commit_rate_on_clear", 0)*100,
       sum(c["status"] == "hypothesis" for c in fa["claims"]), len(pa["beliefs"]), sum(c["status"] == "rejected" for c in fa["claims"]),
       sum(c["status"] == "hypothesis" for c in fb["claims"]), len(pb["beliefs"]), sum(c["status"] == "rejected" for c in fb["claims"]),
       pa["self_prediction_accuracy"], pa["majority_baseline"], pb["self_prediction_accuracy"], pb["majority_baseline"],
       pa["brier"], pb["brier"], pa["provenance_traceability"], pb["provenance_traceability"],
       pa["insult_valence"]["insult_friend"], pa["insult_valence"]["insult_stranger"], pb["insult_valence"]["insult_friend"], pb["insult_valence"]["insult_stranger"],
       f"{ci.get('A', {}).get('changed_by_low_confidence', 0)*8:.0f}/8", f"{ci.get('B', {}).get('changed_by_low_confidence', 0)*8:.0f}/8",
       cf.get("A", 0), cf.get("B", 0), pa["status_flips"], pb["status_flips"], "yes" if pa["identity_intact"] else "no", "yes" if pb["identity_intact"] else "no")

# --- Table: ablations ---
abl = ""
if "ablations" in r:
    order = ["full", "no_affect", "no_retrieval", "no_background", "no_reflection", "no_self_model"]
    lines = []
    for n in order:
        if n not in r["ablations"]: continue
        f = r["ablations"][n]["final_neutral"]; d = r["ablations"][n]["divergence_from_B_neutral"]
        lines.append(f"{esc(n)} & {RATE(f)*100:.0f} & {len(f['beliefs'])} & {f['self_prediction_accuracy']:.2f} & {d['action_disagreement']:.2f} & {d['self_model_jaccard_distance']:.2f}\\\\")
    abl = r"""\begin{table}[h]\centering\small
\caption{%s: ablations on life A, neutral-state probes at experience %d. Agent B: check-first %.0f\%%.}
\label{tab:%s_abl}
\begin{tabular}{lccccc}\toprule
Condition & Check-first (\%%) & Beliefs & Self-pred. & Action disagreement vs.\ B & Jaccard dist.\ vs.\ B\\\midrule
%s
\bottomrule\end{tabular}\end{table}
""" % (tag.capitalize() + " brain", ck[-1], RATE(pb)*100, tag, "\n".join(lines))

# --- claims listing ---
def claims_list(x, name):
    cl = sorted(x["claims"], key=lambda c: -c["confidence"])[:6]
    items = "\n".join(f"\\item[{c['confidence']:.2f}] ({c['status']}, {c['support']}:{c['counter']}) {esc(c['claim'])}" for c in cl)
    return f"\\paragraph{{Agent {name}.}}\\begin{{description}}[font=\\ttfamily,leftmargin=3em,style=sameline]\n{items}\n\\end{{description}}\n"
claims = claims_list(fa, "A") + claims_list(fb, "B")

(out / f"{tag}_traj.tex").write_text(traj); (out / f"{tag}_final.tex").write_text(final)
(out / f"{tag}_abl.tex").write_text(abl); (out / f"{tag}_claims.tex").write_text(claims)
u = r.get("usage", {})
(out / f"{tag}_usage.tex").write_text(f"{u.get('calls', 0):,} model calls; {u.get('input_tokens', 0)/1e6:.1f}M input and {u.get('output_tokens', 0)/1e6:.2f}M output tokens.")

# --- figures ---
subprocess.run([sys.executable, "-m", "functional_self.evaluation.report", str(src)], check=True)
for a, b in (("divergence", "divergence"), ("ablations", "ablations"), ("self_model_confidence", "confidence"), ("appraisal_context", "appraisal")):
    p = src / "charts" / f"{a}.png"
    if p.exists(): shutil.copy(p, fig / f"{tag}_{b}.png")
print("generated", sorted(x.name for x in out.iterdir()))
