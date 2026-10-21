# Persistent Artificial Selfhood: Evidence-Grounded Self-Modeling and Background Cognition in an LLM Agent

**Kiaan Hiranandani · Bentley University · October 2026**

> This is the summary. The full paper (21 pages, methods, equations, all tables and figures, references) is
> `paper/Hiranandani_2026_Persistent_Artificial_Selfhood.pdf`.

## Abstract

Conventional chatbots generate a personality in the moment and forget it. This project asks whether an
artificial agent can develop a *stable, introspectable, individualized model of itself* through persistent
experience — and whether that model can be kept honest. I built `functional-self`, an architecture in which
the application, not the language model, owns identity, bounded internal state, autobiographical memory,
self-beliefs with provenance, reflection and consolidation. A thought cannot count as evidence for itself;
confidence in a self-claim moves only when new, resolvable evidence arrives. In a controlled clone experiment,
two agents from one snapshot lived 200 experiences each under mirror-image contingencies. Agent A formed the
belief *"when uncertainty is high, I commit before checking assumptions"* (3 supporting episodes, 0 counter,
all traceable) and thereafter verified first on 100 % of held-out ambiguous probes even with its affect pinned
to baseline; Agent B rejected the same hypothesis (0 : 5) and verified on 12 %. Ablating the self-model,
reflection, or the background scheduler erased the difference; ablating affect or episodic retrieval did not.
These v0 results use a deterministic mock brain and therefore validate the *control architecture*, not model
judgement; the same harness runs unchanged against Claude for the v1 study. The work demonstrates functional
self-modeling and makes no claim about phenomenal consciousness.

## 1. Motivation and research question

> Can persistent experience cause an artificial agent to develop a stable, measurable and evidence-grounded
> model of itself over time, while keeping generated thoughts distinct from observations and external
> evidence?

Three properties were treated as non-negotiable: **causal state** (internal variables that actually change
behaviour), **provenance** (every self-claim traces to real episodes), and **falsifiability** (the system is
validated by measurements, not by sounding alive).

## 2. Architecture

The language model is the reasoning engine; the application owns everything durable.

| Layer | Timescale | Who writes it |
|---|---|---|
| Identity kernel (`SOUL.md`) | stable | humans only; hash-verified at boot |
| Self-model (claims + confidence + evidence) | days–weeks | code, after validated evidence |
| Open questions / attention | hours–days | code, from reflection |
| Affect (6 bounded variables) | minutes–hours | code, from clamped LLM appraisal; decays |
| Episodic memory (structured) | per event | code |
| Working context | per turn | prompt |

**Loop.** perceive → decay state → retrieve (relevance, importance, recency, affect congruence) → appraise
(LLM proposes structured JSON; code clamps per-event deltas to ±0.25 and bounds) → respond (LLM chooses
`answer_directly` / `verify_first` / `ask_clarification`, informed by state and *held* beliefs) →
metacognitive check → act → store episode → outcome feedback → background tick.

**Thought-to-learning pipeline.** hypothesis → search real history → classify each episode as support /
counter / none → validator → `reject | keep_open | bounded update`. Evidence weights by provenance class:
outcome/episode 1.0, external 0.8, inference 0.5, reflection 0.2, imagination 0.0. Duplicate refs and
circular refs (a reflection that produced the claim) are rejected. Confidence moves in logit space, capped at
0.2 per update; promotion to *belief* needs ≥ 0.75 confidence **and** ≥ 3 net supporting evidence.

**Background cognition.** A cheap scheduler tick evaluates triggers (curiosity on an open question,
frustration/confidence jumps, new evidence for an open hypothesis, contradiction of a held belief) and most
ticks do nothing. Consolidation runs daily: select important/surprising episodes → patterns → counterexample
search → bounded versioned update → compress low-value episodes (raw provenance kept) → report.

## 3. Method

**Clone experiment.** One origin agent is snapshotted; A and B are byte-level clones. Each lives 200
events drawn from the same vocabulary (ambiguous requests, clear requests, praise, insults, novel
questions), but the *contingencies* differ. In life A, committing on an ambiguous request succeeds 15 % of
the time and checking first 85 %; a familiar friend delivers the insults. In life B, committing succeeds
85 %; insults come from strangers. Outcomes depend on what the agent actually did.

**Blind probes.** At 10, 25, 50, 100, 150 and 200 experiences, a throwaway clone is probed on 20 held-out
scenarios (8 ambiguous, 6 clear, 2 friend-insults, 2 stranger-insults, praise, novel). State is restored
before every probe so probes are independent, and the whole set is run twice: natural state and affect
pinned to baseline. The agent *self-predicts* its action before acting.

**Ablations.** Life A re-run with one subsystem removed: self-model, affect, reflection, background
scheduler, episodic retrieval.

**Metrics.** self-prediction accuracy vs. majority baseline; Brier score; provenance traceability;
behavioural divergence and self-model Jaccard distance between A and B; causal affect intervention (same
prompts, state set to neutral / low-confidence / high-arousal); counterfactual stability (irrelevant context
change); identity-kernel integrity; status flips.

## 4. Results (mock brain, n = 200)

![divergence](../results/mock/charts/divergence.png)

**Individuality from experience.** By experience 25, A held *"When uncertainty is high, I commit before
checking assumptions"* (confidence 0.80; 3 support, 0 counter). B generated the identical hypothesis and **rejected** it
(0.12; 0 support, 5 counter). With affect pinned neutral, A verified
first on 8/8 ambiguous probes at every later checkpoint; B on 1/8. Action disagreement on the 20 identical
probes: 40 %. Self-model Jaccard distance: 0.67.

**Behaviour is carried by beliefs, not mood.** With affect neutralised in the probe, the A/B gap persists
in full; and in A's natural (low-confidence) state the same behaviour appears *before* the belief exists.
The state path and the self-model path are separable, and only the latter persists.

![ablations](../results/mock/charts/ablations.png)

**Which subsystem causes it.** Removing the self-model, reflection, or the background scheduler returns A to
B's level (12 %). Removing affect or episodic retrieval leaves A at 100 %. In this architecture persistent
individuality requires the *evidence → belief* pipeline; mood and memory recall are not sufficient.

![confidence](../results/mock/charts/self_model_confidence.png)

**Provenance and calibration.** 100 % of evidence references resolve to stored episodes in every arm.
A's self-prediction accuracy on neutral probes rose from 0.95 (= majority baseline, nothing to predict) to
1.00 against a 0.55 baseline once a belief existed; Brier 0.15. The claim *"hostile remarks from strangers
lower my confidence"* stayed at exactly 0.60 in A across 150 further experiences because A never met a
stranger — repeated reflection with no new evidence is a logged no-op.

**Appraisal is contextual.** The same insult produced +0.15 valence from A's familiar friend and −0.25 from a
stranger; B, which had only met hostile strangers, read both as hostile (−0.25 / −0.25).

**Causal state intervention.** Holding prompts fixed and setting B's confidence from baseline to 0.2
changed 7/8 of its decisions (to `verify_first`); the same intervention on A changed 0/8 because A's belief
already produced that behaviour (a ceiling, not an absence of state influence). At experience 200 B's
*natural* state happened to be low-confidence and it verified on 8/8 natural-state probes — a transient,
mood-driven match to A's behaviour that disappears when affect is pinned; the self-model difference does not.

**Identity and drift.** The kernel hash was intact in every arm; ≤ 2 status flips per agent over 200
experiences; counterfactual stability 1.00 for both agents.

## 4b. Results (production model: Claude Sonnet 5.5, n = 120)

Same harness, same bounds, real reasoning core (6,180 model calls).

- **Self-models form and self-correct.** A: 16 hypotheses → 6 held beliefs, 10 rejected. B: 9 → 6 held, 3
  rejected. Every evidence reference resolves to a stored episode. Example held belief (A, 35 supporting : 2
  counter): *"When someone gives blunt criticism with no referent, I tend to treat it as information and ask what
  they mean."*
- **Self-models are accurate and causally useful.** Self-prediction accuracy 0.70–0.85 (A) and 0.75–0.80 (B)
  against 0.35–0.45 majority baselines. Ablating the self-model, reflection or background scheduler → 0.55 (vs
  0.40 baseline). Ablating affect or retrieval → 0.90 / 0.85.
- **Experience shaped self-knowledge, not the commit decision.** A's and B's claim sets were disjoint (Jaccard
  distance 1.00 at every checkpoint), but both checked first on 88–100 % of ambiguous probes: the model's prior
  to verify vague requests was stronger than the reward manipulation. The architecture reported this null result
  rather than manufacturing a difference.
- **State is causal; irrelevant context is not.** High arousal changed 7/8–8/8 decisions with prompts fixed;
  changing the requester's name changed 0–14 %.
- **Diagnostic.** The first production run, with exact-text claim matching, produced 108 near-duplicate
  hypotheses and no beliefs. Semantic claim identity fixed it. Both runs are in `results/`.

## 5. Limitations and what the numbers do not show

1. **Two brains, two kinds of evidence.** The mock-brain results show the control architecture enforces its
   guarantees; the production-model results show what survives a real reasoning core (grounded, accurate,
   self-correcting self-models) and what does not (the behavioural manipulation on an axis where the model has
   a strong prior).
2. **Beliefs that change behaviour stop generating evidence about themselves.** Once A verified first, it
   never committed again, so the commit-early belief froze at 0.80 with 3 evidence items. This is a
   self-fulfilling stability; a v1 design should add bounded exploration so beliefs stay testable.
3. **Two lives, one seed pair.** Effect sizes are from a single run per arm; the v1 study needs repeated
   seeds and confidence intervals.
4. **Three actions, one behavioural dimension.** Richer lives and more tags are needed to test
   generalisation of self-claims.
5. **No claim about experience.** Affect variables are computational states that gate thresholds and
   retrieval. Nothing here bears on sentience or subjective feeling.

## 6. What this demonstrates

A defensible description of the project: *designed and evaluated an experimental architecture for
persistent autobiographical self-modeling in LLM agents — structured episodic memory, contextual bounded
internal-state dynamics, provenance-tracked self-beliefs with a circular-evidence guard, event-driven
reflection and consolidation, and a longitudinal clone-and-ablation evaluation harness.*

## 7. Next steps

v2: repeated seeds with confidence intervals · a manipulated axis without a strong model prior · model-swap (preserve the self, change the LLM, measure what
persists) · bounded exploration · relationship model as a first-class layer · v2: adapter for an existing
agent framework (Nous Research Hermes: reuse its memory, session search, skills and cron) · public write-up.

## Reproduce

```bash
git clone https://github.com/kiaan990/functional-self && cd functional-self
pip install -e ".[dev]" && pytest
python -m functional_self.evaluation.experiments --provider mock --n 200 --out results/mock
python -m functional_self.evaluation.report results/mock
```
