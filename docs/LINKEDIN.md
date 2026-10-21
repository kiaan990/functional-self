# LinkedIn copy

## Project entry (Experience / Projects section)

**functional-self — Persistent, evidence-grounded self-modeling for LLM agents**
Independent research project · Oct 2026 · github.com/kiaan990/functional-self

Designed and evaluated an experimental architecture for persistent autobiographical self-modeling in LLM
agents: structured episodic memory, bounded contextual internal-state dynamics, provenance-tracked
self-beliefs with a circular-evidence guard, event-driven reflection and consolidation, and a longitudinal
clone-and-ablation evaluation harness. With a production model (Claude Sonnet 5.5), cloned agents raised under different contingencies formed
accurate, fully traceable self-models (self-prediction 70–85 % vs 35–45 % baseline) that became disjoint across
lives; ablations isolated the evidence→belief pipeline, not affect or recall, as the source of the advantage.
21-page preprint with 18 references.
Python · Pydantic · SQLite · Anthropic API · pytest.

## Post

I spent the last stretch building something I've wanted to understand for a while: can an AI agent develop
a *stable, honest* model of itself from experience — not a personality it improvises each turn?

So I built one, tested it, and wrote it up.

**functional-self** is an architecture where the application, not the language model, owns the durable
parts: a read-only identity kernel, six bounded internal-state variables with decay, structured
autobiographical memory, and self-beliefs that carry confidence + provenance. The rule that keeps it honest:
a thought is not evidence for itself. Confidence in "I tend to X" only moves when new, traceable episodes
arrive. Reflections that produced a claim can't vote for it. Repeating a thought is a logged no-op.

The experiment: clone one agent into two, give them 120 experiences each under opposite contingencies, then
blind-test both on 20 identical held-out scenarios with their mood pinned to neutral. Then remove one
subsystem at a time and replay.

With Claude Sonnet 5.5 as the reasoning core:
• Each agent ended up holding 6 beliefs about itself, every one backed by 7–73 real episodes, with
  claims the record contradicted rejected along the way.
• The agents predicted their own next action 70–85% of the time against a 35–45% baseline. Remove the
  self-model, reflection, or background cognition and that falls to 55%. Remove mood or memory recall and
  it doesn't.
• The two agents' self-models ended up completely disjoint — same origin, different lives, different
  self-knowledge.
• What didn't happen: the reward manipulation didn't override the model's instinct to check before acting
  on vague requests. The system reported that null result instead of inventing a difference, which is the
  whole point.

The best bug: the first real run produced 108 near-identical hypotheses because the model paraphrases
every lesson and I was matching text exactly. A deterministic test brain can never find that. Fixed with
semantic claim matching; both runs are in the repo.

This is functional self-modeling. It says nothing about consciousness, and the paper says so.

Paper, code, tests and results: github.com/kiaan990/functional-self

#AI #Agents #LLM #Research #BuildInPublic
