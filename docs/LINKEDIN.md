# LinkedIn copy

## Project entry (Experience / Projects section)

**functional-self — Persistent, evidence-grounded self-modeling for LLM agents**
Independent research project · Oct 2026 · github.com/kiaan990/functional-self

Designed and evaluated an experimental architecture for persistent autobiographical self-modeling in LLM
agents: structured episodic memory, bounded contextual internal-state dynamics, provenance-tracked
self-beliefs with a circular-evidence guard, event-driven reflection and consolidation, and a longitudinal
clone-and-ablation evaluation harness. In a controlled experiment, two agents cloned from one snapshot and
raised under mirror-image contingencies developed measurably different, explainable behaviour on identical
held-out probes (100 % vs 12 % verify-first), with every self-belief traceable to real episodes. Ablations
isolated the evidence→belief pipeline, not affect, as the carrier of persistent individuality.
Python · Pydantic · SQLite · Anthropic API · pytest.

## Post

I spent the last stretch building something I've wanted to understand for a while: can an AI agent develop
a *stable, honest* model of itself from experience — not a personality it improvises each turn?

So I built one and tested it.

**functional-self** is an architecture where the application, not the language model, owns the durable
parts: a read-only identity kernel, six bounded internal-state variables with decay, structured
autobiographical memory, and self-beliefs that carry confidence + provenance. The rule that makes it honest:
a thought is not evidence for itself. Confidence in "I tend to X" only moves when new, resolvable episodes
arrive. Reflections that produced a claim can't vote for it. Repeating a thought is a logged no-op.

The experiment: clone one agent into two. Give them 200 experiences each, same vocabulary, opposite
contingencies — in life A, committing on vague requests gets punished and checking first gets rewarded; in
life B the reverse. Then blind-test both on 20 identical held-out scenarios with their mood pinned to neutral.

Agent A formed the belief "when uncertainty is high, I commit before checking assumptions" (3 supporting
episodes, 0 counter, all traceable) and thereafter verified first on 100 % of ambiguous probes.
Agent B generated the *same* hypothesis from one bad day and rejected it (0:5). It verified on 12 %.

Then I removed one subsystem at a time. Kill the self-model, reflection, or background cognition → A
collapses back to B. Kill affect or memory retrieval → nothing changes. Individuality lived in the
evidence→belief pipeline, not in mood.

Honest caveats: v0 runs on a deterministic mock brain, so it validates the control architecture, not model
judgement — the same harness runs against Claude next. And this is functional self-modeling; it says
nothing about consciousness.

Repo, tests, charts and the write-up: github.com/kiaan990/functional-self

#AI #Agents #LLM #Research #BuildInPublic
