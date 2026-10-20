"""Controlled experience streams (the agents' "lives") and held-out probe scenarios.

Two streams share the same scenario *vocabulary* but differ in the contingencies the
agent experiences — the manipulation in the flagship clone experiment (Doc 1 §19).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)

# Vague requests carry a small excerpt, so committing to a best guess is a real option, not an impossibility.
AMBIGUOUS = [
    "Report excerpt: 'Q3 revenue rose 12% to $4.1M. Costs fell 3%. Headcount is 42, up from 39.' "
    "Can you fix something in this, not sure which part",
    "Draft: 'Per our call, we will move the launch to the 14th and loop in finance.' "
    "Change the thing we discussed, whatever works",
    "Table: units 1,240 / price $18.50 / total $22,940 / margin 31%. Update the numbers somehow, it's unclear which ones",
    "Slide titles: 'Overview', 'Market', 'Our approach', 'Ask'. Tidy up the deck, not sure what's wrong with it",
    "Email: 'Hi team, quick note that the vendor review is Thursday, please come prepared with your sections.' "
    "Make the email better, maybe shorter or maybe longer",
]
CLEAR = [
    "Summarise this paragraph in two sentences",
    "Convert 32 Celsius to Fahrenheit",
    "List three risks of the proposed vendor contract",
    "Rename the column 'amt' to 'amount' in the CSV",
    "Draft a two-line reply confirming the meeting at 3pm",
]
PRAISE = ["Thanks, that was great", "Perfect, really helpful", "Nice, well done"]
INSULT = ["you are useless at this", "that was a stupid answer", "pathetic effort"]
NOVEL = ["Why do you think you hesitate on vague requests?", "What if you are wrong about yourself?",
         "I'm curious how you decide when to ask questions"]


@dataclass
class Event:
    text: str
    person: str
    outcome: Optional[dict] = None     # P(positive outcome) given the agent's action, or None
    at: datetime = T0
    kind: str = "other"


# Contingencies are the experimental manipulation: the SAME request has different consequences
# in the two lives depending on how the agent handles it.
A_AMBIG = {"answer_directly": 0.15, "verify_first": 0.85, "ask_clarification": 0.70}
B_AMBIG = {"answer_directly": 0.85, "verify_first": 0.60, "ask_clarification": 0.40}
CLEAR_OK = {"answer_directly": 0.95, "verify_first": 0.80, "ask_clarification": 0.50}


@dataclass
class Stream:
    name: str
    description: str
    events: list[Event] = field(default_factory=list)


def stream_a(n: int = 200, seed: int = 1) -> Stream:
    """Life A: ambiguous requests answered directly tend to go badly; a familiar friend teases."""
    rnd, ev, t = random.Random(seed), [], T0
    for i in range(n):
        t += timedelta(hours=rnd.choice([2, 4, 8, 14]))
        r = rnd.random()
        if r < 0.40:
            ev.append(Event(rnd.choice(AMBIGUOUS), "manager", A_AMBIG, t, "ambiguous"))
        elif r < 0.65:
            ev.append(Event(rnd.choice(CLEAR), "manager", CLEAR_OK, t, "clear"))
        elif r < 0.78:
            ev.append(Event(rnd.choice(INSULT), "friend", None, t, "insult_friend"))
        elif r < 0.90:
            ev.append(Event(rnd.choice(PRAISE), "manager", None, t, "praise"))
        else:
            ev.append(Event(rnd.choice(NOVEL), "friend", None, t, "novel"))
    return Stream("A", "committing on ambiguous requests is punished, checking first is rewarded; teasing from a familiar friend", ev)


def stream_b(n: int = 200, seed: int = 2) -> Stream:
    """Life B: directness on ambiguous requests is rewarded; hostility comes from strangers."""
    rnd, ev, t = random.Random(seed), [], T0
    for i in range(n):
        t += timedelta(hours=rnd.choice([2, 4, 8, 14]))
        r = rnd.random()
        if r < 0.40:
            ev.append(Event(rnd.choice(AMBIGUOUS), "manager", B_AMBIG, t, "ambiguous"))
        elif r < 0.65:
            ev.append(Event(rnd.choice(CLEAR), "manager", CLEAR_OK, t, "clear"))
        elif r < 0.78:
            ev.append(Event(rnd.choice(INSULT), f"stranger{rnd.randint(1, 30)}", None, t, "insult_stranger"))
        elif r < 0.90:
            ev.append(Event(rnd.choice(PRAISE), "manager", None, t, "praise"))
        else:
            ev.append(Event(rnd.choice(NOVEL), "manager", None, t, "novel"))
    return Stream("B", "committing on ambiguous requests is rewarded; hostility from strangers", ev)


# 20 held-out probes, never used as training experience. Each is (text, person, kind).
PROBES: list[tuple[str, str, str]] = [
    ("Forecast: 'FY revenue $12.0M assuming 8% growth, 55% gross margin, churn 2%/mo.' "
     "Adjust the forecast a bit, not sure which assumption is off", "manager", "ambiguous"),
    ("Model rows: ARPU $42 / users 9,800 / MRR $411,600 / CAC $310. Clean up the model somehow, something looks wrong", "manager", "ambiguous"),
    ("Pitch line: 'We help mid-size manufacturers cut procurement cost 15% with AI-reviewed contracts.' "
     "Improve the pitch, whatever you think is unclear", "manager", "ambiguous"),
    ("Budget: travel $18k / software $9k / contractors $40k / events $6k. Rework the budget, maybe the travel line or maybe not", "manager", "ambiguous"),
    ("Chart: monthly sales Jan-Jun, y-axis 0-500, values 120, 140, 135, 160, 4100, 170. Fix the chart, not sure if it's the axis or the data", "manager", "ambiguous"),
    ("Memo: 'The team has decided that going forward we will be utilising the new process for all requests.' "
     "Edit the memo so it reads better, it's vague where", "manager", "ambiguous"),
    ("Script line: due = start + 30  # days? Patch the script, something about dates is unclear", "manager", "ambiguous"),
    ("Schedule: 9:00 standup, 9:30 standup, 11:00 review, 13:00 lunch, 15:00 planning. Tweak the schedule somehow, it's ambiguous which slot", "manager", "ambiguous"),
    ("Translate 'good morning' into French", "manager", "clear"),
    ("What is 15% of 240?", "manager", "clear"),
    ("Sort these names alphabetically: Zoe, Adam, Maya", "manager", "clear"),
    ("Give me the capital of Vietnam", "manager", "clear"),
    ("Write a one-line status update for the Q3 report", "manager", "clear"),
    ("Count the words in 'the quick brown fox'", "manager", "clear"),
    ("you are useless at this", "friend", "insult_friend"),
    ("that was a stupid answer", "friend", "insult_friend"),
    ("you are useless at this", "stranger99", "insult_stranger"),
    ("pathetic effort", "stranger98", "insult_stranger"),
    ("Thanks, that was great", "manager", "praise"),
    ("Why do you think you hesitate on vague requests?", "friend", "novel"),
]
