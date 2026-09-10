"""Metrics for matching Bob's one-shot guesses to Alice decision events."""
from __future__ import annotations


def score_intent_events(logs):
    """Score Bob once per observed Alice decision.

    ``guess`` and ``cause`` in a trace are state snapshots and may persist for
    many ticks.  Event-only fields plus a decision id prevent those snapshots
    from being accidentally counted as independent observations.
    """
    alice = {}
    bob = {}
    for row in logs:
        ainfo = row.get("info0") or {}
        if ainfo.get("decision_event") and ainfo.get("decision_id") is not None:
            alice[ainfo["decision_id"]] = {
                "intent": row.get("intent0"),
                "cause": ainfo.get("cause_event"),
                "tick": row.get("t"),
            }

        binfo = row.get("info1") or {}
        decision_id = binfo.get("guess_for")
        guess = binfo.get("guess_event")
        if decision_id is not None and guess is not None:
            bob[decision_id] = guess

    paired = [(alice[k], guess) for k, guess in bob.items() if k in alice]
    correct = sum(guess == event["intent"] for event, guess in paired)
    caused = [(event, guess) for event, guess in paired if event["cause"]]
    cause_correct = sum(guess == event["intent"] for event, guess in caused)
    return {
        "alice_decisions": len(alice),
        "guessed_decisions": len(paired),
        "correct_decisions": correct,
        "accuracy": correct / len(paired) if paired else None,
        "cause_events": len(caused),
        "cause_correct": cause_correct,
        "cause_accuracy": cause_correct / len(caused) if caused else None,
    }
