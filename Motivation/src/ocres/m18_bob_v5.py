"""M18 Bob controller v5: intent-level pipeline complement, no target avoidance."""
from __future__ import annotations

from ocres.m18_bob import PublicBob


class PublicBobV5(PublicBob):
    version = "m18-public-bob-v5-pipeline-complement"

    def _choose(self, obs):
        held, now = obs["bob"]["held"], obs["t"]
        known = {name: value["kind"] for name, value in obs["remembered_pots"].items()}
        candidates = []

        def add(intent, names, rank):
            candidates.extend((rank, intent, name) for name in names)

        pots = lambda kinds: [name for name, kind in known.items() if kind in kinds]
        names = lambda kind: [name for name, value in self.vocabulary.items() if value["kind"] == kind]
        if held == "soup":
            add("DELIVER", names("serve"), 0)
        elif held == "onion":
            add("PLACE", pots(("empty", "items1", "items2")), 0)
        elif held == "dish":
            add("PICKUP", pots(("ready",)), 0)
            for name in pots(("cooking",)):
                age = now - obs["remembered_pots"][name]["seen_t"]
                add("INSPECT" if age >= 12 else "PRE", [name], 5)
        else:
            if pots(("ready",)):
                add("GET_DISH", names("dish"), 0)
            add("COOK_START", pots(("items3",)), 5)
            if pots(("empty", "items1", "items2")):
                add("FETCH", names("onion"), 10)
            if pots(("cooking",)):
                add("GET_DISH", names("dish"), 15)
        if not candidates:
            add("INSPECT", names("pot"), 100)

        active_intent = (self.prediction["intent"] if now < self.prediction_until else "unknown")
        preferred = None
        if held is None:
            if active_intent == "COOK_START":
                preferred = "GET_DISH"
            elif active_intent in ("GET_DISH", "PICKUP", "DELIVER", "PRE"):
                preferred = "FETCH"

        def order(entry):
            rank, intent, name = entry
            complement = 0 if preferred is not None and intent == preferred else 1
            # A complement may move ahead of the ordinary empty-hand priority,
            # while held-object work above never reaches this branch.
            effective = rank if preferred is None else complement
            if intent == "INSPECT":
                return (effective, self.inspected_at.get(name, -1),
                        self._distance(obs["bob"]["position"], name), name)
            return (effective, self._distance(obs["bob"]["position"], name), rank, name)

        _, self.task, self.target = min(candidates, key=order)
        self.started = now
