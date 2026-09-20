"""M18 Bob controller v2: conservative facility-level coordination.

Predictions only break ties between otherwise equal tasks. They never suppress
an entire intent class, which made a correct Alice prediction reduce throughput.
"""
from __future__ import annotations

from ocres.m18_bob import PublicBob


class PublicBobV2(PublicBob):
    version = "m18-public-bob-v2-facility-tiebreak"

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

        active = now < self.prediction_until and self.prediction["intent"] not in ("unknown", "HOLD")
        predicted_target = self.prediction["target_facility"] if active else "unknown"

        def order(entry):
            rank, intent, name = entry
            # Avoid Alice's exact facility only among tasks of the same base
            # priority. Distance remains the next tie-breaker, so the signal
            # cannot promote a lower-priority, semantically different task.
            same_facility = int(name == predicted_target and predicted_target in self.vocabulary)
            if intent == "INSPECT":
                return (rank, same_facility, self.inspected_at.get(name, -1),
                        self._distance(obs["bob"]["position"], name), name)
            return (rank, same_facility, self._distance(obs["bob"]["position"], name), name)

        _, self.task, self.target = min(candidates, key=order)
        self.started = now
