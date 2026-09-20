"""M18 Bob controller v4: coordinate only duplicated pickup sources.

Pot reservations caused rare but catastrophic production deadlocks in v2/v3.
This version never changes pot/serve work; it only selects another onion or dish
dispenser when the prediction says Alice is using the same dispenser.
"""
from __future__ import annotations

from ocres.m18_bob_v2 import PublicBobV2


class PublicBobV4(PublicBobV2):
    version = "m18-public-bob-v4-pickup-source-only"

    def _choose(self, obs):
        original = self.prediction
        target = original.get("target_facility", "unknown")
        allowed = (original.get("intent") in ("FETCH", "GET_DISH")
                   and target in self.vocabulary
                   and self.vocabulary[target]["kind"] in ("onion", "dish"))
        if not allowed:
            self.prediction = {"intent": original.get("intent", "unknown"),
                               "target_facility": "unknown"}
        try:
            super()._choose(obs)
        finally:
            self.prediction = original
