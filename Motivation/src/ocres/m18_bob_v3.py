"""M18 Bob controller v3: yield only when visible Alice has a clear lead."""
from __future__ import annotations

from ocres.m18_bob_v2 import PublicBobV2


class PublicBobV3(PublicBobV2):
    version = "m18-public-bob-v3-visible-clear-lead"

    def _choose(self, obs):
        original = self.prediction
        target = original.get("target_facility", "unknown")
        alice = obs.get("alice")
        should_yield = False
        if target in self.vocabulary and alice is not None:
            alice_distance = self._distance(alice["position"], target)
            bob_distance = self._distance(obs["bob"]["position"], target)
            should_yield = alice_distance + 2 <= bob_distance
        if not should_yield:
            self.prediction = {"intent": original.get("intent", "unknown"),
                               "target_facility": "unknown"}
        try:
            super()._choose(obs)
        finally:
            self.prediction = original
