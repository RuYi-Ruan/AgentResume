"""Auditable observation and LLM boundaries for the M17 experiment.

This module deliberately contains no model calls.  It turns simulator state
changes into facts Bob could observe, prevents visible histories from crossing
an out-of-view gap, validates dataset provenance, and strictly validates the
small JSON object returned by the intent predictor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Iterable, Mapping


INTENTS = frozenset({"PRE", "FETCH"})


def partner_visible(alice_position, bob_position, radius=1):
    """Return whether Alice is inside Bob's square local view."""

    return max(
        abs(int(alice_position[0]) - int(bob_position[0])),
        abs(int(alice_position[1]) - int(bob_position[1])),
    ) <= int(radius)


def realized_partner_action(before, after):
    """Describe only the externally observable result of one simulator step.

    ``before`` and ``after`` are small mappings with ``position``,
    ``orientation`` and ``held`` entries.  A requested move that was blocked is
    therefore recorded as ``stay`` rather than as a move.
    """

    before_pos = tuple(before["position"])
    after_pos = tuple(after["position"])
    delta = after_pos[0] - before_pos[0], after_pos[1] - before_pos[1]
    if delta != (0, 0):
        if abs(delta[0]) + abs(delta[1]) != 1:
            raise ValueError(f"non-adjacent partner displacement: {delta!r}")
        return {"kind": "move", "delta": list(delta)}
    if before.get("held") != after.get("held"):
        return {
            "kind": "interact",
            "held_before": before.get("held"),
            "held_after": after.get("held"),
        }
    if tuple(before.get("orientation", ())) != tuple(after.get("orientation", ())):
        return {"kind": "turn", "orientation": list(after["orientation"])}
    return {"kind": "stay"}


@dataclass
class VisibleSegmentRecorder:
    """Record consecutive partner observations without bridging blind gaps."""

    active: dict | None = None
    completed: list[dict] = field(default_factory=list)

    def start(self, frame: Mapping):
        if self.active is not None:
            raise RuntimeError("a visible segment is already active")
        self.active = {
            "start_t": int(frame["t"]),
            "frames": [dict(frame)],
            "outcome": None,
        }

    def observe(self, frame: Mapping | None):
        """Append a visible frame or immediately close on lost visibility."""

        if self.active is None:
            return None
        if frame is None:
            return self.close("lost_visibility")
        if int(frame["t"]) <= int(self.active["frames"][-1]["t"]):
            raise ValueError("segment frames must have increasing timesteps")
        self.active["frames"].append(dict(frame))
        return None

    def close(self, outcome):
        if self.active is None:
            return None
        segment = self.active
        segment["outcome"] = str(outcome)
        segment["end_t"] = int(segment["frames"][-1]["t"])
        segment["visible_steps"] = len(segment["frames"])
        self.completed.append(segment)
        self.active = None
        return segment


def validate_partition_provenance(partitions: Mapping[str, Iterable[Mapping]]):
    """Reject seed/episode/event reuse across history, development and test."""

    owners = {}
    for partition, rows in partitions.items():
        for row in rows:
            missing = [key for key in ("seed", "episode_id", "event_id") if key not in row]
            if missing:
                raise ValueError(f"{partition} row missing provenance: {missing}")
            key = int(row["seed"]), str(row["episode_id"]), str(row["event_id"])
            previous = owners.setdefault(key, partition)
            if previous != partition:
                raise ValueError(
                    f"provenance overlap between {previous} and {partition}: {key!r}"
                )
    return True


def stable_event_id(public_payload):
    """Return a reproducible identifier for the exact public event payload."""

    encoded = json.dumps(
        public_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:20]


def validate_intent_response(value, allowed_fact_ids, allowed_facilities):
    """Validate a Qwen result before it can enter accuracy or control code.

    Target facilities may be outside the current view because they are marked
    as predictions.  Facts, however, must be selected from program-issued IDs.
    """

    errors = []
    required = {"intent", "target_facility", "confidence", "used_fact_ids"}
    if not isinstance(value, dict):
        return False, ["response_not_object"]
    extra = set(value) - required
    missing = required - set(value)
    if missing:
        errors.append("missing_fields:" + ",".join(sorted(missing)))
    if extra:
        errors.append("extra_fields:" + ",".join(sorted(extra)))
    if value.get("intent") not in INTENTS:
        errors.append("invalid_intent")
    facilities = set(allowed_facilities) | {"unknown"}
    if value.get("target_facility") not in facilities:
        errors.append("invalid_target_facility")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append("invalid_confidence")
    fact_ids = value.get("used_fact_ids")
    if not isinstance(fact_ids, list) or not fact_ids:
        errors.append("invalid_used_fact_ids")
    elif any(not isinstance(item, str) for item in fact_ids):
        errors.append("non_string_fact_id")
    else:
        unknown = sorted(set(fact_ids) - set(allowed_fact_ids))
        if unknown:
            errors.append("unknown_fact_ids:" + ",".join(unknown))
        if len(fact_ids) != len(set(fact_ids)):
            errors.append("duplicate_fact_ids")
    return not errors, errors
