"""M18 Bob's observable boundary, eight-intent protocol and public controller.

The sensor alone accepts a simulator state. Prediction and action selection accept
only its returned facts; the executor gets a player-only public proxy.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from types import SimpleNamespace

from ocres.executor import Executor
from ocres.grid import STAY

INTENTS = ("FETCH", "PLACE", "COOK_START", "GET_DISH", "PICKUP", "DELIVER", "PRE", "HOLD")
TARGET_TYPES = {"FETCH": "onion", "PLACE": "pot", "COOK_START": "pot", "GET_DISH": "dish",
                "PICKUP": "pot", "DELIVER": "serve", "PRE": "pot", "HOLD": "none"}


def facilities(grid):
    result = {}
    for kind, locations in (("pot", grid.pot_locs), ("onion", grid.onion_locs),
                            ("dish", grid.dish_locs), ("serve", grid.serve_locs)):
        for index, position in enumerate(sorted(locations)):
            result[f"{kind}_{chr(65+index)}"] = {"kind": kind, "position": list(position)}
    return result


def visible(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= 1


def player_fact(player):
    return {"position": list(player.position), "orientation": list(player.orientation),
            "held": player.held_object.name if player.held_object is not None else None}


class PublicSensor:
    def __init__(self, grid):
        self.facilities = facilities(grid)
        self.memory = {}

    def observe(self, state):
        own = player_fact(state.players[1])
        alice = state.players[0]
        partner = player_fact(alice) if visible(own["position"], alice.position) else None
        local_pots = {}
        for name, facility in self.facilities.items():
            position = tuple(facility["position"])
            if facility["kind"] != "pot" or not visible(own["position"], position):
                continue
            if not state.has_object(position):
                kind = "empty"
            else:
                soup = state.get_object(position)
                kind = ("ready" if soup.is_ready else "cooking" if soup.is_cooking
                        else f"items{len(soup.ingredients)}")
            local_pots[name] = kind
            self.memory[name] = {"kind": kind, "seen_t": int(state.timestep)}
        return {"t": int(state.timestep), "bob": own, "alice": partner,
                "visible_pots": local_pots, "remembered_pots": deepcopy(self.memory)}


def realized_action(before, after):
    """No simulator requested-action or private decision information is read."""
    if before["alice"] is None or after["alice"] is None:
        return None
    a, b = before["alice"], after["alice"]
    delta = [b["position"][i]-a["position"][i] for i in (0, 1)]
    if delta != [0, 0]:
        if sum(abs(d) for d in delta) != 1:
            raise ValueError("non-adjacent observed displacement")
        return {"kind": "move", "delta": delta}
    if a["held"] != b["held"]:
        return {"kind": "object_change", "before": a["held"], "after": b["held"]}
    if a["orientation"] != b["orientation"]:
        return {"kind": "turn", "orientation": b["orientation"]}
    changed = {k: [v, after["visible_pots"][k]] for k, v in before["visible_pots"].items()
               if k in after["visible_pots"] and v != after["visible_pots"][k]}
    if changed:
        # A pot change is observable; attributing it to Alice is still an inference.
        return {"kind": "stationary_with_pot_change", "pot_changes": changed}
    return {"kind": "stay"}


def public_event(before, after):
    action = realized_action(before, after)
    if action is None:
        return None
    return {"before": deepcopy(before), "after": deepcopy(after), "action": action}


def normalized_event(event):
    item = deepcopy(event)
    for key in ("before", "after"):
        frame = item[key]
        now = frame.pop("t")
        for entry in frame["remembered_pots"].values():
            entry["age"] = now-entry.pop("seen_t")
    return item


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def event_hash(event):
    return digest(normalized_event(event))


def near_family(map_name, event):
    frame = event["before"]
    return digest({"map": map_name, "held": frame["alice"]["held"],
                   "pots": frame["visible_pots"], "action": event["action"],
                   "alice_cell": [v//2 for v in frame["alice"]["position"]],
                   "bob_cell": [v//2 for v in frame["bob"]["position"]]})


def validate_response(value, fact_ids, vocabulary):
    required = {"intent", "target_facility", "confidence", "used_fact_ids"}
    if not isinstance(value, dict) or set(value) != required:
        return False, ["invalid_fields"]
    errors = []
    intent, target = value["intent"], value["target_facility"]
    if not isinstance(intent, str) or intent not in INTENTS + ("unknown",):
        return False, ["invalid_intent"]
    if not isinstance(target, str) or target not in set(vocabulary) | {"none", "unknown"}:
        errors.append("invalid_facility")
    elif intent in TARGET_TYPES and target != "unknown":
        kind = vocabulary[target]["kind"] if target in vocabulary else target
        if kind != TARGET_TYPES[intent]:
            errors.append("incompatible_facility")
    elif intent == "unknown" and target != "unknown":
        errors.append("unknown_intent_with_target")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        errors.append("invalid_confidence")
    refs = value["used_fact_ids"]
    if (not isinstance(refs, list) or not refs or any(not isinstance(r, str) for r in refs)
            or len(refs) != len(set(refs)) or any(r not in fact_ids for r in refs)):
        errors.append("invalid_fact_ids")
    return not errors, errors


def majority(predictions):
    """Invalid responses are None, never removed to change the denominator."""
    if len(predictions) != 3:
        raise ValueError("exactly three attempts required")
    counts = Counter(p["intent"] for p in predictions if p is not None)
    winners = [intent for intent, count in counts.items() if count >= 2 and intent != "unknown"]
    if not winners:
        return {"intent": "unknown", "target_facility": "unknown"}
    intent = winners[0]
    targets = Counter(p["target_facility"] for p in predictions if p and p["intent"] == intent)
    target = next((name for name, n in targets.items() if n >= 2), "unknown")
    return {"intent": intent, "target_facility": target}


class PublicTrigger:
    def __init__(self, cooldown=8, limit=12):
        self.cooldown, self.limit = cooldown, limit
        self.last_query = -10**9
        self.queries = 0
        self.was_continuously_visible = False
        self.last_direction = None

    def consider(self, before, after):
        action = realized_action(before, after)
        if action is None:
            self.was_continuously_visible = False
            self.last_direction = None
            return False
        first = not self.was_continuously_visible
        self.was_continuously_visible = True
        direction = action.get("delta")
        turn = direction is not None and self.last_direction is not None and direction != self.last_direction
        if direction is not None:
            self.last_direction = direction
        changed = action["kind"] in ("object_change", "stationary_with_pot_change", "turn") or turn
        if (first or changed) and after["t"]-self.last_query >= self.cooldown and self.queries < self.limit:
            self.last_query = after["t"]
            self.queries += 1
            return True
        return False


class PublicBob:
    """Fixed cooperation policy. action() cannot accept a simulator state."""

    def __init__(self, grid, ttl=12):
        self.grid, self.vocabulary = grid, facilities(grid)
        self.executor = Executor(grid, 1)
        self.ttl = ttl
        self.prediction = {"intent": "unknown", "target_facility": "unknown"}
        self.prediction_until = -1
        self.task, self.target, self.started = None, None, -1
        self.last_position, self.last_action = None, STAY
        self.blocked = 0
        self.block_events = 0
        self.block_active = False
        self.inspected_at = {}

    def update_prediction(self, prediction, now):
        self.prediction = dict(prediction)
        self.prediction_until = now + self.ttl
        # Re-evaluate only on the next public frame; held-object tasks remain priorities.
        self.task = None

    def _position(self, name):
        return tuple(self.vocabulary[name]["position"])

    def _distance(self, own, name):
        target = self._position(name)
        paths = [self.grid.bfs(tuple(own), stand) for stand, _ in self.grid.interact_stand_cells(target)]
        lengths = [len(path) for path in paths if path is not None]
        return min(lengths) if lengths else 10**6

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
                # An old memory motivates looking again, not assuming the soup is ready.
                age = now-obs["remembered_pots"][name]["seen_t"]
                add("INSPECT" if age >= 12 else "PRE", [name], 5)
        else:
            if pots(("ready",)):
                add("GET_DISH", names("dish"), 0)
            add("COOK_START", pots(("items3",)), 5)
            if pots(("empty", "items1", "items2")):
                add("FETCH", names("onion"), 10)
            if pots(("cooking",)):
                add("GET_DISH", names("dish"), 15)
        # Explore unknown/stale pots if there is no currently actionable task.
        if not candidates:
            add("INSPECT", names("pot"), 100)
        def order(entry):
            rank, intent, name = entry
            penalty = 0
            if now < self.prediction_until and self.prediction["intent"] not in ("unknown", "HOLD"):
                if intent == self.prediction["intent"]:
                    penalty += 20
                if name == self.prediction["target_facility"]:
                    penalty += 50
            if intent == "INSPECT":
                return (rank, self.inspected_at.get(name, -1), self._distance(obs["bob"]["position"], name), name)
            return (rank+penalty, self._distance(obs["bob"]["position"], name), 0, name)
        _, self.task, self.target = min(candidates, key=order)
        self.started = now

    def _finished(self, obs):
        if self.task is None or obs["t"]-self.started >= 50:
            return True
        if obs["t"] == self.started:
            return False
        held = obs["bob"]["held"]
        kind = obs["visible_pots"].get(self.target)
        if self.task == "FETCH":
            return held is not None
        if self.task == "GET_DISH":
            return held is not None
        if self.task == "DELIVER":
            return held != "soup"
        if self.task == "PLACE":
            return held != "onion" or (kind is not None and kind not in ("empty", "items1", "items2"))
        if self.task == "PICKUP":
            return held != "dish" or (kind is not None and kind != "ready")
        if self.task == "COOK_START":
            return kind is not None and kind != "items3"
        if self.task == "PRE":
            return (kind is not None and kind != "cooking") or obs["t"]-self.started >= 12
        if self.task == "INSPECT":
            return self.target in obs["visible_pots"]
        return False

    def action(self, obs):
        own = obs["bob"]
        now = obs["t"]
        for name in obs["visible_pots"]:
            self.inspected_at[name] = now
        if now == self.prediction_until:
            self.task = None
        if self._finished(obs):
            self._choose(obs)
        partner = obs["alice"]
        # No hidden partner coordinate is ever passed to pathfinding.
        proxy = SimpleNamespace(players=[
            SimpleNamespace(position=tuple(partner["position"]) if partner else (-100, -100)),
            SimpleNamespace(position=tuple(own["position"]), orientation=tuple(own["orientation"]))])
        if self.task == "PRE":
            entrances = {stand for facility in self.vocabulary.values()
                         for stand, _ in self.grid.interact_stand_cells(tuple(facility["position"]))}
            safe = self.grid.passable-entrances-{proxy.players[0].position}
            target = self._position(self.target)
            goal = min(safe or {tuple(own["position"])},
                       key=lambda p: (abs(p[0]-target[0])+abs(p[1]-target[1]), p))
            action = self.executor.stand_action(proxy, goal)
        elif self.task == "INSPECT":
            positions = [stand for stand, _ in self.grid.interact_stand_cells(self._position(self.target))]
            goal = min(positions, key=lambda p: (abs(p[0]-own["position"][0])+abs(p[1]-own["position"][1]), p))
            action = self.executor.stand_action(proxy, goal)
        else:
            action = self.executor.interact_action(proxy, self._position(self.target))
        action = action if action is not None else STAY
        failed = self.last_position == own["position"] and self.last_action != STAY and isinstance(self.last_action, tuple)
        self.blocked = self.blocked+1 if failed else 0
        if self.blocked >= 3:
            if not self.block_active:
                self.block_events += 1
            self.block_active = True
            position = tuple(own["position"])
            neighbor_cells = [p for p in self.grid.passable
                              if abs(p[0]-position[0])+abs(p[1]-position[1]) == 1
                              and p != proxy.players[0].position]
            if neighbor_cells:
                target = max(neighbor_cells, key=lambda p: (abs(p[0]-proxy.players[0].position[0])+
                                                            abs(p[1]-proxy.players[0].position[1]), p))
                action = (target[0]-position[0], target[1]-position[1])
        elif self.last_position is not None and self.last_position != own["position"]:
            self.block_active = False
        self.last_position, self.last_action = list(own["position"]), action
        return action
