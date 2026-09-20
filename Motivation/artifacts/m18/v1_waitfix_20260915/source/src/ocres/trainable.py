"""PyTorch building blocks for trainable macro-intent policies.

This module deliberately separates policy learning from low-level execution.
The policy sees a fixed structured observation and predicts one of the same
auditable macro intents used by the existing agents.  It never receives the
expert target, policy level, future reward, or post-hoc outcome as an input.
"""
from __future__ import annotations

import ast
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

# The Conda NumPy build and the official PyTorch Windows wheel bundle different
# copies of Intel OpenMP.  Selecting MKL's sequential backend prevents a second
# runtime from being loaded; unlike KMP_DUPLICATE_LIB_OK, this is not an unsafe
# suppression of the duplicate-runtime check.  Torch CUDA kernels are unaffected.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ocres.agents import COOK_START, DELIVER, FETCH, GET_DISH, HOLD, PICKUP, PLACE, PRE
from ocres.executor import Executor
from ocres.grid import STAY
from ocres.recipes import agent_held, pot_kinds

INTENTS = (FETCH, PLACE, COOK_START, GET_DISH, PICKUP, DELIVER, PRE, HOLD)
INTENT_ID = {name: index for index, name in enumerate(INTENTS)}
HELD_KINDS = (None, "onion", "dish", "soup")
POT_KINDS = ("empty", "items1", "items2", "items3", "cooking", "ready")


@dataclass(frozen=True)
class ObservationSpec:
    """Versioned shape of the historical structured observation."""

    width: int = 7
    height: int = 5
    max_pots: int = 2
    version: str = "m11-v1"

    @property
    def dim(self):
        cells = self.width * self.height
        return 2 * cells + 2 * len(HELD_KINDS) + self.max_pots * len(POT_KINDS) + 2


def _value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _held(value):
    value = _value(value)
    return None if value in (None, "None") else str(value)


def _position(value):
    value = _value(value)
    return tuple(int(v) for v in value)


def parse_pots(value):
    """Return sorted ``((x, y), kind)`` entries from a logged pot snapshot."""

    value = _value(value)
    entries = ast.literal_eval(value) if isinstance(value, str) else value
    return [(tuple(int(v) for v in pos), str(kind)) for pos, kind in sorted(entries)]


def encode_log_observation(row, deliveries=0, horizon=700, spec=None):
    """Encode a pre-action historical log row without intent/target leakage."""

    spec = spec or ObservationSpec()
    vector = np.zeros(spec.dim, dtype=np.float32)
    cells = spec.width * spec.height
    offset = 0
    for key in ("p0", "p1"):
        x, y = _position(row[key])
        if not (0 <= x < spec.width and 0 <= y < spec.height):
            raise ValueError(f"position {x, y} is outside observation grid")
        vector[offset + y * spec.width + x] = 1.0
        offset += cells
    for key in ("held0", "held1"):
        held = _held(row[key])
        if held not in HELD_KINDS:
            raise ValueError(f"unknown held object: {held}")
        vector[offset + HELD_KINDS.index(held)] = 1.0
        offset += len(HELD_KINDS)
    pots = parse_pots(row["pot"])
    if len(pots) > spec.max_pots:
        raise ValueError(f"found {len(pots)} pots, schema supports {spec.max_pots}")
    for pot_index, (_, kind) in enumerate(pots):
        if kind not in POT_KINDS:
            raise ValueError(f"unknown pot kind: {kind}")
        vector[offset + pot_index * len(POT_KINDS) + POT_KINDS.index(kind)] = 1.0
    offset += spec.max_pots * len(POT_KINDS)
    vector[offset] = min(float(_value(row["t"])) / max(float(horizon), 1.0), 1.0)
    vector[offset + 1] = min(float(deliveries) / 20.0, 1.0)
    return vector


def live_row(state, grid, me=0):
    """Build the historical-log abstraction from a live Overcooked state."""

    other = 1 - me
    return {
        "t": state.timestep,
        "p0": tuple(state.players[me].position),
        "p1": tuple(state.players[other].position),
        "held0": agent_held(state, me),
        "held1": agent_held(state, other),
        "pot": str(sorted(pot_kinds(state, grid).items())),
    }


def legal_intent_mask(row):
    """Physical feasibility mask, intentionally free of strategy preferences."""

    held = _held(row["held0"])
    kinds = [kind for _, kind in parse_pots(row["pot"])]
    accepting = any(kind in ("empty", "items1", "items2") for kind in kinds)
    full = any(kind == "items3" for kind in kinds)
    cooking = any(kind == "cooking" for kind in kinds)
    ready = any(kind == "ready" for kind in kinds)
    legal = {HOLD}
    if held is None:
        if accepting:
            legal.add(FETCH)
        if full:
            legal.add(COOK_START)
        if cooking or ready:
            legal.add(GET_DISH)
            # M16-V2: an empty-handed conservative cook may actively move to
            # a non-blocking pot-monitoring area. PRE remains a physical,
            # executable macro intent; it is not exposed to Bob.
            legal.add(PRE)
    elif held == "onion":
        if accepting:
            legal.add(PLACE)
    elif held == "dish":
        if ready:
            legal.add(PICKUP)
        if cooking:
            legal.add(PRE)
    elif held == "soup":
        legal.add(DELIVER)
    mask = np.zeros(len(INTENTS), dtype=np.bool_)
    for intent in legal:
        mask[INTENT_ID[intent]] = True
    return mask


class HistoricalIntentDataset(Dataset):
    """Decision-event dataset backed by complete episode NPZ files."""

    def __init__(self, paths, spec=None, event_only=True):
        self.spec = spec or ObservationSpec()
        self.samples = []
        for path in (Path(p) for p in paths):
            archive = np.load(path, allow_pickle=True)
            horizon = max(int(v) for v in archive["t"]) + 1
            deliveries = 0
            previous = None
            for index in range(len(archive["t"])):
                row = {key: archive[key][index] for key in archive.files}
                intent = str(_value(row["intent0"]))
                if intent not in INTENT_ID:
                    raise ValueError(f"unknown intent {intent} in {path}")
                marker = intent, _position(row["target0"])
                is_event = marker != previous
                previous = marker
                if not event_only or is_event:
                    mask = legal_intent_mask(row)
                    label = INTENT_ID[intent]
                    if not mask[label]:
                        raise ValueError(f"expert intent {intent} is illegal at {path}:{index}")
                    self.samples.append(
                        {
                            "x": encode_log_observation(row, deliveries, horizon, self.spec),
                            "y": label,
                            "mask": mask,
                            "episode": path.stem,
                            "source": str(path),
                            "t": int(_value(row["t"])),
                        }
                    )
                if float(_value(row["r"])) > 0:
                    deliveries += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            torch.from_numpy(sample["x"]),
            torch.tensor(sample["y"], dtype=torch.long),
            torch.from_numpy(sample["mask"]),
        )

    @property
    def labels(self):
        return np.asarray([sample["y"] for sample in self.samples], dtype=np.int64)


class MacroIntentPolicy(nn.Module):
    """Small MLP policy over auditable macro intents."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=len(INTENTS)):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, observation, legal_mask=None):
        logits = self.network(observation)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    @torch.no_grad()
    def choose(self, observation, legal_mask, deterministic=True):
        logits = self.forward(observation, legal_mask)
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()


def select_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_policy_checkpoint(path, model, spec, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "agentresume-macro-policy-v1",
        "state_dict": model.state_dict(),
        "model": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "output_dim": model.output_dim,
        },
        "observation_spec": asdict(spec),
        "intents": INTENTS,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_policy_checkpoint(path, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format") != "agentresume-macro-policy-v1":
        raise ValueError("unsupported policy checkpoint format")
    if tuple(payload["intents"]) != INTENTS:
        raise ValueError("checkpoint intent vocabulary does not match runtime")
    model = MacroIntentPolicy(**payload["model"])
    model.load_state_dict(payload["state_dict"])
    spec = ObservationSpec(**payload["observation_spec"])
    return model, spec, payload.get("extra", {})


class TrainableMacroAgent:
    """Closed-loop adapter from a macro-intent policy to the existing executor.

    Target resolution contains geometry only: it chooses the nearest feature
    compatible with the policy's intent.  It does not select the intent or
    encode the scripted L0/Lk scheduling rule.
    """

    def __init__(self, grid, me, model, spec=None, device="cpu", horizon=700, max_goal_ticks=40):
        self.grid = grid
        self.me = me
        self.model = model.to(device).eval()
        self.spec = spec or ObservationSpec()
        self.device = torch.device(device)
        self.horizon = horizon
        self.max_goal_ticks = max_goal_ticks
        self.exec = Executor(grid, me)
        self.intent = None
        self.intent_target = None
        self.decision_id = 0
        self._goal_ticks = 0
        self._decision_signature = None
        self._delivery_started = None
        self._blocked_ticks = 0
        self._yield_cooldown = 0
        self._last_primitive_position = None
        self._last_primitive_action = STAY

        stands = set()
        features = grid.onion_locs + grid.pot_locs + grid.dish_locs + grid.serve_locs
        for location in features:
            stands.update(stand for stand, _ in grid.interact_stand_cells(location))
        self.free_cells = sorted(grid.passable - stands)

    def _distance_to_feature(self, state, feature):
        start = tuple(state.players[self.me].position)
        distances = []
        for stand, _ in self.grid.interact_stand_cells(feature):
            path = self.grid.bfs(start, stand, ())
            if path is not None:
                distances.append(len(path))
        return min(distances) if distances else 10**6

    def _nearest(self, state, candidates):
        candidates = list(candidates)
        return min(candidates, key=lambda target: (self._distance_to_feature(state, target), target)) if candidates else None

    def _resolve_target(self, state, intent):
        kinds = pot_kinds(state, self.grid)
        if intent == FETCH:
            return self._nearest(state, self.grid.onion_locs)
        if intent == PLACE:
            return self._nearest(state, (p for p, k in kinds.items() if k in ("empty", "items1", "items2")))
        if intent == COOK_START:
            return self._nearest(state, (p for p, k in kinds.items() if k == "items3"))
        if intent == GET_DISH:
            return self._nearest(state, self.grid.dish_locs)
        if intent == PICKUP:
            return self._nearest(state, (p for p, k in kinds.items() if k == "ready"))
        if intent == DELIVER:
            return self._nearest(state, self.grid.serve_locs)
        if intent == PRE:
            return self._nearest(state, (p for p, k in kinds.items() if k in ("cooking", "ready")))
        return tuple(state.players[self.me].position)

    def _signature(self, state, deliveries):
        row = live_row(state, self.grid, self.me)
        return row["held0"], row["held1"], row["pot"], int(deliveries)

    def _goal_satisfied(self, state, deliveries):
        if self.intent is None:
            return True
        held = agent_held(state, self.me)
        kinds = pot_kinds(state, self.grid)
        target_kind = kinds.get(self.intent_target)
        if self.intent == FETCH:
            return held == "onion"
        if self.intent == PLACE:
            return held != "onion"
        if self.intent == COOK_START:
            return target_kind != "items3"
        if self.intent == GET_DISH:
            return held == "dish"
        if self.intent == PICKUP:
            return held == "soup"
        if self.intent == DELIVER:
            return held != "soup" or deliveries > self._delivery_started
        if self.intent == PRE:
            return any(kind == "ready" for kind in kinds.values())
        if self.intent == HOLD:
            return self._signature(state, deliveries) != self._decision_signature
        return False

    def _goal_invalid(self, state):
        if self.intent_target is None:
            return True
        kinds = pot_kinds(state, self.grid)
        kind = kinds.get(self.intent_target)
        if self.intent == PLACE:
            return kind not in ("empty", "items1", "items2")
        if self.intent == COOK_START:
            return kind != "items3"
        if self.intent == PICKUP:
            return kind != "ready"
        if self.intent == PRE:
            return not any(value in ("cooking", "ready") for value in kinds.values())
        return False

    @torch.no_grad()
    def _decide(self, state, deliveries):
        observation_array, mask_array = self.observe(state, deliveries)
        observation = torch.from_numpy(observation_array).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask_array).unsqueeze(0).to(self.device)
        logits = self.model(observation, mask)
        choice = int(logits.argmax(dim=1).item())
        confidence = float(torch.softmax(logits, dim=1)[0, choice].item())
        self.begin_intent(state, deliveries, choice)
        return {"decision_id": self.decision_id, "confidence": confidence, "policy": "torch_bc"}

    def observe(self, state, deliveries):
        row = live_row(state, self.grid, self.me)
        return (
            encode_log_observation(row, deliveries, self.horizon, self.spec),
            legal_intent_mask(row),
        )

    def begin_intent(self, state, deliveries, choice):
        """Start an externally selected macro intent (used by PPO rollouts)."""

        choice = int(choice)
        row = live_row(state, self.grid, self.me)
        mask = legal_intent_mask(row)
        if not (0 <= choice < len(INTENTS)) or not mask[choice]:
            raise ValueError(f"illegal macro intent choice: {choice}")
        self.intent = INTENTS[choice]
        self.intent_target = self._resolve_target(state, self.intent)
        self.decision_id += 1
        self._goal_ticks = 0
        self._decision_signature = self._signature(state, deliveries)
        self._delivery_started = deliveries
        self._blocked_ticks = 0
        self._yield_cooldown = 0
        self._last_primitive_position = None
        self._last_primitive_action = STAY

    def macro_complete(self, state, deliveries):
        return (
            self._goal_ticks > 0
            and (
                self._goal_satisfied(state, deliveries)
                or self._goal_invalid(state)
                or self._goal_ticks >= self.max_goal_ticks
            )
        )

    def _parking_action(self, state, preferred):
        if not self.free_cells:
            return STAY
        cell = min(self.free_cells, key=lambda value: (abs(value[0] - preferred[0]) + abs(value[1] - preferred[1]), value))
        action = self.exec.stand_action(state, cell)
        return action if action is not None else STAY

    def _yield_action(self, state):
        """Step aside after repeated blocking without changing macro intent."""

        position = tuple(state.players[self.me].position)
        occupied = tuple(state.players[1 - self.me].position)
        candidates = sorted(
            cell
            for cell in self.grid.passable
            if cell != occupied and abs(cell[0] - position[0]) + abs(cell[1] - position[1]) == 1
        )
        if not candidates:
            return STAY
        # Prefer leaving interaction stands/choke points and increasing the
        # distance to the partner; stable tuple ordering breaks remaining ties.
        chosen = max(
            candidates,
            key=lambda cell: (
                cell in self.free_cells,
                abs(cell[0] - occupied[0]) + abs(cell[1] - occupied[1]),
                -cell[0],
                -cell[1],
            ),
        )
        return chosen[0] - position[0], chosen[1] - position[1]

    def _execute(self, state, deliveries):
        held = agent_held(state, self.me)
        if self.intent == FETCH:
            return self.exec.interact_action(state, self.intent_target) if held != "onion" else STAY
        if self.intent == PLACE:
            return self.exec.interact_action(state, self.intent_target) if held == "onion" else STAY
        if self.intent == COOK_START:
            return self.exec.interact_action(state, self.intent_target)
        if self.intent == GET_DISH:
            return self.exec.interact_action(state, self.intent_target) if held is None else STAY
        if self.intent == PICKUP:
            return self.exec.interact_action(state, self.intent_target) if held == "dish" else STAY
        if self.intent == DELIVER:
            return self.exec.interact_action(state, self.intent_target) if held == "soup" else STAY
        if self.intent == PRE:
            return self._parking_action(state, (3, 2))
        if self.intent == HOLD:
            role = "cook" if self.me == (deliveries % 2) else "serve"
            return self._parking_action(state, (3, 1) if role == "cook" else (5, 2))
        return STAY

    def primitive_action(self, state, deliveries):
        """Execute one primitive tick of the current externally chosen intent."""

        if self.intent is None:
            raise RuntimeError("begin_intent must be called before primitive_action")
        position = tuple(state.players[self.me].position)
        previous_move_failed = (
            self._last_primitive_position == position
            and isinstance(self._last_primitive_action, tuple)
            and self._last_primitive_action != STAY
        )
        if previous_move_failed:
            self._blocked_ticks += 1
        elif self._last_primitive_position is not None and self._last_primitive_position != position:
            self._blocked_ticks = 0
        if self._yield_cooldown > 0:
            self._yield_cooldown -= 1
            self._goal_ticks += 1
            self._last_primitive_position = position
            self._last_primitive_action = STAY
            return STAY
        action = self._execute(state, deliveries)
        if self.intent not in (HOLD, PRE) and action in (None, STAY):
            self._blocked_ticks += 1
        if self._blocked_ticks >= (3 if self.me == 0 else 4):
            action = self._yield_action(state)
            self._blocked_ticks = 0
            if action != STAY:
                self._yield_cooldown = 3
        self._goal_ticks += 1
        action = action if action is not None else STAY
        self._last_primitive_position = position
        self._last_primitive_action = action
        return action

    def action(self, state, deliveries):
        info = None
        if self.intent is None or self.macro_complete(state, deliveries):
            info = self._decide(state, deliveries)
        action = self.primitive_action(state, deliveries)
        return action, self.intent, self.intent_target, info
