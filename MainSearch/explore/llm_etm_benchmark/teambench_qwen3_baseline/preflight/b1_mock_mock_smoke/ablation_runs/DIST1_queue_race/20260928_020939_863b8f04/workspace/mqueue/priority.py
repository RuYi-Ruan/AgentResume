"""
Priority-ordered task message wrapper.

Uses a dataclass with ordering so that tasks can be placed in a heap.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(order=True)
class PriorityTask:
    """Lower urgency number = higher priority (0=critical, 9=low)

    BUG 3: When two tasks have equal urgency, Python's dataclass
    ordering falls through to comparing the `message` field. If the payload
    is a dict, list, or other non-comparable type, this raises TypeError at
    runtime — crashes the priority heap under concurrent load.
    """
    urgency: int   # Primary sort key
    message: Any        # BUG: used as secondary sort key by dataclass ordering
                        # — crashes with TypeError if payload is a dict or list
