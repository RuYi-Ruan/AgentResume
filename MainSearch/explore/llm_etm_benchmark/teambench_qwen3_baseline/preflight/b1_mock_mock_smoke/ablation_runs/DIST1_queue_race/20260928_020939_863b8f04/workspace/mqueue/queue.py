"""
Task message queue with acknowledgment support.

WARNING: This implementation contains known race conditions for testing purposes.
"""
import threading
import uuid
from collections import deque
from typing import Any, Optional, Tuple


class QueueFull(Exception):
    """Raised when the queue has reached its capacity."""


class QueueEmpty(Exception):
    """Raised when the queue is empty."""


class TaskQueue:
    """Thread-safe task queue with configurable capacity."""

    def __init__(self, capacity: int = 500):
        self._capacity = capacity
        self._queue: deque = deque()
        self._lock = threading.Lock()

    def put(self, message: Any) -> None:
        """
        Enqueue a task message.

        Raises QueueFull if the queue is at capacity.
        """
        # BUG 1: TOCTOU — the capacity check and the append are two separate
        # operations with no lock holding both. Two producers can both pass
        # the check before either appends, causing capacity to be exceeded.
        if len(self._queue) >= self._capacity:  # Step 1: check (no lock held)
            raise QueueFull(
                f"TaskQueue at capacity ({self._capacity})"
            )
        self._queue.append(message)  # Step 2: append (another thread may have
                                     # also passed the check between these two lines)

    def get(self) -> Optional[Any]:
        """
        Dequeue and return the next task message, or None if empty.

        BUG 2: The message is removed from the queue immediately. If the
        consumer crashes after get() but before finishing processing, the
        message is permanently lost — there is no way to re-deliver it.
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()  # message gone forever after this line

    def size(self) -> int:
        """Return the current number of tasks in the queue."""
        return len(self._queue)

    def is_empty(self) -> bool:
        """Return True if the queue has no tasks waiting."""
        return len(self._queue) == 0

    def is_full(self) -> bool:
        """Return True if the queue is at capacity."""
        return len(self._queue) >= self._capacity
