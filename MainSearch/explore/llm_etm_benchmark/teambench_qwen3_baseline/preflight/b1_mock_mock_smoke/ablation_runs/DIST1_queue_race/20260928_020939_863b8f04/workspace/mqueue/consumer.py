"""Consumer interface for the task queue.

This consumer does NOT use an acknowledgment pattern — it calls get() and
immediately proceeds with processing. If the consumer crashes mid-processing,
the task is permanently lost.

The consumer needs to be updated to use ack/nack once the queue supports it.
"""
import threading
import time
from typing import Any, Callable, Optional

from mqueue.queue import TaskQueue


class TaskConsumer:
    """
    Consumes tasks from a TaskQueue.

    Current implementation: fire-and-forget (no ack/nack).
    After fixing Bug 2, this should use the ack/nack pattern.
    """

    def __init__(
        self,
        queue: TaskQueue,
        handler: Callable[[Any], None],
        consumer_id: int = 0,
    ):
        self._queue = queue
        self._handler = handler
        self._consumer_id = consumer_id
        self._processed: list = []
        self._lock = threading.Lock()
        self._running = False

    def run_once(self) -> bool:
        """
        Process one task from the queue.

        Returns True if a task was processed, False if the queue was empty.
        """
        # No ack pattern: message removed from queue before handler called.
        # If self._handler raises, the task is permanently lost.
        message = self._queue.get()
        if message is None:
            return False
        self._handler(message)
        with self._lock:
            self._processed.append(message)
        return True

    def run_until_empty(self, max_idle_cycles: int = 10) -> None:
        """Drain the queue, stopping after max_idle_cycles consecutive empty polls."""
        idle = 0
        while idle < max_idle_cycles:
            if self.run_once():
                idle = 0
            else:
                idle += 1
                time.sleep(0.001)

    @property
    def processed_count(self) -> int:
        return len(self._processed)

    @property
    def processed_messages(self) -> list:
        with self._lock:
            return list(self._processed)
