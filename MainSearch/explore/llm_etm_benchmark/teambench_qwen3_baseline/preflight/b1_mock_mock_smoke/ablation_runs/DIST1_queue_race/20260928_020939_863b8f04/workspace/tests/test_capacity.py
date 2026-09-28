"""
Capacity enforcement tests for the task queue under concurrent load.

Verifies Bug 1 (TOCTOU) is fixed: concurrent puts must never push the queue
past its declared capacity.
"""
import threading
import pytest

from mqueue.queue import TaskQueue, QueueFull


def test_capacity_not_exceeded_under_concurrent_puts():
    """
    5 threads each try 40 puts into a capacity-10 queue.
    After all threads finish, queue size must be <= 10.
    """
    capacity = 10
    q = TaskQueue(capacity=capacity)
    errors = []

    def stuffer():
        for _ in range(40):
            try:
                q.put("task_item")
            except QueueFull:
                pass
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=stuffer) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Unexpected errors: {errors}"
    assert q.size() <= capacity, (
        f"Capacity violated: size={q.size()}, capacity={capacity}"
    )


def test_capacity_not_exceeded_high_contention():
    """20 threads hammering a capacity-5 queue must never exceed 5 items."""
    capacity = 5
    q = TaskQueue(capacity=capacity)
    peak_sizes = []
    lock = threading.Lock()

    def worker():
        for _ in range(100):
            try:
                q.put("task")
            except QueueFull:
                pass
            with lock:
                peak_sizes.append(q.size())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    max_seen = max(peak_sizes) if peak_sizes else 0
    assert max_seen <= capacity, (
        f"Capacity {capacity} violated: peak size was {max_seen}"
    )
