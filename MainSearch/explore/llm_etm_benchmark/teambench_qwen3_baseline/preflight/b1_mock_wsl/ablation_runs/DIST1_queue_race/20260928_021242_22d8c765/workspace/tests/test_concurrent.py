"""
Concurrent correctness tests for the task queue.

These tests FAIL with the buggy implementation because they exercise
race conditions that single-threaded tests cannot expose.
"""
import threading
import time
import pytest

from mqueue.queue import TaskQueue, QueueFull
from mqueue.config import N_PRODUCERS, N_CONSUMERS, MESSAGES_PER_PRODUCER


def test_concurrent_no_message_loss():
    """
    Multiple producers and consumers, all messages must arrive with zero loss.

    Exposes Bug 2 (no ack): if a consumer gets a message but crashes before
    processing, the message is gone. We simulate this by tracking all sent
    and received messages.
    """
    total_messages = N_PRODUCERS * MESSAGES_PER_PRODUCER
    q = TaskQueue(capacity=total_messages)

    sent = []
    received = []
    sent_lock = threading.Lock()
    recv_lock = threading.Lock()
    done_event = threading.Event()

    def producer(pid: int):
        for i in range(MESSAGES_PER_PRODUCER):
            msg = f"p{pid}-task{i}"
            with sent_lock:
                sent.append(msg)
            q.put(msg)
            time.sleep(0.00005)

    def consumer():
        while not done_event.is_set() or not q.is_empty():
            result = q.get()
            if isinstance(result, tuple):
                msg, receipt = result
                if msg is None:
                    time.sleep(0.001)
                    continue
                # ack if the queue supports it
                if hasattr(q, 'ack') and receipt is not None:
                    q.ack(receipt)
            else:
                msg = result
                if msg is None:
                    time.sleep(0.001)
                    continue
            with recv_lock:
                received.append(msg)

    threads = []
    for i in range(N_PRODUCERS):
        threads.append(threading.Thread(target=producer, args=(i,), daemon=False))
    consumer_threads = [
        threading.Thread(target=consumer, daemon=True)
        for _ in range(N_CONSUMERS)
    ]

    for t in consumer_threads:
        t.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    done_event.set()
    time.sleep(0.5)  # Let consumers drain

    assert len(received) == total_messages, (
        f"Message loss: sent {total_messages}, received {len(received)}"
    )


def test_capacity_never_exceeded():
    """
    Concurrent puts must not push the queue past capacity.

    Exposes Bug 1 (TOCTOU): two producers both check capacity, both see room,
    both append — queue exceeds declared capacity.
    """
    capacity = 20
    q = TaskQueue(capacity=capacity)
    errors = []

    def aggressive_producer():
        for _ in range(50):
            try:
                q.put(f"task_item")
            except QueueFull:
                pass
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=aggressive_producer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Unexpected errors during concurrent puts: {errors}"
    assert q.size() <= capacity, (
        f"Capacity violated: queue size {q.size()} exceeds capacity {capacity}"
    )
