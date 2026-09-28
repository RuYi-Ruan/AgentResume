"""
Zero-loss guarantee test for the task queue.

Sends 10,000 tasks across 20 threads and verifies every one is received.
"""
import threading
import time

from mqueue.queue import TaskQueue, QueueFull


def test_zero_message_loss_10k():
    """10K tasks across 20 threads must arrive with zero loss."""
    N = 10_000
    n_producers = 10
    n_consumers = 10
    per_producer = N // n_producers

    q = TaskQueue(capacity=N)
    sent = []
    received = []
    s_lock = threading.Lock()
    r_lock = threading.Lock()
    stop = threading.Event()

    def producer(pid: int):
        for i in range(per_producer):
            msg = f"task-{pid}-{i}"
            retry = 0
            while retry < 10:
                try:
                    q.put(msg)
                    with s_lock:
                        sent.append(msg)
                    break
                except QueueFull:
                    time.sleep(0.001)
                    retry += 1

    def consumer():
        while not stop.is_set() or not q.is_empty():
            result = q.get()
            if isinstance(result, tuple):
                msg, receipt = result
                if msg is None:
                    time.sleep(0.0005)
                    continue
                if hasattr(q, 'ack') and receipt is not None:
                    q.ack(receipt)
            else:
                msg = result
                if msg is None:
                    time.sleep(0.0005)
                    continue
            with r_lock:
                received.append(msg)

    producer_threads = [
        threading.Thread(target=producer, args=(i,)) for i in range(n_producers)
    ]
    consumer_threads = [
        threading.Thread(target=consumer, daemon=True) for _ in range(n_consumers)
    ]

    for t in consumer_threads:
        t.start()
    for t in producer_threads:
        t.start()
    for t in producer_threads:
        t.join(timeout=120)

    stop.set()
    time.sleep(1.0)

    assert len(sent) == N, f"Not all tasks were sent: {len(sent)}/{N}"
    assert len(received) == len(sent), (
        f"Message loss: sent {len(sent)}, received {len(received)}"
    )
