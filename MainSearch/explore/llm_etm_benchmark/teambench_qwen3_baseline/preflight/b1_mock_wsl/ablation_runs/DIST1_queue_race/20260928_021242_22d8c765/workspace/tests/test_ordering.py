"""
Priority ordering tests for the task priority queue.

Verifies that urgency ordering is correct and that equal-urgency
messages do not cause TypeError (Bug 3).
"""
import heapq
import pytest

# Import the priority class — name varies by seed
try:
    from mqueue.priority import PriorityTask
    _PRIO_CLS = PriorityTask
except ImportError:
    _PRIO_CLS = None


def _make_item(prio, seq, msg):
    """Construct a priority item, handling both buggy (2-field) and fixed (3-field) class."""
    try:
        return _PRIO_CLS(urgency=prio, seq=seq, message=msg)
    except TypeError:
        # Buggy version has only two fields: urgency and message
        return _PRIO_CLS(urgency=prio, message=msg)


@pytest.mark.skipif(_PRIO_CLS is None, reason="Priority class not found")
def test_higher_priority_comes_first():
    """Lower urgency number must be dequeued first (string payloads — comparable in both versions)."""
    heap = []
    for prio in [5, 1, 3, 0, 4]:
        item = _make_item(prio, prio, f"task-prio{prio}")
        heapq.heappush(heap, item)

    priorities = []
    while heap:
        item = heapq.heappop(heap)
        priorities.append(item.urgency)

    assert priorities == sorted(priorities), (
        f"Priority ordering wrong: {priorities}"
    )


@pytest.mark.skipif(_PRIO_CLS is None, reason="Priority class not found")
def test_equal_priority_no_type_error_with_dict_payload():
    """Equal-urgency tasks with dict payloads must not raise TypeError."""
    heap = []
    for i in range(5):
        item = _make_item(1, i, {"task_id": i, "data": [i, i + 1]})
        heapq.heappush(heap, item)  # Buggy version raises TypeError here

    results = []
    while heap:
        results.append(heapq.heappop(heap))

    assert len(results) == 5


@pytest.mark.skipif(_PRIO_CLS is None, reason="Priority class not found")
def test_equal_priority_no_type_error_with_list_payload():
    """Equal-urgency tasks with list payloads must not raise TypeError."""
    heap = []
    for i in range(3):
        item = _make_item(2, i, [i, "data", {}])
        heapq.heappush(heap, item)

    while heap:
        heapq.heappop(heap)  # Must not raise
