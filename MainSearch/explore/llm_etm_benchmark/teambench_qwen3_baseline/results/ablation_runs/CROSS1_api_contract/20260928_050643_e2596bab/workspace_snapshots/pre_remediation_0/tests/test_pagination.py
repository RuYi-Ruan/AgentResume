import pytest
from client.api import APIClient


def test_pagination_page1(go_server):
    client = APIClient(go_server)
    items, cursor = client.list_users(page=1)
    assert cursor is not None, "page=1 must return a next cursor"
    assert cursor.startswith("cursor_"), f"Unexpected cursor format: {cursor!r}"


def test_pagination_page2(go_server):
    client = APIClient(go_server)
    items1, cursor1 = client.list_users(page=1)
    items2, cursor2 = client.list_users(page=2)
    assert cursor1 != cursor2, "Cursors for different pages must differ"


def test_list_returns_list(go_server):
    client = APIClient(go_server)
    items, cursor = client.list_users()
    assert isinstance(items, list), f"Expected list, got {type(items).__name__}"
