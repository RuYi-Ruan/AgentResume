import pytest
from client.api import APIClient


def test_list_users_returns_items(go_server):
    client = APIClient(go_server)
    items, cursor = client.list_users()
    assert len(items) > 0, "Expected at least one user in response"


def test_list_users_id_is_set(go_server):
    client = APIClient(go_server)
    items, cursor = client.list_users()
    assert items[0].user_id is not None, (
        f"Expected {items[0].user_id!r} to be set — "
        "check that from_dict maps the correct camelCase key from the server"
    )
    assert items[0].user_id > 0, "ID must be a positive integer"


def test_list_users_cursor_present(go_server):
    client = APIClient(go_server)
    items, cursor = client.list_users(page=1)
    assert cursor is not None, (
        "Pagination cursor must not be None — "
        "check that api.py reads the correct key from the server response"
    )
    assert len(cursor) > 0, "Pagination cursor must be non-empty"
