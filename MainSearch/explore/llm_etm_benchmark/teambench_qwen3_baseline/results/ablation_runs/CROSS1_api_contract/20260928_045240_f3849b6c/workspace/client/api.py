import requests
from typing import List, Optional, Tuple


class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def list_users(self, page: int = 1) -> Tuple[List, Optional[str]]:
        resp = requests.get(
            f"{self.base_url}/users",
            params={"page": page},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        from client.models import User
        # BUG: server sends "data" and "next", not "results" and "cursor"
        items = [
            User.from_dict(u)
            for u in data.get("results", [])   # BUG: should be "data"
        ]
        next_cursor = data.get("cursor")        # BUG: should be "next"
        return items, next_cursor

    def create_user(self, payload: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/users",
            json=payload,
            timeout=10,
        )
        if resp.status_code >= 400:
            from client.exceptions import parse_error_response
            raise parse_error_response(resp)
        return resp.json()
