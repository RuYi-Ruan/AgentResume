from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    user_id: int  # BUG: server sends "userId" (camelCase)
    name: str
    email: str

    @classmethod
def from_dict(cls, data: dict) -> 'User':
    return cls(
        user_id=data.get("userId"),  # Fixed to match server's camelCase format
        name=data.get("name", ""),
        email=data.get("email", ""),
    )