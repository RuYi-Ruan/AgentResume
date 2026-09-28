import pytest
import requests
from client.api import APIClient
from client.exceptions import APIError, parse_error_response


def test_invalid_body_raises_api_error(go_server):
    """Sending invalid JSON must raise APIError with status 422."""
    with pytest.raises(APIError) as exc_info:
        import requests as _req
        resp = _req.post(
            f"{go_server}/users",
            data="not-json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code >= 400:
            raise parse_error_response(resp)
    err = exc_info.value
    assert err.status_code == 422, (
        f"Expected status 422, got {err.status_code} -- "
        "check that parse_error_response handles HTTP 422"
    )


def test_error_message_is_string(go_server):
    """The error message extracted from the response must be a non-empty string."""
    import requests as _req
    resp = _req.post(
        f"{go_server}/users",
        data="bad",
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    body = resp.json()
    assert 'errors' in body, f"Response missing 'errors' key: {body}"
    assert isinstance(body["errors"], list), "errors must be an array"
    assert len(body["errors"]) > 0, "errors array must be non-empty"


def test_parse_error_response_422():
    """Unit test: parse_error_response must handle a 422 with errors array."""

    class FakeResponse:
        status_code = 422

        def json(self):
            return {"errors": ["Invalid JSON body", "Request body required"]}

    err = parse_error_response(FakeResponse())
    assert err.status_code == 422
    assert "Invalid JSON" in err.message or "body" in err.message.lower()
