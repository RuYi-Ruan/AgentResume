class APIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


def parse_error_response(response) -> APIError:
    if response.status_code == 400:
        data = response.json()
        return APIError(400, data.get("error", "Unknown error"))
    return APIError(response.status_code, "HTTP error")
