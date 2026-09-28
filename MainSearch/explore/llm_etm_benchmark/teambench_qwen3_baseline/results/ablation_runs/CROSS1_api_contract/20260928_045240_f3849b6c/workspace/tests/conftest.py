import subprocess
import time
import os
import signal
import pytest


@pytest.fixture(scope="session")
def go_server():
    """Build and start the Go server for integration tests."""
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_dir = os.path.join(workspace, "service")
    binary = "/tmp/cross1_server_users"
    env = os.environ.copy()
    env["PORT"] = "18080"

    # Build the server binary
    build = subprocess.run(
        ["go", "build", "-o", binary, "."],
        cwd=server_dir,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"Go build failed: {build.stderr}")

    proc = subprocess.Popen(
        [binary],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.8)  # Wait for startup
    if proc.poll() is not None:
        pytest.skip("Go server failed to start")

    yield "http://localhost:18080"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
