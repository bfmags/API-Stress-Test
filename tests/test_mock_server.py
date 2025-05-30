import pytest
import asyncio
import time
import os
import importlib # Added
from fastapi.testclient import TestClient
# Assuming your FastAPI app instance is named 'app' in mock_server.py
import mock_server # Changed import to allow reload
app = mock_server.app # Explicitly assign app

@pytest.fixture
def mock_env_vars(monkeypatch):
    def _set_vars(min_latency=None, max_latency=None, error_rate=None):
        # Set environment variables
        if min_latency is not None:
            monkeypatch.setenv("MIN_LATENCY", str(min_latency))
        else:
            monkeypatch.delenv("MIN_LATENCY", raising=False)
        if max_latency is not None:
            monkeypatch.setenv("MAX_LATENCY", str(max_latency))
        else:
            monkeypatch.delenv("MAX_LATENCY", raising=False)
        if error_rate is not None:
            monkeypatch.setenv("ERROR_RATE", str(error_rate))
        else:
            monkeypatch.delenv("ERROR_RATE", raising=False)

        # Reload mock_server to pick up changes in environment variables
        importlib.reload(mock_server)
        # Update the global 'app' instance for TestClient, as reload creates a new module object
        global app
        app = mock_server.app
        # Also update the client's app instance directly.
        # This is crucial because TestClient might hold a reference to the old app object.
        client.app = app

    return _set_vars

# client must be initialized after mock_server (and app) potentially reloaded.
# However, client is typically defined globally.
# We will re-assign client.app inside the fixture.
client = TestClient(app)


def test_default_behavior(mock_env_vars):
    mock_env_vars() # Clear any set env vars and reload module
    latencies = []
    errors = 0
    num_requests = 200
    for _ in range(num_requests):
        start_time = time.perf_counter()
        response = client.get("/test")
        end_time = time.perf_counter()
        if response.status_code != 200:
            errors += 1
        else:
            latencies.append(end_time - start_time)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    error_percentage = errors / num_requests

    # Default: min_latency=0.01, max_latency=0.5, error_rate=0.1
    if latencies:
        assert avg_latency >= 0.01 - 0.005, f"Default average latency {avg_latency} too low"
        # Increased upper buffer slightly for avg_latency due to potential system variance
        assert avg_latency <= 0.5 + 0.075, f"Default average latency {avg_latency} too high"
        for l_val in latencies:
            assert l_val >= 0.01 - 0.005, f"Individual latency {l_val} too low for default"
            # Increased upper buffer for individual latency to account for system variance
            assert l_val <= 0.5 + 0.125, f"Individual latency {l_val} too high for default (max allowed: 0.625)"

    assert 0.05 <= error_percentage <= 0.15, f"Default error rate {error_percentage} not close to 0.1 (expected range 0.05-0.15)"

def test_custom_latency(mock_env_vars):
    min_lat, max_lat = 0.1, 0.2
    mock_env_vars(min_latency=min_lat, max_latency=max_lat, error_rate=0)

    latencies = []
    num_requests = 50
    for _ in range(num_requests):
        start_time = time.perf_counter()
        response = client.get("/test")
        end_time = time.perf_counter()
        assert response.status_code == 200, "Status code should be 200 for latency test with 0 error rate"
        request_latency = end_time - start_time
        latencies.append(request_latency)
        # Increased upper buffer slightly
        assert min_lat - 0.005 <= request_latency <= max_lat + 0.075, f"Individual latency {request_latency} out of configured range [{min_lat}, {max_lat}] with buffer"

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    if latencies:
      # Increased upper buffer slightly
      assert min_lat - 0.005 <= avg_latency <= max_lat + 0.075, f"Average latency {avg_latency} out of configured range [{min_lat}, {max_lat}] with buffer"

def test_custom_error_rate(mock_env_vars):
    error_rt = 0.5
    mock_env_vars(min_latency=0.01, max_latency=0.01, error_rate=error_rt)

    errors = 0
    num_requests = 300
    for _ in range(num_requests):
        response = client.get("/test")
        if response.status_code != 200:
            errors += 1
            assert response.status_code == 500, f"Expected status code 500 for errors, got {response.status_code}"

    error_percentage = errors / num_requests
    # Increased tolerance for error rate due to statistical variance.
    assert error_rt - 0.12 <= error_percentage <= error_rt + 0.12, f"Error rate {error_percentage} not close to configured {error_rt}. This can be flaky due to randomness; consider re-running."

def test_zero_error_rate(mock_env_vars):
    mock_env_vars(min_latency=0.01, max_latency=0.01, error_rate=0)

    num_requests = 100
    for _ in range(num_requests):
        response = client.get("/test")
        assert response.status_code == 200, "Status code should always be 200 when error_rate is 0"
