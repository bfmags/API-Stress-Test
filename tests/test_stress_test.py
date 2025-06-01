# Test cases for stress_test.py will be added here.
import pytest
import math
# To run tests from the 'tests' directory, ensure that the main project directory
# (containing stress_test.py) is in Python's path.
# Pytest usually handles this if run from the project root.
# If not, uncomment and adjust the following:
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stress_test import pattern_sin, pattern_linear, TRAFFIC_PATTERNS

# General parameters for testing patterns
T_PROGRESS_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_CYCLES = 3 # Default for sin

def test_pattern_output_range():
    """Test that all pattern functions return values within [0.0, 1.0]."""
    patterns_to_test = {
        "sin": pattern_sin,
        "linear": pattern_linear
    }
    for name, func in patterns_to_test.items():
        for t in T_PROGRESS_POINTS:
            if name == "sin":
                val = func(t, cycles=DEFAULT_CYCLES)
            else: # linear
                val = func(t)
            assert 0.0 <= val <= 1.0, f"{name} pattern output {val} out of range for t={t}"

def test_pattern_linear():
    """Test specific values for the linear pattern."""
    assert pattern_linear(0.0) == pytest.approx(0.0)
    assert pattern_linear(0.5) == pytest.approx(0.5)
    assert pattern_linear(1.0) == pytest.approx(1.0)

def test_pattern_sin_values():
    """Test specific values for the sin pattern (default 3 cycles)."""
    # (sin(t * 3 * 2 * pi) + 1) / 2
    # t=0: (sin(0)+1)/2 = 0.5
    # First peak at t = 1 / (4 * cycles)
    # t=0.25 (1/4th of duration, 3/4th of one cycle if 3 cycles total): sin(3*pi/2) = -1 => (-1+1)/2 = 0
    # t=0.5 (half duration, 1.5 cycles): sin(3*pi) = 0 => (0+1)/2 = 0.5
    # t=0.75 (3/4th duration, 2.25 cycles): sin(4.5*pi) = 1 => (1+1)/2 = 1.0
    # t=1.0 (full duration, 3 cycles): sin(6*pi) = 0 => (0+1)/2 = 0.5
    assert pattern_sin(0.0, cycles=DEFAULT_CYCLES) == pytest.approx(0.5)
    assert pattern_sin(1.0 / (4.0 * DEFAULT_CYCLES), cycles=DEFAULT_CYCLES) == pytest.approx(1.0) # Peak of first cycle part
    assert pattern_sin(0.25, cycles=DEFAULT_CYCLES) == pytest.approx(0.0) # 3/4 cycle point for 3 cycles
    assert pattern_sin(0.5, cycles=DEFAULT_CYCLES) == pytest.approx(0.5)
    assert pattern_sin(0.75, cycles=DEFAULT_CYCLES) == pytest.approx(1.0) # Maxima for 3 cycles
    assert pattern_sin(1.0, cycles=DEFAULT_CYCLES) == pytest.approx(0.5)

# --- Tests for worker_loop concurrency calculation ---

def calculate_concurrency_for_test(traffic_pattern_name, t_progress, max_concurrency):
    """
    Helper function to simulate the concurrency calculation logic from worker_loop.
    """
    pattern_function = TRAFFIC_PATTERNS[traffic_pattern_name]

    if traffic_pattern_name == "sin":
        pattern_scaling_factor = pattern_function(t_progress, cycles=DEFAULT_CYCLES)
    else: # linear
        pattern_scaling_factor = pattern_function(t_progress)

    current_concurrency = max(1, int(pattern_scaling_factor * max_concurrency))
    return current_concurrency

@pytest.mark.parametrize("t_progress, expected_factor", [
    (0.0, 0.0), (0.5, 0.5), (1.0, 1.0)
])
def test_worker_loop_concurrency_linear(t_progress, expected_factor):
    max_conc = 100
    expected_conc = max(1, int(expected_factor * max_conc))
    assert calculate_concurrency_for_test("linear", t_progress, max_conc) == expected_conc

@pytest.mark.parametrize("t_progress_ideal_factor_map", [
    (0.0, 0.5), (0.5, 0.5), (1.0, 0.5)
    # (t_progress, idealized_factor for sin pattern with 3 cycles)
])
def test_worker_loop_concurrency_sin(t_progress_ideal_factor_map):
    t_progress, ideal_factor = t_progress_ideal_factor_map
    max_conc = 50

    # Calculate expected concurrency based on the actual pattern_sin output for this t_progress
    # This accounts for floating point nuances before int() conversion.
    actual_calculated_factor = pattern_sin(t_progress, cycles=DEFAULT_CYCLES)
    # We can still assert that the actual_calculated_factor is close to our ideal_factor
    assert actual_calculated_factor == pytest.approx(ideal_factor)

    expected_conc = max(1, int(actual_calculated_factor * max_conc))
    assert calculate_concurrency_for_test("sin", t_progress, max_conc) == expected_conc

def test_worker_loop_concurrency_always_at_least_1():
    """Concurrency should always be at least 1, even if factor is 0."""
    max_conc = 100
    # For sin pattern at t_progress = 0.25 (with 3 cycles), factor is (sin(1.5pi)+1)/2 = 0
    t_progress_yields_zero_factor_sin = 0.25
    assert calculate_concurrency_for_test("sin", t_progress_yields_zero_factor_sin, max_conc) == 1

# --- Test for request timeout ---
import asyncio # For asyncio.sleep if needed in more complex mocks
import httpx # For httpx.ReadTimeout and httpx.AsyncClient

from stress_test import fetch_once, fixed_backoff # For testing fetch_once directly
from stress_test import results as stress_test_results
from stress_test import status_counts as stress_test_status_counts
from stress_test import latencies as stress_test_latencies

@pytest.mark.asyncio
async def test_request_timeout_triggers():
    """
    Tests that the request_timeout parameter in fetch_once is correctly
    passed to the HTTP client and that a timeout is handled as an error.
    """
    test_url = "http://localhost:12345/slow_endpoint" # Dummy URL, server not actually called due to mock
    headers = {}
    short_timeout = 0.05 # 50ms - very short for testing

    # Store original httpx.AsyncClient.get
    original_async_client_get = httpx.AsyncClient.get

    async def mock_get_that_times_out(self, url, headers, timeout):
        # 'self' is the httpx.AsyncClient instance here
        assert timeout == short_timeout, f"Timeout value expected {short_timeout}, got {timeout}"
        # Simulate a timeout by raising the specific exception httpx uses
        raise httpx.ReadTimeout(f"Simulated read timeout on {url}", request=None)

    # Patch the .get method on the class itself
    httpx.AsyncClient.get = mock_get_that_times_out

    # Use a real AsyncClient for context management; its .get method is now our mock
    async with httpx.AsyncClient() as client:
        # Clear global result accumulators from stress_test.py for an isolated test
        stress_test_results.clear()
        stress_test_status_counts.clear()
        stress_test_latencies.clear()

        await fetch_once(
            client=client,
            url=test_url,
            headers=headers,
            retries=0,  # Critical: 0 retries to ensure timeout error is not masked
            backoff_fn=fixed_backoff, # Function itself doesn't matter with 0 retries
            base=0.1,   # Value doesn't matter with 0 retries
            cap=0.1,    # Value doesn't matter with 0 retries
            request_timeout=short_timeout # The timeout being tested
        )

        assert len(stress_test_results) == 1, "One result should be recorded"
        timestamp, latency, status = stress_test_results[0]

        # Latency might be very small or 0 as the request didn't complete
        # assert latency == 0, "Latency should be 0 for a request that timed out before response"
        # Depending on how time.time() is captured, latency might be > 0, so we check status primarily.

        assert "ERR:ReadTimeout" in str(status), f"Status should indicate ReadTimeout, got {status}"
        assert stress_test_status_counts[status] == 1, "Status count for ReadTimeout error should be 1"
        assert not stress_test_latencies, "Latencies list should be empty for failed request"

    # Restore the original .get method to avoid affecting other tests
    httpx.AsyncClient.get = original_async_client_get
