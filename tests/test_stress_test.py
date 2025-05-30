import pytest
from click.testing import CliRunner
from stress_test import main as stress_test_main # Assuming 'main' is the click entry point
from stress_test import fixed_backoff, exp_backoff, jitter_backoff
import random
# New imports
import asyncio
from collections import Counter, deque
import time
from unittest import mock # Use unittest.mock for async mocking if needed
import httpx # Required for Response and RequestError
import stress_test # Import the module itself to access its global variables and functions


def test_fixed_backoff():
    assert fixed_backoff(1, 1.0, 10.0) == 1.0
    assert fixed_backoff(5, 2.0, 10.0) == 2.0

def test_exp_backoff():
    assert exp_backoff(1, 1.0, 10.0) == 1.0  # base * 2**(1-1) = 1
    assert exp_backoff(2, 1.0, 10.0) == 2.0  # base * 2**(2-1) = 2
    assert exp_backoff(3, 1.0, 10.0) == 4.0  # base * 2**(3-1) = 4
    assert exp_backoff(4, 1.0, 10.0) == 8.0  # base * 2**(4-1) = 8
    assert exp_backoff(5, 1.0, 10.0) == 10.0 # Capped at 10.0 (base * 2**(5-1) = 16)

def test_jitter_backoff():
    # Seed random for predictable jitter in tests
    random.seed(0) # seed used: 0
    base = 1.0
    cap = 30.0
    # First attempt (prev is None)
    val1 = jitter_backoff(1, base, cap)
    assert base <= val1 <= base * 3

    # Subsequent attempts
    prev_sleep = val1
    val2 = jitter_backoff(2, base, cap, prev_sleep)
    assert base <= val2 <= prev_sleep * 3

    # Test cap
    high_prev_sleep = cap / 2
    val_capped = jitter_backoff(3, base, cap, high_prev_sleep)
    assert val_capped <= cap

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_missing_keys_or_file(runner):
    result = runner.invoke(stress_test_main, ['-e', '/test'])
    assert result.exit_code != 0 # Expect error
    assert "Error: provide -f or -k" in result.output

def test_cli_both_keys_and_file(runner, tmp_path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("key1")
    result = runner.invoke(stress_test_main, ['-f', str(key_file), '-k', 'key1,key2', '-e', '/test'])
    assert result.exit_code != 0 # Expect error
    assert "Error: choose only one of -f/-k" in result.output

def test_cli_no_api_keys_in_file(runner, tmp_path):
    key_file = tmp_path / "empty_keys.txt"
    key_file.write_text("") # Empty file
    result = runner.invoke(stress_test_main, ['-f', str(key_file), '-e', '/test'])
    assert result.exit_code != 0
    assert "No API keys!" in result.output

# Test for endpoint requirement
def test_cli_missing_endpoint(runner):
    result = runner.invoke(stress_test_main, ['-k', 'somekey'])
    assert result.exit_code != 0
    # Click's default error message for missing required options
    assert "Error: Missing option '-e' / '--endpoint'." in result.output

# A minimal successful invocation (won't actually run async loop, just parse args)
# We'll need to mock the actual worker loop for deeper tests later.
def test_cli_minimal_success_parse(runner, monkeypatch):
    # Prevent the actual worker loop and plotting from running for this parsing test
    monkeypatch.setattr("stress_test.run_async_tasks_in_thread", lambda *args, **kwargs: None)
    monkeypatch.setattr("stress_test.start_plot", lambda *args, **kwargs: None)
    # Also mock print_summary_stats as it might be called even if loop doesn't run,
    # depending on how main is structured after loop.
    monkeypatch.setattr("stress_test.print_summary_stats", lambda *args, **kwargs: None)

    result = runner.invoke(stress_test_main, ['-k', 'key1', '-e', '/test', '--no-live-plot'])
    assert result.exit_code == 0, f"CLI parsing failed: {result.output}"
    # We can add more assertions here if main() returned specific values or had side effects
    # For now, exit_code 0 is enough to show basic parsing and option handling worked.

# --- New tests below ---

def test_print_summary_stats_no_results(capsys):
    results_deque = deque()
    latencies_list = []
    status_counts_counter = Counter()
    stress_test.print_summary_stats(results_deque, latencies_list, status_counts_counter)
    captured = capsys.readouterr()
    assert "No requests were completed." in captured.out

def test_print_summary_stats_with_data(capsys):
    # (timestamp, latency, status)
    results_deque = deque([
        (time.time(), 0.1, 200),
        (time.time(), 0.2, 200),
        (time.time(), 0.0, "ERR:ConnectError") # 0 latency for error
    ])
    latencies_list = [0.1, 0.2] # Only successful latencies
    status_counts_counter = Counter({200: 2, "ERR:ConnectError": 1})

    stress_test.print_summary_stats(results_deque, latencies_list, status_counts_counter)
    captured = capsys.readouterr()

    assert "Total Requests Attempted (approx.): 3" in captured.out
    assert "Successful Requests: 2" in captured.out
    assert "Failed Requests (Errors): 1" in captured.out
    assert "Error Rate: 33.33%" in captured.out # 1/3
    assert "Average Latency (successful requests): 0.1500 s" in captured.out # (0.1+0.2)/2
    # Note: p50 for 2 items [0.1, 0.2] is 0.1 (the first element after sorting at index int(0.5*2-1)=0 if 0-indexed, or simply the lower of the two for even N)
    # The current implementation `lat[int(0.5 * len(lat))]` for len=2 becomes `lat[1]` which is 0.2.
    # If we want p50 to be the lower for even, it should be `lat[int(0.5 * len(lat)) -1]` for even or average.
    # The provided code for p50 `lat[int(0.5 * len(lat))]` is a common way to get upper median for even lists.
    # For [0.1, 0.2], p50 will be 0.2 using this. Let's adjust the test to expect this or clarify p50 definition for 2 items.
    # Given stress_test.py logic: `p50 = lat[int(0.5 * len(lat))]` -> `lat[1]` -> 0.2
    # if len(lat) == 1, p50=avg. If len(lat) >=2, this formula is used.
    # For [0.1, 0.2], p50=0.2, p90=0.2, p99=0.2
    assert "p50 Latency (Median): 0.2000 s" in captured.out
    assert "p90 Latency: 0.2000 s" in captured.out
    assert "p99 Latency: 0.2000 s" in captured.out
    assert "  200: 2" in captured.out
    assert "  ERR:ConnectError: 1" in captured.out

# Mock for httpx.Response
class MockHTTPXResponse:
    def __init__(self, status_code, content=None, text=""):
        self.status_code = status_code
        self.content = content
        self.text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("Error", request=None, response=self)

# Mock for httpx.AsyncClient
class MockAsyncClient:
    def __init__(self, behavior="success", no_internal_sleep=False): # Added no_internal_sleep
        self.behavior = behavior # "success", "fail_once", "always_fail", "timeout"
        self.request_count = 0
        self.no_internal_sleep = no_internal_sleep # Flag to control internal sleep

    async def get(self, url, headers, timeout):
        self.request_count += 1
        # Simulate network delay only if not skipping internal sleeps
        # Use a very small, almost negligible delay if not skipping, to represent processing
        simulated_delay = 0.0001

        if self.behavior == "success":
            if not self.no_internal_sleep: await asyncio.sleep(simulated_delay)
            return MockHTTPXResponse(200, content={"status": "ok"})
        elif self.behavior == "fail_once" and self.request_count == 1:
            # Errors typically don't have deliberate sleep
            # if not self.no_internal_sleep: await asyncio.sleep(simulated_delay)
            raise httpx.RequestError("Simulated network error", request=None)
        elif self.behavior == "fail_once" and self.request_count > 1:
            if not self.no_internal_sleep: await asyncio.sleep(simulated_delay)
            return MockHTTPXResponse(200, content={"status": "ok_after_retry"})
        elif self.behavior == "always_fail":
            # Errors typically don't have deliberate sleep
            # if not self.no_internal_sleep: await asyncio.sleep(simulated_delay)
            raise httpx.RequestError("Simulated persistent network error", request=None)
        elif self.behavior == "timeout":
            if not self.no_internal_sleep:
                await asyncio.sleep(timeout + 0.1) # Sleep longer than timeout to simulate it
            # If no_internal_sleep is True, timeout is effectively instant for this mock's sleep part
            # but the ReadTimeout exception itself implies the timeout duration was exceeded.
            raise httpx.ReadTimeout("Simulated read timeout", request=None)
        # Add more behaviors if needed

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

@pytest.mark.asyncio
async def test_fetch_once_success():
    stress_test.results.clear()
    stress_test.status_counts.clear()
    stress_test.latencies.clear()
    stress_test.stop_event.clear()

    # For this test, internal sleeps in mock_client don't matter as much as we don't mock asyncio.sleep
    # However, good practice to be explicit if a test assumes no internal sleeps.
    mock_client = MockAsyncClient(behavior="success", no_internal_sleep=True)
    await stress_test.fetch_once(mock_client, "http://test.com/api", {},
                                 retries=3, backoff_fn=stress_test.fixed_backoff,
                                 base=0.01, cap=0.1)

    assert len(stress_test.results) == 1
    timestamp, latency, status = stress_test.results[0]
    assert status == 200
    # With no_internal_sleep=True in MockAsyncClient, latency will be very small (just processing time)
    assert latency >= 0 and latency < 0.01, f"Latency {latency} was not small and positive."
    assert stress_test.status_counts[200] == 1
    assert len(stress_test.latencies) == 1

@pytest.mark.asyncio
async def test_fetch_once_retry_then_success():
    stress_test.results.clear()
    stress_test.status_counts.clear()
    stress_test.latencies.clear()
    stress_test.stop_event.clear()

    mock_client = MockAsyncClient(behavior="fail_once", no_internal_sleep=True)
    with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep:
        await stress_test.fetch_once(mock_client, "http://test.com/api", {},
                                     retries=3, backoff_fn=stress_test.fixed_backoff,
                                     base=0.01, cap=0.1)

    assert mock_client.request_count == 2 # 1 fail, 1 success
    assert mock_sleep.call_count == 1 # Called once for backoff
    assert len(stress_test.results) == 1
    _, _, status = stress_test.results[0]
    assert status == 200
    assert stress_test.status_counts[200] == 1

@pytest.mark.asyncio
async def test_fetch_once_persistent_failure():
    stress_test.results.clear()
    stress_test.status_counts.clear()
    stress_test.latencies.clear()
    stress_test.stop_event.clear()

    mock_client = MockAsyncClient(behavior="always_fail", no_internal_sleep=True)
    with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep: # Typo fixed
        await stress_test.fetch_once(mock_client, "http://test.com/api", {},
                                     retries=2, backoff_fn=stress_test.fixed_backoff,
                                     base=0.01, cap=0.1)

    assert mock_client.request_count == 3 # 1 initial + 2 retries
    assert mock_sleep.call_count == 2 # Called for each retry
    assert len(stress_test.results) == 1
    _, latency, status_str = stress_test.results[0]
    assert status_str.startswith("ERR:RequestError")
    # Latency is recorded as 0 in results deque due to `latency or 0`
    assert latency == 0, "Latency for a fully failed request should be recorded as 0 in results."
    assert stress_test.status_counts[status_str] == 1
    assert len(stress_test.latencies) == 0 # Ensure it wasn't added to actual latencies list
