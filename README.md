# API Stress-Test

## Features
- Async + HTTPX for high throughput
- Backoff strategies (fixed, exponential, decorrelated jitter)
- Real-time plotting: cumulative requests, avg/p50/p90/p99 latencies, error rate
- Run by count, duration, or Ctrl+C
- CSV & PNG export plot graphs

### Backoff Strategies
- **Fixed**: sleep a constant `base` seconds each retry.
- **Exponential**: sleep `base * 2**(attempt-1)`, capped at `cap`.
- **Decorrelated Jitter**: random between `base` and `prev_sleep * 3`, capped.

### Real-Time Plotting (Optional)
When enabled (default), the tool displays:
- **Cumulative requests**
- **Avg latency**, **p50**, **p90**, **p99**
- **Status counts** & **error rate** (failures/total)

## Understanding Latency Metrics
The latency metrics help you understand the performance characteristics of your API under load:
- **Average Latency (avg):** The mean response time of all successful requests. While useful, it can be skewed by a small number of very slow or very fast requests.
- **p50 Latency (Median):** The value below which 50% of the successful request latencies fall. This represents the typical latency experienced by users and is often more representative than the average.
- **p90 Latency:** The value below which 90% of the successful request latencies fall. This helps understand the tail latency for the majority of requests, indicating what a typical "worst-case" experience might be for most users.
- **p99 Latency:** The value below which 99% of the successful request latencies fall. This indicates the latency experienced by the slowest 1% of requests, highlighting extreme "worst-case" scenarios.

## Setup
```
git clone <repository_url>
cd <repository_directory_name>
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage
Example:
```
python stress_test.py -f api_keys.txt -n 2000 -c 100 \
  --traffic-pattern linear -e /users --base-url https://api.example.com \
  --retries 5 --backoff-method jitter --backoff-base 0.5 \
  --backoff-cap 20 --duration 60 \
  --export-csv data.csv --no-live-plot
```

### Command-Line Options
- `-f, --api-keys-file PATH`: File with one API key per line.
- `-k, --api-keys TEXT`: Comma-separated API keys.
- `-n, --total-requests INTEGER`: Total number of requests to send. If omitted, the test runs for the specified `--duration` or until manually interrupted (if plotting is also disabled).
- `-c, --concurrency INTEGER`: Number of concurrent requests to maintain. Default: 50.
- --traffic-pattern [linear|sin]: Defines the pattern of requests over time. Default: `linear`.
    - `linear`: A linear ramp-up of traffic. This is the default behavior if `--traffic-pattern` is not specified.
    - `sin`: Traffic follows a sine wave pattern. The load will oscillate, completing a few cycles over the test duration.
- `-e, --endpoint TEXT`: API endpoint path (e.g., `/users`). Required.
- `--base-url TEXT`: API base URL (e.g., `http://127.0.0.1:8000`). Default: `http://127.0.0.1:8000`.
- `--retries INTEGER`: Number of retries for failed requests. Default: 3.
- `--backoff-method [fixed|exponential|jitter]`: Retry backoff strategy. Default: `exponential`.
- `--backoff-base FLOAT`: Base time for backoff in seconds. Default: 1.0.
- `--backoff-cap FLOAT`: Maximum backoff time in seconds. Default: 30.0.
- `--duration INTEGER`: Duration to run the test in seconds. If `total-requests` is also set, the test stops when either limit is reached.
- `--live-plot / --no-live-plot`: Enable or disable the live plotting GUI window. Defaults to enabled (`--live-plot`). If disabled (`--no-live-plot`), the test relies on `--duration` or `--total-requests` for completion, or manual interruption (Ctrl+C if those are not set).
- `--export-csv PATH`: Export raw metrics (timestamp, latency, status) to a CSV file.

## Mock Server for Local Testing

Included is a simple **FastAPI** mock server you can run locally to test the stress-test client.
It simulates random latency and occasional errors. You can configure its behavior using the following environment variables when starting the server:

-   `MIN_LATENCY`: Minimum simulated latency in seconds (float, default: 0.01).
-   `MAX_LATENCY`: Maximum simulated latency in seconds (float, default: 0.5).
-   `ERROR_RATE`: Probability of the server returning an error (float, 0.0 to 1.0, default: 0.1, meaning 10% error rate).

Example of running the mock server with custom settings:
```bash
MIN_LATENCY=0.1 MAX_LATENCY=0.3 ERROR_RATE=0.05 python mock_server.py
```

Start Mock Server: 
- `python mock_server.py`

Example Stress Test against Mock Server:
- `python stress_test.py -k your_api_key -n 1000 -c 50 --traffic-pattern linear -e /test --base-url http://127.0.0.1:8000 --duration 30 --export-csv mock_test_data.csv`
