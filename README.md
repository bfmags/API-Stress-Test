# API Stress-Test

## Features
- Async + HTTPX for high throughput
- Backoff strategies (fixed, exponential, decorrelated jitter)
- Real-time plotting: cumulative requests, avg/p50/p90/p99 latencies, error rate
- Run by count, duration, or user keypress
- CSV & PNG export plot graphs

### Backoff Strategies
- **Fixed**: sleep a constant `base` seconds each retry.
- **Exponential**: sleep `base * 2**(attempt-1)`, capped at `cap`.
- **Decorrelated Jitter**: random between `base` and `prev_sleep * 3`, capped.

### Real-Time Plotting
- **Cumulative requests**
- **Avg latency**, **p50**, **p90**, **p99**
- **Status counts** & **error rate** (failures/total)

## Setup
```
git clone ...
cd api-stress-test
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Requirements
* click
* httpx[http2]
* matplotlib

*Mock Server*
* fastapi
* uvicorn

## Usage
```
python stress_test.py -f api_keys.txt -n 2000 -c 100 --crescendo 
  -e /users --base-url https://api.example.com 
  --retries 5 --backoff-method jitter --backoff-base 0.5 
  --backoff-cap 20 --duration 60 --stop-on-key 
  --export-csv data.csv
```

- `-f`/`-k`: API keys
- `-n`: total requests
- `-c`: concurrency
- `--crescendo`: ramp
- `--duration`: seconds to run
- `--stop-on-key`: Enter to end
- `--export-csv`: raw metrics

## Mock Server for Local Testing

Included is a simple **FastAPI** mock server you can run locally to test the stress-test client.
It simulates random latency and occasional errors.

Start Mock Server: 
- `python mock_server.py`

Start Stress Test: 
- `python stress_test.py -k api_key -n 2000 -c 100 --crescendo 
  -e /users --base-url http://127.0.0.1:8000 
  --retries 5 --backoff-method jitter --backoff-base 0.5 
  --backoff-cap 20 --duration 60 --stop-on-key 
  --export-csv data.csv`
