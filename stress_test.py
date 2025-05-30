import asyncio
import csv
import sys
import random
import time
from pathlib import Path
from itertools import cycle
from collections import Counter, deque

import click
import httpx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def fixed_backoff(attempt: int, base: float, cap: float) -> float:
    """Return a constant backoff of `base` seconds."""
    return base


def exp_backoff(attempt: int, base: float, cap: float) -> float:
    """Return exponential backoff, doubling each retry up to `cap`."""
    return min(cap, base * (2 ** (attempt - 1)))


def jitter_backoff(attempt: int, base: float, cap: float, prev: float = None) -> float:
    """
    Decorrelated jitter: random between `base` and `prev * 3` (or `base` if first), capped.
    Helps avoid synchronized retries.
    """
    if prev is None:
        prev = base
    sleep = random.uniform(base, prev * 3)
    return min(cap, sleep)


BACKOFF_METHODS = {
    'fixed': fixed_backoff,
    'exponential': exp_backoff,
    'jitter': jitter_backoff,
}

stop_event = asyncio.Event()
results = deque()
status_counts = Counter()
latencies = []
start_time = None


async def fetch_once(client: httpx.AsyncClient, url: str,
                     headers: dict, retries: int,
                     backoff_fn, base: float, cap: float) -> None:
    """
    Send one GET request; on failure retry up to `retries` times with `backoff_fn`.
    Record (timestamp, latency, status) in shared deques.
    """
    attempt = 0
    prev_sleep = base
    while not stop_event.is_set():
        attempt += 1
        t0 = time.time()
        try:
            resp = await client.get(url, headers=headers, timeout=30.0)
            latency = time.time() - t0
            status = resp.status_code
        except Exception as e:
            if attempt > retries:
                latency = None
                status = f"ERR:{type(e).__name__}"
            else:
                sleep = backoff_fn(attempt, base, cap) if backoff_fn != jitter_backoff else jitter_backoff(attempt, base, cap, prev_sleep)
                prev_sleep = sleep
                await asyncio.sleep(sleep)
                continue
        # record metrics
        now = time.time()
        results.append((now, latency or 0, status))
        status_counts[status] += 1
        if latency is not None:
            latencies.append(latency)
        return


async def worker_loop(keys, url, total, concurrency, crescendo,
                      retries, backoff_method, base, cap):
    """Continuously schedule `fetch_once` tasks until stop_event."""
    key_cycle = cycle(keys)
    count = 0
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set() and (total is None or count < total):
            count += 1
            # determine current concurrency
            if crescendo and total:
                current = max(1, min(concurrency, int(count/total * concurrency)))
            else:
                current = concurrency

            # schedule batch
            tasks = []
            for _ in range(current):
                key = next(key_cycle)
                headers = {'Authorization': f"Bearer {key}"}
                tasks.append(
                    fetch_once(client, url, headers,
                               retries, BACKOFF_METHODS[backoff_method], base, cap)
                )
            # fire and forget
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)


def start_plot():
    """Real-Time Plotting of metrics, latency, errors"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    def update(frame):
        if not results:
            return
        data = list(results)
        t0 = data[0][0]
        times = [t - t0 for t, *_ in data]
        total = len(times)
        # latency stats
        lat = sorted(latencies)
        p50 = lat[int(0.5*len(lat))]
        p90 = lat[int(0.9*len(lat))]
        p99 = lat[int(0.99*len(lat))]
        avg = sum(lat)/len(lat)
        # plot metrics
        ax1.clear()
        ax1.plot(times, list(range(1, total+1)), label='Cumulative Requests')
        ax1.plot(times, [avg]*total, '--', label='Avg Latency')
        ax1.plot(times, [p50]*total, ':', label='p50 Latency')
        ax1.plot(times, [p90]*total, '-.', label='p90 Latency')
        ax1.plot(times, [p99]*total, (0, (3, 1, 1, 1)), label='p99 Latency')
        ax1.legend()
        ax1.set_ylabel('Requests / Latency (s)')
        ax1.set_xlabel('Time (s)')

        # status & error rate
        ax2.clear()
        codes = list(status_counts.keys())
        counts = list(status_counts.values())
        errors = sum(v for k,v in status_counts.items() if isinstance(k,str) and k.startswith('ERR'))
        rate = errors/total if total else 0
        ax2.bar(codes, counts)
        ax2.set_title(f'Status Counts & Error Rate: {rate:.1%}')
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Status Code')

    ani = FuncAnimation(fig, update, interval=1000)
    plt.tight_layout()
    plt.show()


@click.command()
@click.option('-f', '--api-keys-file', type=click.Path(exists=True), help='File with one API key per line')
@click.option('-k', '--api-keys', help='Comma-separated API keys')
@click.option('-n', '--total-requests', type=int, default=None, help='Total requests (omit for duration)')
@click.option('-c', '--concurrency', type=int, default=50, help='Concurrent requests')
@click.option('--crescendo/--no-crescendo', default=False, help='Ramp up concurrency')
@click.option('-e', '--endpoint', required=True, help='API endpoint (e.g. /users)')
@click.option('--base-url', default='http://127.0.0.1:8000 ', help='API root URL')
@click.option('--retries', type=int, default=3, help='Retry count')
@click.option('--backoff-method', type=click.Choice(['fixed','exponential','jitter']), default='exponential')
@click.option('--backoff-base', type=float, default=1.0, help='Base backoff (s)')
@click.option('--backoff-cap', type=float, default=30.0, help='Max backoff (s)')
@click.option('--duration', type=int, default=None, help='Run for N seconds')
@click.option('--stop-on-key', is_flag=True, help='Stop on Enter')
@click.option('--export-csv', type=click.Path(), default=None, help='Export raw CSV')
def main(api_keys_file, api_keys, total_requests, concurrency, crescendo,
         endpoint, base_url, retries, backoff_method, backoff_base,
         backoff_cap, duration, stop_on_key, export_csv):
    """CLI wrapper: parse, set stop conditions, launch worker & plotting."""
    global start_time
    # Load keys
    if api_keys and api_keys_file:
        click.echo('Error: choose only one of -f/-k', err=True); sys.exit(1)
    if api_keys_file:
        keys = [l.strip() for l in Path(api_keys_file).read_text().splitlines() if l.strip()]
    elif api_keys:
        keys = [k.strip() for k in api_keys.split(',') if k.strip()]
    else:
        click.echo('Error: provide -f or -k', err=True); sys.exit(1)
    if not keys:
        click.echo('No API keys!', err=True); sys.exit(1)

    url = base_url.rstrip('/') + endpoint
    start_time = time.time()

    # stop triggers
    if stop_on_key:
        asyncio.get_event_loop().add_reader(sys.stdin, lambda: stop_event.set())
    if duration:
        asyncio.get_event_loop().call_later(duration, stop_event.set)

    # schedule tasks
    task = worker_loop(keys, url, total_requests, concurrency, crescendo,
                       retries, backoff_method, backoff_base, backoff_cap)
    # run producer + plotter
    plot_task = asyncio.to_thread(start_plot)

    asyncio.run(asyncio.gather(task, plot_task))

    # export CSV if asked
    if export_csv:
        with open(export_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','latency','status'])
            writer.writerows(results)
        click.echo(f'CSV saved to {export_csv}')
    click.echo('Done!')


if __name__ == '__main__':
    main()
