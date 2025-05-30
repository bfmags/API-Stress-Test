import asyncio
import csv
import sys
import random
import threading
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
        lat = sorted(latencies) # lat will be an empty list if latencies is empty

        p50, p90, p99, avg = None, None, None, None # Initialize stats

        if not lat:
            # No latency data to calculate statistics from
            pass # avg, p50, p90, p99 remain None
        elif len(lat) < 2:
            # Not enough data for all percentiles, handle avg separately
            avg = sum(lat) / len(lat)
            # p50, p90, p99 remain None (or could be set to avg if preferred)
        else: # Sufficient data
            p50 = lat[int(0.5 * len(lat))]
            p90 = lat[int(0.9 * len(lat))]
            p99 = lat[int(0.99 * len(lat))]
            avg = sum(lat) / len(lat)

        # plot metrics
        ax1.clear()
        ax1.plot(times, list(range(1, total+1)), label='Cumulative Requests')
        if avg is not None:
            ax1.plot(times, [avg]*total, '--', label='Avg Latency')
        if p50 is not None:
            ax1.plot(times, [p50]*total, ':', label='p50 Latency')
        if p90 is not None:
            ax1.plot(times, [p90]*total, '-.', label='p90 Latency')
        if p99 is not None:
            ax1.plot(times, [p99]*total, '-', label='p99 Latency')
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

    ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def run_async_tasks_in_thread(keys, url, total_requests, concurrency, crescendo,
                              retries, backoff_method, backoff_base, backoff_cap,
                              stop_event_arg, duration_arg, shared_context_arg):
    """Runs the asyncio worker_loop in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shared_context_arg['loop'] = loop

    # Ensure global stop_event is used by this loop
    # stop_event is already global, asyncio.set_event_loop ensures it's associated.

    if duration_arg:
        loop.call_later(duration_arg, stop_event_arg.set)

    task = worker_loop(keys, url, total_requests, concurrency, crescendo,
                       retries, backoff_method, backoff_base, backoff_cap)
    try:
        loop.run_until_complete(task)
    finally:
        loop.close()


def print_summary_stats(results_deque, latencies_list, status_counts_counter):
    """Prints a summary of stress test results."""
    total_completed_requests = len(results_deque)

    if total_completed_requests == 0:
        print("\n--- Stress Test Summary ---")
        print("No requests were completed.")
        print("-------------------------")
        return

    errors = sum(v for k, v in status_counts_counter.items() if isinstance(k, str) and k.startswith('ERR'))
    error_rate = errors / total_completed_requests if total_completed_requests else 0
    successful_requests = total_completed_requests - errors

    avg_latency, p50_latency, p90_latency, p99_latency = None, None, None, None

    # Use the pre-populated latencies_list which should only contain latencies from successful requests
    sorted_lat = sorted(latencies_list)

    if sorted_lat:
        avg_latency = sum(sorted_lat) / len(sorted_lat)
        if len(sorted_lat) >= 2:
            p50_latency = sorted_lat[int(0.5 * len(sorted_lat))]
            p90_latency = sorted_lat[int(0.9 * len(sorted_lat))]
            p99_latency = sorted_lat[int(0.99 * len(sorted_lat))]
        elif len(sorted_lat) == 1: # Only avg is truly representative
            p50_latency = avg_latency
            p90_latency = avg_latency
            p99_latency = avg_latency

    print("\n--- Stress Test Summary ---")
    print(f"Total Requests Attempted (approx.): {total_completed_requests}") # Name changed for clarity
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests (Errors): {errors}")
    print(f"Error Rate: {error_rate:.2%}")

    if avg_latency is not None:
        print(f"Average Latency (successful requests): {avg_latency:.4f} s")
    if p50_latency is not None:
        print(f"p50 Latency (Median): {p50_latency:.4f} s")
    if p90_latency is not None:
        print(f"p90 Latency: {p90_latency:.4f} s")
    if p99_latency is not None:
        print(f"p99 Latency: {p99_latency:.4f} s")

    print("Status Code Counts:")
    # Sort items by status code for consistent order
    sorted_status_counts = sorted(status_counts_counter.items(), key=lambda item: str(item[0]))
    for code, count in sorted_status_counts:
        print(f"  {code}: {count}")
    print("-------------------------")


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
@click.option('--live-plot/--no-live-plot', 'live_plot_enabled', default=True, help='Enable or disable live plotting window.')
@click.option('--export-csv', type=click.Path(), default=None, help='Export raw CSV')
def main(api_keys_file, api_keys, total_requests, concurrency, crescendo,
         endpoint, base_url, retries, backoff_method, backoff_base,
         backoff_cap, duration, live_plot_enabled, export_csv):
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

    shared_context = {}
    # Start the asynchronous tasks in a new thread
    thread = threading.Thread(
        target=run_async_tasks_in_thread,
        args=(
            keys, url, total_requests, concurrency, crescendo,
            retries, backoff_method, backoff_base, backoff_cap,
            stop_event,  # Pass the global asyncio.Event
            duration,    # Pass duration
            shared_context # Pass shared_context
        )
    )
    thread.daemon = True # So it exits when main thread exits, if not joined
    thread.start()

    if live_plot_enabled:
        time.sleep(0.1) # Give thread a chance to start and populate shared_context
        background_loop = shared_context.get('loop')

        start_plot() # This will block the main thread until plot window is closed

        # After plot window is closed by user (or if duration timer already fired in thread)
        if background_loop and background_loop.is_running():
            click.echo("Plot closed, signaling worker thread to stop...")
            background_loop.call_soon_threadsafe(stop_event.set)
    else:
        # If plotting is disabled, the main thread still needs to wait for the duration or total_requests.
        # The stop_event will be set by the duration timer in the worker thread,
        # or the worker_loop will exit naturally after total_requests.
        # If neither duration nor total_requests is set, Ctrl+C will be needed.
        pass # No specific action needed here for main thread to signal stop

    click.echo("Waiting for worker thread to finish...")
    thread.join() # Wait for the background thread to complete, regardless of plotting

    # export CSV if asked
    if export_csv:
        with open(export_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','latency','status'])
            writer.writerows(results)
        click.echo(f'CSV saved to {export_csv}')

    print_summary_stats(results, latencies, status_counts)
    click.echo('Done!')


if __name__ == '__main__':
    main()
