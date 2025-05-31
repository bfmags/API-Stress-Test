# Test cases for stress_test.py will be added here.
import pytest
import math
import random # Added for helper function, though mock_random_choice bypasses direct use
# To run tests from the 'tests' directory, ensure that the main project directory
# (containing stress_test.py) is in Python's path.
# Pytest usually handles this if run from the project root.
# If not, uncomment and adjust the following:
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stress_test import pattern_sin, pattern_cos, pattern_fourier_simple, pattern_linear, TRAFFIC_PATTERNS

# General parameters for testing patterns
T_PROGRESS_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_CYCLES = 3 # Default for sin/cos
FOURIER_CYCLES = 2 # Default for fourier

def test_pattern_output_range():
    """Test that all pattern functions return values within [0.0, 1.0]."""
    patterns_to_test = {
        "sin": pattern_sin,
        "cos": pattern_cos,
        "fourier": pattern_fourier_simple,
        "linear": pattern_linear
    }
    for name, func in patterns_to_test.items():
        for t in T_PROGRESS_POINTS:
            if name in ["sin", "cos"]:
                val = func(t, cycles=DEFAULT_CYCLES)
            elif name == "fourier":
                val = func(t, cycles=FOURIER_CYCLES)
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


def test_pattern_cos_values():
    """Test specific values for the cos pattern (default 3 cycles)."""
    # (cos(t * 3 * 2 * pi) + 1) / 2
    # t=0: (cos(0)+1)/2 = 1.0
    # t=0.25: (cos(1.5*pi)+1)/2 = (0+1)/2 = 0.5
    # t=0.5: (cos(3*pi)+1)/2 = (-1+1)/2 = 0.0
    # t=0.75: (cos(4.5*pi)+1)/2 = (0+1)/2 = 0.5
    # t=1.0: (cos(6*pi)+1)/2 = (1+1)/2 = 1.0
    assert pattern_cos(0.0, cycles=DEFAULT_CYCLES) == pytest.approx(1.0)
    assert pattern_cos(0.25, cycles=DEFAULT_CYCLES) == pytest.approx(0.5)
    assert pattern_cos(0.5, cycles=DEFAULT_CYCLES) == pytest.approx(0.0)
    assert pattern_cos(0.75, cycles=DEFAULT_CYCLES) == pytest.approx(0.5)
    assert pattern_cos(1.0, cycles=DEFAULT_CYCLES) == pytest.approx(1.0)

def test_pattern_fourier_simple_values():
    """Test specific values for the fourier pattern (default 2 cycles for fundamental)."""
    # val1 = 0.6 * (math.sin(t * cycles * 2 * math.pi) + 1) / 2
    # val2 = 0.4 * (math.sin(t * (cycles * 2) * 2 * math.pi) + 1) / 2
    # t=0: v1=0.6*(0+1)/2=0.3, v2=0.4*(0+1)/2=0.2. Sum = 0.5
    assert pattern_fourier_simple(0.0, cycles=FOURIER_CYCLES) == pytest.approx(0.5)
    # t=1.0 (end of cycle): Same as t=0
    assert pattern_fourier_simple(1.0, cycles=FOURIER_CYCLES) == pytest.approx(0.5)
    # Fundamental peak at t = 1 / (4 * cycles)
    # If cycles = 2, t = 1/8.
    # Fundamental: sin(pi/2)=1 => val1 = 0.6 * (1+1)/2 = 0.6
    # Harmonic: sin(pi)=0 => val2 = 0.4 * (0+1)/2 = 0.2
    # Sum = 0.8
    t_at_peak_fundamental = 1.0 / (4.0 * FOURIER_CYCLES) # (1/8 for cycles=2)
    assert pattern_fourier_simple(t_at_peak_fundamental, cycles=FOURIER_CYCLES) == pytest.approx(0.8)
    # Max value should be 1.0, min 0.0. These are harder to calculate by hand simply.
    # The general range test (test_pattern_output_range) covers that they don't go out of [0,1].

# --- Tests for worker_loop concurrency calculation ---

def calculate_concurrency_for_test(traffic_pattern_name, t_progress, max_concurrency, mock_random_choice=None):
    """
    Helper function to simulate the concurrency calculation logic from worker_loop.
    Allows mocking random.choice for predictable testing of 'random_cycle'.
    """

    selected_pattern_key = traffic_pattern_name
    if traffic_pattern_name == "random_cycle":
        available_patterns = [p for p in TRAFFIC_PATTERNS.keys() if p != "linear"]
        if not available_patterns:
             selected_pattern_key = "linear"
        elif mock_random_choice:
            selected_pattern_key = mock_random_choice
        else:
            selected_pattern_key = random.choice(available_patterns) # Fallback to actual random if not mocked

    pattern_function = TRAFFIC_PATTERNS[selected_pattern_key]

    if selected_pattern_key in ["sin", "cos"]:
        pattern_scaling_factor = pattern_function(t_progress, cycles=DEFAULT_CYCLES)
    elif selected_pattern_key == "fourier":
        pattern_scaling_factor = pattern_function(t_progress, cycles=FOURIER_CYCLES)
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

def test_worker_loop_concurrency_random_cycle(): # Removed mocker fixture
    """Test random_cycle selects a pattern and uses it."""
    max_conc = 100
    t_progress = 0.5

    # Expected concurrency if 'cos' is chosen at t_progress=0.5 (default 3 cycles)
    # cos(0.5 * 3 * 2 * pi) = cos(3pi) = -1. Factor = (-1+1)/2 = 0.
    # Concurrency = max(1, int(0 * 100)) = 1
    expected_conc_if_cos = 1
    assert calculate_concurrency_for_test("random_cycle", t_progress, max_conc, mock_random_choice="cos") == expected_conc_if_cos

    # Expected concurrency if 'sin' is chosen at t_progress=0.5
    # sin(0.5 * 3 * 2* pi) = sin(3pi) = 0. Factor = (0+1)/2 = 0.5
    # Concurrency = max(1, int(0.5*100)) = 50
    expected_conc_if_sin = 50
    assert calculate_concurrency_for_test("random_cycle", t_progress, max_conc, mock_random_choice="sin") == expected_conc_if_sin

def test_worker_loop_concurrency_always_at_least_1():
    """Concurrency should always be at least 1, even if factor is 0."""
    max_conc = 100
    # For sin pattern at t_progress = 0.25 (with 3 cycles), factor is (sin(1.5pi)+1)/2 = 0
    t_progress_yields_zero_factor_sin = 0.25
    assert calculate_concurrency_for_test("sin", t_progress_yields_zero_factor_sin, max_conc) == 1

    # For cos pattern at t_progress = 0.5 (with 3 cycles), factor is (cos(3pi)+1)/2 = 0
    t_progress_yields_zero_factor_cos = 0.5
    assert calculate_concurrency_for_test("cos", t_progress_yields_zero_factor_cos, max_conc) == 1

# --- Test for sequential random_cycle behavior ---

def test_random_cycle_sequential_behavior(mocker):
    """
    Tests the sequential cycling behavior of 'random_cycle' mode.
    It simulates time passing and checks if patterns switch correctly and
    if t_progress and concurrency are calculated as expected for each segment.
    """
    # --- Test Setup ---
    mock_time = mocker.patch('stress_test.time.time') # Patch time in stress_test module

    # Parameters similar to what worker_loop would receive/use
    initial_start_time = 1000.0 # Arbitrary start time
    test_duration = 30.0        # Total test duration
    max_concurrency = 100

    pattern_sequence = ["sin", "cos", "fourier"]
    num_patterns_in_cycle = len(pattern_sequence)
    segment_duration = test_duration / num_patterns_in_cycle # Should be 10.0s

    # Simulate state variables from worker_loop
    _active_pattern_key = pattern_sequence[0]
    _pattern_function = TRAFFIC_PATTERNS[_active_pattern_key]
    _current_pattern_index = 0
    _segment_start_time = initial_start_time

    # --- Simulation Loop ---
    # Simulate calls at different points in time
    # t_values are relative to initial_start_time
    simulated_time_points = [
        # Start of sin pattern
        0.0,        # active: sin, seg_t_progress: 0.0
        5.0,        # active: sin, seg_t_progress: 0.5
        9.9,        # active: sin, seg_t_progress: ~0.99
        # Transition to cos pattern
        10.0,       # active: cos, seg_t_progress: 0.0
        15.0,       # active: cos, seg_t_progress: 0.5
        19.9,       # active: cos, seg_t_progress: ~0.99
        # Transition to fourier pattern
        20.0,       # active: fourier, seg_t_progress: 0.0
        25.0,       # active: fourier, seg_t_progress: 0.5
        29.9,       # active: fourier, seg_t_progress: ~0.99
        # Optional: check one more transition back to sin if duration allows
        30.0        # active: sin (loops back), seg_t_progress: 0.0
    ]

    expected_states = [
        # (expected_active_pattern, expected_segment_t_progress_approx)
        ("sin", 0.0), ("sin", 0.5), ("sin", 0.99),
        ("cos", 0.0), ("cos", 0.5), ("cos", 0.99),
        ("fourier", 0.0), ("fourier", 0.5), ("fourier", 0.99),
        ("sin", 0.0) # Loops back
    ]

    assert len(simulated_time_points) == len(expected_states), "Test data length mismatch"

    for i, time_elapsed_in_test in enumerate(simulated_time_points):
        current_simulated_time = initial_start_time + time_elapsed_in_test
        mock_time.return_value = current_simulated_time

        # --- Logic mirrored from worker_loop's cycle & t_progress ---
        # 1. Check for pattern switch
        if segment_duration > 0: # segment_duration is positive here
            current_segment_elapsed_time = current_simulated_time - _segment_start_time
            if current_segment_elapsed_time >= segment_duration:
                _current_pattern_index += 1
                if _current_pattern_index >= num_patterns_in_cycle:
                    _current_pattern_index = 0 # Loop back

                _active_pattern_key = pattern_sequence[_current_pattern_index]
                _pattern_function = TRAFFIC_PATTERNS[_active_pattern_key]
                _segment_start_time = current_simulated_time
                # When a switch happens, the elapsed time for the NEW segment is 0
                current_segment_elapsed_time = 0

        # 2. Calculate current_pattern_t_progress for the active segment
        # (Simplified: assumes cycle_patterns_sequentially=True and segment_duration > 0)
        _current_pattern_t_progress = min(current_segment_elapsed_time / segment_duration, 1.0)
        if segment_duration == 0: _current_pattern_t_progress = 1.0 # Avoid division by zero if somehow segment_duration is 0

        # --- Assertions ---
        expected_pattern, expected_t_progress = expected_states[i]
        assert _active_pattern_key == expected_pattern, \
            f"Time {time_elapsed_in_test:.1f}s: Expected pattern '{expected_pattern}', got '{_active_pattern_key}'"
        assert _current_pattern_t_progress == pytest.approx(expected_t_progress, abs=0.01), \
            f"Time {time_elapsed_in_test:.1f}s ({_active_pattern_key}): Expected t_progress {expected_t_progress:.2f}, got {_current_pattern_t_progress:.2f}"

        # 3. Calculate and check concurrency
        # (Using default cycles for pattern functions as worker_loop does)
        # Determine cycles based on active pattern key
        if _active_pattern_key in ["sin", "cos"]:
            cycles_for_pattern = DEFAULT_CYCLES
        elif _active_pattern_key == "fourier":
            cycles_for_pattern = FOURIER_CYCLES
        else: # linear or other patterns that don't use cycles
            pattern_scaling_factor = _pattern_function(_current_pattern_t_progress)
            calculated_concurrency = max(1, int(pattern_scaling_factor * max_concurrency))
            assert 1 <= calculated_concurrency <= max_concurrency, \
                f"Time {time_elapsed_in_test:.1f}s ({_active_pattern_key}): Concurrency {calculated_concurrency} out of range [1, {max_concurrency}]"
            continue # Skip cycle-based calculation for linear

        pattern_scaling_factor = _pattern_function(_current_pattern_t_progress, cycles=cycles_for_pattern)
        calculated_concurrency = max(1, int(pattern_scaling_factor * max_concurrency))

        assert 1 <= calculated_concurrency <= max_concurrency, \
            f"Time {time_elapsed_in_test:.1f}s ({_active_pattern_key}): Concurrency {calculated_concurrency} out of range [1, {max_concurrency}]"

        # print(f"T={time_elapsed_in_test:.1f}s, MockT={current_simulated_time:.1f}, Active='{_active_pattern_key}', SegStartT={_segment_start_time:.1f}, SegElapsed={current_segment_elapsed_time:.2f}, SegTProgress={_current_pattern_t_progress:.2f}, Concurrency={calculated_concurrency}")
