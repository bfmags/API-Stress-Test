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
