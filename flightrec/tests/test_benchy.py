# File: test_benchy.py
# Location: flightrec/tests/test_benchy.py
# Purpose: Unit tests for flightrec.benchy.run_capturing.
# Dependencies: flightrec.benchy, pytest

"""Tests for benchy.run_capturing — live-tee subprocess with optional float parsing."""

from flightrec.benchy import run_capturing


def test_run_capturing_extracts_float():
    exit_code, tok_s = run_capturing(
        ["echo", "throughput: 42.5 tok/s"],
        pattern=r"throughput: (\d+\.?\d*)",
    )
    assert exit_code == 0
    assert tok_s == 42.5


def test_run_capturing_no_match():
    exit_code, tok_s = run_capturing(
        ["echo", "no match here"],
        pattern=r"tok/s=(\d+)",
    )
    assert exit_code == 0
    assert tok_s is None
