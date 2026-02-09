"""Shared test configuration and pytest markers."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: loads real ML models (slow, needs GPU/CPU)"
    )
    config.addinivalue_line(
        "markers", "evaluation: end-to-end scoring evaluation (very slow)"
    )
