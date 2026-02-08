"""Shared dependencies for API routes."""

from services.gemini_client import get_client


def get_gemini_client():
    return get_client()
