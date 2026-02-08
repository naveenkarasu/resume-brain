"""Google Gemini API wrapper with error handling."""

import json
import logging

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_client() -> genai.Client | None:
    global _client
    if not settings.gemini_api_key:
        logger.warning("No GEMINI_API_KEY set - Gemini features disabled")
        return None
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


async def generate_json(prompt: str) -> dict | None:
    """Send a prompt to Gemini and parse the JSON response."""
    client = get_client()
    if client is None:
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=4096,
            ),
        )

        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        return json.loads(text)

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON: %s", e)
        return None
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return None
