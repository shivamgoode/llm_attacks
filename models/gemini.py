"""
models/gemini.py — Google Gemini client using the new google-genai SDK.
"""
import time
import random
import logging
from typing import Tuple

from google import genai
from google.genai import errors as genai_errors

import config

logger = logging.getLogger(__name__)


class GeminiModel:
    """Wraps the Google Gemini API using the official google-genai SDK."""

    name = "Gemini-2.0-Flash"

    def __init__(self) -> None:
        if not config.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is not set. Gemini calls will fail.")
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)

    def generate(self, prompt: str) -> Tuple[str, float]:
        """
        Send *prompt* to Gemini and return (response_text, latency_seconds).
        Retries up to config.MAX_RETRIES times with exponential back-off.
        Returns ("ERROR: <reason>", 0.0) on persistent failure.
        """
        delay = config.RETRY_BASE_DELAY
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                start = time.perf_counter()
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                )
                latency = time.perf_counter() - start

                # Safety filter: no candidates means the content was blocked
                if response.candidates:
                    text = response.candidates[0].content.parts[0].text or ""
                else:
                    text = "I cannot assist with that request due to safety guidelines."

                return text.strip(), round(latency, 3)

            except genai_errors.APIError as e:
                status = getattr(e, "status_code", None)
                if status == 429:
                    logger.warning("[Gemini] Rate-limited (attempt %d/%d): %s", attempt, config.MAX_RETRIES, e)
                elif status in (502, 503, 504):
                    logger.warning("[Gemini] Service unavailable (attempt %d/%d): %s", attempt, config.MAX_RETRIES, e)
                else:
                    logger.error("[Gemini] API error: %s", e)
                    return f"ERROR: {e}", 0.0

            except Exception as e:  # noqa: BLE001
                logger.error("[Gemini] Unexpected error: %s", e)
                return f"ERROR: {e}", 0.0

            if attempt < config.MAX_RETRIES:
                sleep_time = delay + random.uniform(0, 1)
                logger.info("[Gemini] Waiting %.1fs before retry %d…", sleep_time, attempt + 1)
                time.sleep(sleep_time)
                delay *= 2

        return "ERROR: max retries exceeded", 0.0
