"""
models/deepseek.py — DeepSeek client using the OpenAI-compatible API.
"""
import time
import random
import logging
from typing import Tuple

import openai

import config

logger = logging.getLogger(__name__)


class DeepSeekModel:
    """Wraps the DeepSeek API via the OpenAI-compatible endpoint."""

    name = "DeepSeek-Chat"

    def __init__(self) -> None:
        if not config.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY is not set. DeepSeek calls will fail.")
        self._client = openai.OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
        )

    def generate(self, prompt: str) -> Tuple[str, float]:
        """
        Send *prompt* to DeepSeek and return (response_text, latency_seconds).
        Retries up to config.MAX_RETRIES times with exponential back-off.
        Returns ("ERROR: <reason>", latency) on persistent failure.
        """
        delay = config.RETRY_BASE_DELAY
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                start = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                )
                latency = time.perf_counter() - start
                text = response.choices[0].message.content or ""
                return text.strip(), round(latency, 3)

            except openai.RateLimitError as e:
                logger.warning("[DeepSeek] Rate-limited (attempt %d/%d): %s", attempt, config.MAX_RETRIES, e)
            except openai.APITimeoutError as e:
                logger.warning("[DeepSeek] Timeout (attempt %d/%d): %s", attempt, config.MAX_RETRIES, e)
            except openai.APIConnectionError as e:
                logger.warning("[DeepSeek] Connection error (attempt %d/%d): %s", attempt, config.MAX_RETRIES, e)
            except openai.APIStatusError as e:
                logger.error("[DeepSeek] API error %s: %s", e.status_code, e.message)
                return f"ERROR: {e.status_code} {e.message}", 0.0

            if attempt < config.MAX_RETRIES:
                sleep_time = delay + random.uniform(0, 1)
                logger.info("[DeepSeek] Waiting %.1fs before retry %d…", sleep_time, attempt + 1)
                time.sleep(sleep_time)
                delay *= 2

        return "ERROR: max retries exceeded", 0.0
