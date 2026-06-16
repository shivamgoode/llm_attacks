"""
models/groq_model.py — Groq client using the OpenAI-compatible API.
Groq provides ultra-fast inference for open-source models.
API docs: https://console.groq.com/docs/openai
"""
import time
import random
import logging
from typing import Tuple

import openai

import config

logger = logging.getLogger(__name__)


class GroqModel:
    """Wraps the Groq Chat Completions API (OpenAI-compatible)."""

    def __init__(self, model_id: str, display_name: str) -> None:
        """
        Args:
            model_id:     Groq model identifier, e.g. "llama-3.3-70b-versatile"
            display_name: Human-readable name used in benchmark results.
        """
        self.name = display_name
        self._model_id = model_id
        if not config.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is not set. Groq calls will fail.")
        self._client = openai.OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
        )

    def generate(self, prompt: str) -> Tuple[str, float]:
        """
        Send *prompt* to Groq and return (response_text, latency_seconds).
        Retries up to config.MAX_RETRIES times with exponential back-off.
        Returns ("ERROR: <reason>", 0.0) on persistent failure.
        """
        delay = config.RETRY_BASE_DELAY
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                start = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=self._model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                )
                latency = time.perf_counter() - start
                text = response.choices[0].message.content or ""
                return text.strip(), round(latency, 3)

            except openai.RateLimitError as e:
                logger.warning("[Groq:%s] Rate-limited (attempt %d/%d): %s", self.name, attempt, config.MAX_RETRIES, e)
            except openai.APITimeoutError as e:
                logger.warning("[Groq:%s] Timeout (attempt %d/%d): %s", self.name, attempt, config.MAX_RETRIES, e)
            except openai.APIConnectionError as e:
                logger.warning("[Groq:%s] Connection error (attempt %d/%d): %s", self.name, attempt, config.MAX_RETRIES, e)
            except openai.APIStatusError as e:
                logger.error("[Groq:%s] API error %s: %s", self.name, e.status_code, e.message)
                return f"ERROR: {e.status_code} {e.message}", 0.0

            if attempt < config.MAX_RETRIES:
                sleep_time = delay + random.uniform(0, 1)
                logger.info("[Groq:%s] Waiting %.1fs before retry %d…", self.name, sleep_time, attempt + 1)
                time.sleep(sleep_time)
                delay *= 2

        return "ERROR: max retries exceeded", 0.0
