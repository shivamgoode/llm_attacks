"""Configuration for the focus shifting benchmark."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent

DATASET_PATH = os.getenv("FOCUS_SHIFTING_DATASET", str(PACKAGE_DIR / "sample_dataset.csv"))
RESULTS_PATH = os.getenv("FOCUS_SHIFTING_RESULTS", str(PACKAGE_DIR / "results.csv"))
MODEL_COMPARISON_PATH = os.getenv(
    "FOCUS_SHIFTING_MODEL_COMPARISON",
    str(PACKAGE_DIR / "model_comparison.csv"),
)

DATASET_CHUNK_SIZE = int(os.getenv("DATASET_CHUNK_SIZE", "10000"))
RESULTS_FLUSH_EVERY = int(os.getenv("RESULTS_FLUSH_EVERY", "1000"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
RESULT_RESPONSE_CHARS = int(os.getenv("RESULT_RESPONSE_CHARS", "1000"))

DEFAULT_ATTACK = os.getenv("FOCUS_SHIFTING_ATTACK", "focus_shift")
DEFAULT_INJECTION_STRATEGY = os.getenv("FOCUS_SHIFTING_INJECTION_STRATEGY", "suffix")
DEFAULT_ATTACK_STRENGTH = os.getenv("FOCUS_SHIFTING_ATTACK_STRENGTH", "medium")
DEFAULT_RANDOM_SEED = int(os.getenv("FOCUS_SHIFTING_RANDOM_SEED", "7"))

SUPPORTED_INJECTION_STRATEGIES = {
    "prefix",
    "suffix",
    "mid_conversation",
    "last_message",
    "random_position",
}

DETECTION_WEIGHT = 0.6
ASR_WEIGHT = 0.4
