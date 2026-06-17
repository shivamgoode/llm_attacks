#!/usr/bin/env python3
"""
run_benchmark.py — Main orchestrator for the LLM security benchmark pipeline.

Usage:
    # Full benchmark (requires API keys in environment or .env)
    python run_benchmark.py

    # Dry run (no API calls, uses mock responses) — good for testing pipeline
    python run_benchmark.py --dry-run

    # Limit to first N prompts (useful for quick spot-checks)
    python run_benchmark.py --limit 10

    # Combine flags
    python run_benchmark.py --dry-run --limit 20

    # Skip specific models
    python run_benchmark.py --skip-models GPT-4o DeepSeek-Chat
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from evaluator import classify
from metrics import compute_metrics, compute_category_metrics, print_summary_tables

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


# ---------------------------------------------------------------------------
# Dry-run mock
# ---------------------------------------------------------------------------
MOCK_REFUSALS = [
    "I'm sorry, but I can't assist with that request.",
    "I cannot help with this. It violates my ethical guidelines.",
    "I'm unable to provide that information as it could be harmful.",
    "As an AI assistant, I must decline this request.",
    "I won't provide this. It goes against my guidelines.",
]

MOCK_ANSWERS = [
    "Sure! Here is a detailed explanation of how to do that: ...",
    "Of course. The steps are as follows: first, you need to...",
    "Happy to help! Here's the information you requested: ...",
    "Certainly. Here's a working implementation: ...",
    "Here is what you asked for, step by step: ...",
]


def mock_generate(prompt: str) -> tuple[str, float]:
    """Return a fake response with a realistic random latency (0.1–2.5s)."""
    latency = round(random.uniform(0.1, 2.5), 3)
    time.sleep(0)  # no actual sleep; we just fake the latency number
    if random.random() < 0.55:
        return random.choice(MOCK_REFUSALS), latency
    else:
        return random.choice(MOCK_ANSWERS), latency


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Security Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock responses instead of live API calls (no API key required).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit evaluation to the first N prompts from the dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.DATASET_PATH,
        help=f"Path to the input CSV file (default: {config.DATASET_PATH}).",
    )
    parser.add_argument(
        "--results-out",
        type=str,
        default=config.RESULTS_PATH,
        help=f"Path for the raw results CSV (default: {config.RESULTS_PATH}).",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=config.MODEL_COMPARISON_PATH,
        help=f"Path for the model comparison CSV (default: {config.MODEL_COMPARISON_PATH}).",
    )
    parser.add_argument(
        "--skip-models",
        nargs="+",
        default=[],
        metavar="MODEL_NAME",
        help="Model names to exclude (e.g. --skip-models GPT-4o).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # ---- Load dataset -------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    df = pd.read_csv(dataset_path, encoding="utf-8", encoding_errors="replace")
    df.rename(columns={
        "Prompt_ID": "prompt_id",
        "Prompt": "prompt",
        "Attack type": "attack_type"
    }, inplace=True)
    required_cols = {"prompt_id", "prompt", "attack_type"}
    if not required_cols.issubset(df.columns):
        logger.error("Dataset must contain columns: %s", required_cols)
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limiting to first %d prompts.", args.limit)

    logger.info("Loaded %d prompts from %s", len(df), dataset_path)

    # ---- Load models --------------------------------------------------------
    if args.dry_run:
        logger.info("DRY RUN mode — using mock responses (no API calls).")
        # Create lightweight mock model objects
        class MockModel:
            def __init__(self, name):
                self.name = name
            def generate(self, prompt):
                return mock_generate(prompt)

        models = [
            MockModel("Sarvam-105B"),
            MockModel("Groq-LLaMA3.3-70B"),
            MockModel("Groq-Mixtral-8x7B"),
        ]
    else:
        from models import MODELS
        models = MODELS

    # Apply --skip-models filter
    if args.skip_models:
        skip_set = {m.lower() for m in args.skip_models}
        models = [m for m in models if m.name.lower() not in skip_set]
        logger.info("Running with models: %s", [m.name for m in models])

    if not models:
        logger.error("No models selected. Exiting.")
        sys.exit(1)

    # ---- Run benchmark ------------------------------------------------------
    results: list[dict] = []
    total_calls = len(df) * len(models)

    logger.info(
        "Starting benchmark: %d prompts × %d models = %d API calls",
        len(df), len(models), total_calls,
    )

    with tqdm(total=total_calls, desc="Benchmarking", unit="call") as pbar:
        for _, row in df.iterrows():
            prompt_id   = row["prompt_id"]
            prompt      = row["prompt"]
            attack_type = row["attack_type"]

            for model in models:
                pbar.set_postfix(model=model.name, prompt=prompt_id)
                try:
                    response_text, latency = model.generate(prompt)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Unhandled error for model=%s prompt_id=%s: %s",
                        model.name, prompt_id, exc,
                    )
                    response_text = f"ERROR: {exc}"
                    latency = 0.0

                label = classify(response_text)

                results.append({
                    "prompt_id":     prompt_id,
                    "attack_type":   attack_type,
                    "model":         model.name,
                    "response_type": label,
                    "latency":       latency,
                    "response":      response_text[:500],  # truncate for CSV readability
                })
                pbar.update(1)

    # ---- Save raw results ---------------------------------------------------
    results_df = pd.DataFrame(results)
    results_out = Path(args.results_out)
    results_df.to_csv(results_out, index=False)
    logger.info("Raw results saved → %s (%d rows)", results_out, len(results_df))

    # ---- Compute & save metrics --------------------------------------------
    summary_df = compute_metrics(results_df)
    summary_out = Path(args.summary_out)
    summary_df.to_csv(summary_out, index=False)
    logger.info("Model comparison saved → %s", summary_out)

    # Per-category breakdown
    category_df = compute_category_metrics(results_df)
    category_out = summary_out.parent / "category_breakdown.csv"
    category_df.to_csv(category_out, index=False)
    logger.info("Category breakdown saved → %s", category_out)

    # ---- Generate visualizations -------------------------------------------
    from visualizations import generate_plots
    plots_dir = summary_out.parent / "plots"
    generate_plots(summary_df, category_df, plots_dir)

    # ---- Print summary table to stdout -------------------------------------
    print_summary_tables(summary_df, category_df)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
