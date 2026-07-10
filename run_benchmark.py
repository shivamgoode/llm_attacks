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

    # Run one API model or all API models
    python run_benchmark.py --models Sarvam-105B --limit 10
    python run_benchmark.py --models all --limit 10

    # Large dataset run with streamed results and resume support
    python run_benchmark.py --dataset dataset.csv --chunk-size 10000 --resume --no-plots
"""

import argparse
import csv
import logging
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from evaluator import classify
from metrics import MetricAccumulator, accumulate_results_csv, print_summary_tables

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")

RESULT_COLUMNS = [
    "prompt_id",
    "attack_type",
    "model",
    "response_type",
    "latency",
    "response",
]

DATASET_COLUMN_MAP = {
    "Prompt_ID": "prompt_id",
    "Prompt": "prompt",
    "Attack type": "attack_type",
}
REQUIRED_DATASET_COLUMNS = {"prompt_id", "prompt", "attack_type"}


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
        "--models",
        nargs="+",
        default=["all"],
        metavar="MODEL_NAME",
        help="Model names to run, or 'all' for every configured API model.",
    )
    parser.add_argument(
        "--skip-models",
        nargs="+",
        default=[],
        metavar="MODEL_NAME",
        help="Model names to exclude (e.g. --skip-models GPT-4o).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.DATASET_CHUNK_SIZE,
        metavar="N",
        help=f"Read the dataset in chunks of N rows (default: {config.DATASET_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=config.RESULTS_FLUSH_EVERY,
        metavar="N",
        help=f"Flush the results CSV after every N rows (default: {config.RESULTS_FLUSH_EVERY}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.MAX_WORKERS,
        metavar="N",
        help=f"Number of concurrent model calls (default: {config.MAX_WORKERS}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing results CSV and skip completed prompt/model pairs.",
    )
    parser.add_argument(
        "--response-chars",
        type=int,
        default=config.RESULT_RESPONSE_CHARS,
        metavar="N",
        help=f"Characters of each response to store in CSV (default: {config.RESULT_RESPONSE_CHARS}).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation after metrics are written.",
    )
    return parser.parse_args()


def normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize accepted dataset column names to the pipeline schema."""
    return df.rename(columns=DATASET_COLUMN_MAP)


def validate_dataset_header(dataset_path: Path) -> None:
    """Fail fast if the input CSV does not contain required columns."""
    try:
        header = pd.read_csv(
            dataset_path,
            nrows=0,
            encoding="utf-8",
            encoding_errors="replace",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Unable to read dataset header from %s: %s", dataset_path, exc)
        sys.exit(1)

    header = normalize_dataset_columns(header)
    if not REQUIRED_DATASET_COLUMNS.issubset(header.columns):
        logger.error("Dataset must contain columns: %s", REQUIRED_DATASET_COLUMNS)
        sys.exit(1)


def count_prompts(dataset_path: Path, limit: int | None, chunk_size: int) -> int:
    """Count input prompts in chunks so tqdm can show bounded progress."""
    total = 0
    for chunk in pd.read_csv(
        dataset_path,
        usecols=[0],
        chunksize=chunk_size,
        encoding="utf-8",
        encoding_errors="replace",
    ):
        total += len(chunk)
        if limit is not None and total >= limit:
            return limit
    return total


def iter_prompt_rows(dataset_path: Path, limit: int | None, chunk_size: int):
    """Yield normalized prompt rows without loading the full CSV."""
    remaining = limit
    for chunk in pd.read_csv(
        dataset_path,
        chunksize=chunk_size,
        encoding="utf-8",
        encoding_errors="replace",
    ):
        chunk = normalize_dataset_columns(chunk)
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)

        for row in chunk[["prompt_id", "prompt", "attack_type"]].itertuples(index=False):
            prompt_id = "" if pd.isna(row.prompt_id) else str(row.prompt_id)
            prompt = "" if pd.isna(row.prompt) else str(row.prompt)
            attack_type = "" if pd.isna(row.attack_type) else str(row.attack_type)
            yield prompt_id, prompt, attack_type


def run_model_call(model, prompt_id: str, attack_type: str, prompt: str, response_chars: int) -> dict:
    """Call one model and return a CSV-ready result row."""
    try:
        response_text, latency = model.generate(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unhandled error for model=%s prompt_id=%s: %s",
            model.name,
            prompt_id,
            exc,
        )
        response_text = f"ERROR: {exc}"
        latency = 0.0

    label = classify(response_text)
    return {
        "prompt_id": prompt_id,
        "attack_type": attack_type,
        "model": model.name,
        "response_type": label,
        "latency": latency,
        "response": response_text[:response_chars],
    }


def select_models(models: list[object], requested: list[str], skipped: list[str]) -> list[object]:
    """Select configured models by display name."""
    available = {model.name.lower(): model for model in models}
    requested_names = {name.lower() for name in requested}
    selected = list(models)

    if "all" not in requested_names:
        missing = sorted(name for name in requested if name.lower() not in available)
        if missing:
            logger.error("Unknown model(s): %s", ", ".join(missing))
            logger.error("Available models: %s", ", ".join(model.name for model in models))
            sys.exit(1)
        selected = [available[name.lower()] for name in requested]

    if skipped:
        skip_set = {model.lower() for model in skipped}
        selected = [model for model in selected if model.name.lower() not in skip_set]

    logger.info("Running with models: %s", [model.name for model in selected])
    return selected


def write_completed_results(
    completed: set[Future],
    writer: csv.DictWriter,
    accumulator: MetricAccumulator,
    pbar: tqdm,
    result_file,
    flush_every: int,
    write_state: dict[str, int],
) -> None:
    """Write completed futures, update metrics, and periodically flush output."""
    for future in completed:
        result = future.result()
        writer.writerow(result)
        accumulator.update(
            result["model"],
            result["attack_type"],
            result["response_type"],
            result["latency"],
        )
        write_state["rows_since_flush"] += 1
        if write_state["rows_since_flush"] >= flush_every:
            result_file.flush()
            write_state["rows_since_flush"] = 0
        pbar.set_postfix(model=result["model"], prompt=result["prompt_id"])
        pbar.update(1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        logger.error("--limit must be non-negative.")
        sys.exit(1)

    chunk_size = max(1, args.chunk_size)
    flush_every = max(1, args.flush_every)
    workers = max(1, args.workers)
    response_chars = max(0, args.response_chars)

    # ---- Load dataset -------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    validate_dataset_header(dataset_path)
    prompt_count = count_prompts(dataset_path, args.limit, chunk_size)
    if args.limit is not None:
        logger.info("Limiting to first %d prompts.", args.limit)
    logger.info("Found %d prompts in %s", prompt_count, dataset_path)

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
            MockModel("Groq-LLaMA3.1-8B"),
        ]
    else:
        from models import MODELS
        models = MODELS

    models = select_models(models, args.models, args.skip_models)

    if not models:
        logger.error("No models selected. Exiting.")
        sys.exit(1)

    # ---- Run benchmark ------------------------------------------------------
    total_calls = prompt_count * len(models)
    results_out = Path(args.results_out)
    results_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        accumulator, processed = accumulate_results_csv(
            results_out,
            chunksize=chunk_size,
            model_filter={model.name for model in models},
        )
        logger.info("Resume mode: found %d completed prompt/model pairs.", len(processed))
    else:
        accumulator = MetricAccumulator()
        processed: set[tuple[str, str]] = set()

    logger.info(
        "Starting benchmark: %d prompts × %d models = %d model calls",
        prompt_count, len(models), total_calls,
    )
    logger.info(
        "Streaming with chunk_size=%d, workers=%d, flush_every=%d",
        chunk_size, workers, flush_every,
    )

    file_mode = "a" if args.resume and results_out.exists() else "w"
    write_header = file_mode == "w" or results_out.stat().st_size == 0
    max_pending = max(workers * 4, len(models), 1)
    write_state = {"rows_since_flush": 0}

    with results_out.open(file_mode, newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: set[Future] = set()
            with tqdm(total=total_calls, desc="Benchmarking", unit="call") as pbar:
                for prompt_id, prompt, attack_type in iter_prompt_rows(dataset_path, args.limit, chunk_size):
                    for model in models:
                        if (prompt_id, model.name) in processed:
                            pbar.update(1)
                            continue

                        while len(pending) >= max_pending:
                            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                            write_completed_results(
                                completed,
                                writer,
                                accumulator,
                                pbar,
                                result_file,
                                flush_every,
                                write_state,
                            )

                        pending.add(
                            executor.submit(
                                run_model_call,
                                model,
                                prompt_id,
                                attack_type,
                                prompt,
                                response_chars,
                            )
                        )

                while pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    write_completed_results(
                        completed,
                        writer,
                        accumulator,
                        pbar,
                        result_file,
                        flush_every,
                        write_state,
                    )

        result_file.flush()

    logger.info("Raw results saved → %s", results_out)

    # ---- Compute & save metrics --------------------------------------------
    summary_df = accumulator.to_summary_df()
    summary_df.to_csv(summary_out, index=False)
    logger.info("Model comparison saved → %s", summary_out)

    # Per-category breakdown
    category_df = accumulator.to_category_df()
    category_out = summary_out.parent / "category_breakdown.csv"
    category_df.to_csv(category_out, index=False)
    logger.info("Category breakdown saved → %s", category_out)

    # ---- Generate visualizations -------------------------------------------
    if not args.no_plots:
        from visualizations import generate_plots
        plots_dir = summary_out.parent / "plots"
        generate_plots(summary_df, category_df, plots_dir)

    # ---- Save summary tables -----------------------------------------------
    tables_dir = summary_out.parent / "tables"
    print_summary_tables(summary_df, category_df, tables_dir)

    logger.info("Summary tables saved → %s", tables_dir)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
