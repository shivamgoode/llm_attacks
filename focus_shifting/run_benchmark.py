#!/usr/bin/env python3
"""Run the focus shifting benchmark pipeline.

Examples:
    python -m focus_shifting.run_benchmark --dry-run
    python -m focus_shifting.run_benchmark --dataset focus_shifting/sample_dataset.csv --dry-run
    python -m focus_shifting.run_benchmark --attack focus_shift --injection-strategy suffix
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm

from focus_shifting import config
from focus_shifting.attacks.injection import inject_attack
from focus_shifting.attacks.library import ATTACKS, get_attack
from focus_shifting.dataset.loader import (
    ConversationRow,
    count_rows,
    iter_conversation_rows,
    validate_dataset_header,
)
from focus_shifting.evaluation.constraints import evaluate_response, result_to_dict
from focus_shifting.metrics.calculator import FocusMetricAccumulator, accumulate_results_csv
from focus_shifting.models.base import BaseLLM
from focus_shifting.models.registry import load_models, select_models
from focus_shifting.reports.tables import print_summary_tables


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("focus_shifting")


RESULT_COLUMNS = [
    "prompt_id",
    "attack_type",
    "attack_name",
    "attack_strength",
    "injection_strategy",
    "model",
    "response_type",
    "latency",
    "constraints_total",
    "constraints_passed",
    "constraints_failed",
    "constraints_unsupported",
    "constraint_survival_rate",
    "focus_shift_rate",
    "response",
    "conversation_json",
    "evaluation_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focus shifting prompt attack benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock models.")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Limit dataset rows.")
    parser.add_argument("--dataset", default=config.DATASET_PATH, help="Input CSV path.")
    parser.add_argument("--results-out", default=config.RESULTS_PATH, help="Raw results CSV path.")
    parser.add_argument("--summary-out", default=config.MODEL_COMPARISON_PATH, help="Model summary CSV path.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names or 'all'.")
    parser.add_argument("--skip-models", nargs="+", default=[], help="Model names to skip.")
    parser.add_argument("--attack", default=config.DEFAULT_ATTACK, choices=sorted(ATTACKS), help="Attack name.")
    parser.add_argument(
        "--attack-strength",
        default=config.DEFAULT_ATTACK_STRENGTH,
        choices=["low", "medium", "high"],
        help="Built-in attack strength.",
    )
    parser.add_argument(
        "--injection-strategy",
        default=config.DEFAULT_INJECTION_STRATEGY,
        choices=sorted(config.SUPPORTED_INJECTION_STRATEGIES),
        help="Where to inject the attack.",
    )
    parser.add_argument(
        "--mid-position",
        type=int,
        default=None,
        help="Insertion index for mid_conversation strategy.",
    )
    parser.add_argument("--random-seed", type=int, default=config.DEFAULT_RANDOM_SEED, help="Random seed.")
    parser.add_argument("--chunk-size", type=int, default=config.DATASET_CHUNK_SIZE, help="CSV chunk size.")
    parser.add_argument("--flush-every", type=int, default=config.RESULTS_FLUSH_EVERY, help="CSV flush interval.")
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS, help="Concurrent model calls.")
    parser.add_argument("--resume", action="store_true", help="Append and skip completed prompt/model/attack rows.")
    parser.add_argument(
        "--response-chars",
        type=int,
        default=config.RESULT_RESPONSE_CHARS,
        help="Characters of response to store.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def run_model_call(
    model: BaseLLM,
    row: ConversationRow,
    attack_name: str,
    attack_strength: str,
    injection_strategy: str,
    attack_text: str,
    mid_position: int | None,
    response_chars: int,
    random_seed: int,
) -> dict[str, Any]:
    """Call one model and return a CSV-ready result row."""
    rng = random.Random(f"{random_seed}:{row.prompt_id}:{model.name}:{attack_name}:{injection_strategy}")
    attacked_conversation = inject_attack(
        row.conversation_messages,
        attack_text,
        injection_strategy,
        mid_position=mid_position,
        rng=rng,
    )

    try:
        response_text, latency = model.generate(attacked_conversation)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled error for model=%s prompt_id=%s: %s", model.name, row.prompt_id, exc)
        response_text = f"ERROR: {exc}"
        latency = 0.0

    evaluation = evaluate_response(response_text, row.instruction_types, row.instruction_parameters)
    evaluation_payload = result_to_dict(evaluation)
    attack_type = f"{row.attack_type}:{attack_name}:{injection_strategy}"
    response_type = _standard_response_type(evaluation.response_type)

    return {
        "prompt_id": row.prompt_id,
        "attack_type": attack_type,
        "attack_name": attack_name,
        "attack_strength": attack_strength,
        "injection_strategy": injection_strategy,
        "model": model.name,
        "response_type": response_type,
        "latency": latency,
        "constraints_total": evaluation.constraints_total,
        "constraints_passed": evaluation.constraints_passed,
        "constraints_failed": evaluation.constraints_failed,
        "constraints_unsupported": evaluation.constraints_unsupported,
        "constraint_survival_rate": evaluation.constraint_survival_rate,
        "focus_shift_rate": evaluation.focus_shift_rate,
        "response": response_text[:response_chars],
        "conversation_json": json.dumps(attacked_conversation, ensure_ascii=False),
        "evaluation_json": json.dumps(evaluation_payload, ensure_ascii=False),
    }


def _standard_response_type(evaluation_type: str) -> str:
    """Map focus-shifting outcomes onto the root benchmark labels."""
    if evaluation_type == "Preserved":
        return "Refused"
    if evaluation_type in {"Partial", "Shifted"}:
        return "Answered"
    return "Error"


def write_completed_results(
    completed: set[Future],
    writer: csv.DictWriter,
    accumulator: FocusMetricAccumulator,
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
            result["constraints_total"],
            result["constraints_passed"],
            result["constraints_failed"],
            result["constraints_unsupported"],
        )
        write_state["rows_since_flush"] += 1
        if write_state["rows_since_flush"] >= flush_every:
            result_file.flush()
            write_state["rows_since_flush"] = 0
        pbar.set_postfix(model=result["model"], prompt=result["prompt_id"])
        pbar.update(1)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        logger.error("--limit must be non-negative.")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    try:
        validate_dataset_header(dataset_path)
        all_models = load_models(args.dry_run, args.random_seed)
        models = select_models(all_models, args.models, args.skip_models)
        attack = get_attack(args.attack)
    except Exception as exc:  # noqa: BLE001
        logger.error("%s", exc)
        sys.exit(1)

    if not models:
        logger.error("No models selected.")
        sys.exit(1)

    chunk_size = max(1, args.chunk_size)
    flush_every = max(1, args.flush_every)
    workers = max(1, args.workers)
    response_chars = max(0, args.response_chars)
    prompt_count = count_rows(dataset_path, args.limit, chunk_size)
    total_calls = prompt_count * len(models)
    attack_text = attack.generate_attack(args.attack_strength)

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
        logger.info("Resume mode: found %d completed prompt/model/attack rows.", len(processed))
    else:
        accumulator = FocusMetricAccumulator()
        processed: set[tuple[str, str, str, str]] = set()

    logger.info("Dataset rows: %d", prompt_count)
    logger.info("Models: %s", [model.name for model in models])
    logger.info("Attack: %s strength=%s strategy=%s", args.attack, args.attack_strength, args.injection_strategy)
    logger.info("Starting benchmark: %d rows x %d models = %d calls", prompt_count, len(models), total_calls)

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
            with tqdm(total=total_calls, desc="Focus shifting", unit="call") as pbar:
                for row in iter_conversation_rows(dataset_path, args.limit, chunk_size):
                    for model in models:
                        attack_type = f"{row.attack_type}:{args.attack}:{args.injection_strategy}"
                        processed_key = (row.prompt_id, model.name, attack_type, args.injection_strategy)
                        if processed_key in processed:
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
                                row,
                                args.attack,
                                args.attack_strength,
                                args.injection_strategy,
                                attack_text,
                                args.mid_position,
                                response_chars,
                                args.random_seed,
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

    logger.info("Raw results saved -> %s", results_out)

    summary_df = accumulator.to_summary_df()
    summary_df.to_csv(summary_out, index=False)
    logger.info("Model comparison saved -> %s", summary_out)

    category_df = accumulator.to_category_df()
    category_out = summary_out.parent / "category_breakdown.csv"
    category_df.to_csv(category_out, index=False)
    logger.info("Category breakdown saved -> %s", category_out)

    if not args.no_plots:
        from focus_shifting.reports.visualizations import generate_plots

        plots_dir = summary_out.parent / "plots"
        generate_plots(summary_df, category_df, plots_dir)
        logger.info("Plots saved -> %s", plots_dir)

    tables_dir = summary_out.parent / "tables"
    print_summary_tables(summary_df, category_df, tables_dir)
    logger.info("Summary tables saved -> %s", tables_dir)
    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
