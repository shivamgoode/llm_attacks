"""Pretty text tables for focus shifting benchmark summaries."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from focus_shifting.metrics.calculator import slugify


def print_summary_tables(summary_df: pd.DataFrame, category_df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    """Print and optionally save overall and per-attack summary tables."""
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    overall_text = _format_table(summary_df, "FOCUS SHIFTING BENCHMARK - OVERALL SUMMARY", has_tabulate)
    print(overall_text, end="")
    if output_path is not None:
        (output_path / "overall_summary.txt").write_text(overall_text, encoding="utf-8")

    if not category_df.empty:
        for attack_type, group in category_df.groupby("Attack_Type"):
            display_df = group.drop(columns=["Attack_Type"])
            category_text = _format_table(display_df, f"ATTACK: {attack_type}", has_tabulate)
            print(category_text, end="")
            if output_path is not None:
                (output_path / f"category_{slugify(attack_type)}.txt").write_text(category_text, encoding="utf-8")


def _format_table(df: pd.DataFrame, title: str, has_tabulate: bool) -> str:
    parts = [
        "\n" + "=" * 80,
        f"  {title}",
        "=" * 80,
    ]
    if df.empty:
        parts.append("(no rows)")
    elif has_tabulate:
        from tabulate import tabulate

        parts.append(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    else:
        parts.append(df.to_string(index=False))
    parts.append("=" * 80 + "\n")
    return "\n".join(parts)

