"""Load conversation-level focus shifting benchmark rows."""
from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


COLUMN_MAP = {
    "id": "prompt_id",
    "prompt_id": "prompt_id",
    "Prompt_ID": "prompt_id",
    "Prompt_id": "prompt_id",
    "Prompt": "prompt",
    "prompt": "prompt",
    "Attack type": "attack_type",
    "attack_type": "attack_type",
    "conversation": "conversation_messages",
    "conversation_json": "conversation_messages",
    "conversation_messages": "conversation_messages",
    "messages": "conversation_messages",
    "input": "conversation_messages",
    "instruction_types": "instruction_types",
    "instruction_parameters": "instruction_parameters",
    "instruction_params": "instruction_parameters",
    "active_instruction_ids_json": "instruction_types",
    "active_kwargs_json": "instruction_parameters",
}


@dataclass(frozen=True)
class ConversationRow:
    prompt_id: str
    conversation_messages: list[dict[str, str]]
    instruction_texts: list[str]
    instruction_types: list[str]
    instruction_parameters: list[dict[str, Any]]
    attack_type: str


def normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize accepted CSV header names to the focus shifting schema."""
    columns: dict[Any, str] = {}
    for column in df.columns:
        column_text = str(column).lstrip("\ufeff")
        canonical = _canonical(column_text)
        if column_text in COLUMN_MAP:
            columns[column] = COLUMN_MAP[column_text]
        elif canonical in COLUMN_MAP:
            columns[column] = COLUMN_MAP[canonical]
        elif re.match(r"^turn_\d+_user$", canonical):
            columns[column] = canonical.replace("turn_", "prompt_").replace("_user", "")
        else:
            columns[column] = column_text
    return df.rename(columns=columns)


def validate_dataset_header(dataset_path: Path) -> None:
    """Validate that the CSV has either a conversation column or a legacy prompt column."""
    header = _read_dataset_sample(dataset_path, nrows=1)
    header = _normalize_dataset_frame(header)
    if "conversation_messages" not in header.columns and "prompt" not in header.columns:
        raise ValueError(
            "Dataset must contain 'conversation_messages' for focus shifting rows "
            "or use positional columns: id, conversation_messages, prompt_1..prompt_n, "
            "instruction_types, instruction_parameters."
        )


def count_rows(dataset_path: Path, limit: int | None, chunk_size: int) -> int:
    """Count dataset rows in chunks."""
    total = 0
    for chunk in _read_dataset_chunks(dataset_path, chunk_size, usecols=[0]):
        total += len(chunk)
        if limit is not None and total >= limit:
            return limit
    return total


def iter_conversation_rows(dataset_path: Path, limit: int | None, chunk_size: int) -> Iterator[ConversationRow]:
    """Yield normalized conversation rows without loading the whole CSV."""
    remaining = limit
    for chunk in _read_dataset_chunks(dataset_path, chunk_size):
        chunk = _normalize_dataset_frame(chunk)
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)

        for index, row in chunk.iterrows():
            yield _row_to_conversation(row, index)


def _row_to_conversation(row: pd.Series, index: int) -> ConversationRow:
    prompt_id = _string_value(row.get("prompt_id"), str(index + 1))
    attack_type = _string_value(row.get("attack_type"), "Focus Shifting")
    messages = _parse_messages(row.get("conversation_messages"), row.get("prompt"))
    instruction_texts = _parse_instruction_texts(row, messages)
    instruction_types = _parse_instruction_types(row.get("instruction_types"))
    if not instruction_types:
        instruction_types = _infer_instruction_types(instruction_texts)
    instruction_parameters = _parse_instruction_parameters(
        row.get("instruction_parameters"),
        instruction_types,
        instruction_texts,
    )
    return ConversationRow(
        prompt_id=prompt_id,
        conversation_messages=messages,
        instruction_texts=instruction_texts,
        instruction_types=instruction_types,
        instruction_parameters=instruction_parameters,
        attack_type=attack_type,
    )


def _read_dataset_sample(dataset_path: Path, nrows: int) -> pd.DataFrame:
    layout = _detect_layout(dataset_path)
    return pd.read_csv(
        dataset_path,
        nrows=nrows,
        **_read_csv_kwargs(layout),
    )


def _read_dataset_chunks(
    dataset_path: Path,
    chunk_size: int,
    usecols: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    layout = _detect_layout(dataset_path)
    kwargs = _read_csv_kwargs(layout)
    if usecols is not None:
        kwargs["usecols"] = usecols
    yield from pd.read_csv(dataset_path, chunksize=chunk_size, **kwargs)


def _read_csv_kwargs(layout: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "sep": layout["separator"],
        "header": 0 if layout["has_header"] else None,
        "encoding": "utf-8",
        "encoding_errors": "replace",
    }
    if layout["separator"] == "\t":
        kwargs["engine"] = "python"
        kwargs["quoting"] = csv.QUOTE_NONE
    return kwargs


def _detect_layout(dataset_path: Path) -> dict[str, Any]:
    with dataset_path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline().strip("\n")
    separator = "\t" if "\t" in first_line else ","
    first_cell = first_line.split(separator, 1)[0].strip().strip('"')
    has_header = not first_cell.isdigit()
    return {"separator": separator, "has_header": has_header}


def _normalize_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    if all(isinstance(column, int) for column in df.columns):
        return _normalize_positional_columns(df)
    return normalize_dataset_columns(df)


def _normalize_positional_columns(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) < 4:
        return df

    columns: dict[Any, str] = {
        df.columns[0]: "prompt_id",
        df.columns[1]: "conversation_messages",
        df.columns[-2]: "instruction_types",
        df.columns[-1]: "instruction_parameters",
    }
    for prompt_number, column in enumerate(df.columns[2:-2], start=1):
        columns[column] = f"instruction_text_{prompt_number}"
    return df.rename(columns=columns)


def _string_value(value: Any, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _parse_messages(value: Any, fallback_prompt: Any) -> list[dict[str, str]]:
    parsed = _parse_structured(value)
    if parsed is None:
        prompt = "" if fallback_prompt is None or pd.isna(fallback_prompt) else str(fallback_prompt)
        return [{"role": "user", "content": prompt}]

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return [{"role": "user", "content": str(parsed)}]

    messages: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict):
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
        else:
            role = "user"
            content = str(item).strip()
        if content:
            messages.append({"role": role, "content": content})

    return messages or [{"role": "user", "content": ""}]


def _parse_instruction_types(value: Any) -> list[str]:
    parsed = _parse_structured(value)
    if parsed is None:
        return []
    if isinstance(parsed, str):
        separators = ["|", ";", ","]
        values = [parsed]
        for separator in separators:
            if separator in parsed:
                values = parsed.split(separator)
                break
        return [_normalize_instruction_type(item) for item in values if item.strip()]
    if isinstance(parsed, list):
        return [_normalize_instruction_type(item) for item in parsed if str(item).strip()]
    if isinstance(parsed, dict):
        return [_normalize_instruction_type(key) for key in parsed if str(key).strip()]
    return [_normalize_instruction_type(parsed)]


def _normalize_instruction_type(value: Any) -> str:
    return str(value).strip().split(":", 1)[0]


def _infer_instruction_types(instruction_texts: list[str]) -> list[str]:
    """Infer supported checker names from natural-language sample dataset turns."""
    instruction_types: list[str] = []
    for text in instruction_texts:
        lowered = text.lower()
        if any(phrase in lowered for phrase in ("lowercase", "lower case", "no capital")):
            instruction_types.append("change_case")
        elif any(phrase in lowered for phrase in ("uppercase", "upper case", "all capital", "no lowercase")):
            instruction_types.append("change_case")
        elif "end with" in lowered or "should end" in lowered:
            instruction_types.append("startend")
        elif "start with" in lowered or "should start" in lowered:
            instruction_types.append("startend")
        elif "include the following keywords" in lowered or "containing keywords" in lowered:
            instruction_types.append("keywords")
        elif "should not include" in lowered or "should not contain" in lowered:
            instruction_types.append("forbidden_keywords")
        elif "at least" in lowered and "word" in lowered:
            instruction_types.append("length_constraints")
        elif ("less than" in lowered or "at least" in lowered) and "sentence" in lowered:
            instruction_types.append("sentence_count")
        elif "exactly" in lowered and "paragraph" in lowered:
            instruction_types.append("paragraph_count")
        elif "wrap your whole response with double quotation marks" in lowered:
            instruction_types.append("quotation_wrapper")
        elif "postscript" in lowered or "p.s." in lowered or "p.p.s" in lowered:
            instruction_types.append("detectable_format")
        elif "title wrapped in double angular brackets" in lowered:
            instruction_types.append("detectable_content")
    return instruction_types


def _parse_instruction_texts(row: pd.Series, messages: list[dict[str, str]]) -> list[str]:
    text_columns = [
        column
        for column in row.index
        if _is_instruction_text_column(str(column))
    ]
    text_columns = sorted(text_columns, key=lambda column: _instruction_text_sort_key(str(column)))
    texts = [_string_value(row.get(column), "") for column in text_columns]
    texts = [text for text in texts if text]
    if texts:
        return texts
    return [message["content"] for message in messages if message.get("role") == "user" and message.get("content")]


def _is_instruction_text_column(column: str) -> bool:
    canonical = _canonical(column)
    return bool(re.match(r"^instruction_text_\d+$", canonical) or re.match(r"^prompt_\d+$", canonical))


def _instruction_text_sort_key(column: str) -> int:
    canonical = _canonical(column)
    match = re.search(r"_(\d+)$", canonical)
    return int(match.group(1)) if match else 0


def _parse_instruction_parameters(
    value: Any,
    instruction_types: list[str],
    instruction_texts: list[str],
) -> list[dict[str, Any]]:
    parsed = _parse_structured(value)
    if not instruction_types:
        return []

    if parsed is None:
        return _attach_instruction_texts(
            _infer_instruction_parameters(instruction_types, instruction_texts),
            instruction_texts,
        )

    if isinstance(parsed, list):
        parameters = [item if isinstance(item, dict) else {"value": item} for item in parsed]
        parameters = [_normalize_instruction_parameter(parameter) for parameter in parameters]
        return _attach_instruction_texts(_pad_parameters(parameters, len(instruction_types)), instruction_texts)

    if isinstance(parsed, dict):
        if len(instruction_types) == 1 and not _has_instruction_keys(parsed, instruction_types):
            return _attach_instruction_texts([_normalize_instruction_parameter(parsed)], instruction_texts)
        parameters = []
        for instruction_type in instruction_types:
            value = parsed.get(instruction_type, parsed.get(_canonical(instruction_type), {}))
            parameter = value if isinstance(value, dict) else {"value": value}
            parameters.append(_normalize_instruction_parameter(parameter))
        return _attach_instruction_texts(parameters, instruction_texts)

    if len(instruction_types) == 1:
        return _attach_instruction_texts([{"value": parsed}], instruction_texts)
    return _attach_instruction_texts(_pad_parameters([{"value": parsed}], len(instruction_types)), instruction_texts)


def _normalize_instruction_parameter(parameter: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parameter)
    if "postscript_marker" in normalized:
        normalized.setdefault("marker", normalized["postscript_marker"])
        normalized.setdefault("phrase", normalized["postscript_marker"])
    if "num_sections" in normalized:
        normalized.setdefault("n", normalized["num_sections"])
    return normalized


def _attach_instruction_texts(
    parameters: list[dict[str, Any]],
    instruction_texts: list[str],
) -> list[dict[str, Any]]:
    enriched = [dict(parameter) for parameter in parameters]
    for index, parameter in enumerate(enriched):
        if index < len(instruction_texts):
            parameter.setdefault("instruction_text", instruction_texts[index])
    return enriched


def _infer_instruction_parameters(
    instruction_types: list[str],
    instruction_texts: list[str],
) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for index, instruction_type in enumerate(instruction_types):
        text = instruction_texts[index] if index < len(instruction_texts) else ""
        lowered = text.lower()
        parameter: dict[str, Any] = {}

        if instruction_type == "length_constraints":
            parameter.update(_infer_count_parameters(lowered, "word"))
        elif instruction_type == "sentence_count":
            parameter.update(_infer_count_parameters(lowered, "sentence"))
        elif instruction_type == "paragraph_count":
            parameter.update(_infer_count_parameters(lowered, "paragraph"))
        elif instruction_type == "keywords":
            keywords = _extract_list_after_marker(text, "keywords:")
            if keywords:
                parameter["keywords"] = keywords
        elif instruction_type == "forbidden_keywords":
            keywords = _extract_list_after_marker(text, "words:")
            if keywords:
                parameter["keywords"] = keywords
        elif instruction_type == "detectable_format":
            if "p.p.s" in lowered:
                parameter["marker"] = "P.P.S"
            elif "p.s." in lowered:
                parameter["marker"] = "P.S."
        elif instruction_type == "detectable_content" and "double angular brackets" in lowered:
            parameter["phrases"] = ["<<", ">>"]

        parameters.append(parameter)
    return parameters


def _infer_count_parameters(text: str, unit: str) -> dict[str, Any]:
    match = re.search(rf"\bat least\s+(\d+)\s+{unit}s?\b", text)
    if match:
        return {"relation": "at least", "num_words": int(match.group(1))}
    match = re.search(rf"\bat most\s+(\d+)\s+{unit}s?\b", text)
    if match:
        return {"relation": "at most", "num_words": int(match.group(1))}
    match = re.search(rf"\bless than\s+(\d+)\s+{unit}s?\b", text)
    if match:
        return {"relation": "less than", "num_words": int(match.group(1))}
    match = re.search(rf"\bexactly\s+(\d+)\s+{unit}s?\b", text)
    if match:
        return {"relation": "exactly", "num_words": int(match.group(1))}
    return {}


def _extract_list_after_marker(text: str, marker: str) -> list[str]:
    marker_index = text.lower().find(marker)
    if marker_index == -1:
        return []
    tail = text[marker_index + len(marker):]
    tail = tail.split(".", 1)[0]
    return [item.strip(" \"'") for item in tail.split(",") if item.strip(" \"'")]


def _pad_parameters(parameters: list[dict[str, Any]], expected: int) -> list[dict[str, Any]]:
    if len(parameters) >= expected:
        return parameters[:expected]
    return parameters + [{} for _ in range(expected - len(parameters))]


def _has_instruction_keys(value: dict[str, Any], instruction_types: list[str]) -> bool:
    keys = {_canonical(key) for key in value}
    return any(_canonical(instruction_type) in keys for instruction_type in instruction_types)


def _canonical(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _parse_structured(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
