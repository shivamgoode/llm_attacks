"""Constraint registry and checkers for focus shifting evaluation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


Checker = Callable[[str, dict[str, Any]], tuple[bool, str]]
CHECKERS: dict[str, Checker] = {}


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


@dataclass(frozen=True)
class ConstraintResult:
    instruction_type: str
    passed: bool
    supported: bool
    detail: str


@dataclass(frozen=True)
class EvaluationResult:
    response_type: str
    constraints_total: int
    constraints_passed: int
    constraints_failed: int
    constraints_unsupported: int
    constraint_survival_rate: float | None
    focus_shift_rate: float | None
    constraint_results: list[ConstraintResult]


def register(*names: str) -> Callable[[Checker], Checker]:
    """Register a checker under one or more instruction names."""
    def decorator(func: Checker) -> Checker:
        for name in names:
            CHECKERS[_canonical(name)] = func
        return func
    return decorator


def evaluate_response(
    response: str,
    instruction_types: list[str],
    instruction_parameters: list[dict[str, Any]],
) -> EvaluationResult:
    """Evaluate response against all supported constraints."""
    if not response or response.startswith("ERROR:"):
        return EvaluationResult("Error", 0, 0, 0, 0, None, None, [])

    results: list[ConstraintResult] = []
    for index, instruction_type in enumerate(instruction_types):
        parameters = instruction_parameters[index] if index < len(instruction_parameters) else {}
        checker = CHECKERS.get(_canonical(instruction_type))
        if checker is None:
            results.append(
                ConstraintResult(
                    instruction_type=instruction_type,
                    passed=False,
                    supported=False,
                    detail="No checker registered for this instruction type.",
                )
            )
            continue

        try:
            passed, detail = checker(response, parameters)
        except Exception as exc:  # noqa: BLE001
            passed = False
            detail = f"Checker error: {exc}"
        results.append(
            ConstraintResult(
                instruction_type=instruction_type,
                passed=passed,
                supported=True,
                detail=detail,
            )
        )

    supported = [result for result in results if result.supported]
    passed_count = sum(1 for result in supported if result.passed)
    failed_count = len(supported) - passed_count
    unsupported_count = len(results) - len(supported)

    if not supported:
        return EvaluationResult("Unsupported", 0, 0, 0, unsupported_count, None, None, results)

    survival_rate = passed_count / len(supported)
    focus_shift_rate = failed_count / len(supported)
    if passed_count == len(supported):
        response_type = "Preserved"
    elif passed_count == 0:
        response_type = "Shifted"
    else:
        response_type = "Partial"

    return EvaluationResult(
        response_type=response_type,
        constraints_total=len(supported),
        constraints_passed=passed_count,
        constraints_failed=failed_count,
        constraints_unsupported=unsupported_count,
        constraint_survival_rate=survival_rate,
        focus_shift_rate=focus_shift_rate,
        constraint_results=results,
    )


def result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    """Serialize an evaluation result for CSV/JSON output."""
    return {
        "response_type": result.response_type,
        "constraints_total": result.constraints_total,
        "constraints_passed": result.constraints_passed,
        "constraints_failed": result.constraints_failed,
        "constraints_unsupported": result.constraints_unsupported,
        "constraint_survival_rate": result.constraint_survival_rate,
        "focus_shift_rate": result.focus_shift_rate,
        "constraint_results": [
            {
                "instruction_type": item.instruction_type,
                "passed": item.passed,
                "supported": item.supported,
                "detail": item.detail,
            }
            for item in result.constraint_results
        ],
    }


@register("lowercase", "lower_case")
def _check_lowercase(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    text = _trim(response, parameters)
    return text == text.lower(), "Response must be lowercase."


@register("uppercase", "upper_case")
def _check_uppercase(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    text = _trim(response, parameters)
    return text == text.upper(), "Response must be uppercase."


@register("change_case")
def _check_change_case(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    case = str(_parameter(parameters, "case", "value", default="")).lower()
    instruction_text = str(parameters.get("instruction_text", "")).lower()
    if not case:
        if any(
            phrase in instruction_text
            for phrase in ("uppercase", "upper case", "all capital", "all caps", "no lowercase", "no lower case")
        ):
            case = "uppercase"
        elif any(phrase in instruction_text for phrase in ("lowercase", "lower case", "no capital")):
            case = "lowercase"

    if case in {"lower", "lowercase", "lower_case"}:
        return _check_lowercase(response, parameters)
    if case in {"upper", "uppercase", "upper_case", "capital"}:
        return _check_uppercase(response, parameters)
    return False, "Unable to infer required case from parameters or instruction text."


@register("keyword", "keywords", "keyword_existence", "required_keywords")
def _check_keywords(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    keywords = _list_parameter(parameters, "keywords", "keyword", "value")
    haystack = response.lower()
    missing = [keyword for keyword in keywords if keyword.lower() not in haystack]
    return not missing, f"Missing keywords: {missing}" if missing else "All required keywords found."


@register("forbidden_keywords", "forbidden_keyword", "excluded_keywords")
def _check_forbidden_keywords(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    keywords = _list_parameter(parameters, "keywords", "keyword", "value")
    haystack = response.lower()
    present = [keyword for keyword in keywords if keyword.lower() in haystack]
    return not present, f"Forbidden keywords present: {present}" if present else "No forbidden keywords found."


@register("word_count", "words", "length_constraints")
def _check_word_count(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    count = len(re.findall(r"\b\w+\b", response))
    return _check_count(count, parameters), f"Word count: {count}."


@register("sentence_count", "sentences")
def _check_sentence_count(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    count = len([item for item in re.split(r"[.!?]+", response) if item.strip()])
    return _check_count(count, parameters), f"Sentence count: {count}."


@register("paragraph_count", "paragraphs")
def _check_paragraph_count(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    count = len([item for item in re.split(r"\n\s*\n", response.strip()) if item.strip()])
    return _check_count(count, parameters), f"Paragraph count: {count}."


@register("exact_ending", "ending", "ends_with")
def _check_exact_ending(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    expected = str(_parameter(parameters, "ending", "end_phrase", "text", "value", default=""))
    if not expected:
        expected = _extract_quoted_phrase(str(parameters.get("instruction_text", "")))
    return response.rstrip().endswith(expected), f"Response must end with {expected!r}."


@register("exact_beginning", "beginning", "starts_with")
def _check_exact_beginning(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    expected = str(_parameter(parameters, "beginning", "start_phrase", "text", "value", default=""))
    if not expected:
        expected = _extract_quoted_phrase(str(parameters.get("instruction_text", "")))
    return response.lstrip().startswith(expected), f"Response must begin with {expected!r}."


@register("startend", "start_end")
def _check_startend(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    start_phrase = _parameter(parameters, "start_phrase", "beginning", "starts_with")
    end_phrase = _parameter(parameters, "end_phrase", "ending", "ends_with")
    if start_phrase is None and end_phrase is None:
        instruction_text = str(parameters.get("instruction_text", ""))
        quoted = _extract_quoted_phrase(instruction_text)
        if "end" in instruction_text.lower() and quoted:
            end_phrase = quoted
        elif "begin" in instruction_text.lower() and quoted:
            start_phrase = quoted

    if start_phrase is not None and not response.lstrip().startswith(str(start_phrase)):
        return False, f"Response must begin with {str(start_phrase)!r}."
    if end_phrase is not None and not response.rstrip().endswith(str(end_phrase)):
        return False, f"Response must end with {str(end_phrase)!r}."
    if start_phrase is None and end_phrase is None:
        return False, "No start or end phrase configured."
    return True, "Start/end constraint satisfied."


@register("markdown_title", "title")
def _check_markdown_title(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    first_line = response.lstrip().splitlines()[0] if response.strip() else ""
    return first_line.startswith("# "), "First non-empty line must be an H1 markdown title."


@register("quotation_wrapper", "quoted")
def _check_quotation_wrapper(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    quote = str(_parameter(parameters, "quote", "value", default='"'))
    text = response.strip()
    return text.startswith(quote) and text.endswith(quote), f"Response must be wrapped in {quote!r}."


@register("number_of_sections", "section_count", "sections")
def _check_section_count(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    count = len(re.findall(r"(?m)^#{1,6}\s+\S+", response))
    if count == 0:
        count = len([item for item in re.split(r"\n\s*\n", response.strip()) if item.strip()])
    return _check_count(count, parameters), f"Section count: {count}."


@register("bullet_count", "bullets")
def _check_bullet_count(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    count = len(re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S+", response))
    return _check_count(count, parameters), f"Bullet count: {count}."


@register("placeholder_count", "placeholders")
def _check_placeholder_count(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    count = len(re.findall(r"\{[^{}]+\}|\[[^\[\]]+\]", response))
    return _check_count(count, parameters), f"Placeholder count: {count}."


@register("detectable_format")
def _check_detectable_format(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    marker = _parameter(parameters, "marker", "phrase", "postscript", "value")
    splitter = _parameter(parameters, "section_spliter", "section_splitter", "section_marker")
    expected_count = _parameter(parameters, "n", "count", "value")

    if marker is None:
        instruction_text = str(parameters.get("instruction_text", ""))
        if "p.s." in instruction_text.lower():
            marker = "P.S."
        else:
            marker = _extract_quoted_phrase(instruction_text)

    if marker:
        present = str(marker).lower() in response.lower()
        if not present:
            return False, f"Required detectable marker {str(marker)!r} not found."

    if splitter and expected_count is not None:
        count = len(re.findall(rf"\b{re.escape(str(splitter))}\b", response, flags=re.IGNORECASE))
        if count != int(expected_count):
            return False, f"Expected {expected_count} occurrences of {str(splitter)!r}; found {count}."

    if marker or splitter:
        return True, "Detectable format constraint satisfied."
    return False, "No detectable format marker configured."


@register("detectable_content")
def _check_detectable_content(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    phrases = _list_parameter(parameters, "phrases", "phrase", "content", "value")
    if not phrases:
        phrase = _extract_quoted_phrase(str(parameters.get("instruction_text", "")))
        phrases = [phrase] if phrase else []
    if not phrases:
        return False, "No detectable content phrase configured."

    haystack = response.lower()
    missing = [phrase for phrase in phrases if phrase.lower() not in haystack]
    return not missing, f"Missing detectable content: {missing}" if missing else "Detectable content found."


@register("language")
def _check_language(response: str, parameters: dict[str, Any]) -> tuple[bool, str]:
    language = str(_parameter(parameters, "language", "value", default="english")).lower()
    ascii_ratio = sum(1 for char in response if ord(char) < 128) / max(len(response), 1)
    if language in {"en", "eng", "english"}:
        return ascii_ratio >= 0.9, f"ASCII ratio for English heuristic: {ascii_ratio:.2f}."
    return True, f"No deterministic checker for language {language!r}; treated as supported."


def _trim(response: str, parameters: dict[str, Any]) -> str:
    return response.strip() if parameters.get("trim", True) else response


def _parameter(parameters: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in parameters:
            return parameters[name]
    return default


def _list_parameter(parameters: dict[str, Any], *names: str) -> list[str]:
    value = _parameter(parameters, *names, default=[])
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        if "|" in value:
            return [item.strip() for item in value.split("|") if item.strip()]
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        return [value]
    return [str(value)] if value is not None else []


def _check_count(count: int, parameters: dict[str, Any]) -> bool:
    relation = str(parameters.get("relation", "")).strip().lower()
    if "num_words" in parameters:
        target = int(parameters["num_words"])
        if relation in {"at least", "minimum", "min", ">=", "greater than or equal to"}:
            return count >= target
        if relation in {"at most", "maximum", "max", "<=", "less than or equal to"}:
            return count <= target
        if relation in {"more than", "greater than", ">"}:
            return count > target
        if relation in {"less than", "<"}:
            return count < target
        return count == target
    if "exact" in parameters:
        return count == int(parameters["exact"])
    if "count" in parameters:
        return count == int(parameters["count"])
    if "value" in parameters:
        return count == int(parameters["value"])
    minimum = parameters.get("min", parameters.get("minimum"))
    maximum = parameters.get("max", parameters.get("maximum"))
    if minimum is not None and count < int(minimum):
        return False
    if maximum is not None and count > int(maximum):
        return False
    return True


def _extract_quoted_phrase(text: str) -> str:
    matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    if not matches:
        return ""
    first = matches[-1]
    return first[0] or first[1]
