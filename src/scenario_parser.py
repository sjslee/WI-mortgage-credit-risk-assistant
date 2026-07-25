from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class ScenarioParseError(ValueError):
    """Raised when a scenario request cannot be parsed safely."""


FIELD_RULES = {
    "credit_score": {
        "label": "Credit score",
        "minimum": 300,
        "maximum": 850,
        "integer": True,
        "patterns": [
            (
                r"(?:credit\s*score|fico)"
                r"(?:\s*(?:is|was|to|of|=|becomes|became|"
                r"increase(?:s|d)?\s+to|decrease(?:s|d)?\s+to|"
                r"drop(?:s|ped)?\s+to|rise(?:s|n)?\s+to|"
                r"were|was set to))?"
                r"\s*(\d{3})"
            ),
        ],
    },
    "original_combined_loan_to_value": {
        "label": "Combined loan-to-value ratio",
        "minimum": 0.0,
        "maximum": 200.0,
        "integer": False,
        "patterns": [
            (
                r"(?:combined\s+loan[-\s]?to[-\s]?value"
                r"(?:\s+ratio)?|cltv)"
                r"(?:\s*(?:is|was|to|of|=|becomes|became|"
                r"increase(?:s|d)?\s+to|decrease(?:s|d)?\s+to|"
                r"drop(?:s|ped)?\s+to|rise(?:s|n)?\s+to|"
                r"were|was set to))?"
                r"\s*(\d+(?:\.\d+)?)\s*%?"
            ),
        ],
    },
    "original_debt_to_income_ratio": {
        "label": "Debt-to-income ratio",
        "minimum": 0.0,
        "maximum": 100.0,
        "integer": False,
        "patterns": [
            (
                r"(?:debt[-\s]?to[-\s]?income"
                r"(?:\s+ratio)?|dti)"
                r"(?:\s*(?:is|was|to|of|=|becomes|became|"
                r"increase(?:s|d)?\s+to|decrease(?:s|d)?\s+to|"
                r"drop(?:s|ped)?\s+to|rise(?:s|n)?\s+to|"
                r"were|was set to))?"
                r"\s*(\d+(?:\.\d+)?)\s*%?"
            ),
        ],
    },
    "original_interest_rate": {
        "label": "Original interest rate",
        "minimum": 0.0,
        "maximum": 20.0,
        "integer": False,
        "patterns": [
            (
                r"(?:original\s+interest\s+rate|interest\s+rate|"
                r"mortgage\s+rate|rate)"
                r"(?:\s*(?:is|was|to|of|=|becomes|became|"
                r"increase(?:s|d)?\s+to|decrease(?:s|d)?\s+to|"
                r"drop(?:s|ped)?\s+to|rise(?:s|n)?\s+to|"
                r"were|was set to))?"
                r"\s*(\d+(?:\.\d+)?)\s*%?"
            ),
        ],
    },
    "number_of_borrowers": {
        "label": "Number of borrowers",
        "minimum": 1,
        "maximum": 6,
        "integer": True,
        "patterns": [
            (
                r"(?:number\s+of\s+borrowers|borrowers?)"
                r"(?:\s*(?:is|was|to|of|=|becomes|became|"
                r"increase(?:s|d)?\s+to|decrease(?:s|d)?\s+to|"
                r"drop(?:s|ped)?\s+to|rise(?:s|n)?\s+to|"
                r"were|was set to))?"
                r"\s*(\d+)"
            ),
            r"(\d+)\s+borrowers?",
        ],
    },
}


@dataclass(frozen=True)
class ScenarioIntent:
    """
    Structured interpretation of a user scenario request.

    intent values:
    - scenario
    - target_pd
    - target_tier
    - comparison
    - optimization
    """

    intent: str
    updates: dict[str, int | float] | None = None
    variable: str | None = None
    target_pd_percent: float | None = None
    target_tier: str | None = None
    comparison_updates: list[dict[str, int | float]] | None = None
    raw_question: str = ""


def clean_question(question: str) -> str:
    """Normalize spacing and capitalization."""
    return " ".join(
        question.lower().strip().split()
    )


def validate_field_value(
    field: str,
    raw_value: float,
) -> int | float:
    """Validate and cast a parsed field value."""
    rule = FIELD_RULES[field]

    if rule["integer"]:
        if not raw_value.is_integer():
            raise ScenarioParseError(
                f"{rule['label']} must be a whole number."
            )

        value: int | float = int(raw_value)

    else:
        value = float(raw_value)

    if (
        value < rule["minimum"]
        or value > rule["maximum"]
    ):
        raise ScenarioParseError(
            f"{rule['label']} must be between "
            f"{rule['minimum']} and {rule['maximum']}."
        )

    return value


def extract_field_updates(
    cleaned_question: str,
) -> dict[str, int | float]:
    """
    Extract all explicit borrower-input values from a question.
    """
    updates: dict[str, int | float] = {}

    for field, rule in FIELD_RULES.items():
        for pattern in rule["patterns"]:
            match = re.search(
                pattern,
                cleaned_question,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            raw_value = float(match.group(1))

            updates[field] = validate_field_value(
                field,
                raw_value,
            )

            break

    return updates


def identify_requested_field(
    cleaned_question: str,
) -> str | None:
    """
    Identify a model input mentioned without requiring a value.

    This is used for target searches such as:
    "What credit score gets PD below 0.5%?"
    """
    aliases = {
        "credit_score": [
            r"\bcredit\s*score\b",
            r"\bfico\b",
        ],
        "original_combined_loan_to_value": [
            r"\bcltv\b",
            r"\bcombined\s+loan[-\s]?to[-\s]?value\b",
        ],
        "original_debt_to_income_ratio": [
            r"\bdti\b",
            r"\bdebt[-\s]?to[-\s]?income\b",
        ],
        "original_interest_rate": [
            r"\boriginal\s+interest\s+rate\b",
            r"\binterest\s+rate\b",
            r"\bmortgage\s+rate\b",
        ],
        "number_of_borrowers": [
            r"\bnumber\s+of\s+borrowers\b",
            r"\bborrowers?\b",
        ],
    }

    matches: list[tuple[int, str]] = []

    for field, patterns in aliases.items():
        for pattern in patterns:
            match = re.search(
                pattern,
                cleaned_question,
                flags=re.IGNORECASE,
            )

            if match:
                matches.append(
                    (match.start(), field)
                )
                break

    if not matches:
        return None

    matches.sort(
        key=lambda item: item[0]
    )

    return matches[0][1]


def parse_scenario_question(
    question: str,
) -> dict[str, int | float]:
    """
    Extract explicit borrower-input changes from a what-if question.

    This function is preserved for compatibility with the
    current Streamlit app.
    """
    cleaned = clean_question(question)

    if not cleaned:
        raise ScenarioParseError(
            "Enter a what-if question."
        )

    updates = extract_field_updates(cleaned)

    if not updates:
        raise ScenarioParseError(
            "I could not find a specific value to change. Try: "
            "'What if the credit score increased to 760?'"
        )

    return updates


def parse_pd_target(
    cleaned_question: str,
) -> float | None:
    """
    Extract a requested PD threshold expressed as a percentage.

    Examples:
    - PD below 0.50%
    - default risk under 1%
    - probability of default at or below 0.75%
    """
    patterns = [
        (
            r"(?:pd|probability\s+of\s+default|default\s+risk)"
            r".{0,40}?"
            r"(?:below|under|at\s+or\s+below|less\s+than|to)"
            r"\s*(\d+(?:\.\d+)?)\s*%"
        ),
        (
            r"(?:below|under|at\s+or\s+below|less\s+than)"
            r"\s*(\d+(?:\.\d+)?)\s*%"
            r".{0,40}?"
            r"(?:pd|probability\s+of\s+default|default\s+risk)"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            cleaned_question,
            flags=re.IGNORECASE,
        )

        if match:
            target = float(match.group(1))

            if target < 0:
                raise ScenarioParseError(
                    "The target PD cannot be negative."
                )

            return target

    return None


def parse_tier_target(
    cleaned_question: str,
) -> str | None:
    """
    Extract a requested risk tier.

    A tier word alone is not enough. The question must also
    contain target-oriented wording.
    """
    tier_match = re.search(
        r"\b(low|moderate|elevated|high)\b"
        r"(?:\s+risk)?"
        r"(?:\s+tier)?",
        cleaned_question,
        flags=re.IGNORECASE,
    )

    if not tier_match:
        return None

    target_phrases = [
        "reach",
        "move",
        "get",
        "into",
        "achieve",
        "target",
        "what would",
        "what value",
        "needed",
        "required",
    ]

    if not any(
        phrase in cleaned_question
        for phrase in target_phrases
    ):
        return None

    return tier_match.group(1).title()


def is_optimization_request(
    cleaned_question: str,
) -> bool:
    """Detect a best-change or largest-impact question."""
    phrases = [
        "which variable",
        "which factor",
        "which change",
        "what change",
        "best way",
        "helps the most",
        "help the most",
        "largest reduction",
        "biggest reduction",
        "reduce risk the most",
        "lowest pd",
        "change first",
        "greatest impact",
        "biggest impact",
    ]

    return any(
        phrase in cleaned_question
        for phrase in phrases
    )


def is_comparison_request(
    cleaned_question: str,
    updates: dict[str, int | float],
) -> bool:
    """Detect a comparison between two or more explicit changes."""
    comparison_phrases = [
        "compare",
        "which helps more",
        "which is better",
        "greater impact",
        "bigger impact",
        "more effective",
        "versus",
        " vs ",
        " or ",
    ]

    return (
        len(updates) >= 2
        and any(
            phrase in cleaned_question
            for phrase in comparison_phrases
        )
    )


def build_comparison_updates(
    updates: dict[str, int | float],
) -> list[dict[str, int | float]]:
    """
    Convert combined explicit updates into separate
    single-variable scenarios.

    Example:

    {
        "credit_score": 760,
        "original_debt_to_income_ratio": 30
    }

    becomes:

    [
        {"credit_score": 760},
        {"original_debt_to_income_ratio": 30}
    ]
    """
    return [
        {field: value}
        for field, value in updates.items()
    ]


def parse_scenario_intent(
    question: str,
) -> ScenarioIntent:
    """
    Route a natural-language question into a deterministic
    scenario operation.
    """
    cleaned = clean_question(question)

    if not cleaned:
        raise ScenarioParseError(
            "Enter a scenario or optimization question."
        )

    if is_optimization_request(cleaned):
        return ScenarioIntent(
            intent="optimization",
            raw_question=question,
        )

    updates = extract_field_updates(cleaned)

    if is_comparison_request(
        cleaned,
        updates,
    ):
        return ScenarioIntent(
            intent="comparison",
            comparison_updates=build_comparison_updates(
                updates
            ),
            raw_question=question,
        )

    target_pd = parse_pd_target(cleaned)

    if target_pd is not None:
        variable = identify_requested_field(
            cleaned
        )

        if variable is None:
            raise ScenarioParseError(
                "Specify which borrower input should be searched. "
                "For example: "
                "'What credit score gets PD below 0.50%?'"
            )

        return ScenarioIntent(
            intent="target_pd",
            variable=variable,
            target_pd_percent=target_pd,
            raw_question=question,
        )

    target_tier = parse_tier_target(
        cleaned
    )

    if target_tier is not None:
        variable = identify_requested_field(
            cleaned
        )

        if variable is None:
            raise ScenarioParseError(
                "Specify which borrower input should be searched. "
                "For example: "
                "'What DTI reaches the Low risk tier?'"
            )

        return ScenarioIntent(
            intent="target_tier",
            variable=variable,
            target_tier=target_tier,
            raw_question=question,
        )

    if updates:
        return ScenarioIntent(
            intent="scenario",
            updates=updates,
            raw_question=question,
        )

    raise ScenarioParseError(
        "I could not identify a supported request. Try one of these:\n\n"
        "- What if the credit score increased to 760?\n"
        "- What credit score gets PD below 0.50%?\n"
        "- What DTI reaches the Low risk tier?\n"
        "- Compare credit score 760 and DTI 30.\n"
        "- Which variable produces the largest PD reduction?"
    )


def apply_scenario_updates(
    current_application: dict[str, Any],
    updates: dict[str, int | float],
) -> dict[str, Any]:
    """
    Return a new application with only the requested fields changed.
    """
    scenario_application = dict(
        current_application
    )

    scenario_application.update(
        updates
    )

    return scenario_application


def describe_changes(
    current_application: dict[str, Any],
    scenario_application: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create display-ready descriptions of changed inputs."""
    changes = []

    for field, rule in FIELD_RULES.items():
        old_value = current_application.get(
            field
        )

        new_value = scenario_application.get(
            field
        )

        if old_value == new_value:
            continue

        changes.append(
            {
                "field": field,
                "label": rule["label"],
                "old_value": old_value,
                "new_value": new_value,
            }
        )

    return changes