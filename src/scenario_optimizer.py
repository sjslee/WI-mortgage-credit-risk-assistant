from __future__ import annotations

from typing import Any


class ScenarioOptimizationError(ValueError):
    """Raised when a scenario optimization request is invalid."""


VARIABLE_CONFIG = {
    "credit_score": {
        "label": "Credit score",
        "minimum": 300,
        "maximum": 850,
        "step": 1,
        "direction": "increase",
        "integer": True,
    },
    "original_combined_loan_to_value": {
        "label": "Combined loan-to-value ratio",
        "minimum": 0.0,
        "maximum": 200.0,
        "step": 1.0,
        "direction": "decrease",
        "integer": False,
    },
    "original_debt_to_income_ratio": {
        "label": "Debt-to-income ratio",
        "minimum": 0.0,
        "maximum": 100.0,
        "step": 1.0,
        "direction": "decrease",
        "integer": False,
    },
    "original_interest_rate": {
        "label": "Original interest rate",
        "minimum": 0.0,
        "maximum": 20.0,
        "step": 0.1,
        "direction": "decrease",
        "integer": False,
    },
    "number_of_borrowers": {
        "label": "Number of borrowers",
        "minimum": 1,
        "maximum": 6,
        "step": 1,
        "direction": "increase",
        "integer": True,
    },
}


def _cast_value(
    variable: str,
    value: float,
) -> int | float:
    config = VARIABLE_CONFIG[variable]

    if config["integer"]:
        return int(round(value))

    return round(float(value), 4)


def _generate_candidate_values(
    variable: str,
    current_value: float,
) -> list[int | float]:
    """
    Generate increasingly favorable values for one model input.

    Candidate order starts with the smallest possible change so
    the first successful result is the minimum tested change.
    """
    config = VARIABLE_CONFIG[variable]

    minimum = float(config["minimum"])
    maximum = float(config["maximum"])
    step = float(config["step"])
    direction = config["direction"]

    candidates = []

    if direction == "increase":
        candidate = current_value + step

        while candidate <= maximum + 1e-9:
            candidates.append(
                _cast_value(variable, candidate)
            )
            candidate += step

    else:
        candidate = current_value - step

        while candidate >= minimum - 1e-9:
            candidates.append(
                _cast_value(variable, candidate)
            )
            candidate -= step

    # Remove duplicates caused by rounding.
    return list(dict.fromkeys(candidates))


def evaluate_single_change(
    application: dict[str, Any],
    service: Any,
    variable: str,
    new_value: float,
) -> dict[str, Any]:
    """
    Change one model input, score the scenario, and compare it with
    the original borrower.
    """
    if variable not in VARIABLE_CONFIG:
        raise ScenarioOptimizationError(
            f"Unsupported variable: {variable}"
        )

    original_result = service.evaluate(application)

    scenario_application = dict(application)
    scenario_application[variable] = _cast_value(
        variable,
        new_value,
    )

    scenario_result = service.evaluate(
        scenario_application
    )

    original_pd = float(
        original_result["calibrated_pd_percent"]
    )
    scenario_pd = float(
        scenario_result["calibrated_pd_percent"]
    )

    return {
        "variable": variable,
        "label": VARIABLE_CONFIG[variable]["label"],
        "original_value": application[variable],
        "scenario_value": scenario_application[variable],
        "original_application": dict(application),
        "scenario_application": scenario_application,
        "original_result": original_result,
        "scenario_result": scenario_result,
        "original_pd_percent": original_pd,
        "scenario_pd_percent": scenario_pd,
        "pd_change_percentage_points": (
            scenario_pd - original_pd
        ),
        "original_risk_tier": original_result[
            "risk_tier"
        ],
        "scenario_risk_tier": scenario_result[
            "risk_tier"
        ],
    }


def find_value_for_pd_target(
    application: dict[str, Any],
    service: Any,
    variable: str,
    target_pd_percent: float,
) -> dict[str, Any]:
    """
    Find the smallest tested change in one variable that produces
    a calibrated PD at or below the requested target.
    """
    if variable not in VARIABLE_CONFIG:
        raise ScenarioOptimizationError(
            f"Unsupported variable: {variable}"
        )

    if target_pd_percent < 0:
        raise ScenarioOptimizationError(
            "The target PD cannot be negative."
        )

    original_result = service.evaluate(application)

    original_pd = float(
        original_result["calibrated_pd_percent"]
    )

    if original_pd <= target_pd_percent:
        return {
            "success": True,
            "already_meets_target": True,
            "target_type": "pd",
            "target_pd_percent": target_pd_percent,
            "tested_candidates": 0,
            "best_candidate": None,
            "message": (
                "The current borrower already meets the "
                "requested PD target."
            ),
        }

    candidate_values = _generate_candidate_values(
        variable,
        float(application[variable]),
    )

    for tested_count, candidate_value in enumerate(
        candidate_values,
        start=1,
    ):
        candidate = evaluate_single_change(
            application=application,
            service=service,
            variable=variable,
            new_value=candidate_value,
        )

        if (
            candidate["scenario_pd_percent"]
            <= target_pd_percent
        ):
            return {
                "success": True,
                "already_meets_target": False,
                "target_type": "pd",
                "target_pd_percent": target_pd_percent,
                "tested_candidates": tested_count,
                "best_candidate": candidate,
                "message": (
                    "The smallest tested change reached the "
                    "requested PD target."
                ),
            }

    return {
        "success": False,
        "already_meets_target": False,
        "target_type": "pd",
        "target_pd_percent": target_pd_percent,
        "tested_candidates": len(candidate_values),
        "best_candidate": None,
        "message": (
            f"No tested value for "
            f"{VARIABLE_CONFIG[variable]['label'].lower()} "
            "reached the requested PD target."
        ),
    }


def find_value_for_tier_target(
    application: dict[str, Any],
    service: Any,
    variable: str,
    target_tier: str,
) -> dict[str, Any]:
    """
    Find the smallest tested change in one variable that reaches
    the requested model risk tier.
    """
    if variable not in VARIABLE_CONFIG:
        raise ScenarioOptimizationError(
            f"Unsupported variable: {variable}"
        )

    normalized_target = target_tier.strip().lower()

    valid_tiers = {
        "low",
        "moderate",
        "elevated",
        "high",
    }

    if normalized_target not in valid_tiers:
        raise ScenarioOptimizationError(
            "Target tier must be Low, Moderate, Elevated, or High."
        )

    original_result = service.evaluate(application)

    if (
        str(original_result["risk_tier"]).lower()
        == normalized_target
    ):
        return {
            "success": True,
            "already_meets_target": True,
            "target_type": "tier",
            "target_tier": target_tier.title(),
            "tested_candidates": 0,
            "best_candidate": None,
            "message": (
                "The current borrower already meets the "
                "requested risk-tier target."
            ),
        }

    candidate_values = _generate_candidate_values(
        variable,
        float(application[variable]),
    )

    for tested_count, candidate_value in enumerate(
        candidate_values,
        start=1,
    ):
        candidate = evaluate_single_change(
            application=application,
            service=service,
            variable=variable,
            new_value=candidate_value,
        )

        if (
            str(candidate["scenario_risk_tier"]).lower()
            == normalized_target
        ):
            return {
                "success": True,
                "already_meets_target": False,
                "target_type": "tier",
                "target_tier": target_tier.title(),
                "tested_candidates": tested_count,
                "best_candidate": candidate,
                "message": (
                    "The smallest tested change reached the "
                    "requested risk tier."
                ),
            }

    return {
        "success": False,
        "already_meets_target": False,
        "target_type": "tier",
        "target_tier": target_tier.title(),
        "tested_candidates": len(candidate_values),
        "best_candidate": None,
        "message": (
            f"No tested value for "
            f"{VARIABLE_CONFIG[variable]['label'].lower()} "
            "reached the requested risk tier."
        ),
    }


def compare_single_changes(
    application: dict[str, Any],
    service: Any,
    changes: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """
    Compare explicit single-variable scenarios and rank them by
    resulting calibrated PD.
    """
    comparisons = []

    for change in changes:
        if len(change) != 1:
            raise ScenarioOptimizationError(
                "Each comparison scenario must contain exactly "
                "one variable and one value."
            )

        variable, new_value = next(
            iter(change.items())
        )

        comparison = evaluate_single_change(
            application=application,
            service=service,
            variable=variable,
            new_value=float(new_value),
        )

        comparisons.append(comparison)

    return sorted(
        comparisons,
        key=lambda item: item["scenario_pd_percent"],
    )


def find_best_single_variable_change(
    application: dict[str, Any],
    service: Any,
) -> dict[str, Any]:
    """
    Find the tested single-variable scenario producing the lowest
    calibrated PD.

    This compares model outputs only. It does not consider the
    feasibility, cost, fairness, or practicality of the changes.
    """
    original_result = service.evaluate(application)

    original_pd = float(
        original_result["calibrated_pd_percent"]
    )

    best_by_variable = []
    total_tested = 0

    for variable in VARIABLE_CONFIG:
        candidate_values = _generate_candidate_values(
            variable,
            float(application[variable]),
        )

        best_candidate = None

        for candidate_value in candidate_values:
            total_tested += 1

            candidate = evaluate_single_change(
                application=application,
                service=service,
                variable=variable,
                new_value=candidate_value,
            )

            if (
                best_candidate is None
                or candidate["scenario_pd_percent"]
                < best_candidate["scenario_pd_percent"]
            ):
                best_candidate = candidate

        if (
            best_candidate is not None
            and best_candidate["scenario_pd_percent"]
            < original_pd
        ):
            best_by_variable.append(best_candidate)

    best_by_variable.sort(
        key=lambda item: item["scenario_pd_percent"]
    )

    return {
        "success": bool(best_by_variable),
        "original_pd_percent": original_pd,
        "tested_candidates": total_tested,
        "best_candidate": (
            best_by_variable[0]
            if best_by_variable
            else None
        ),
        "alternatives": best_by_variable[1:4],
        "message": (
            "Results rank model-estimated single-variable "
            "scenarios only. They do not represent lending or "
            "underwriting recommendations."
        ),
    }