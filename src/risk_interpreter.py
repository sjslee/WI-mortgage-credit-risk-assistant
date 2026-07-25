from __future__ import annotations

from dataclasses import dataclass


FEATURE_LABELS = {
    "credit_score_woe": "Credit score",
    "original_combined_loan_to_value_woe": "Combined loan-to-value ratio",
    "original_debt_to_income_ratio_woe": "Debt-to-income ratio",
    "original_interest_rate_woe": "Interest rate",
    "number_of_borrowers_woe": "Number of borrowers",
}


@dataclass
class RiskInterpretation:
    risk_tier: str
    risk_summary: str
    top_risk_increasers: list[dict]
    top_risk_reducers: list[dict]


def assign_risk_tier(calibrated_pd: float) -> tuple[str, str]:
    if calibrated_pd < 0.0025:
        return (
            "Low",
            "The estimated probability of default is below the typical level observed in the model portfolio.",
        )

    if calibrated_pd < 0.0075:
        return (
            "Moderate",
            "The estimated probability of default is near the typical range observed in the model portfolio.",
        )

    if calibrated_pd < 0.015:
        return (
            "Elevated",
            "The estimated probability of default is above the typical range observed in the model portfolio.",
        )

    return (
        "High",
        "The estimated probability of default is among the higher-risk outcomes on this model's scale.",
    )


def interpret_score(score_result) -> RiskInterpretation:
    risk_tier, risk_summary = assign_risk_tier(
        score_result.calibrated_pd
    )

    drivers = []

    for feature, details in score_result.feature_contributions.items():
        contribution = float(details["log_odds_contribution"])

        drivers.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "contribution": contribution,
                "absolute_contribution": abs(contribution),
                "direction": details["direction"],
                "woe": float(details["woe"]),
                "coefficient": float(details["coefficient"]),
            }
        )

    increasers = sorted(
        [
            item
            for item in drivers
            if item["contribution"] > 0
        ],
        key=lambda item: item["contribution"],
        reverse=True,
    )

    reducers = sorted(
        [
            item
            for item in drivers
            if item["contribution"] < 0
        ],
        key=lambda item: item["contribution"],
    )

    return RiskInterpretation(
        risk_tier=risk_tier,
        risk_summary=risk_summary,
        top_risk_increasers=increasers,
        top_risk_reducers=reducers,
    )