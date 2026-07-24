from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import pickle

from src.woe_transformer import WOETransformer


MODEL_FEATURES = [
    "credit_score_woe",
    "original_combined_loan_to_value_woe",
    "original_debt_to_income_ratio_woe",
    "original_interest_rate_woe",
    "number_of_borrowers_woe",
]


@dataclass
class ScoreResult:
    raw_pd: float
    calibrated_pd: float
    raw_logit: float
    calibrated_logit: float
    woe_values: dict
    feature_contributions: dict


class MortgageScoringEngine:
    def __init__(
        self,
        model_path: str | Path,
        calibration_path: str | Path,
        bins_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)
        self.calibration_path = Path(calibration_path)
        self.bins_path = Path(bins_path)

        self.model = self._load_pickle(self.model_path)
        self.calibration = self._load_json(self.calibration_path)
        self.transformer = WOETransformer(self.bins_path)

        self._validate_model_features()

    def score(self, application: dict) -> ScoreResult:
        woe_values = self.transformer.transform_record(application)

        raw_logit = float(self.model.params["const"])
        feature_contributions = {}

        for feature in MODEL_FEATURES:
            coefficient = float(self.model.params[feature])
            woe_value = float(woe_values[feature])
            contribution = coefficient * woe_value

            raw_logit += contribution

            feature_contributions[feature] = {
                "woe": woe_value,
                "coefficient": coefficient,
                "log_odds_contribution": contribution,
                "direction": self._get_direction(contribution),
            }

        raw_pd = self._sigmoid(raw_logit)

        calibration_delta = float(self.calibration["delta"])
        calibrated_logit = raw_logit + calibration_delta
        calibrated_pd = self._sigmoid(calibrated_logit)

        return ScoreResult(
            raw_pd=raw_pd,
            calibrated_pd=calibrated_pd,
            raw_logit=raw_logit,
            calibrated_logit=calibrated_logit,
            woe_values=woe_values,
            feature_contributions=feature_contributions,
        )

    def _validate_model_features(self) -> None:
        model_parameters = set(self.model.params.index)
        expected_parameters = {"const", *MODEL_FEATURES}

        missing = expected_parameters - model_parameters

        if missing:
            raise ValueError(
                "Saved model is missing expected parameters: "
                f"{sorted(missing)}"
            )

    @staticmethod
    def _get_direction(contribution: float) -> str:
        if contribution > 0:
            return "increases risk"

        if contribution < 0:
            return "decreases risk"

        return "neutral"

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)

        z = math.exp(value)
        return z / (1.0 + z)

    @staticmethod
    def _load_pickle(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)