from __future__ import annotations

from pathlib import Path
import math

import pandas as pd


NUMERIC_VARIABLES = {
    "credit_score",
    "original_combined_loan_to_value",
    "original_debt_to_income_ratio",
    "original_interest_rate",
}

CATEGORICAL_VARIABLES = {
    "number_of_borrowers",
}


class WOETransformer:
    def __init__(self, bins_path: str | Path) -> None:
        self.bins_path = Path(bins_path)

        if not self.bins_path.exists():
            raise FileNotFoundError(
                f"WOE bins file not found: {self.bins_path}"
            )

        self.bins = pd.read_csv(self.bins_path)

        required_columns = {
            "variable",
            "bin",
            "woe",
            "breaks",
        }

        missing = required_columns - set(self.bins.columns)

        if missing:
            raise ValueError(
                f"WOE file is missing columns: {sorted(missing)}"
            )

    def transform_record(self, record: dict) -> dict:
        transformed = {}

        for variable in NUMERIC_VARIABLES:
            if variable not in record:
                raise KeyError(f"Missing required input: {variable}")

            value = float(record[variable])

            transformed[f"{variable}_woe"] = (
                self._transform_numeric(variable, value)
            )

        for variable in CATEGORICAL_VARIABLES:
            if variable not in record:
                raise KeyError(f"Missing required input: {variable}")

            value = record[variable]

            transformed[f"{variable}_woe"] = (
                self._transform_categorical(variable, value)
            )

        return transformed

    def _transform_numeric(
        self,
        variable: str,
        value: float,
    ) -> float:
        variable_bins = (
            self.bins.loc[self.bins["variable"] == variable]
            .reset_index(drop=True)
        )

        if variable_bins.empty:
            raise ValueError(
                f"No WOE bins found for variable: {variable}"
            )

        lower_bound = -math.inf

        for _, row in variable_bins.iterrows():
            upper_bound = self._parse_upper_bound(row["breaks"])

            if lower_bound <= value < upper_bound:
                return float(row["woe"])

            lower_bound = upper_bound

        raise ValueError(
            f"Value {value} could not be mapped for {variable}"
        )

    def _transform_categorical(
        self,
        variable: str,
        value,
    ) -> float:
        variable_bins = (
            self.bins.loc[self.bins["variable"] == variable]
            .reset_index(drop=True)
        )

        if variable_bins.empty:
            raise ValueError(
                f"No WOE bins found for variable: {variable}"
            )

        value_string = str(value).strip()

        for _, row in variable_bins.iterrows():
            bin_string = str(row["bin"]).strip()

            categories = [
                item.strip()
                for item in bin_string.split("%,%")
            ]

            if value_string in categories:
                return float(row["woe"])

        raise ValueError(
            f"Value {value!r} could not be mapped for {variable}"
        )

    @staticmethod
    def _parse_upper_bound(value) -> float:
        value_string = str(value).strip().lower()

        if value_string in {"inf", "+inf", "infinity"}:
            return math.inf

        return float(value_string)