import json
import os
from typing import Any

import requests


class AICreditAnalyst:
    """
    Generates a natural-language explanation of the mortgage
    credit-risk model output.

    The scoring model calculates:
    - probability of default
    - calibrated probability
    - risk tier
    - feature contributions

    The LLM only explains those results.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        timeout: int = 180,
    ) -> None:
        """
        Configure the LLM provider.

        Supported providers:
        - ollama: local development
        - openrouter: hosted deployment
        """

        self.provider = (
            provider
            or os.getenv("LLM_PROVIDER", "ollama")
        ).lower()

        self.timeout = timeout

        if self.provider == "ollama":
            self.model = (
                model
                or os.getenv(
                    "OLLAMA_MODEL",
                    "llama3.2:3b",
                )
            )

            self.base_url = os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )

        elif self.provider == "openrouter":
            self.model = (
                model
                or os.getenv(
                    "OPENROUTER_MODEL",
                    "meta-llama/llama-3.2-3b-instruct:free",
                )
            )

            self.base_url = (
                "https://openrouter.ai/api/v1"
            )

        else:
            raise ValueError(
                "Unsupported LLM provider. "
                "Use 'ollama' or 'openrouter'."
            )

    def generate_explanation(
        self,
        result: dict[str, Any],
    ) -> str:
        """
        Generate the AI credit-risk explanation.

        If the LLM fails, return a deterministic fallback
        explanation so the application still works.
        """

        prompt = self.build_prompt(result)

        try:
            if self.provider == "ollama":
                return self._call_ollama(prompt)

            if self.provider == "openrouter":
                return self._call_openrouter(prompt)

        except Exception as error:
            print(
                "AI explanation unavailable. "
                f"Using fallback explanation: {error}"
            )

        return self.build_fallback_explanation(result)

    def build_prompt(
        self,
        result: dict[str, Any],
    ) -> str:
        """
        Build a strict prompt that tells the LLM how to explain
        the model result accurately.
        """

        context = self._build_llm_context(result)

        context_json = json.dumps(
            context,
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are an experienced mortgage credit-risk analyst.

Explain the model results below clearly and professionally for a
business user with limited technical knowledge.

STRICT ACCURACY RULES:

1. Use only the information provided in the structured model output.

2. The field "calibrated_12_month_pd_percent" is already expressed
   as a percentage.

3. For example, a value of 0.6592 means 0.6592%.

4. Always include the percent sign when stating the probability
   of default.

5. Do not calculate a new probability of default.

6. Do not change or reinterpret the displayed probability.

7. The values labeled "log_odds_contribution" are contributions
   to the logistic model's log odds.

8. Log-odds contributions are not:
   - percentages
   - probabilities
   - percentage-point changes in PD
   - direct additions to the PD

9. A positive log-odds contribution increased the model's
   estimated risk.

10. A negative log-odds contribution reduced the model's
    estimated risk.

11. If number of borrowers has a negative contribution, say:

    "The number-of-borrowers category reduced the model's
    estimated risk for this borrower, based on patterns learned
    from the modeling data."

    Do not say that multiple borrowers always or generally reduce
    mortgage default risk..

12. Do not claim that multiple borrowers necessarily means:
    - multiple incomes
    - greater repayment ability
    - shared repayment responsibility

13. Do not invent:
    - borrower income
    - assets
    - employment information
    - payment history
    - documentation
    - collateral condition
    - loan purpose
    - any other borrower facts

14. Do not recommend approval or denial.

15. Treat model effects as historical associations, not proof of
    causation.

16. Do not suggest that an input is inaccurate unless the
    structured output specifically identifies a data-quality issue.

17. Keep the explanation concise, conservative, and readable.

Use exactly these sections:

### Overall Assessment
State the calibrated 12-month PD with a percent sign and explain
the risk tier in neutral language.

### Primary Risk Factors
List the risk-increasing factors in ranked order. Describe their
relative importance. Do not describe contribution values as direct
changes in PD.

### Positive Factors
Explain the risk-reducing factors. A negative contribution must be
described as reducing estimated risk in this case.

### Underwriting Considerations
Identify model variables that could be explored through scenario
analysis. Do not recommend approval, denial, or specific loan terms.

Structured model output:

{context_json}
""".strip()

    def _build_llm_context(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Select and organize only the model information needed
        by the LLM.
        """

        borrower_inputs = result.get(
            "borrower_inputs",
            {},
        )

        increasers = self._normalize_drivers(
            result.get(
                "top_risk_increasers",
                [],
            ),
            reverse=True,
        )

        reducers = self._normalize_drivers(
            result.get(
                "top_risk_reducers",
                [],
            ),
            reverse=False,
        )

        return {
            "borrower_inputs": borrower_inputs,
            "calibrated_12_month_pd_percent": round(
                float(
                    result.get(
                        "calibrated_pd_percent",
                        0.0,
                    )
                ),
                4,
            ),
            "risk_tier": result.get(
                "risk_tier",
                "Unknown",
            ),
            "risk_summary": result.get(
                "risk_summary",
                "",
            ),
            "risk_increasing_factors": increasers,
            "risk_reducing_factors": reducers,
        }

    @staticmethod
    def _normalize_drivers(
        drivers: list[dict[str, Any]],
        reverse: bool,
    ) -> list[dict[str, Any]]:
        """
        Sort and simplify feature contributions before sending
        them to the LLM.
        """

        ordered_drivers = sorted(
            drivers,
            key=lambda item: float(
                item.get(
                    "contribution",
                    0.0,
                )
            ),
            reverse=reverse,
        )

        normalized_drivers = []

        for rank, driver in enumerate(
            ordered_drivers,
            start=1,
        ):
            normalized_drivers.append(
                {
                    "rank": rank,
                    "factor": driver.get(
                        "label",
                        "Unknown factor",
                    ),
                    "log_odds_contribution": round(
                        float(
                            driver.get(
                                "contribution",
                                0.0,
                            )
                        ),
                        4,
                    ),
                }
            )

        return normalized_drivers

    def _call_ollama(
        self,
        prompt: str,
    ) -> str:
        """
        Send the prompt to a locally running Ollama model.
        """

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        explanation = data.get(
            "response",
            "",
        ).strip()

        if not explanation:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return explanation

    def _call_openrouter(
        self,
        prompt: str,
    ) -> str:
        """
        Send the prompt to OpenRouter.
        """

        api_key = os.getenv(
            "OPENROUTER_API_KEY",
        )

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured."
            )

        url = (
            f"{self.base_url}/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        app_url = os.getenv(
            "APP_URL",
        )

        app_name = os.getenv(
            "APP_NAME",
            "AI Mortgage Credit Risk Assistant",
        )

        if app_url:
            headers["HTTP-Referer"] = app_url

        if app_name:
            headers["X-Title"] = app_name

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You explain mortgage credit-risk "
                        "model outputs accurately and "
                        "conservatively."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        choices = data.get(
            "choices",
            [],
        )

        if not choices:
            raise RuntimeError(
                "OpenRouter returned no response choices."
            )

        explanation = (
            choices[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not explanation:
            raise RuntimeError(
                "OpenRouter returned an empty response."
            )

        return explanation


    def generate_scenario_explanation(
        self,
        original_result: dict[str, Any],
        scenario_result: dict[str, Any],
        changes: list[dict[str, Any]],
    ) -> str:
        """Explain a deterministic what-if comparison."""
        prompt = self.build_scenario_prompt(
            original_result=original_result,
            scenario_result=scenario_result,
            changes=changes,
        )

        try:
            if self.provider == "ollama":
                return self._call_ollama(prompt)

            if self.provider == "openrouter":
                return self._call_openrouter(prompt)

        except Exception as error:
            print(
                "AI scenario explanation unavailable. "
                f"Using fallback explanation: {error}"
            )

        return self.build_fallback_scenario_explanation(
            original_result=original_result,
            scenario_result=scenario_result,
            changes=changes,
        )

    def build_scenario_prompt(
        self,
        original_result: dict[str, Any],
        scenario_result: dict[str, Any],
        changes: list[dict[str, Any]],
    ) -> str:
        """Build a strict prompt for explaining a what-if comparison."""
        original_pd = float(
            original_result.get("calibrated_pd_percent", 0.0)
        )
        scenario_pd = float(
            scenario_result.get("calibrated_pd_percent", 0.0)
        )
        change_pp = scenario_pd - original_pd
        relative_change = (
            (change_pp / original_pd) * 100
            if original_pd > 0
            else None
        )

        context = {
            "changes": changes,
            "original_calibrated_pd_percent": round(original_pd, 4),
            "scenario_calibrated_pd_percent": round(scenario_pd, 4),
            "absolute_change_percentage_points": round(change_pp, 4),
            "relative_change_percent": (
                round(relative_change, 2)
                if relative_change is not None
                else None
            ),
            "original_risk_tier": original_result.get(
                "risk_tier", "Unknown"
            ),
            "scenario_risk_tier": scenario_result.get(
                "risk_tier", "Unknown"
            ),
        }

        context_json = json.dumps(
            context,
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are an experienced mortgage credit-risk analyst.

Explain the deterministic what-if scenario below for a business user.

STRICT RULES:

1. Use only the supplied comparison.
2. Do not calculate or invent any probability.
3. All PD values are already percentages.
4. Always include the percent sign when stating a PD.
5. Clearly distinguish percentage-point change from relative percent change.
6. Do not mention log odds, coefficients, WOE, or internal model scores.
7. Do not claim causation. Describe the change as a model-based association.
8. Do not recommend approval, denial, or loan terms.
9. Do not introduce borrower facts that are not supplied.
10. Keep the explanation concise and professional.

Use exactly these sections:

### Scenario Tested
State which borrower input or inputs changed.

### Result
State the original PD, scenario PD, absolute percentage-point change,
and whether the risk tier changed.

### Interpretation
Explain whether estimated risk increased, decreased, or stayed the same.
State that the result reflects historical relationships learned by the model.

### Important Note
State that this is a model-based scenario analysis for decision support,
not an approval or denial decision.

Structured scenario comparison:

{context_json}
""".strip()

    def build_fallback_scenario_explanation(
        self,
        original_result: dict[str, Any],
        scenario_result: dict[str, Any],
        changes: list[dict[str, Any]],
    ) -> str:
        """Build a deterministic scenario explanation without an LLM."""
        original_pd = float(
            original_result.get("calibrated_pd_percent", 0.0)
        )
        scenario_pd = float(
            scenario_result.get("calibrated_pd_percent", 0.0)
        )
        change_pp = scenario_pd - original_pd

        original_tier = original_result.get("risk_tier", "Unknown")
        scenario_tier = scenario_result.get("risk_tier", "Unknown")

        change_lines = []
        for change in changes:
            change_lines.append(
                f"- **{change['label']}**: "
                f"{change['old_value']} → {change['new_value']}"
            )

        if change_pp < 0:
            direction_text = "decreased"
        elif change_pp > 0:
            direction_text = "increased"
        else:
            direction_text = "did not change"

        tier_text = (
            f"The risk tier changed from **{original_tier}** "
            f"to **{scenario_tier}**."
            if original_tier != scenario_tier
            else f"The risk tier remained **{original_tier}**."
        )

        lines = [
            "### Scenario Tested",
            *change_lines,
            "",
            "### Result",
            (
                f"The calibrated 12-month PD changed from "
                f"**{original_pd:.4f}%** to **{scenario_pd:.4f}%**."
            ),
            (
                f"The absolute change was **{change_pp:+.4f} "
                "percentage points**."
            ),
            tier_text,
            "",
            "### Interpretation",
            (
                f"The model's estimated default risk {direction_text} "
                "under this scenario. This result reflects historical "
                "relationships learned during model development."
            ),
            "",
            "### Important Note",
            (
                "This scenario analysis supports human review and does "
                "not represent an approval or denial decision."
            ),
        ]

        return "\n".join(lines)

    def build_fallback_explanation(
        self,
        result: dict[str, Any],
    ) -> str:
        """
        Build a reliable explanation without using an LLM.

        This is used when Ollama or OpenRouter is unavailable.
        """

        pd_percent = float(
            result.get(
                "calibrated_pd_percent",
                0.0,
            )
        )

        risk_tier = result.get(
            "risk_tier",
            "Unknown",
        )

        increasers = self._normalize_drivers(
            result.get(
                "top_risk_increasers",
                [],
            ),
            reverse=True,
        )

        reducers = self._normalize_drivers(
            result.get(
                "top_risk_reducers",
                [],
            ),
            reverse=False,
        )

        lines = [
            "### Overall Assessment",
            (
                f"The model estimates a calibrated 12-month "
                f"probability of default of **{pd_percent:.4f}%**, "
                f"placing the borrower in the **{risk_tier}** "
                "risk tier."
            ),
            "",
            "### Primary Risk Factors",
        ]

        if increasers:
            for driver in increasers[:4]:
                lines.append(
                    f"- **{driver['factor']}** increased the "
                    "model's estimated risk."
                )
        else:
            lines.append(
                "- No material risk-increasing factors were "
                "identified."
            )

        lines.extend(
            [
                "",
                "### Positive Factors",
            ]
        )

        if reducers:
            for driver in reducers[:3]:
                lines.append(
                    f"- **{driver['factor']}** reduced the "
                    "model's estimated risk in this case, based "
                    "on patterns learned from the modeling data."
                )
        else:
            lines.append(
                "- No material risk-reducing factors were "
                "identified."
            )

        lines.extend(
            [
                "",
                "### Underwriting Considerations",
                (
                    "The largest model drivers can be explored "
                    "through scenario analysis to understand how "
                    "changes in the model inputs affect the "
                    "estimated probability of default. This "
                    "assessment supports human review and does "
                    "not represent an approval or denial "
                    "decision."
                ),
            ]
        )

        return "\n".join(lines)