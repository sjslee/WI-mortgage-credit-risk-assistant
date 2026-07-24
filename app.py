from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.credit_risk_service import CreditRiskService
from src.ai_credit_analyst import AICreditAnalyst
from src.scenario_parser import (
    ScenarioParseError,
    apply_scenario_updates,
    describe_changes,
    parse_scenario_intent,
)

from src.scenario_optimizer import (
    compare_single_changes,
    find_best_single_variable_change,
    find_value_for_pd_target,
    find_value_for_tier_target,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
AI_DRIVER_THRESHOLD = 0.05


@st.cache_resource
def load_service() -> CreditRiskService:
    return CreditRiskService(ARTIFACT_DIR)


@st.cache_resource
def load_ai_analyst() -> AICreditAnalyst:
    return AICreditAnalyst()


def get_risk_style(risk_tier: str) -> dict:
    styles = {
        "Low": {
            "background": "#dcfce7",
            "border": "#22c55e",
            "text": "#166534",
            "icon": "●",
        },
        "Moderate": {
            "background": "#fef3c7",
            "border": "#f59e0b",
            "text": "#92400e",
            "icon": "●",
        },
        "Elevated": {
            "background": "#ffedd5",
            "border": "#f97316",
            "text": "#9a3412",
            "icon": "●",
        },
        "High": {
            "background": "#fee2e2",
            "border": "#ef4444",
            "text": "#991b1b",
            "icon": "●",
        },
    }

    return styles.get(
        risk_tier,
        {
            "background": "#f1f5f9",
            "border": "#64748b",
            "text": "#334155",
            "icon": "●",
        },
    )


def join_feature_names(names: list[str]) -> str:
    if not names:
        return "no material risk-increasing factors"

    if len(names) == 1:
        return names[0]

    if len(names) == 2:
        return f"{names[0]} and {names[1]}"

    return f"{names[0]}, {names[1]}, and {names[2]}"


def build_plain_english_explanation(result: dict) -> str:
    increasers = result["top_risk_increasers"]
    reducers = result["top_risk_reducers"]

    top_names = [
        driver["label"].lower()
        for driver in increasers[:3]
    ]

    increaser_text = join_feature_names(top_names)

    if reducers:
        reducer_names = [
            driver["label"].lower()
            for driver in reducers[:2]
        ]

        reducer_text = " and ".join(reducer_names)

        reducer_sentence = (
            f"{reducer_text.capitalize()} partially offsets "
            "the estimated risk."
        )
    else:
        reducer_sentence = (
            "The model did not identify a material "
            "risk-reducing factor for this borrower."
        )

    return (
        f"This borrower has an estimated 12-month probability "
        f"of default of {result['calibrated_pd_percent']:.3f}%, "
        f"placing the borrower in the "
        f"{result['risk_tier'].lower()} risk tier. "
        f"The strongest risk-increasing factors are "
        f"{increaser_text}. {reducer_sentence}"
    )


def render_metric_card(
    label: str,
    value: str,
    detail: str,
    value_class: str = "metric-value",
) -> None:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{escape(label)}</div>
    <div class="{value_class}">{escape(value)}</div>
    <div class="metric-detail">{escape(detail)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_risk_card(
    risk_tier: str,
    risk_style: dict,
) -> None:
    st.markdown(
        f"""
<div
    class="risk-card"
    style="
        background-color: {risk_style['background']};
        border-left-color: {risk_style['border']};
        color: {risk_style['text']};
    "
>
    <div class="risk-label">Risk Tier</div>
    <div class="risk-value">
        {risk_style['icon']} {escape(risk_tier)}
    </div>
    <div class="risk-detail">
        Based on calibrated model thresholds
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_driver_bars(drivers: list[dict]) -> None:
    if not drivers:
        st.success(
            "No model factors increased the estimated risk."
        )
        return

    ordered_drivers = sorted(
        drivers,
        key=lambda item: float(item["contribution"]),
        reverse=True,
    )

    max_contribution = max(
        abs(float(driver["contribution"]))
        for driver in ordered_drivers
    )

    for driver in ordered_drivers:
        label = str(driver["label"])
        contribution = float(driver["contribution"])

        normalized_value = (
            abs(contribution) / max_contribution
            if max_contribution > 0
            else 0.0
        )

        st.markdown(f"**{label}**")
        st.progress(normalized_value)
        st.write("")


def render_reducer_cards(drivers: list[dict]) -> None:
    if not drivers:
        st.warning(
            "No model factors reduced the estimated risk."
        )
        return

    ordered_drivers = sorted(
        drivers,
        key=lambda item: float(item["contribution"]),
    )

    for driver in ordered_drivers:
        label = str(driver["label"])
        contribution = float(driver["contribution"])

        with st.container(border=True):
            icon_column, text_column = st.columns(
                [0.6, 5.2]
            )

            with icon_column:
                st.markdown("### ↓")

            with text_column:
                st.markdown(f"**{label}**")
                st.caption(
                    "Reduces the model's estimated default risk."
                )


def build_driver_table(
    increasers: list[dict],
    reducers: list[dict],
) -> pd.DataFrame:
    rows = []

    for driver in increasers:
        rows.append(
            {
                "Risk factor": driver["label"],
                "Contribution": float(
                    driver["contribution"]
                ),
                "Effect": "Increases risk",
            }
        )

    for driver in reducers:
        rows.append(
            {
                "Risk factor": driver["label"],
                "Contribution": float(
                    driver["contribution"]
                ),
                "Effect": "Decreases risk",
            }
        )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        return dataframe

    dataframe["Magnitude"] = dataframe[
        "Contribution"
    ].abs()

    dataframe = (
        dataframe.sort_values(
            by="Magnitude",
            ascending=False,
        )
        .drop(columns="Magnitude")
        .reset_index(drop=True)
    )

    return dataframe


def build_technical_table(result: dict) -> pd.DataFrame:
    rows = []

    for feature, details in result[
        "feature_contributions"
    ].items():
        rows.append(
            {
                "Feature": feature,
                "WOE": details["woe"],
                "Coefficient": details["coefficient"],
                "Direction": details["direction"],
            }
        )

    return pd.DataFrame(rows)


def prepare_ai_result(
    result: dict,
    threshold: float = AI_DRIVER_THRESHOLD,
) -> dict:
    """
    Create a copy of the scoring result for the LLM.

    Only meaningful model drivers are included in the AI context.
    The original result remains unchanged for the dashboard and
    technical tables.
    """
    ai_result = dict(result)

    increasers = [
        driver
        for driver in result.get("top_risk_increasers", [])
        if abs(float(driver.get("contribution", 0.0)))
        >= threshold
    ]

    reducers = [
        driver
        for driver in result.get("top_risk_reducers", [])
        if abs(float(driver.get("contribution", 0.0)))
        >= threshold
    ]

    # Preserve at least the strongest driver when contributions exist
    # but all fall below the materiality threshold.
    if (
        not increasers
        and result.get("top_risk_increasers")
    ):
        increasers = [
            max(
                result["top_risk_increasers"],
                key=lambda item: abs(
                    float(item.get("contribution", 0.0))
                ),
            )
        ]

    if (
        not reducers
        and result.get("top_risk_reducers")
    ):
        strongest_reducer = max(
            result["top_risk_reducers"],
            key=lambda item: abs(
                float(item.get("contribution", 0.0))
            ),
        )

        if abs(
            float(
                strongest_reducer.get(
                    "contribution",
                    0.0,
                )
            )
        ) >= threshold:
            reducers = [strongest_reducer]

    ai_result["top_risk_increasers"] = increasers
    ai_result["top_risk_reducers"] = reducers

    return ai_result

def format_target_search_result(
    optimization_result: dict,
) -> str:
    if optimization_result["already_meets_target"]:
        return (
            f"{optimization_result['message']}\n\n"
            f"**Scenarios evaluated:** "
            f"{optimization_result['tested_candidates']}"
        )

    if not optimization_result["success"]:
        return (
            "**Target not reached.**\n\n"
            "Changing only the selected borrower input was not "
            "sufficient to reach the requested target while all "
            "other borrower characteristics remained unchanged.\n\n"
            f"{optimization_result['message']}\n\n"
            f"**Scenarios evaluated:** "
            f"{optimization_result['tested_candidates']}"
        )

    candidate = optimization_result["best_candidate"]

    change_pp = float(
        candidate["pd_change_percentage_points"]
    )

    return (
        f"**Model-based target:** "
        f"{candidate['label']} = "
        f"{candidate['scenario_value']}\n\n"
        f"This was the first evaluated value that reached the "
        f"requested target while all other borrower inputs remained "
        f"unchanged.\n\n"
        f"**Original value:** "
        f"{candidate['original_value']}\n\n"
        f"**Original PD:** "
        f"{candidate['original_pd_percent']:.4f}%\n\n"
        f"**Scenario PD:** "
        f"{candidate['scenario_pd_percent']:.4f}%\n\n"
        f"**Change:** "
        f"{change_pp:+.4f} percentage points\n\n"
        f"**Risk tier:** "
        f"{candidate['original_risk_tier']} → "
        f"{candidate['scenario_risk_tier']}\n\n"
        f"**Scenarios evaluated:** "
        f"{optimization_result['tested_candidates']}"
    )


def format_comparison_result(
    comparisons: list[dict],
) -> str:
    if not comparisons:
        return "No comparison results were available."

    lines = [
        "### Scenario Comparison",
        "",
    ]

    for rank, comparison in enumerate(
        comparisons,
        start=1,
    ):
        lines.extend(
            [
                (
                    f"**{rank}. {comparison['label']} = "
                    f"{comparison['scenario_value']}**"
                ),
                (
                    f"- Scenario PD: "
                    f"{comparison['scenario_pd_percent']:.4f}%"
                ),
                (
                    f"- PD change: "
                    f"{comparison['pd_change_percentage_points']:+.4f} "
                    "percentage points"
                ),
                (
                    f"- Risk tier: "
                    f"{comparison['scenario_risk_tier']}"
                ),
                "",
            ]
        )

    best = comparisons[0]

    lines.extend(
        [
            "### Best Tested Scenario",
            (
                f"Among these explicit scenarios, "
                f"**{best['label']} = "
                f"{best['scenario_value']}** produced the "
                f"lowest calibrated PD of "
                f"**{best['scenario_pd_percent']:.4f}%**."
            ),
            "",
            (
                "This comparison reflects model outputs only and "
                "does not account for feasibility, cost, or an "
                "approval decision."
            ),
        ]
    )

    return "\n".join(lines)


def format_best_change_result(
    optimization_result: dict,
) -> str:
    if not optimization_result["success"]:
        return (
            "No evaluated single-variable scenario reduced the "
            "model-estimated PD."
        )

    candidate = optimization_result["best_candidate"]

    lines = [
        "### Largest Tested PD Reduction",
        "",
        (
            f"**Variable:** {candidate['label']}"
        ),
        (
            f"**Tested value:** "
            f"{candidate['scenario_value']}"
        ),
        (
            f"**Original PD:** "
            f"{candidate['original_pd_percent']:.4f}%"
        ),
        (
            f"**Scenario PD:** "
            f"{candidate['scenario_pd_percent']:.4f}%"
        ),
        (
            f"**PD change:** "
            f"{candidate['pd_change_percentage_points']:+.4f} "
            "percentage points"
        ),
        (
            f"**Risk tier:** "
            f"{candidate['original_risk_tier']} → "
            f"{candidate['scenario_risk_tier']}"
        ),
        "",
        (
            f"**Scenarios evaluated:** "
            f"{optimization_result['tested_candidates']}"
        ),
    ]

    alternatives = optimization_result.get(
        "alternatives",
        [],
    )

    if alternatives:
        lines.extend(
            [
                "",
                "### Other Strong Tested Changes",
            ]
        )

        for alternative in alternatives:
            lines.append(
                f"- {alternative['label']} = "
                f"{alternative['scenario_value']}: "
                f"{alternative['scenario_pd_percent']:.4f}% PD"
            )

    lines.extend(
        [
            "",
            optimization_result["message"],
        ]
    )

    return "\n".join(lines)


st.set_page_config(
    page_title="AI Mortgage Credit Risk Assistant",
    page_icon="🏠",
    layout="wide",
)


st.markdown(
    """
<style>
    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    .app-header {
        padding: 0.25rem 0 1.25rem 0;
    }

    .app-title {
        color: #0f172a;
        font-size: 2.55rem;
        font-weight: 750;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }

    .app-subtitle {
        color: #475569;
        font-size: 1.05rem;
        line-height: 1.5;
        margin-bottom: 0.3rem;
    }

    .portfolio-caption {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .section-title {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 1.1rem;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        min-height: 138px;
        padding: 1.25rem 1.4rem;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
    }

    .primary-driver-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.25;
    }

    .metric-detail {
        color: #94a3b8;
        font-size: 0.84rem;
        line-height: 1.4;
        margin-top: 0.65rem;
    }

    .risk-card {
        border-left: 6px solid;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        min-height: 138px;
        padding: 1.25rem 1.4rem;
    }

    .risk-label {
        font-size: 0.88rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }

    .risk-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
    }

    .risk-detail {
        font-size: 0.84rem;
        line-height: 1.4;
        margin-top: 0.65rem;
        opacity: 0.8;
    }

    .explanation-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        margin-top: 1rem;
        padding: 1.25rem 1.4rem;
    }

    .explanation-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .explanation-body {
        color: #334155;
        line-height: 1.65;
    }

    .driver-heading {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .driver-row {
        margin-bottom: 1.25rem;
    }

    .driver-row:last-child {
        margin-bottom: 0.25rem;
    }

    .driver-row-header {
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }

    .driver-name {
        color: #334155;
        font-size: 0.94rem;
        font-weight: 600;
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .driver-value {
        color: #64748b;
        font-size: 0.88rem;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }

    .driver-track {
        background-color: #e2e8f0;
        border-radius: 999px;
        height: 13px;
        overflow: hidden;
        width: 100%;
    }

    .driver-fill {
        background:
            linear-gradient(90deg, #2563eb, #60a5fa);
        border-radius: 999px;
        height: 100%;
    }

    .reducer-card {
        align-items: center;
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        display: flex;
        gap: 0.9rem;
        margin-bottom: 0.85rem;
        padding: 1rem;
    }

    .reducer-icon {
        align-items: center;
        background-color: #dcfce7;
        border-radius: 50%;
        color: #15803d;
        display: flex;
        flex-shrink: 0;
        font-size: 1.2rem;
        font-weight: 700;
        height: 34px;
        justify-content: center;
        width: 34px;
    }

    .reducer-content {
        flex-grow: 1;
    }

    .reducer-label {
        color: #166534;
        font-size: 0.98rem;
        font-weight: 700;
    }

    .reducer-description {
        color: #4b7c5a;
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }

    .reducer-value {
        color: #166534;
        font-size: 0.95rem;
        font-variant-numeric: tabular-nums;
        font-weight: 700;
        white-space: nowrap;
    }

    .disclaimer {
        color: #64748b;
        font-size: 0.82rem;
        padding-top: 1rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


service = load_service()
ai_analyst = load_ai_analyst()


if "risk_result" not in st.session_state:
    st.session_state.risk_result = None

if "current_application" not in st.session_state:
    st.session_state.current_application = None

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

if "scenario_messages" not in st.session_state:
    st.session_state.scenario_messages = []

if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None


with st.sidebar:
    st.header("Borrower Inputs")

    st.caption(
        "Enter the borrower characteristics used by the "
        "12-month default model."
    )

    credit_score = st.number_input(
        "Credit score",
        min_value=300,
        max_value=850,
        value=725,
        step=1,
        help="Original borrower credit score.",
    )

    cltv = st.number_input(
        "Combined loan-to-value ratio (%)",
        min_value=0.0,
        max_value=200.0,
        value=85.0,
        step=1.0,
        help=(
            "Total mortgage debt divided by the "
            "property value."
        ),
    )

    dti = st.number_input(
        "Debt-to-income ratio (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=1.0,
        help=(
            "Borrower's monthly debt obligations divided "
            "by monthly income."
        ),
    )

    interest_rate = st.number_input(
        "Original interest rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=3.5,
        step=0.1,
        format="%.2f",
    )

    number_of_borrowers = st.selectbox(
        "Number of borrowers",
        options=[1, 2, 3, 4, 5, 6],
        index=1,
    )

    score_button = st.button(
        "Evaluate Borrower",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        "The application estimates default risk only. "
        "It does not make an approval or denial decision."
    )


if score_button:
    application = {
        "credit_score": int(credit_score),
        "original_combined_loan_to_value": float(cltv),
        "original_debt_to_income_ratio": float(dti),
        "original_interest_rate": float(interest_rate),
        "number_of_borrowers": int(
            number_of_borrowers
        ),
    }

    try:
        st.session_state.risk_result = service.evaluate(
            application
        )

        st.session_state.current_application = application
        st.session_state.ai_explanation = None
        st.session_state.scenario_messages = []
        st.session_state.scenario_result = None

    except Exception as error:
        st.error(f"Unable to score borrower: {error}")
        st.stop()


st.markdown(
    """
<div class="app-header">
    <div class="app-title">
        AI Mortgage Credit Risk Assistant
    </div>
    <div class="app-subtitle">
        Estimate and explain a borrower's 12-month probability
        of mortgage default.
    </div>
    <div class="portfolio-caption">
        Freddie Mac Wisconsin mortgage originations, 2020–2022
    </div>
</div>
""",
    unsafe_allow_html=True,
)


result = st.session_state.risk_result
application = st.session_state.current_application


if result is None:
    st.info(
        "Enter borrower information in the sidebar and select "
        "Evaluate Borrower to generate a risk assessment."
    )

else:
    risk_style = get_risk_style(result["risk_tier"])

    explanation = build_plain_english_explanation(
        result
    )

    ai_result = prepare_ai_result(result)

    increasers = sorted(
        result["top_risk_increasers"],
        key=lambda item: float(item["contribution"]),
        reverse=True,
    )

    reducers = sorted(
        result["top_risk_reducers"],
        key=lambda item: float(item["contribution"]),
    )

    primary_driver = (
        increasers[0]["label"]
        if increasers
        else "No material driver"
    )

    st.markdown(
        '<div class="section-title">'
        "Risk Assessment"
        "</div>",
        unsafe_allow_html=True,
    )

    pd_column, tier_column, driver_column = st.columns(3)

    with pd_column:
        render_metric_card(
            label="Calibrated 12-Month PD",
            value=(
                f"{result['calibrated_pd_percent']:.3f}%"
            ),
            detail=(
                "Estimated chance of default within the first 12 months"
            ),
        )

    with tier_column:
        render_risk_card(
            risk_tier=result["risk_tier"],
            risk_style=risk_style,
        )

    with driver_column:
        render_metric_card(
            label="Primary Risk Driver",
            value=primary_driver,
            detail=(
                "Largest model-based contributor to estimated default risk"
            ),
            value_class="primary-driver-value",
        )

    st.markdown(
        f"""
<div class="explanation-card">
    <div class="explanation-title">
        Credit Risk Summary
    </div>
    <div class="explanation-body">
        {escape(explanation)}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        "Model Drivers"
        "</div>",
        unsafe_allow_html=True,
    )

    driver_left, driver_right = st.columns(
        [1.35, 1],
        gap="large",
    )

    with driver_left:
        st.markdown(
            '<div class="driver-heading">'
            "Risk-Increasing Factors"
            "</div>",
            unsafe_allow_html=True,
        )

        render_driver_bars(increasers)

    with driver_right:
        st.markdown(
            '<div class="driver-heading">'
            "Risk-Reducing Factors"
            "</div>",
            unsafe_allow_html=True,
        )

        render_reducer_cards(reducers)

    with st.expander("View Ranked Driver Details"):
        ranked_rows = []

        for rank, driver in enumerate(
            increasers,
            start=1,
        ):
            ranked_rows.append(
                {
                    "Rank": rank,
                    "Risk factor": driver["label"],
                    "Effect": "Increases estimated risk",
                }
            )

        for driver in reducers:
            ranked_rows.append(
                {
                    "Rank": "—",
                    "Risk factor": driver["label"],
                    "Effect": "Reduces estimated risk",
                }
            )

        ranked_driver_table = pd.DataFrame(
            ranked_rows
        )

        if ranked_driver_table.empty:
            st.write("No model drivers are available.")

        else:
            st.dataframe(
                ranked_driver_table,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Drivers are ordered by their relative influence on "
            "the model prediction. Numeric internal model scores "
            "are intentionally not displayed."
        )

    st.markdown(
        '<div class="section-title">'
        "AI Credit Analysis"
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Generate a structured, plain-language explanation of "
        "the existing model result. The AI explains the model "
        "output only; it does not calculate or change the PD, "
        "and internal model score units are not shown."
    )

    if st.session_state.ai_explanation is None:
        if st.button(
            "Generate AI Explanation",
            use_container_width=True,
        ):
            with st.spinner(
                "Analyzing the model result..."
            ):
                try:
                    st.session_state.ai_explanation = (
                        ai_analyst.generate_explanation(
                            ai_result
                        )
                    )

                except Exception as error:
                    st.error(
                        "The AI explanation could not be "
                        f"generated: {error}"
                    )

            if st.session_state.ai_explanation:
                st.rerun()

    else:
        with st.container(border=True):
            st.markdown(
                st.session_state.ai_explanation
            )

        st.caption(
            "The explanation is based only on the supplied model "
            "output and supports human review. It does not "
            "constitute an approval or denial decision."
        )

        if st.button(
            "Regenerate Explanation",
            use_container_width=True,
        ):
            with st.spinner(
                "Regenerating the explanation..."
            ):
                try:
                    st.session_state.ai_explanation = (
                        ai_analyst.generate_explanation(
                            ai_result
                        )
                    )

                except Exception as error:
                    st.error(
                        "The AI explanation could not be "
                        f"regenerated: {error}"
                    )

            if st.session_state.ai_explanation:
                st.rerun()

    st.markdown(
        '<div class="section-title">'
        "Conversational Scenario Analysis"
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Ask an explicit what-if question, search for a PD or "
        "risk-tier target, compare two changes, or ask which "
        "single model input produces the largest tested PD reduction."
    )

    for message in st.session_state.scenario_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    scenario_question = st.chat_input(
        "Ask a scenario, target-search, comparison, or optimization question"
    )

    if scenario_question:
        st.session_state.scenario_messages.append(
            {
                "role": "user",
                "content": scenario_question,
            }
        )

        try:
            intent = parse_scenario_intent(
                scenario_question
            )

            if intent.intent == "scenario":
                scenario_application = apply_scenario_updates(
                    application,
                    intent.updates or {},
                )

                changes = describe_changes(
                    application,
                    scenario_application,
                )

                if not changes:
                    raise ScenarioParseError(
                        "The scenario matches the current borrower "
                        "values, so there is no change to evaluate."
                    )

                scenario_result = service.evaluate(
                    scenario_application
                )

                scenario_explanation = (
                    ai_analyst.generate_scenario_explanation(
                        original_result=result,
                        scenario_result=scenario_result,
                        changes=changes,
                    )
                )

                original_pd = float(
                    result["calibrated_pd_percent"]
                )

                new_pd = float(
                    scenario_result[
                        "calibrated_pd_percent"
                    ]
                )

                change_pp = new_pd - original_pd

                comparison = (
                    f"**Original PD:** {original_pd:.4f}%  \n"
                    f"**Scenario PD:** {new_pd:.4f}%  \n"
                    f"**Change:** "
                    f"{change_pp:+.4f} percentage points"
                )

                assistant_message = (
                    f"{comparison}\n\n"
                    f"{scenario_explanation}"
                )

                st.session_state.scenario_result = {
                    "application": scenario_application,
                    "result": scenario_result,
                    "changes": changes,
                }

            elif intent.intent == "target_pd":
                if (
                    intent.variable is None
                    or intent.target_pd_percent is None
                ):
                    raise ScenarioParseError(
                        "The PD target request is incomplete."
                    )

                optimization_result = (
                    find_value_for_pd_target(
                        application=application,
                        service=service,
                        variable=intent.variable,
                        target_pd_percent=float(
                            intent.target_pd_percent
                        ),
                    )
                )

                assistant_message = (
                    format_target_search_result(
                        optimization_result
                    )
                )

                st.session_state.scenario_result = (
                    optimization_result
                )

            elif intent.intent == "target_tier":
                if (
                    intent.variable is None
                    or intent.target_tier is None
                ):
                    raise ScenarioParseError(
                        "The risk-tier target request is incomplete."
                    )

                optimization_result = (
                    find_value_for_tier_target(
                        application=application,
                        service=service,
                        variable=intent.variable,
                        target_tier=intent.target_tier,
                    )
                )

                assistant_message = (
                    format_target_search_result(
                        optimization_result
                    )
                )

                st.session_state.scenario_result = (
                    optimization_result
                )

            elif intent.intent == "comparison":
                comparison_updates = (
                    intent.comparison_updates or []
                )

                if len(comparison_updates) < 2:
                    raise ScenarioParseError(
                        "Provide at least two explicit scenarios "
                        "to compare."
                    )

                comparisons = compare_single_changes(
                    application=application,
                    service=service,
                    changes=comparison_updates,
                )

                assistant_message = (
                    format_comparison_result(
                        comparisons
                    )
                )

                st.session_state.scenario_result = {
                    "comparisons": comparisons,
                }

            elif intent.intent == "optimization":
                optimization_result = (
                    find_best_single_variable_change(
                        application=application,
                        service=service,
                    )
                )

                assistant_message = (
                    format_best_change_result(
                        optimization_result
                    )
                )

                st.session_state.scenario_result = (
                    optimization_result
                )

            else:
                raise ScenarioParseError(
                    "The requested scenario type is not supported."
                )

            st.session_state.scenario_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                }
            )

        except ScenarioParseError as error:
            st.session_state.scenario_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I could not evaluate that request. "
                        f"{error}"
                    ),
                }
            )

        except Exception as error:
            st.session_state.scenario_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "The scenario analysis could not be "
                        f"completed. Details: {error}"
                    ),
                }
            )

        st.rerun()

    if st.session_state.scenario_messages:
        if st.button(
            "Clear Scenario Conversation",
            use_container_width=True,
        ):
            st.session_state.scenario_messages = []
            st.session_state.scenario_result = None
            st.rerun()

    st.markdown(
        '<div class="section-title">'
        "Borrower Profile"
        "</div>",
        unsafe_allow_html=True,
    )

    profile_data = pd.DataFrame(
        [
            {
                "Credit score": application[
                    "credit_score"
                ],
                "CLTV (%)": application[
                    "original_combined_loan_to_value"
                ],
                "DTI (%)": application[
                    "original_debt_to_income_ratio"
                ],
                "Interest rate (%)": application[
                    "original_interest_rate"
                ],
                "Borrowers": application[
                    "number_of_borrowers"
                ],
            }
        ]
    )

    st.dataframe(
        profile_data,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Technical Model Details"):
        st.write(
            "The raw probability is produced directly by "
            "the logistic regression model. The calibrated "
            "probability includes the validation-based "
            "intercept adjustment and is used for the "
            "displayed risk interpretation."
        )

        technical_left, technical_right = st.columns(2)

        technical_left.metric(
            "Raw Model PD",
            f"{result['raw_pd_percent']:.6f}%",
        )

        technical_right.metric(
            "Calibrated PD",
            f"{result['calibrated_pd_percent']:.6f}%",
        )

        technical_table = build_technical_table(result)

        st.dataframe(
            technical_table.style.format(
                {
                    "WOE": "{:.4f}",
                    "Coefficient": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


st.divider()

st.markdown(
    """
<div class="disclaimer">
    Demonstration model only. This application is not intended
    to make real lending, underwriting, approval, or denial
    decisions.
</div>
""",
    unsafe_allow_html=True,
)