from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.scoring_engine import MortgageScoringEngine
from src.risk_interpreter import interpret_score


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


engine = MortgageScoringEngine(
    model_path=ARTIFACT_DIR / "logistic_model.pkl",
    calibration_path=ARTIFACT_DIR / "calibration_config.json",
    bins_path=ARTIFACT_DIR / "final_woe_bins.csv",
)


application = {
    "credit_score": 725,
    "original_combined_loan_to_value": 85,
    "original_debt_to_income_ratio": 40,
    "original_interest_rate": 3.5,
    "number_of_borrowers": 2,
}


score = engine.score(application)
interpretation = interpret_score(score)


print(f"Calibrated PD: {score.calibrated_pd:.4%}")
print(f"Risk tier: {interpretation.risk_tier}")
print(f"Summary: {interpretation.risk_summary}")


print("\nTop risk increasers:")

for driver in interpretation.top_risk_increasers:
    print(
        f"{driver['label']}: "
        f"{driver['contribution']:.6f}"
    )


print("\nRisk reducers:")

for driver in interpretation.top_risk_reducers:
    print(
        f"{driver['label']}: "
        f"{driver['contribution']:.6f}"
    )