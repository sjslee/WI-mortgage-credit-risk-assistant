from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.scoring_engine import MortgageScoringEngine


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


engine = MortgageScoringEngine(
    model_path=ARTIFACT_DIR / "logistic_model.pkl",
    calibration_path=ARTIFACT_DIR / "calibration_config.json",
    bins_path=ARTIFACT_DIR / "final_woe_bins.csv",
)


sample_application = {
    "credit_score": 725,
    "original_combined_loan_to_value": 85,
    "original_debt_to_income_ratio": 40,
    "original_interest_rate": 3.5,
    "number_of_borrowers": 2,
}


result = engine.score(sample_application)


print("Raw borrower inputs:")

for key, value in sample_application.items():
    print(f"{key}: {value}")


print("\nWOE values:")

for key, value in result.woe_values.items():
    print(f"{key}: {value:.6f}")


print("\nFeature contributions:")

for feature, details in result.feature_contributions.items():
    print(
        f"{feature}: "
        f"{details['log_odds_contribution']:.6f} "
        f"({details['direction']})"
    )


print("\nScore results:")
print(f"Raw logit: {result.raw_logit:.6f}")
print(f"Raw PD: {result.raw_pd:.6%}")
print(f"Calibrated logit: {result.calibrated_logit:.6f}")
print(f"Calibrated PD: {result.calibrated_pd:.6%}")