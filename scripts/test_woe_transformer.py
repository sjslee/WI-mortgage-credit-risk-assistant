from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.woe_transformer import WOETransformer


BINS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "final_woe_bins.csv"
)


transformer = WOETransformer(BINS_PATH)


sample_application = {
    "credit_score": 725,
    "original_combined_loan_to_value": 85,
    "original_debt_to_income_ratio": 40,
    "original_interest_rate": 3.5,
    "number_of_borrowers": 2,
}


transformed = transformer.transform_record(
    sample_application
)


print("Raw application:")
for key, value in sample_application.items():
    print(f"{key}: {value}")


print("\nWOE-transformed application:")
for key, value in transformed.items():
    print(f"{key}: {value:.6f}")


boundary_tests = [
    {
        "credit_score": 709,
        "original_combined_loan_to_value": 53,
        "original_debt_to_income_ratio": 25,
        "original_interest_rate": 2.99,
        "number_of_borrowers": 1,
    },
    {
        "credit_score": 710,
        "original_combined_loan_to_value": 54,
        "original_debt_to_income_ratio": 26,
        "original_interest_rate": 3.0,
        "number_of_borrowers": 2,
    },
    {
        "credit_score": 740,
        "original_combined_loan_to_value": 64,
        "original_debt_to_income_ratio": 36,
        "original_interest_rate": 6.5,
        "number_of_borrowers": 3,
    },
    {
        "credit_score": 775,
        "original_combined_loan_to_value": 82,
        "original_debt_to_income_ratio": 43,
        "original_interest_rate": 7.0,
        "number_of_borrowers": 6,
    },
]


print("\nBoundary tests:")

for index, application in enumerate(boundary_tests, start=1):
    result = transformer.transform_record(application)

    print(f"\nTest {index}")

    for key, value in result.items():
        print(f"{key}: {value:.6f}")