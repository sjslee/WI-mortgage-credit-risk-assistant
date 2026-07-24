import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ai_credit_analyst import AICreditAnalyst
from src.credit_risk_service import CreditRiskService


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


service = CreditRiskService(ARTIFACT_DIR)

borrower = {
    "credit_score": 725,
    "original_combined_loan_to_value": 85.0,
    "original_debt_to_income_ratio": 40.0,
    "original_interest_rate": 3.5,
    "number_of_borrowers": 2,
}


result = service.evaluate(borrower)

analyst = AICreditAnalyst(
    provider="ollama",
)

explanation = analyst.generate_explanation(result)

print("\nAI CREDIT ANALYST")
print("=" * 60)
print(explanation)