from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.credit_risk_service import CreditRiskService


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

service = CreditRiskService(ARTIFACT_DIR)

application = {
    "credit_score": 725,
    "original_combined_loan_to_value": 85,
    "original_debt_to_income_ratio": 40,
    "original_interest_rate": 3.5,
    "number_of_borrowers": 2,
}

result = service.evaluate(application)

print(json.dumps(result, indent=2))