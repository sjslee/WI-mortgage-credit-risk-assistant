from __future__ import annotations

from pathlib import Path

from src.scoring_engine import MortgageScoringEngine
from src.risk_interpreter import interpret_score


class CreditRiskService:
    def __init__(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)

        self.engine = MortgageScoringEngine(
            model_path=artifact_dir / "logistic_model.pkl",
            calibration_path=artifact_dir / "calibration_config.json",
            bins_path=artifact_dir / "final_woe_bins.csv",
        )

    def evaluate(self, application: dict) -> dict:
        score = self.engine.score(application)
        interpretation = interpret_score(score)

        return {
            "borrower_inputs": application,
            "raw_pd": score.raw_pd,
            "calibrated_pd": score.calibrated_pd,
            "raw_pd_percent": score.raw_pd * 100,
            "calibrated_pd_percent": score.calibrated_pd * 100,
            "risk_tier": interpretation.risk_tier,
            "risk_summary": interpretation.risk_summary,
            "woe_values": score.woe_values,
            "feature_contributions": score.feature_contributions,
            "top_risk_increasers": interpretation.top_risk_increasers,
            "top_risk_reducers": interpretation.top_risk_reducers,
        }