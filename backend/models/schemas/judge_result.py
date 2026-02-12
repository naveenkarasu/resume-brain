"""Model 5 output: overall scoring judgment combining M3 + M4 signals."""

from pydantic import BaseModel

from models.responses import ScoreBreakdown


class JudgeResult(BaseModel):
    """Structured output of the Judge (Model 5).

    LightGBM regressor combining skill and experience signals into
    a single 0-100 overall score with category breakdown.
    """
    overall_score: int = 0  # 0-100
    score_breakdown: ScoreBreakdown = ScoreBreakdown()

    # 13-dim input feature vector (for interpretability / debugging)
    feature_vector: list[float] = []
    feature_names: list[str] = []
    feature_importances: dict[str, float] = {}  # from trained model
