"""Model 4 output: experience and education comparison scores."""

from pydantic import BaseModel


class ExpEduComparison(BaseModel):
    """Structured output of the Exp/Edu Comparator (Model 4).

    LightGBM regressor on 14 handcrafted features producing 4 sub-scores.
    """
    experience_score: float = 0.0  # 0-100
    education_score: float = 0.0  # 0-100
    domain_score: float = 0.0  # 0-100, how well career domain matches JD
    title_score: float = 0.0  # 0-100, job title similarity

    # Feature details for interpretability
    years_gap: float = 0.0  # resume_years - required_years
    title_cosine_sim: float = 0.0  # JobBERT cosine of most recent title vs JD title
    edu_gap: int = 0  # ordinal diff (resume_level - required_level)
    field_match: bool = False  # education field matches JD domain
    career_velocity: float = 0.0  # promotions per year
    domain_sim: float = 0.0  # domain embedding similarity
