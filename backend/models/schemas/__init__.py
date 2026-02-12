"""Inter-model Pydantic contracts for the 6-model pipeline."""

from models.schemas.jd_extracted import JDExtracted
from models.schemas.resume_extracted import ResumeExtracted
from models.schemas.skills_comparison import SkillsComparison
from models.schemas.exp_edu_comparison import ExpEduComparison
from models.schemas.judge_result import JudgeResult
from models.schemas.verdict_result import VerdictResult

__all__ = [
    "JDExtracted",
    "ResumeExtracted",
    "SkillsComparison",
    "ExpEduComparison",
    "JudgeResult",
    "VerdictResult",
]
