"""Model 3 output: taxonomy-aware skill comparison between resume and JD."""

from pydantic import BaseModel


class SkillMatch(BaseModel):
    """A single skill match with similarity details."""
    jd_skill: str
    resume_skill: str = ""  # empty if unmatched
    similarity: float = 0.0  # 0.0-1.0 cosine similarity
    match_type: str = "none"  # exact, semantic, taxonomy, partial, none
    is_required: bool = False


class SkillsComparison(BaseModel):
    """Structured output of the Skills Comparator (Model 3).

    Provides taxonomy-aware matching between resume skills and JD skills
    using contrastive skill embeddings trained on ESCO/O*NET.
    """
    matched_skills: list[SkillMatch] = []
    missing_required: list[str] = []
    missing_preferred: list[str] = []
    extra_skills: list[str] = []  # resume skills not in JD
    skill_coverage: float = 0.0  # 0.0-1.0 weighted coverage
    required_coverage: float = 0.0  # coverage of required-only skills
    category_coverage: dict[str, float] = {}  # per-domain coverage
