"""Model 1 output: structured entities extracted from a job description."""

from pydantic import BaseModel


class JDEntity(BaseModel):
    """A single named entity span from the JD."""
    text: str
    label: str  # e.g. SKILL, QUALIFICATION, EXPERIENCE_REQ, ...
    start: int = 0
    end: int = 0
    confidence: float = 1.0


class JDExtracted(BaseModel):
    """Structured output of the JD Extractor (Model 1).

    Entity types (9 BIO-tagged categories):
        SKILL, SOFT_SKILL, QUALIFICATION, EXPERIENCE_REQ, EDUCATION_REQ,
        CERTIFICATION, RESPONSIBILITY, TOOL, DOMAIN
    """
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    soft_skills: list[str] = []
    qualifications: list[str] = []
    experience_requirements: list[str] = []
    education_requirements: list[str] = []
    certifications: list[str] = []
    responsibilities: list[str] = []
    tools: list[str] = []
    domain: list[str] = []
    entities: list[JDEntity] = []  # raw NER spans for debugging
    required_years: float = 0.0  # parsed from experience_requirements
