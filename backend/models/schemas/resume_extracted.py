"""Model 2 output: structured entities extracted from a resume."""

from pydantic import BaseModel


class WorkExperience(BaseModel):
    """A single work experience entry."""
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_months: int = 0
    description: str = ""
    skills_used: list[str] = []


class Education(BaseModel):
    """A single education entry."""
    institution: str = ""
    degree: str = ""  # e.g. "bachelors", "masters", "phd"
    field: str = ""
    graduation_year: str = ""


class Project(BaseModel):
    """A single project entry."""
    name: str = ""
    description: str = ""
    technologies: list[str] = []


class ResumeExtracted(BaseModel):
    """Structured output of the Resume Extractor (Model 2).

    Extends yashpwr's 11 entity types to 14:
        +CERTIFICATION, +PROJECT_NAME, +PROJECT_TECHNOLOGY
    """
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    skills: list[str] = []
    work_experience: list[WorkExperience] = []
    education: list[Education] = []
    certifications: list[str] = []
    projects: list[Project] = []
    total_years_experience: float = 0.0
    highest_education: str = ""  # phd, masters, bachelors, associate
    sections_found: list[str] = []
