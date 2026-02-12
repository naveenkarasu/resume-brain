"""Model 2: Resume Extractor - Token-level NER for resumes.

Extends yashpwr/resume-ner-bert-v2 (11 types, 90.87% F1) to 14 types:
    +CERTIFICATION, +PROJECT_NAME, +PROJECT_TECHNOLOGY

Base model: yashpwr/resume-ner-bert-v2 with additional fine-tuning.
Falls back to existing resume_ner.py + section_parser.py if model not available.
"""

import logging
import re
from pathlib import Path
from typing import Any

from models.schemas.resume_extracted import (
    Education,
    Project,
    ResumeExtracted,
    WorkExperience,
)
from services.pipeline.base import BaseModelService

logger = logging.getLogger(__name__)


class ResumeExtractorService(BaseModelService):
    model_name = "m2_resume_extractor"

    def __init__(self) -> None:
        self._pipeline = None
        self._use_fallback = False

    def load(self) -> None:
        model_dir = Path("training/models/m2_resume_extractor")
        if model_dir.exists():
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "token-classification",
                    model=str(model_dir),
                    aggregation_strategy="simple",
                )
                logger.info("M2 Resume Extractor model loaded from %s", model_dir)
                return
            except Exception as e:
                logger.warning("Failed to load M2 model: %s", e)

        logger.info("M2 model not found, using fallback extraction")
        self._use_fallback = True

    def predict(self, **kwargs: Any) -> ResumeExtracted:
        self.ensure_loaded()
        resume_text: str = kwargs["resume_text"]

        if self._use_fallback:
            return self._fallback_extract(resume_text)
        return self._model_extract(resume_text)

    def _model_extract(self, resume_text: str) -> ResumeExtracted:
        """Extract entities using the fine-tuned NER model."""
        from services.resume_ner import _chunk_text

        chunks = _chunk_text(resume_text)
        raw_entities: dict[str, list[str]] = {}

        for chunk in chunks:
            results = self._pipeline(chunk)
            for r in results:
                if r["score"] < 0.5:
                    continue
                label = r["entity_group"].lower()
                text = r["word"].strip()
                text = re.sub(r"\s*##\s*", "", text).strip()
                if text and len(text) > 1:
                    raw_entities.setdefault(label, []).append(text)

        # Deduplicate
        for key in raw_entities:
            seen: set[str] = set()
            deduped: list[str] = []
            for val in raw_entities[key]:
                lower = val.lower()
                if lower not in seen:
                    seen.add(lower)
                    deduped.append(val)
            raw_entities[key] = deduped

        return _entities_to_schema(raw_entities, resume_text)

    def _fallback_extract(self, resume_text: str) -> ResumeExtracted:
        """Fallback: combine existing resume_ner + section_parser + skill_extractor."""
        from services.resume_ner import extract_resume_entities
        from services.section_parser import (
            extract_education_level,
            extract_experience_years,
            parse_sections,
        )
        from services.skill_extractor import extract_skills_combined

        ner = extract_resume_entities(resume_text)
        sections = parse_sections(resume_text)
        skills = sorted(extract_skills_combined(resume_text))
        years = extract_experience_years(resume_text, ner_entities=ner)
        edu_level = extract_education_level(resume_text, ner_entities=ner)

        # Build education entries from NER
        educations = []
        if ner.get("degrees") or ner.get("colleges"):
            educations.append(Education(
                institution=ner["colleges"][0] if ner.get("colleges") else "",
                degree=edu_level,
                graduation_year=ner["graduation_years"][0] if ner.get("graduation_years") else "",
            ))

        # Build work experience from NER
        experiences = []
        companies = ner.get("companies", [])
        designations = ner.get("designations", [])
        for i in range(max(len(companies), len(designations))):
            experiences.append(WorkExperience(
                company=companies[i] if i < len(companies) else "",
                title=designations[i] if i < len(designations) else "",
            ))

        return ResumeExtracted(
            name=ner["name"][0] if ner.get("name") else "",
            email=ner["email"][0] if ner.get("email") else "",
            phone=ner["phone"][0] if ner.get("phone") else "",
            location=ner["locations"][0] if ner.get("locations") else "",
            skills=skills,
            work_experience=experiences,
            education=educations,
            total_years_experience=years,
            highest_education=edu_level,
            sections_found=sorted(sections.keys()),
        )


def _entities_to_schema(raw: dict[str, list[str]], text: str) -> ResumeExtracted:
    """Convert raw NER entity dict to ResumeExtracted schema."""
    from services.section_parser import (
        extract_education_level,
        extract_experience_years,
        parse_sections,
    )

    sections = parse_sections(text)
    years = extract_experience_years(text)
    edu_level = extract_education_level(text)

    # Map NER labels to schema fields
    name = raw.get("name", [""])[0] if raw.get("name") else ""
    email = raw.get("email address", raw.get("email", [""]))[0] if raw.get("email address", raw.get("email")) else ""
    phone = raw.get("phone number", raw.get("phone", [""]))[0] if raw.get("phone number", raw.get("phone")) else ""
    location = raw.get("location", [""])[0] if raw.get("location") else ""

    skills = sorted(set(raw.get("skills", [])))
    certifications = raw.get("certification", [])

    # Build work experience entries
    companies = raw.get("companies worked at", raw.get("companies", []))
    designations = raw.get("designation", raw.get("designations", []))
    experiences = []
    for i in range(max(len(companies), len(designations), 0)):
        experiences.append(WorkExperience(
            company=companies[i] if i < len(companies) else "",
            title=designations[i] if i < len(designations) else "",
        ))

    # Build education entries
    colleges = raw.get("college name", raw.get("colleges", []))
    degrees = raw.get("degree", raw.get("degrees", []))
    grad_years = raw.get("graduation year", raw.get("graduation_years", []))
    educations = []
    for i in range(max(len(colleges), len(degrees), 0)):
        educations.append(Education(
            institution=colleges[i] if i < len(colleges) else "",
            degree=degrees[i] if i < len(degrees) else "",
            graduation_year=grad_years[i] if i < len(grad_years) else "",
        ))

    # Build project entries from new entity types
    project_names = raw.get("project_name", raw.get("project name", []))
    project_techs = raw.get("project_technology", raw.get("project technology", []))
    projects = []
    for i, name_val in enumerate(project_names):
        projects.append(Project(
            name=name_val,
            technologies=project_techs if i == 0 else [],
        ))

    return ResumeExtracted(
        name=name,
        email=email,
        phone=phone,
        location=location,
        skills=skills,
        work_experience=experiences,
        education=educations,
        certifications=certifications,
        projects=projects,
        total_years_experience=years,
        highest_education=edu_level,
        sections_found=sorted(sections.keys()),
    )
