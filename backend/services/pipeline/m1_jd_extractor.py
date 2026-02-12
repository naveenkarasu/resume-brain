"""Model 1: JD Extractor - Token-level NER for job descriptions.

Extracts 9 entity types from job descriptions using BIO tagging:
    SKILL, SOFT_SKILL, QUALIFICATION, EXPERIENCE_REQ, EDUCATION_REQ,
    CERTIFICATION, RESPONSIBILITY, TOOL, DOMAIN

Base model: bert-base-cased (110M params), fine-tuned on unified JD NER data.
Falls back to keyword_extractor + section heuristics if model not available.
"""

import logging
import re
from pathlib import Path
from typing import Any

from models.schemas.jd_extracted import JDEntity, JDExtracted
from services.pipeline.base import BaseModelService

logger = logging.getLogger(__name__)

# BIO label set for 9 entity types (19 tags + O)
LABEL_LIST = [
    "O",
    "B-SKILL", "I-SKILL",
    "B-SOFT_SKILL", "I-SOFT_SKILL",
    "B-QUALIFICATION", "I-QUALIFICATION",
    "B-EXPERIENCE_REQ", "I-EXPERIENCE_REQ",
    "B-EDUCATION_REQ", "I-EDUCATION_REQ",
    "B-CERTIFICATION", "I-CERTIFICATION",
    "B-RESPONSIBILITY", "I-RESPONSIBILITY",
    "B-TOOL", "I-TOOL",
    "B-DOMAIN", "I-DOMAIN",
]

ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

# Map BIO entity types to JDExtracted fields
_ENTITY_FIELD_MAP = {
    "SKILL": "required_skills",
    "SOFT_SKILL": "soft_skills",
    "QUALIFICATION": "qualifications",
    "EXPERIENCE_REQ": "experience_requirements",
    "EDUCATION_REQ": "education_requirements",
    "CERTIFICATION": "certifications",
    "RESPONSIBILITY": "responsibilities",
    "TOOL": "tools",
    "DOMAIN": "domain",
}

# Experience years pattern for parsing experience_requirements
_EXP_YEARS_RE = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp\b)",
    re.IGNORECASE,
)


class JDExtractorService(BaseModelService):
    model_name = "m1_jd_extractor"

    def __init__(self) -> None:
        self._pipeline = None
        self._use_fallback = False

    def load(self) -> None:
        model_dir = Path("training/models/m1_jd_extractor")
        if model_dir.exists():
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "token-classification",
                    model=str(model_dir),
                    aggregation_strategy="simple",
                )
                logger.info("M1 JD Extractor model loaded from %s", model_dir)
                return
            except Exception as e:
                logger.warning("Failed to load M1 model: %s", e)

        logger.info("M1 model not found, using fallback extraction")
        self._use_fallback = True

    def predict(self, **kwargs: Any) -> JDExtracted:
        self.ensure_loaded()
        jd_text: str = kwargs["jd_text"]

        if self._use_fallback:
            return self._fallback_extract(jd_text)
        return self._model_extract(jd_text)

    def _model_extract(self, jd_text: str) -> JDExtracted:
        """Extract entities using the trained NER model."""
        chunks = _chunk_text(jd_text)
        all_entities: list[JDEntity] = []

        for chunk in chunks:
            results = self._pipeline(chunk)
            for r in results:
                if r["score"] < 0.5:
                    continue
                label = r["entity_group"]
                text = r["word"].strip()
                text = re.sub(r"\s*##\s*", "", text).strip()
                if text and len(text) > 1:
                    all_entities.append(JDEntity(
                        text=text,
                        label=label,
                        start=r.get("start", 0),
                        end=r.get("end", 0),
                        confidence=r["score"],
                    ))

        return _entities_to_schema(all_entities, jd_text)

    def _fallback_extract(self, jd_text: str) -> JDExtracted:
        """Fallback: use existing keyword_extractor + heuristics."""
        from services import keyword_extractor
        from services.skill_extractor import extract_skills_combined
        from services.section_parser import extract_required_years

        jd_keywords = keyword_extractor.extract_keywords_combined(jd_text)
        jd_skills = list(extract_skills_combined(jd_text))
        required_text, preferred_text = keyword_extractor.extract_jd_priority_sections(jd_text)
        required_years = extract_required_years(jd_text)

        # Split skills into required vs preferred based on JD sections
        required_skills_set = set(extract_skills_combined(required_text)) if required_text != jd_text else set(jd_skills)
        preferred_skills = [s for s in jd_skills if s not in required_skills_set]

        return JDExtracted(
            required_skills=sorted(required_skills_set),
            preferred_skills=sorted(preferred_skills),
            qualifications=[kw for kw in jd_keywords if kw not in jd_skills],
            required_years=required_years,
        )


def _chunk_text(text: str, max_words: int = 300) -> list[str]:
    """Split text into chunks for BERT's 512 token limit."""
    paragraphs = re.split(r"\n\n+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        pw = len(para.split())
        if current_words + pw > max_words and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_words = pw
        else:
            current.append(para)
            current_words += pw
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:2000]]


def _entities_to_schema(entities: list[JDEntity], jd_text: str) -> JDExtracted:
    """Convert raw NER entities into the JDExtracted schema."""
    fields: dict[str, list[str]] = {v: [] for v in _ENTITY_FIELD_MAP.values()}
    seen: dict[str, set[str]] = {v: set() for v in _ENTITY_FIELD_MAP.values()}

    for ent in entities:
        field = _ENTITY_FIELD_MAP.get(ent.label)
        if field and ent.text.lower() not in seen[field]:
            fields[field].append(ent.text)
            seen[field].add(ent.text.lower())

    # Parse required years from experience_requirements
    required_years = 0.0
    for req in fields.get("experience_requirements", []):
        for m in _EXP_YEARS_RE.finditer(req):
            years = float(m.group(1))
            if years > required_years:
                required_years = years

    # Fallback: parse from full text
    if required_years == 0.0:
        for m in _EXP_YEARS_RE.finditer(jd_text):
            years = float(m.group(1))
            if years > required_years:
                required_years = years

    return JDExtracted(
        required_skills=fields["required_skills"],
        soft_skills=fields["soft_skills"],
        qualifications=fields["qualifications"],
        experience_requirements=fields["experience_requirements"],
        education_requirements=fields["education_requirements"],
        certifications=fields["certifications"],
        responsibilities=fields["responsibilities"],
        tools=fields["tools"],
        domain=fields["domain"],
        entities=entities,
        required_years=required_years,
    )
