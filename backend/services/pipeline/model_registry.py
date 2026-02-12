"""Lazy-loading model registry for the 6-model pipeline.

Follows the same pattern as resume_ner.py: global singleton, loaded on first use.
"""

import logging
from typing import TYPE_CHECKING

from services.pipeline.base import BaseModelService

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_registry: dict[str, BaseModelService] = {}


def _create_model(name: str) -> BaseModelService:
    """Factory: create a model service by name with deferred imports."""
    if name == "m1_jd_extractor":
        from services.pipeline.m1_jd_extractor import JDExtractorService
        return JDExtractorService()
    elif name == "m2_resume_extractor":
        from services.pipeline.m2_resume_extractor import ResumeExtractorService
        return ResumeExtractorService()
    elif name == "m3_skills_comparator":
        from services.pipeline.m3_skills_comparator import SkillsComparatorService
        return SkillsComparatorService()
    elif name == "m4_exp_edu_comparator":
        from services.pipeline.m4_exp_edu_comparator import ExpEduComparatorService
        return ExpEduComparatorService()
    elif name == "m5_judge":
        from services.pipeline.m5_judge import JudgeService
        return JudgeService()
    elif name == "m6_verdict":
        from services.pipeline.m6_verdict import VerdictService
        return VerdictService()
    else:
        raise ValueError(f"Unknown model: {name}")


def get_model(name: str) -> BaseModelService:
    """Get a model service by name, creating and loading it on first access."""
    if name not in _registry:
        _registry[name] = _create_model(name)
    svc = _registry[name]
    svc.ensure_loaded()
    return svc


def preload(*names: str) -> None:
    """Pre-load multiple models (e.g. at startup)."""
    for name in names:
        get_model(name)


def clear() -> None:
    """Unload all models. Useful for testing."""
    _registry.clear()
