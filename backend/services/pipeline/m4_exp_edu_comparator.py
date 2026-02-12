"""Model 4: Experience/Education Comparator - LightGBM regressor.

Scores experience and education match using 14 handcrafted features:
    years_gap, title_cosine_sim, edu_gap, field_match, domain_sim,
    career_velocity, num_roles, avg_tenure, has_leadership, etc.

Falls back to section_parser.compute_experience_match() if model not available.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from models.schemas.exp_edu_comparison import ExpEduComparison
from models.schemas.jd_extracted import JDExtracted
from models.schemas.resume_extracted import ResumeExtracted
from services.pipeline.base import BaseModelService

logger = logging.getLogger(__name__)

# Education level ordinal mapping
_EDU_ORDINAL = {"": 0, "associate": 1, "bachelors": 2, "masters": 3, "phd": 4}

# Leadership title keywords
_LEADERSHIP_KEYWORDS = {"lead", "senior", "principal", "staff", "director", "vp",
                         "head", "chief", "manager", "architect"}

FEATURE_NAMES = [
    "years_gap",
    "resume_years",
    "required_years",
    "title_cosine_sim",
    "edu_gap",
    "field_match",
    "num_roles",
    "avg_tenure_months",
    "has_leadership",
    "career_velocity",
    "domain_sim",
    "num_skills",
    "edu_level_ordinal",
    "jd_edu_ordinal",
]


class ExpEduComparatorService(BaseModelService):
    model_name = "m4_exp_edu_comparator"

    def __init__(self) -> None:
        self._model = None
        self._use_fallback = False
        self._sbert = None

    def load(self) -> None:
        model_path = Path("training/models/m4_exp_edu_comparator/model.txt")
        if model_path.exists():
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=str(model_path))
                logger.info("M4 LightGBM model loaded from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load M4 model: %s", e)
                self._use_fallback = True
        else:
            logger.info("M4 model not found, using fallback scoring")
            self._use_fallback = True

        # Load JobBERT for title similarity (shared with similarity.py)
        try:
            from services.similarity import _get_sbert_model
            self._sbert = _get_sbert_model()
        except Exception:
            pass

    def predict(self, **kwargs: Any) -> ExpEduComparison:
        self.ensure_loaded()
        resume_extracted: ResumeExtracted = kwargs["resume_extracted"]
        jd_extracted: JDExtracted = kwargs["jd_extracted"]

        features = self._extract_features(resume_extracted, jd_extracted)

        if self._use_fallback:
            return self._fallback_score(resume_extracted, jd_extracted, features)
        return self._model_score(features)

    def _extract_features(
        self, resume: ResumeExtracted, jd: JDExtracted
    ) -> dict[str, float]:
        """Extract 14 handcrafted features for the LightGBM model."""
        resume_years = resume.total_years_experience
        required_years = jd.required_years
        years_gap = resume_years - required_years

        # Title similarity via JobBERT
        title_sim = 0.0
        if self._sbert and resume.work_experience:
            most_recent_title = resume.work_experience[0].title
            if most_recent_title and jd.responsibilities:
                try:
                    jd_title_text = " ".join(jd.responsibilities[:3])
                    embs = self._sbert.encode(
                        [most_recent_title, jd_title_text], convert_to_numpy=True
                    )
                    from sklearn.metrics.pairwise import cosine_similarity
                    title_sim = float(cosine_similarity(embs[0:1], embs[1:2])[0][0])
                except Exception:
                    pass

        # Education gap
        resume_edu = _EDU_ORDINAL.get(resume.highest_education, 0)
        jd_edu_reqs = jd.education_requirements
        jd_edu = 0
        for req in jd_edu_reqs:
            req_lower = req.lower()
            for level, ordinal in _EDU_ORDINAL.items():
                if level and level in req_lower:
                    jd_edu = max(jd_edu, ordinal)
        edu_gap = resume_edu - jd_edu

        # Field match (heuristic: check if JD domain keywords appear in education)
        field_match = 0.0
        if resume.education and jd.domain:
            edu_text = " ".join(
                (e.field + " " + e.degree) for e in resume.education
            ).lower()
            for d in jd.domain:
                if d.lower() in edu_text:
                    field_match = 1.0
                    break

        # Role count and tenure
        num_roles = len(resume.work_experience)
        tenures = [we.duration_months for we in resume.work_experience if we.duration_months > 0]
        avg_tenure = np.mean(tenures) if tenures else 0.0

        # Leadership detection
        has_leadership = 0.0
        for we in resume.work_experience:
            title_words = set(we.title.lower().split())
            if title_words & _LEADERSHIP_KEYWORDS:
                has_leadership = 1.0
                break

        # Career velocity (promotions per year)
        career_velocity = num_roles / max(resume_years, 1.0) if resume_years > 0 else 0.0

        # Domain similarity via JobBERT
        domain_sim = 0.0
        if self._sbert and resume.work_experience and jd.domain:
            try:
                resume_domain = " ".join(
                    we.title + " " + we.description
                    for we in resume.work_experience[:3]
                )
                jd_domain = " ".join(jd.domain)
                if resume_domain.strip() and jd_domain.strip():
                    embs = self._sbert.encode(
                        [resume_domain, jd_domain], convert_to_numpy=True
                    )
                    from sklearn.metrics.pairwise import cosine_similarity
                    domain_sim = float(cosine_similarity(embs[0:1], embs[1:2])[0][0])
            except Exception:
                pass

        return {
            "years_gap": years_gap,
            "resume_years": resume_years,
            "required_years": required_years,
            "title_cosine_sim": title_sim,
            "edu_gap": float(edu_gap),
            "field_match": field_match,
            "num_roles": float(num_roles),
            "avg_tenure_months": float(avg_tenure),
            "has_leadership": has_leadership,
            "career_velocity": career_velocity,
            "domain_sim": domain_sim,
            "num_skills": float(len(resume.skills)),
            "edu_level_ordinal": float(resume_edu),
            "jd_edu_ordinal": float(jd_edu),
        }

    def _model_score(self, features: dict[str, float]) -> ExpEduComparison:
        """Score using trained LightGBM model."""
        feature_vec = np.array([[features[name] for name in FEATURE_NAMES]])
        predictions = self._model.predict(feature_vec)[0]

        # Model outputs 4 scores: experience, education, domain, title
        if hasattr(predictions, "__len__") and len(predictions) >= 4:
            exp_score, edu_score, domain_score, title_score = predictions[:4]
        else:
            # Single output - distribute heuristically
            overall = float(predictions) if not hasattr(predictions, "__len__") else float(predictions[0])
            exp_score = overall
            edu_score = overall
            domain_score = overall * 0.8
            title_score = features["title_cosine_sim"] * 100

        return ExpEduComparison(
            experience_score=round(max(0, min(100, float(exp_score))), 1),
            education_score=round(max(0, min(100, float(edu_score))), 1),
            domain_score=round(max(0, min(100, float(domain_score))), 1),
            title_score=round(max(0, min(100, float(title_score))), 1),
            years_gap=features["years_gap"],
            title_cosine_sim=features["title_cosine_sim"],
            edu_gap=int(features["edu_gap"]),
            field_match=features["field_match"] > 0.5,
            career_velocity=round(features["career_velocity"], 2),
            domain_sim=features["domain_sim"],
        )

    def _fallback_score(
        self,
        resume: ResumeExtracted,
        jd: JDExtracted,
        features: dict[str, float],
    ) -> ExpEduComparison:
        """Fallback: use section_parser heuristics."""
        from services.section_parser import compute_experience_match

        exp_score = float(compute_experience_match(
            resume.total_years_experience, jd.required_years
        ))

        # Education score from ordinal gap
        edu_gap = int(features["edu_gap"])
        if edu_gap >= 0:
            edu_score = min(100.0, 70.0 + edu_gap * 15)
        else:
            edu_score = max(0.0, 70.0 + edu_gap * 20)

        # Domain/title scores from cosine similarity
        domain_score = features["domain_sim"] * 100
        title_score = features["title_cosine_sim"] * 100

        return ExpEduComparison(
            experience_score=round(exp_score, 1),
            education_score=round(edu_score, 1),
            domain_score=round(max(0, min(100, domain_score)), 1),
            title_score=round(max(0, min(100, title_score)), 1),
            years_gap=features["years_gap"],
            title_cosine_sim=features["title_cosine_sim"],
            edu_gap=edu_gap,
            field_match=features["field_match"] > 0.5,
            career_velocity=round(features["career_velocity"], 2),
            domain_sim=features["domain_sim"],
        )
