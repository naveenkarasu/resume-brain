"""Model 5: Judge - Overall scoring from M3 + M4 signals.

LightGBM regressor combining 13 features from skills comparison and
experience/education comparison into a single 0-100 overall score.

Falls back to the legacy weighted formula if model not available.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from models.responses import ScoreBreakdown
from models.schemas.exp_edu_comparison import ExpEduComparison
from models.schemas.judge_result import JudgeResult
from models.schemas.skills_comparison import SkillsComparison
from services.pipeline.base import BaseModelService

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "skill_coverage",
    "required_coverage",
    "n_matched_skills",
    "n_missing_required",
    "n_missing_preferred",
    "n_extra_skills",
    "avg_match_similarity",
    "experience_score",
    "education_score",
    "domain_score",
    "title_score",
    "years_gap",
    "career_velocity",
]


class JudgeService(BaseModelService):
    model_name = "m5_judge"

    def __init__(self) -> None:
        self._model = None
        self._use_fallback = False

    def load(self) -> None:
        model_path = Path("training/models/m5_judge/model.txt")
        if model_path.exists():
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=str(model_path))
                logger.info("M5 Judge model loaded from %s", model_path)
                return
            except Exception as e:
                logger.warning("Failed to load M5 model: %s", e)

        logger.info("M5 model not found, using fallback scoring")
        self._use_fallback = True

    def predict(self, **kwargs: Any) -> JudgeResult:
        self.ensure_loaded()
        skills_comparison: SkillsComparison = kwargs["skills_comparison"]
        exp_edu_comparison: ExpEduComparison = kwargs["exp_edu_comparison"]

        features = self._extract_features(skills_comparison, exp_edu_comparison)

        if self._use_fallback:
            return self._fallback_score(skills_comparison, exp_edu_comparison, features)
        return self._model_score(features, skills_comparison, exp_edu_comparison)

    def _extract_features(
        self,
        skills: SkillsComparison,
        exp_edu: ExpEduComparison,
    ) -> dict[str, float]:
        """Extract 13-dim feature vector from M3 + M4 outputs."""
        matched_sims = [m.similarity for m in skills.matched_skills]
        avg_sim = float(np.mean(matched_sims)) if matched_sims else 0.0

        return {
            "skill_coverage": skills.skill_coverage,
            "required_coverage": skills.required_coverage,
            "n_matched_skills": float(len(skills.matched_skills)),
            "n_missing_required": float(len(skills.missing_required)),
            "n_missing_preferred": float(len(skills.missing_preferred)),
            "n_extra_skills": float(len(skills.extra_skills)),
            "avg_match_similarity": avg_sim,
            "experience_score": exp_edu.experience_score,
            "education_score": exp_edu.education_score,
            "domain_score": exp_edu.domain_score,
            "title_score": exp_edu.title_score,
            "years_gap": exp_edu.years_gap,
            "career_velocity": exp_edu.career_velocity,
        }

    def _model_score(
        self,
        features: dict[str, float],
        skills: SkillsComparison,
        exp_edu: ExpEduComparison,
    ) -> JudgeResult:
        """Score using trained LightGBM model."""
        feature_vec = np.array([[features[name] for name in FEATURE_NAMES]])
        prediction = self._model.predict(feature_vec)[0]
        overall = int(round(max(0, min(100, float(prediction)))))

        breakdown = ScoreBreakdown(
            skills_match=int(round(skills.skill_coverage * 100)),
            experience_match=int(round(exp_edu.experience_score)),
            education_match=int(round(exp_edu.education_score)),
            keywords_match=int(round(skills.required_coverage * 100)),
        )

        # Get feature importances from model
        importances = {}
        try:
            raw_imp = self._model.feature_importance(importance_type="gain")
            for name, imp in zip(FEATURE_NAMES, raw_imp):
                importances[name] = float(imp)
        except Exception:
            pass

        return JudgeResult(
            overall_score=overall,
            score_breakdown=breakdown,
            feature_vector=[features[name] for name in FEATURE_NAMES],
            feature_names=list(FEATURE_NAMES),
            feature_importances=importances,
        )

    def _fallback_score(
        self,
        skills: SkillsComparison,
        exp_edu: ExpEduComparison,
        features: dict[str, float],
    ) -> JudgeResult:
        """Fallback: weighted combination matching legacy formula proportions."""
        # Weights inspired by legacy: 30% semantic, 20% tfidf, 20% keyword,
        # 20% skill, 10% section. Mapped to M3+M4 signals:
        skill_score = skills.skill_coverage * 100
        exp_score = exp_edu.experience_score
        edu_score = exp_edu.education_score
        domain_score = exp_edu.domain_score

        overall = round(
            0.35 * skill_score
            + 0.30 * exp_score
            + 0.20 * edu_score
            + 0.15 * domain_score
        )
        overall = max(0, min(100, overall))

        breakdown = ScoreBreakdown(
            skills_match=int(round(skill_score)),
            experience_match=int(round(exp_score)),
            education_match=int(round(edu_score)),
            keywords_match=int(round(skills.required_coverage * 100)),
        )

        return JudgeResult(
            overall_score=overall,
            score_breakdown=breakdown,
            feature_vector=[features[name] for name in FEATURE_NAMES],
            feature_names=list(FEATURE_NAMES),
        )
