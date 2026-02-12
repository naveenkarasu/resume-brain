"""Model 3: Skills Comparator - Taxonomy-aware skill matching.

Uses a fine-tuned SentenceTransformer (JobBERT-v2 + Router projection)
trained on ESCO/O*NET/Nesta triplets for semantic skill matching.

Falls back to existing skill_extractor.compute_skill_overlap() if model not available.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from models.schemas.jd_extracted import JDExtracted
from models.schemas.resume_extracted import ResumeExtracted
from models.schemas.skills_comparison import SkillMatch, SkillsComparison
from services.pipeline.base import BaseModelService

logger = logging.getLogger(__name__)

TRAINED_MODEL_DIR = Path("training/models/m3_skills_comparator")


class SkillsComparatorService(BaseModelService):
    model_name = "m3_skills_comparator"

    def __init__(self) -> None:
        self._encoder = None
        self._use_fallback = False

    def load(self) -> None:
        # Try loading the fine-tuned model from training output
        if (TRAINED_MODEL_DIR / "model.safetensors").exists():
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(str(TRAINED_MODEL_DIR))
                logger.info("M3 Skills Comparator loaded fine-tuned model from %s", TRAINED_MODEL_DIR)
                return
            except Exception as e:
                logger.warning("Failed to load fine-tuned M3 model: %s", e)

        # Fall back to shared base JobBERT-v2 (no fine-tuning)
        try:
            from services.similarity import _get_sbert_model
            self._encoder = _get_sbert_model()
            if self._encoder is not None:
                logger.info("M3 using base JobBERT-v2 embeddings (no fine-tuning)")
                return
        except Exception as e:
            logger.warning("Failed to load JobBERT for M3: %s", e)

        logger.info("M3 model not available, using fallback skill matching")
        self._use_fallback = True

    def predict(self, **kwargs: Any) -> SkillsComparison:
        self.ensure_loaded()
        resume_extracted: ResumeExtracted = kwargs["resume_extracted"]
        jd_extracted: JDExtracted = kwargs["jd_extracted"]

        resume_skills = set(resume_extracted.skills)
        jd_required = set(jd_extracted.required_skills)
        jd_preferred = set(jd_extracted.preferred_skills)

        if self._use_fallback:
            return self._fallback_compare(resume_skills, jd_required, jd_preferred)

        return self._semantic_compare(resume_skills, jd_required, jd_preferred)

    def _encode_skills(self, skills: list[str]) -> np.ndarray:
        """Encode skills into embedding vectors."""
        if not skills:
            return np.array([])
        embeddings = self._encoder.encode(skills, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def _semantic_compare(
        self,
        resume_skills: set[str],
        jd_required: set[str],
        jd_preferred: set[str],
    ) -> SkillsComparison:
        """Compare skills using semantic embeddings."""
        jd_all = sorted(jd_required | jd_preferred)
        resume_list = sorted(resume_skills)

        if not jd_all:
            return SkillsComparison(
                extra_skills=resume_list,
                skill_coverage=0.5,
                required_coverage=0.5,
            )

        if not resume_list:
            return SkillsComparison(
                missing_required=sorted(jd_required),
                missing_preferred=sorted(jd_preferred),
                skill_coverage=0.0,
                required_coverage=0.0,
            )

        jd_emb = self._encode_skills(jd_all)
        resume_emb = self._encode_skills(resume_list)

        # Compute cosine similarity matrix
        sim_matrix = resume_emb @ jd_emb.T  # (n_resume, n_jd)

        # Build exact-match index for O(1) lookup
        resume_lower_to_idx = {s.lower(): i for i, s in enumerate(resume_list)}

        # Match each JD skill to best resume skill
        matched_skills: list[SkillMatch] = []
        missing_required: list[str] = []
        missing_preferred: list[str] = []
        matched_resume_indices: set[int] = set()

        MATCH_THRESHOLD = 0.75
        SEMANTIC_THRESHOLD = 0.85

        for j, jd_skill in enumerate(jd_all):
            is_req = jd_skill in jd_required
            best_idx = int(np.argmax(sim_matrix[:, j]))
            best_sim = float(sim_matrix[best_idx, j])

            # Check exact match first
            exact_match = resume_lower_to_idx.get(jd_skill.lower())

            if exact_match is not None:
                matched_skills.append(SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_list[exact_match],
                    similarity=1.0,
                    match_type="exact",
                    is_required=is_req,
                ))
                matched_resume_indices.add(exact_match)
            elif best_sim >= MATCH_THRESHOLD:
                match_type = "semantic" if best_sim >= SEMANTIC_THRESHOLD else "partial"
                matched_skills.append(SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_list[best_idx],
                    similarity=best_sim,
                    match_type=match_type,
                    is_required=is_req,
                ))
                matched_resume_indices.add(best_idx)
            else:
                if is_req:
                    missing_required.append(jd_skill)
                else:
                    missing_preferred.append(jd_skill)

        # Extra skills (in resume but not matched to JD)
        extra_skills = [
            resume_list[i] for i in range(len(resume_list))
            if i not in matched_resume_indices
        ]

        # Compute coverage
        n_matched = len(matched_skills)
        n_total = len(jd_all)
        skill_coverage = n_matched / n_total if n_total > 0 else 0.5

        n_req_matched = sum(1 for m in matched_skills if m.is_required)
        n_req_total = len(jd_required)
        required_coverage = n_req_matched / n_req_total if n_req_total > 0 else 0.5

        return SkillsComparison(
            matched_skills=matched_skills,
            missing_required=sorted(missing_required),
            missing_preferred=sorted(missing_preferred),
            extra_skills=sorted(extra_skills),
            skill_coverage=round(skill_coverage, 3),
            required_coverage=round(required_coverage, 3),
        )

    def _fallback_compare(
        self,
        resume_skills: set[str],
        jd_required: set[str],
        jd_preferred: set[str],
    ) -> SkillsComparison:
        """Fallback: use existing skill_extractor overlap logic."""
        from services.skill_extractor import compute_skill_overlap, _expand_skills

        jd_all = jd_required | jd_preferred
        expanded_resume = _expand_skills(resume_skills)

        matched_skills: list[SkillMatch] = []
        missing_required: list[str] = []
        missing_preferred: list[str] = []

        for skill in sorted(jd_all):
            is_req = skill in jd_required
            if skill in resume_skills:
                matched_skills.append(SkillMatch(
                    jd_skill=skill, resume_skill=skill,
                    similarity=1.0, match_type="exact", is_required=is_req,
                ))
            elif skill in expanded_resume:
                matched_skills.append(SkillMatch(
                    jd_skill=skill, resume_skill=skill,
                    similarity=0.5, match_type="taxonomy", is_required=is_req,
                ))
            else:
                if is_req:
                    missing_required.append(skill)
                else:
                    missing_preferred.append(skill)

        extra_skills = sorted(resume_skills - jd_all)
        coverage = compute_skill_overlap(resume_skills, jd_all, required_skills=jd_required)

        n_req_matched = sum(1 for m in matched_skills if m.is_required)
        n_req_total = len(jd_required)
        required_coverage = n_req_matched / n_req_total if n_req_total > 0 else 0.5

        return SkillsComparison(
            matched_skills=matched_skills,
            missing_required=missing_required,
            missing_preferred=missing_preferred,
            extra_skills=extra_skills,
            skill_coverage=round(coverage, 3),
            required_coverage=round(required_coverage, 3),
        )
