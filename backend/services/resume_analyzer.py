"""Orchestrator: multi-layer hybrid analysis pipeline.

Pipeline:
1. Section parsing (structural analysis)
2. TF-IDF + SBERT similarity (lexical + semantic scoring)
3. Keyword extraction from JD (TF-IDF + dictionary combined)
4. Keyword matching against resume
5. Hybrid local score computation
6. Gemini qualitative analysis (optional, enhances results)
7. Combine local + LLM scores into final response
"""

import logging

from models.responses import (
    AnalysisResponse,
    BulletRewrite,
    ScoreBreakdown,
    SectionAnalysis,
)
from services import gemini_client, keyword_extractor, pdf_parser, prompt_builder
from services.section_parser import (
    compute_experience_match,
    compute_section_completeness,
    extract_education_level,
    extract_experience_years,
    extract_required_years,
    parse_sections,
)
from services.similarity import hybrid_similarity, section_similarities

logger = logging.getLogger(__name__)

# Weights for computing hybrid local score
W_SEMANTIC = 0.35
W_TFIDF = 0.30
W_KEYWORD = 0.20
W_SECTION = 0.15


def _compute_local_overall(
    tfidf_score: float,
    semantic_score: float,
    keyword_overlap: float,
    section_completeness: float,
) -> int:
    """Compute weighted local score from all signals. Returns 0-100."""
    raw = (
        W_SEMANTIC * semantic_score
        + W_TFIDF * tfidf_score
        + W_KEYWORD * keyword_overlap
        + W_SECTION * section_completeness
    )
    return min(100, max(0, round(raw * 100)))


async def analyze(resume_text: str, job_description: str) -> AnalysisResponse:
    """Run full hybrid analysis pipeline."""
    bullets = pdf_parser.extract_bullets(resume_text)

    # --- Layer 1: Section parsing ---
    sections = parse_sections(resume_text)
    section_completeness = compute_section_completeness(sections)
    section_analysis = SectionAnalysis(
        detected_sections=sorted(sections.keys()),
        completeness=round(section_completeness, 2),
    )

    # --- Layer 1b: Education level detection ---
    education_level = extract_education_level(resume_text)

    # --- Layer 1c: Experience years extraction ---
    resume_years = extract_experience_years(resume_text)
    required_years = extract_required_years(job_description)
    experience_match_score = compute_experience_match(resume_years, required_years)

    # --- Layer 2: Similarity scores ---
    hybrid_score, tfidf_score, semantic_score = hybrid_similarity(
        resume_text, job_description
    )

    # --- Layer 2b: Section-level similarity ---
    section_sims = section_similarities(sections, job_description)

    # --- Layer 3: Keyword extraction (TF-IDF + dictionary combined) ---
    jd_keywords = keyword_extractor.extract_keywords_combined(job_description)

    # --- Layer 4: Keyword matching ---
    matched_keywords, missing_keywords = keyword_extractor.match_keywords(
        resume_text, jd_keywords
    )
    keyword_overlap = keyword_extractor.compute_keyword_overlap(
        matched_keywords, missing_keywords
    )

    # --- Layer 4b: Bullet quality scoring ---
    bullet_scores = pdf_parser.score_bullets(bullets, set(jd_keywords))

    # --- Layer 4c: Keyword density analysis ---
    all_keywords = matched_keywords + missing_keywords
    keyword_density = keyword_extractor.compute_keyword_density(resume_text, all_keywords)

    # --- Layer 5: Compute local hybrid score ---
    local_overall = _compute_local_overall(
        tfidf_score, semantic_score, keyword_overlap, section_completeness
    )

    # Build local score breakdown using section-level SBERT + factual experience
    skills_sim = section_sims.get("skills", max(0, semantic_score))
    education_sim = section_sims.get("education", section_completeness)

    local_breakdown = ScoreBreakdown(
        skills_match=min(100, round(max(keyword_overlap, skills_sim) * 100)),
        experience_match=experience_match_score,
        education_match=min(100, round(education_sim * 100)),
        keywords_match=min(100, round(tfidf_score * 100)),
    )

    # --- Layer 6: Gemini qualitative analysis ---
    scoring_prompt = prompt_builder.build_scoring_prompt(
        resume_text,
        job_description,
        local_tfidf_score=tfidf_score,
        local_semantic_score=semantic_score,
        local_matched=matched_keywords,
        local_missing=missing_keywords,
    )
    scoring_data = await gemini_client.generate_json(scoring_prompt)

    rewrite_prompt = prompt_builder.build_rewrite_prompt(
        resume_text, job_description, bullets
    )
    rewrite_data = await gemini_client.generate_json(rewrite_prompt)

    # --- Layer 7: Combine local + LLM results ---
    degraded = False
    scoring_method = "hybrid"

    if scoring_data:
        # Use LLM scores for qualitative fields, blend with local for overall
        llm_overall = scoring_data.get("overall_score", 0)
        breakdown_raw = scoring_data.get("score_breakdown", {})
        score_breakdown = ScoreBreakdown(
            skills_match=breakdown_raw.get("skills_match", local_breakdown.skills_match),
            experience_match=breakdown_raw.get("experience_match", local_breakdown.experience_match),
            education_match=breakdown_raw.get("education_match", local_breakdown.education_match),
            keywords_match=breakdown_raw.get("keywords_match", local_breakdown.keywords_match),
        )
        # Blend: 60% LLM + 40% local NLP for the overall score
        overall_score = round(0.6 * llm_overall + 0.4 * local_overall)

        # Merge keyword lists: prefer LLM's richer analysis, supplement with local
        llm_matched = scoring_data.get("matched_keywords", [])
        llm_missing = scoring_data.get("missing_keywords", [])
        matched_keywords = sorted(set(matched_keywords + llm_matched))
        missing_keywords = sorted(set(llm_missing + missing_keywords))

        summary = scoring_data.get("summary", "")
        strengths = scoring_data.get("strengths", [])
        weaknesses = scoring_data.get("weaknesses", [])
    else:
        # Fallback: local scores are still meaningful
        logger.warning("Gemini scoring unavailable, using local NLP pipeline")
        degraded = True
        scoring_method = "local_only"
        overall_score = local_overall
        score_breakdown = local_breakdown
        summary = (
            f"Analysis performed using local NLP pipeline (AI service unavailable). "
            f"TF-IDF similarity: {tfidf_score:.0%}, "
            f"Semantic similarity: {semantic_score:.0%}, "
            f"Keyword overlap: {keyword_overlap:.0%}."
        )
        strengths = [f"Matched keyword: {kw}" for kw in matched_keywords[:5]]
        weaknesses = [f"Missing keyword: {kw}" for kw in missing_keywords[:5]]

    # Parse rewrite results
    if rewrite_data:
        bullet_rewrites = [
            BulletRewrite(
                original=br.get("original", ""),
                rewritten=br.get("rewritten", ""),
                reason=br.get("reason", ""),
            )
            for br in rewrite_data.get("bullet_rewrites", [])
        ]
        ats_optimized_resume = rewrite_data.get("ats_optimized_resume", "")
    else:
        if not degraded:
            logger.warning("Gemini rewrite unavailable")
            degraded = True
        bullet_rewrites = []
        ats_optimized_resume = ""

    return AnalysisResponse(
        overall_score=overall_score,
        score_breakdown=score_breakdown,
        missing_keywords=missing_keywords,
        matched_keywords=matched_keywords,
        keyword_density=keyword_density,
        bullet_rewrites=bullet_rewrites,
        ats_optimized_resume=ats_optimized_resume,
        summary=summary,
        strengths=strengths,
        weaknesses=weaknesses,
        degraded=degraded,
        tfidf_score=round(tfidf_score, 4),
        semantic_score=round(max(0, semantic_score), 4),
        scoring_method=scoring_method,
        section_analysis=section_analysis,
        experience_years=resume_years,
        bullet_scores=bullet_scores,
        education_level=education_level,
    )
