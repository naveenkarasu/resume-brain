"""All prompt templates for Gemini API calls."""


def build_scoring_prompt(
    resume_text: str,
    job_description: str,
    local_tfidf_score: float = 0.0,
    local_semantic_score: float = 0.0,
    local_matched: list[str] | None = None,
    local_missing: list[str] | None = None,
) -> str:
    """Call A: Scoring + keyword extraction.

    Includes local NLP scores as context to help calibrate the LLM's judgment.
    """
    context_section = ""
    if local_tfidf_score > 0 or local_semantic_score > 0:
        context_section = f"""
LOCAL NLP PRE-ANALYSIS (use as calibration reference, not as final scores):
- TF-IDF lexical similarity: {local_tfidf_score:.0%}
- Semantic embedding similarity: {local_semantic_score:.0%}
- Keywords already matched: {', '.join(local_matched or [])}
- Keywords detected as missing: {', '.join(local_missing or [])}
---
"""

    return f"""You are an expert ATS (Applicant Tracking System) and resume analyst.

Analyze this resume against the job description using the scoring rubric below.

SCORING RUBRIC (follow strictly):
- 0-20:  No relevant match. Resume is for a completely different field.
- 20-40: Weak match. Some transferable skills but major gaps in core requirements.
- 40-60: Moderate match. Meets some key requirements but missing several important ones.
- 60-80: Strong match. Meets most requirements with minor gaps.
- 80-100: Exceptional match. Meets or exceeds nearly all requirements.

CATEGORY WEIGHTS:
- skills_match (40% weight): Technical skills and tools alignment
- experience_match (30% weight): Relevant experience level and domain
- education_match (15% weight): Education requirements alignment
- keywords_match (15% weight): Keyword coverage and density

{context_section}RESUME:
---
{resume_text}
---

JOB DESCRIPTION:
---
{job_description}
---

Respond with ONLY valid JSON (no markdown, no code fences) in this exact structure:
{{
  "overall_score": <integer 0-100, weighted average of categories>,
  "confidence": <integer 1-10, how confident you are in this assessment>,
  "score_breakdown": {{
    "skills_match": <integer 0-100>,
    "experience_match": <integer 0-100>,
    "education_match": <integer 0-100>,
    "keywords_match": <integer 0-100>
  }},
  "matched_keywords": [<list of specific keywords/skills found in BOTH resume and JD>],
  "missing_keywords": [<list of important JD keywords/skills NOT found in resume>],
  "summary": "<2-3 sentence analysis explaining the score and key factors>",
  "strengths": [<3-5 specific strengths with evidence from resume>],
  "weaknesses": [<3-5 specific gaps with reference to JD requirements>]
}}"""


def build_rewrite_prompt(resume_text: str, job_description: str, bullets: list[str]) -> str:
    """Call B: Bullet rewrites + ATS-optimized resume."""
    bullets_text = "\n".join(f"- {b}" for b in bullets[:10])

    return f"""You are an expert resume writer and ATS optimization specialist.

Given this resume and target job description, provide:
1. Rewritten bullet points optimized for the job
2. A full ATS-optimized version of the resume

ATS OPTIMIZATION RULES:
- Use standard section headers: "Work Experience", "Education", "Skills", "Summary"
- Include keywords from the job description naturally (target 1-3% density for primary keywords)
- Lead bullets with strong action verbs and quantified results (metrics, percentages, dollar amounts)
- Keep formatting simple: no tables, no columns, no text boxes, no graphics
- Use standard bullet characters
- Match job title terminology where truthful
- Include a skills section with keywords listed explicitly

RESUME:
---
{resume_text}
---

JOB DESCRIPTION:
---
{job_description}
---

ORIGINAL BULLET POINTS TO REWRITE:
{bullets_text}

Respond with ONLY valid JSON (no markdown, no code fences) in this exact structure:
{{
  "bullet_rewrites": [
    {{
      "original": "<original bullet text>",
      "rewritten": "<improved version with relevant keywords, action verbs, and metrics>",
      "reason": "<specific changes made: keywords added, metrics included, etc.>"
    }}
  ],
  "ats_optimized_resume": "<full ATS-optimized resume with standard sections, keywords, and formatting>"
}}"""
