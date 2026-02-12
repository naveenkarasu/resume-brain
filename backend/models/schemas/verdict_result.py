"""Model 6 output: user-facing verdict with feedback and rewrites."""

from pydantic import BaseModel

from models.responses import BulletRewrite


class VerdictResult(BaseModel):
    """Structured output of the Verdict engine (Model 6).

    Tier 1: Template-based rules engine (deterministic, no ML).
    Tier 2 (future): Fine-tuned FLAN-T5-base for richer generation.
    """
    summary: str = ""
    strengths: list[str] = []
    weaknesses: list[str] = []
    bullet_rewrites: list[BulletRewrite] = []
    ats_optimized_resume: str = ""
    recommendations: list[str] = []
