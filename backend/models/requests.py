from pydantic import BaseModel, Field


class QuickAnalyzeRequest(BaseModel):
    resume_text: str = Field(..., max_length=50000, description="Plain text resume content")
    job_description: str = Field(..., max_length=10000, description="Job description text")
