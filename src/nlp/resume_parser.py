import re
from pathlib import Path
import pandas as pd

from src.nlp.skill_extractor import SkillExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResumeParser:
    """
    Parses a PDF or text resume and extracts skills.
    Compares against market demand to give a readiness score.
    """

    def __init__(self):
        self._extractor = SkillExtractor(use_spacy=False)

    def parse_pdf(self, file_bytes: bytes) -> dict:
        """Extract text and skills from a PDF resume (bytes)."""
        try:
            import pdfplumber
            import io

            text = ""
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            return self._analyse_text(text)

        except Exception as exc:
            logger.error("PDF parsing failed: %s", exc)
            return {"error": str(exc), "skills": [], "text": ""}

    def parse_text(self, text: str) -> dict:
        """Extract skills from plain text resume."""
        return self._analyse_text(text)

    def _analyse_text(self, text: str) -> dict:
        skills = self._extractor.extract(text)
        word_count = len(text.split())

        return {
            "skills": skills,
            "skill_count": len(skills),
            "word_count": word_count,
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
        }

    def get_market_readiness(self, resume_skills: list,
                              target_role: str,
                              jobs_df: pd.DataFrame = None) -> dict:
        """
        Compare resume skills against market demand.
        Returns a detailed gap analysis.
        """
        from src.recommendations.career_recommender import CareerRecommender

        recommender = CareerRecommender(jobs_df=jobs_df)
        rec = recommender.recommend(
            user_skills=resume_skills,
            target_role=target_role,
        )

        return {
            "readiness_score": rec.readiness_score,
            "readiness_label": rec.readiness_label,
            "matched_skills": rec.matched_skills,
            "missing_skills": rec.missing_skills,
            "total_learning_hours": rec.total_learning_hours,
        }