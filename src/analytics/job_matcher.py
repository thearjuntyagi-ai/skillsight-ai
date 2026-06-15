import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from src.nlp.skill_extractor import SkillExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JobMatcher:
    """
    Compares a user profile against a job description
    and returns a match score with detailed breakdown.
    """

    def __init__(self):
        self._extractor = SkillExtractor(use_spacy=False)
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    def match(self, user_skills: list, job_description: str,
              experience: float = 0.0) -> dict:
        """
        Returns a full match analysis between user and job.
        """
        # Extract skills from JD
        jd_skills = self._extractor.extract(job_description)

        # Skill overlap
        user_set = {s.lower() for s in user_skills}
        jd_set   = {s.lower() for s in jd_skills}

        matched  = [s for s in jd_skills if s.lower() in user_set]
        missing  = [s for s in jd_skills if s.lower() not in user_set]
        extra    = [s for s in user_skills if s.lower() not in jd_set]

        # Skill match score (0-100)
        if jd_skills:
            skill_score = len(matched) / len(jd_skills) * 100
        else:
            skill_score = 50.0

        # Text similarity score using TF-IDF cosine similarity
        user_text = " ".join(user_skills)
        try:
            tfidf = self._vectorizer.fit_transform([job_description, user_text])
            text_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]) * 100
        except Exception:
            text_score = skill_score

        # Experience match
        required_exp = self._extract_experience(job_description)
        if required_exp is None:
            exp_score = 80.0
        elif experience >= required_exp:
            exp_score = 100.0
        elif experience >= required_exp * 0.7:
            exp_score = 70.0
        else:
            exp_score = 40.0

        # Overall score (weighted)
        overall = (skill_score * 0.55) + (text_score * 0.25) + (exp_score * 0.20)
        overall = min(round(overall, 1), 100.0)

        # Grade
        if overall >= 80:
            grade = "Excellent Match ⭐⭐⭐"
            color = "#22c55e"
        elif overall >= 60:
            grade = "Good Match ⭐⭐"
            color = "#f59e0b"
        elif overall >= 40:
            grade = "Partial Match ⭐"
            color = "#f97316"
        else:
            grade = "Low Match ❌"
            color = "#ef4444"

        return {
            "overall_score":  overall,
            "skill_score":    round(skill_score, 1),
            "text_score":     round(text_score, 1),
            "exp_score":      round(exp_score, 1),
            "grade":          grade,
            "color":          color,
            "matched_skills": matched,
            "missing_skills": missing,
            "extra_skills":   extra,
            "jd_skills":      jd_skills,
            "required_exp":   required_exp,
        }

    def _extract_experience(self, text: str):
        """Extract required years of experience from JD text."""
        patterns = [
            r"(\d+)\+?\s*years?\s+of\s+experience",
            r"(\d+)\+?\s*years?\s+experience",
            r"experience\s+of\s+(\d+)\+?\s*years?",
            r"minimum\s+(\d+)\s+years?",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return None