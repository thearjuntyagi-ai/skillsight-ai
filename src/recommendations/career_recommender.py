from dataclasses import dataclass, field

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

CERT_RESOURCES = {
    "Python": [
        {"title": "Python for Everybody", "platform": "Coursera", "url": "https://coursera.org"},
    ],
    "SQL": [
        {"title": "SQL for Data Science", "platform": "Coursera", "url": "https://coursera.org"},
    ],
    "Power BI": [
        {"title": "PL-300 Power BI Analyst", "platform": "Microsoft Learn",
         "url": "https://learn.microsoft.com"},
    ],
    "Tableau": [
        {"title": "Tableau Desktop Specialist", "platform": "Tableau",
         "url": "https://tableau.com/learn/certification"},
    ],
    "Machine Learning": [
        {"title": "ML Specialization", "platform": "Coursera", "url": "https://coursera.org"},
    ],
    "AWS": [
        {"title": "AWS Cloud Practitioner", "platform": "AWS",
         "url": "https://aws.amazon.com/certification"},
    ],
    "Apache Spark": [
        {"title": "Databricks Spark Developer", "platform": "Databricks",
         "url": "https://databricks.com/learn/certification"},
    ],
    "dbt": [
        {"title": "dbt Fundamentals", "platform": "dbt Learn",
         "url": "https://courses.getdbt.com"},
    ],
}

SKILL_HOURS = {
    "Python": 80, "SQL": 60, "Power BI": 40, "Tableau": 40,
    "Machine Learning": 120, "Deep Learning": 100, "AWS": 80,
    "Apache Spark": 60, "dbt": 30, "Airflow": 40, "Docker": 30,
    "Statistics": 60, "A/B Testing": 20, "Git": 15,
}

ROLE_DEFAULTS = {
    "Data Analyst": {
        "SQL": 0.89, "Excel": 0.76, "Python": 0.68, "Tableau": 0.52,
        "Power BI": 0.48, "Statistics": 0.45, "Communication": 0.42,
        "Problem Solving": 0.40, "Git": 0.35, "A/B Testing": 0.30,
    },
    "Data Scientist": {
        "Python": 0.94, "Machine Learning": 0.88, "SQL": 0.72,
        "Statistics": 0.70, "Deep Learning": 0.55, "TensorFlow": 0.48,
        "PyTorch": 0.45, "Git": 0.60, "scikit-learn": 0.75, "NLP": 0.42,
    },
    "Data Engineer": {
        "Python": 0.90, "SQL": 0.88, "Apache Spark": 0.72,
        "Apache Kafka": 0.55, "Airflow": 0.60, "AWS": 0.65,
        "Docker": 0.58, "dbt": 0.45, "Git": 0.70, "Snowflake": 0.48,
    },
    "Business Analyst": {
        "Excel": 0.85, "SQL": 0.70, "Power BI": 0.60, "Tableau": 0.45,
        "Communication": 0.80, "Problem Solving": 0.75, "Statistics": 0.40,
        "Python": 0.35,
    },
    "ML Engineer": {
        "Python": 0.96, "Machine Learning": 0.92, "Deep Learning": 0.78,
        "TensorFlow": 0.62, "PyTorch": 0.68, "Docker": 0.65,
        "AWS": 0.60, "Git": 0.75,
    },
}


@dataclass
class CareerRecommendation:
    target_role: str
    user_skills: list
    matched_skills: list
    missing_skills: list
    readiness_score: float
    readiness_label: str
    total_learning_hours: int
    top_missing: list = field(default_factory=list)


class CareerRecommender:

    def __init__(self, jobs_df=None):
        self._jobs_df = jobs_df
        self._cache = {}

    def recommend(self, user_skills: list, target_role: str, top_n=10) -> CareerRecommendation:
        logger.info("Generating recommendation: role=%s skills=%d", target_role, len(user_skills))

        user_set = {s.strip().lower() for s in user_skills}
        role_demand = self._get_role_demand(target_role)

        matched = [s for s in role_demand.index
                   if any(s.lower() == u or s.lower() in u or u in s.lower()
                          for u in user_set)]
        missing_names = [s for s in role_demand.index if s not in matched]

        missing_records = []
        for skill in missing_names[:top_n]:
            demand_pct = float(role_demand.get(skill, 0.01))
            missing_records.append({
                "skill": skill,
                "demand_pct": round(demand_pct * 100, 1),
                "priority_score": round(demand_pct, 4),
                "estimated_hours": SKILL_HOURS.get(skill, 25),
                "resources": CERT_RESOURCES.get(skill, []),
            })

        missing_records.sort(key=lambda x: x["priority_score"], reverse=True)

        total_demand = role_demand.sum()
        matched_demand = sum(role_demand.get(s, 0) for s in matched)
        readiness = float(matched_demand / total_demand * 100) if total_demand > 0 else 0.0
        readiness = min(round(readiness, 1), 100.0)

        return CareerRecommendation(
            target_role=target_role,
            user_skills=sorted(user_skills),
            matched_skills=sorted(matched),
            missing_skills=missing_records,
            readiness_score=readiness,
            readiness_label=self._label(readiness),
            total_learning_hours=sum(r["estimated_hours"] for r in missing_records),
            top_missing=[r["skill"] for r in missing_records[:5]],
        )

    def _get_role_demand(self, role: str) -> pd.Series:
        if role in self._cache:
            return self._cache[role]

        if self._jobs_df is not None and "skills" in self._jobs_df.columns:
            role_df = self._jobs_df[
                self._jobs_df["normalized_title"].str.contains(role, case=False, na=False)
            ]
            if len(role_df) > 10:
                counts = role_df["skills"].explode().value_counts(normalize=True)
                self._cache[role] = counts
                return counts

        defaults = ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["Data Analyst"])
        series = pd.Series(defaults, dtype=float)
        self._cache[role] = series
        return series

    def _label(self, score: float) -> str:
        if score >= 80:
            return "Expert — Ready to Apply"
        if score >= 60:
            return "Proficient — Nearly There"
        if score >= 40:
            return "Developing — Keep Learning"
        return "Beginner — Build Core Skills First"