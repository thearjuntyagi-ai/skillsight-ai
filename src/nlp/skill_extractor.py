import re

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

SKILL_TAXONOMY = {
    "Python":           ["python"],
    "SQL":              ["sql", "mysql", "postgresql", "sqlite"],
    "R":                [r"\br\b", "r programming"],
    "Java":             ["java"],
    "Scala":            ["scala"],
    "JavaScript":       ["javascript", r"\bjs\b"],
    "Tableau":          ["tableau"],
    "Power BI":         [r"power\s*bi", "powerbi"],
    "Looker":           ["looker"],
    "Excel":            ["excel", "ms excel", "advanced excel"],
    "Google Sheets":    ["google sheets"],
    "Apache Spark":     ["spark", "pyspark", "apache spark"],
    "Apache Kafka":     ["kafka", "apache kafka"],
    "Airflow":          ["airflow", "apache airflow"],
    "dbt":              [r"\bdbt\b"],
    "Hadoop":           ["hadoop"],
    "Snowflake":        ["snowflake"],
    "BigQuery":         ["bigquery"],
    "Redshift":         ["redshift"],
    "AWS":              [r"\baws\b", "amazon web services"],
    "Azure":            [r"\bazure\b", "microsoft azure"],
    "GCP":              [r"\bgcp\b", "google cloud"],
    "Machine Learning": ["machine learning", r"\bml\b"],
    "Deep Learning":    ["deep learning"],
    "TensorFlow":       ["tensorflow"],
    "PyTorch":          ["pytorch"],
    "scikit-learn":     ["scikit-learn", "sklearn"],
    "XGBoost":          ["xgboost"],
    "NLP":              [r"\bnlp\b", "natural language processing"],
    "Docker":           ["docker"],
    "Kubernetes":       ["kubernetes", r"\bk8s\b"],
    "Git":              [r"\bgit\b", "github", "gitlab"],
    "Statistics":       ["statistics", "statistical analysis"],
    "A/B Testing":      [r"a/b test", "ab testing", "hypothesis testing"],
    "MongoDB":          ["mongodb"],
    "Redis":            ["redis"],
    "Communication":    ["communication skills", "communication"],
    "Problem Solving":  ["problem solving", "analytical thinking"],
    "Leadership":       ["leadership", "mentoring"],
}


class SkillExtractor:

    def __init__(self, use_spacy=False):
        self._compiled = []
        self._use_spacy = use_spacy
        self._nlp = None
        self._compile_patterns()
        if use_spacy:
            self._load_spacy()

    def extract(self, description: str) -> list:
        found = set()
        text = description.lower()

        for canonical, pattern in self._compiled:
            if pattern.search(text):
                found.add(canonical)

        return sorted(found)

    def extract_batch(self, descriptions: pd.Series) -> pd.Series:
        return descriptions.fillna("").apply(self.extract)

    def _compile_patterns(self):
        for canonical, aliases in SKILL_TAXONOMY.items():
            parts = []
            for alias in aliases:
                if re.search(r"[\\^$.*+?\[\]{}()|]", alias):
                    parts.append(alias)
                else:
                    parts.append(re.escape(alias))
            pattern = re.compile(
                r"(?<![a-z])(?:" + "|".join(parts) + r")(?![a-z])",
                re.IGNORECASE,
            )
            self._compiled.append((canonical, pattern))

    def _load_spacy(self):
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy loaded successfully")
        except Exception:
            logger.warning("spaCy not available — using pattern matching only")
            self._nlp = None