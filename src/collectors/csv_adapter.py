import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.collectors.base_collector import BaseCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)

COLUMN_MAP = {
    "job_id": "external_id",
    "title": "title",
    "company_name": "company",
    "location": "location",
    "description": "description",
    "max_salary": "salary_raw",
    "formatted_experience_level": "experience_raw",
    "work_type": "employment_type",
    "remote_allowed": "work_mode",
    "original_listed_time": "posted_at",
    "job_posting_url": "url",
    "job_title": "title",
    "company": "company",
    "salary": "salary_raw",
    "job_description": "description",
}


class CSVAdapter(BaseCollector):

    SOURCE_NAME = "csv"

    def __init__(self, file_path=None):
        super().__init__()
        cfg_path = self.config.get("sources", {}).get("csv", {}).get("path")
        self.file_path = Path(file_path or cfg_path or "data/external/sample_jobs.csv")

    def collect(self, query="", location="", max_results=500) -> list:
        logger.info("[csv] Loading from %s", self.file_path)

        if not self.file_path.exists():
            logger.warning("[csv] File not found: %s", self.file_path)
            return []

        df = pd.read_csv(self.file_path, low_memory=False)
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

        if query and "title" in df.columns:
            df = df[df["title"].str.contains(query, case=False, na=False)]
        if location and "location" in df.columns:
            df = df[df["location"].str.contains(location, case=False, na=False)]

        df = df.head(max_results)
        records = []

        for _, row in df.iterrows():
            record = self._row_to_record(row)
            validated = self._validate(record)
            if validated:
                records.append(validated)

        self.log_stats()
        return records

    def _row_to_record(self, row) -> dict:
        def safe(col):
            val = row.get(col)
            if val is None:
                return None
            if isinstance(val, float):
                import math
                if math.isnan(val):
                    return None
            return str(val).strip()

        work_mode = safe("work_mode")
        if work_mode is not None:
            work_mode = "remote" if work_mode.lower() in ("1", "true", "yes") else "onsite"

        return {
            "external_id": safe("external_id"),
            "title": safe("title"),
            "company": safe("company"),
            "location": safe("location"),
            "description": safe("description"),
            "salary_raw": safe("salary_raw"),
            "experience_raw": safe("experience_raw"),
            "employment_type": safe("employment_type"),
            "work_mode": work_mode,
            "posted_at": safe("posted_at"),
            "url": safe("url"),
            "source": self.SOURCE_NAME,
        }

    @staticmethod
    def generate_sample_csv(output_path="data/external/sample_jobs.csv") -> Path:
        titles = [
            "Data Analyst", "Senior Data Analyst", "Business Analyst",
            "Data Scientist", "ML Engineer", "Analytics Engineer",
            "BI Developer", "Product Analyst", "Data Engineer",
        ]
        companies = [
            "Infosys", "Wipro", "TCS", "Accenture", "Flipkart",
            "Swiggy", "Razorpay", "PhonePe", "Meesho", "Zepto",
        ]
        locations = [
            "Bengaluru, Karnataka", "Mumbai, Maharashtra",
            "Hyderabad, Telangana", "Delhi, Delhi",
            "Pune, Maharashtra", "Chennai, Tamil Nadu",
            "Gurugram, Haryana", "Noida, Uttar Pradesh",
        ]
        descriptions = [
            "We are looking for a {title} with expertise in Python, SQL, Tableau, "
            "and Power BI. Experience with AWS and data pipelines is a plus. "
            "Strong communication and Excel skills required.",
            "Seeking a {title} skilled in machine learning, pandas, scikit-learn, "
            "and deep learning. Must know SQL and Python.",
            "Join our analytics team as a {title}. You will work with Looker, "
            "BigQuery, dbt, Airflow, and build dashboards in Tableau.",
            "We need a {title} proficient in R, Python, A/B testing, statistics, "
            "Excel, and business intelligence tools like Power BI.",
            "Exciting {title} role — must know SQL, Python, Git, Docker. "
            "Experience with Kafka or Spark is highly desirable.",
        ]
        salaries = [
            "6-10 LPA", "10-18 LPA", "18-28 LPA",
            "5-8 LPA", "25-40 LPA", "8-14 LPA", "12-20 LPA",
        ]

        rows = []
        for i in range(500):
            title = random.choice(titles)
            rows.append({
                "job_id": f"sample_{i:05d}",
                "title": title,
                "company_name": random.choice(companies),
                "location": random.choice(locations),
                "description": random.choice(descriptions).format(title=title),
                "max_salary": random.choice(salaries),
                "formatted_experience_level": random.choice(
                    ["Entry level", "Mid-Senior level", "Senior", "Director"]
                ),
                "work_type": random.choice(["FULL_TIME", "CONTRACT", "PART_TIME"]),
                "remote_allowed": random.choice([0, 1]),
                "original_listed_time": (
                    datetime.now() - timedelta(days=random.randint(0, 180))
                ).strftime("%Y-%m-%d"),
            })

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Sample CSV written: %s (%d rows)", out, len(rows))
        return out