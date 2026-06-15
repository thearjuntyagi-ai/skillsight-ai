import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.collectors.base_collector import BaseCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maps Kaggle column names to SkillSight standard names
COLUMN_MAP = {
    "job_id":                     "external_id",
    "title":                      "title",
    "company_name":               "company",
    "location":                   "location",
    "description":                "description",
    "max_salary":                 "salary_raw",
    "normalized_salary":          "salary_normalized",
    "formatted_experience_level": "experience_raw",
    "work_type":                  "employment_type",
    "remote_allowed":             "work_mode",
    "original_listed_time":       "posted_at",
    "job_posting_url":            "url",
    "formatted_work_type":        "work_type_formatted",
    "med_salary":                 "salary_median",
    "min_salary":                 "salary_min_raw",
}


class CSVAdapter(BaseCollector):

    SOURCE_NAME = "csv"

    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = Path(file_path or "data/external/postings.csv")

        # Fall back to sample if real data not found
        if not self.file_path.exists():
            self.file_path = Path("data/external/sample_jobs.csv")

    def collect(self, query="", location="", max_results=2000) -> list:
        logger.info("[csv] Loading from %s", self.file_path)

        if not self.file_path.exists():
            logger.warning("[csv] No data file found, generating sample data")
            sample_path = Path("data/external/sample_jobs.csv")
            self.generate_sample_csv(sample_path)
            self.file_path = sample_path

        # Read only needed columns to save memory
        needed_cols = list(COLUMN_MAP.keys())

        try:
            df = pd.read_csv(
                self.file_path,
                usecols=[c for c in needed_cols
                         if c in pd.read_csv(self.file_path, nrows=0).columns],
                low_memory=False,
                nrows=max_results * 2,  # Read extra to account for filtering
            )
        except Exception as exc:
            logger.error("[csv] Failed to read file: %s", exc)
            return []

        # Rename columns
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items()
                                 if k in df.columns})

        # Filter by query/location
        if query and "title" in df.columns:
            df = df[df["title"].str.contains(query, case=False, na=False)]
        if location and "location" in df.columns:
            df = df[df["location"].str.contains(location, case=False, na=False)]

        # Drop rows with no title or description
        df = df.dropna(subset=["title", "description"])
        df = df.head(max_results)

        records = []
        for _, row in df.iterrows():
            record = self._row_to_record(row)
            validated = self._validate(record)
            if validated:
                records.append(validated)

        self.log_stats()
        logger.info("[csv] Loaded %d records", len(records))
        return records

    def _row_to_record(self, row) -> dict:
        def safe(col):
            val = row.get(col)
            if val is None:
                return None
            try:
                if pd.isna(val):
                    return None
            except Exception:
                pass
            return str(val).strip()

        # Handle work mode
        work_mode = safe("work_mode")
        if work_mode is not None:
            work_mode = "remote" if work_mode.lower() in ("1", "true", "yes") else "onsite"

        # Handle salary — use normalized_salary if available
        salary_raw = safe("salary_raw")
        salary_normalized = safe("salary_normalized")
        salary_median = safe("salary_median")

        # Build a readable salary string
        if salary_normalized:
            try:
                sal_val = float(salary_normalized)
                if sal_val > 0:
                    salary_raw = f"${sal_val:,.0f}"
            except Exception:
                pass
        elif salary_median:
            try:
                sal_val = float(salary_median)
                if sal_val > 0:
                    salary_raw = f"${sal_val:,.0f}"
            except Exception:
                pass

        # Handle timestamp posted_at
        posted_at = safe("posted_at")
        if posted_at:
            try:
                ts = float(posted_at)
                if ts > 1e10:
                    ts /= 1000
                posted_at = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            except Exception:
                pass

        return {
            "external_id":    safe("external_id"),
            "title":          safe("title"),
            "company":        safe("company"),
            "location":       safe("location"),
            "description":    safe("description"),
            "salary_raw":     salary_raw,
            "experience_raw": safe("experience_raw"),
            "employment_type": safe("employment_type"),
            "work_mode":      work_mode,
            "posted_at":      posted_at,
            "url":            safe("url"),
            "source":         self.SOURCE_NAME,
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
                "job_id":                     f"sample_{i:05d}",
                "title":                      title,
                "company_name":               random.choice(companies),
                "location":                   random.choice(locations),
                "description":                random.choice(descriptions).format(title=title),
                "max_salary":                 random.choice(salaries),
                "formatted_experience_level": random.choice(
                    ["Entry level", "Mid-Senior level", "Senior", "Director"]
                ),
                "work_type":                  random.choice(["FULL_TIME", "CONTRACT", "PART_TIME"]),
                "remote_allowed":             random.choice([0, 1]),
                "original_listed_time":       (
                    datetime.now() - timedelta(days=random.randint(0, 180))
                ).strftime("%Y-%m-%d"),
            })

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Sample CSV written: %s (%d rows)", out, len(rows))
        return out