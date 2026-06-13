import hashlib
import re
from datetime import datetime, timedelta

import pandas as pd

from src.utils.helpers import clean_text, parse_salary_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

LOCATION_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "delhi ncr": "Delhi",
    "ncr": "Delhi",
    "gurgaon": "Gurugram",
    "new delhi": "Delhi",
}

REMOTE_SIGNALS = ["remote", "work from home", "wfh", "anywhere", "fully remote"]
HYBRID_SIGNALS = ["hybrid", "partially remote", "flexible location"]


class DataCleaner:

    def __init__(self):
        self._seen_hashes = set()

    def clean(self, raw_records: list) -> pd.DataFrame:
        logger.info("Cleaning %d raw records", len(raw_records))
        cleaned = []
        dupes = 0

        for rec in raw_records:
            try:
                c = self._clean_record(rec)
                if c is None:
                    continue
                h = c["content_hash"]
                if h in self._seen_hashes:
                    dupes += 1
                    continue
                self._seen_hashes.add(h)
                cleaned.append(c)
            except Exception as exc:
                logger.debug("Failed to clean record: %s", exc)

        df = pd.DataFrame(cleaned)
        logger.info("Clean complete: %d records (%d dupes removed)", len(df), dupes)
        return df

    def _clean_record(self, rec: dict):
        title = rec.get("title", "")
        description = rec.get("description", "")

        if not title or not description or len(description) < 50:
            return None

        title_clean = clean_text(title)
        desc_clean = clean_text(description)

        content_hash = hashlib.sha256(
            f"{title_clean.lower()}{desc_clean[:200].lower()}".encode()
        ).hexdigest()

        norm_title = self._normalise_title(title_clean)
        salary = parse_salary_text(rec.get("salary_raw") or "")
        exp_min, exp_max = self._parse_experience(rec.get("experience_raw") or "")
        city, state, country = self._parse_location(rec.get("location") or "")
        work_mode = rec.get("work_mode") or self._detect_work_mode(desc_clean)
        posted_at = self._parse_date(rec.get("posted_at"))
        emp_type = self._normalise_employment_type(rec.get("employment_type") or "")

        return {
            "title": title_clean,
            "normalized_title": norm_title,
            "company": rec.get("company", ""),
            "normalized_company": self._normalise_company(rec.get("company") or ""),
            "location_raw": rec.get("location", ""),
            "city": city,
            "state": state,
            "country": country,
            "description": description,
            "description_clean": desc_clean,
            "salary_min": salary["min"],
            "salary_max": salary["max"],
            "salary_currency": salary["currency"],
            "experience_min": exp_min,
            "experience_max": exp_max,
            "employment_type": emp_type,
            "work_mode": work_mode,
            "posted_at": posted_at,
            "source": rec.get("source", "unknown"),
            "external_id": rec.get("external_id"),
            "url": rec.get("url"),
            "content_hash": content_hash,
        }

    def _normalise_title(self, title: str) -> str:
        t = title.lower()
        t = re.sub(r"\bsr\.?\s+", "Senior ", t, flags=re.IGNORECASE)
        t = re.sub(r"\bjr\.?\s+", "Junior ", t, flags=re.IGNORECASE)
        return t.title().strip()

    def _normalise_company(self, name: str) -> str:
        n = name.lower()
        n = re.sub(r"\b(pvt\.?|ltd\.?|llp|inc\.?|corp\.?)\b", "", n)
        return n.strip().title()

    def _parse_experience(self, raw: str):
        if not raw:
            return None, None
        raw = raw.lower()
        if any(w in raw for w in ("fresher", "entry", "0-1")):
            return 0.0, 1.0
        m = re.search(r"(\d+)\s*[-–]+\s*(\d+)", raw)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r"(\d+)\+?", raw)
        if m:
            val = float(m.group(1))
            return val, val + 3
        return None, None

    def _parse_location(self, raw: str):
        parts = [p.strip() for p in raw.split(",")]
        city_raw = parts[0].lower() if parts else ""
        city = LOCATION_ALIASES.get(city_raw, parts[0].title() if parts else "")
        state = parts[1].title() if len(parts) > 1 else ""
        country = parts[2].title() if len(parts) > 2 else "India"
        return city, state, country

    def _detect_work_mode(self, description: str) -> str:
        desc_lower = description.lower()
        if any(s in desc_lower for s in REMOTE_SIGNALS):
            return "remote"
        if any(s in desc_lower for s in HYBRID_SIGNALS):
            return "hybrid"
        return "onsite"

    def _normalise_employment_type(self, raw: str) -> str:
        r = raw.upper()
        if "FULL" in r:
            return "full-time"
        if "CONTRACT" in r or "FREELANCE" in r:
            return "contract"
        if "PART" in r:
            return "part-time"
        if "INTERN" in r:
            return "internship"
        return "full-time"

    def _parse_date(self, raw):
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw
        raw = str(raw).strip()

        m = re.search(r"(\d+)\s+(day|week|month)s?\s+ago", raw, re.IGNORECASE)
        if m:
            n, unit = int(m.group(1)), m.group(2).lower()
            delta = {"day": 1, "week": 7, "month": 30}[unit]
            return datetime.utcnow() - timedelta(days=n * delta)

        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(raw[:10], fmt)
            except ValueError:
                continue
        return None