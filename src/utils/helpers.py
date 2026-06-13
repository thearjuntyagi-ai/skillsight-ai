import hashlib
import re
import time
from functools import wraps
from pathlib import Path

import yaml


def load_config() -> dict:
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_salary_text(raw: str) -> dict:
    result = {"min": None, "max": None, "currency": "INR"}
    if not raw:
        return result

    raw = raw.lower().replace(",", "")

    # Range like "8-12 LPA"
    m = re.search(r"([\d.]+)\s*-\s*([\d.]+)\s*(lpa|lakhs?)", raw)
    if m:
        result["min"] = float(m.group(1)) * 100_000
        result["max"] = float(m.group(2)) * 100_000
        return result

    # Single like "8 LPA"
    m = re.search(r"([\d.]+)\s*(lpa|lakhs?)", raw)
    if m:
        val = float(m.group(1)) * 100_000
        result["min"] = result["max"] = val
        return result

    # USD like "$90,000"
    m = re.search(r"\$\s*([\d.]+)\s*k?", raw)
    if m:
        val = float(m.group(1))
        val *= 1000 if "k" in raw else 1
        result["min"] = result["max"] = val
        result["currency"] = "USD"
        return result

    return result


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def retry(max_attempts=3, delay=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator