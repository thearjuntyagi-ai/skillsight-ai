import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests

from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = ["title", "company", "location", "description", "source"]
OPTIONAL_FIELDS = ["url", "salary_raw", "experience_raw", "employment_type",
                   "work_mode", "posted_at", "external_id"]


@dataclass
class CollectionStats:
    source: str
    attempted: int = 0
    collected: int = 0
    failed: int = 0

    @property
    def success_rate(self):
        if self.attempted == 0:
            return 0.0
        return self.collected / self.attempted * 100


class BaseCollector(ABC):

    SOURCE_NAME = "base"

    def __init__(self):
        self.config = load_config().get("collection", {})
        self.delay = self.config.get("request_delay_seconds", 2)
        self.max_retries = self.config.get("max_retries", 3)
        self.timeout = self.config.get("timeout_seconds", 30)
        self.stats = CollectionStats(source=self.SOURCE_NAME)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; SkillSightBot/1.0)"
        })

    @abstractmethod
    def collect(self, query="", location="", max_results=100) -> list:
        pass

    def _validate(self, record: dict) -> dict:
        self.stats.attempted += 1

        for field_name in REQUIRED_FIELDS:
            if not record.get(field_name):
                self.stats.failed += 1
                return {}

        for field_name in OPTIONAL_FIELDS:
            record.setdefault(field_name, None)

        record["source"] = self.SOURCE_NAME
        self.stats.collected += 1
        return record

    def log_stats(self):
        logger.info(
            "[%s] collected=%d failed=%d success_rate=%.1f%%",
            self.SOURCE_NAME,
            self.stats.collected,
            self.stats.failed,
            self.stats.success_rate,
        )