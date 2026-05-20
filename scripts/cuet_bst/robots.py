from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests


@dataclass
class RobotsCache:
    user_agent: str
    timeout: int = 20
    logger: logging.Logger | None = None
    parsers: dict[str, robotparser.RobotFileParser] = field(default_factory=dict)
    last_request_at: dict[str, float] = field(default_factory=dict)

    def can_fetch(self, url: str) -> bool:
        parser = self._parser_for(url)
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str, default: float) -> float:
        parser = self._parser_for(url)
        if parser is None:
            return default
        delay = parser.crawl_delay(self.user_agent)
        return float(delay) if delay is not None else default

    def wait(self, url: str, default_delay: float) -> None:
        parsed = urlparse(url)
        key = f"{parsed.scheme}://{parsed.netloc}"
        delay = self.crawl_delay(url, default_delay)
        elapsed = time.monotonic() - self.last_request_at.get(key, 0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self.last_request_at[key] = time.monotonic()

    def _parser_for(self, url: str) -> robotparser.RobotFileParser | None:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base in self.parsers:
            return self.parsers[base]
        robots_url = urljoin(base, "/robots.txt")
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                parser.parse([])
            else:
                parser.parse(response.text.splitlines())
        except requests.RequestException as exc:
            if self.logger:
                self.logger.warning("Could not read robots.txt for %s: %s", base, exc)
            parser.parse([])
        self.parsers[base] = parser
        return parser
