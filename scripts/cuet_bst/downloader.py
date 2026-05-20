from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .robots import RobotsCache
from .settings import CONFIG_DIR, MANUAL_IMPORT_DIR, RAW_DIR, ensure_dirs, load_json
from .util import read_jsonl, safe_filename, stable_hash, utc_now, url_extension, write_jsonl


MANIFEST = RAW_DIR / "manifest.jsonl"


def load_sources() -> dict[str, Any]:
    return load_json(CONFIG_DIR / "cuet_sources.json")


def collect_data(
    *,
    extra_urls: list[str] | None = None,
    include_secondary: bool = True,
    include_manual_discovered: bool = True,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    ensure_dirs()
    logger = logger or logging.getLogger(__name__)
    config = load_sources()
    user_agent = config["user_agent"]
    delay = float(config.get("request_delay_seconds", 2.0))
    timeout = int(config.get("timeout_seconds", 30))
    robots = RobotsCache(user_agent=user_agent, timeout=timeout, logger=logger)
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    existing = read_jsonl(MANIFEST)
    seen_urls = {row.get("source_url"): row for row in existing if row.get("status") == "downloaded"}
    rows = list(existing)

    source_specs: list[dict[str, Any]] = []
    source_specs.extend(config.get("official_sources", []))
    if include_secondary:
        source_specs.extend(config.get("secondary_sources", []))
    for url in extra_urls or []:
        source_specs.append({"name": "extra-url", "kind": "auto", "url": url, "priority": 3})
    if include_manual_discovered:
        source_specs.extend(_manual_discovered_sources(config.get("manual_discovery_file")))

    for spec in source_specs:
        url = spec["url"]
        if url in seen_urls:
            logger.info("Cached: %s", url)
            continue
        kind = spec.get("kind", "auto")
        if kind == "crawl_page":
            rows.extend(_download_page_and_linked_docs(spec, session, robots, delay, timeout, logger))
        else:
            row = _download_url(spec, session, robots, delay, timeout, logger)
            rows.append(row)
            if row.get("status") == "downloaded" and row.get("raw_file_path", "").endswith((".html", ".htm")):
                rows.extend(_download_links_from_html(row, spec, session, robots, delay, timeout, logger))
        write_jsonl(MANIFEST, _dedupe_manifest(rows))
    return _dedupe_manifest(rows)


def _manual_discovered_sources(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.is_absolute():
        path = MANUAL_IMPORT_DIR.parent.parent / path
    if not path.exists():
        return []
    specs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            specs.append({"name": "manual-discovered-url", "kind": "auto", "url": line, "priority": 3})
    return specs


def _download_page_and_linked_docs(
    spec: dict[str, Any],
    session: requests.Session,
    robots: RobotsCache,
    delay: float,
    timeout: int,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    page = _download_url(spec, session, robots, delay, timeout, logger)
    rows = [page]
    if page.get("status") == "downloaded":
        rows.extend(_download_links_from_html(page, spec, session, robots, delay, timeout, logger))
    return rows


def _download_links_from_html(
    page_row: dict[str, Any],
    spec: dict[str, Any],
    session: requests.Session,
    robots: RobotsCache,
    delay: float,
    timeout: int,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    raw_path = Path(page_row["raw_file_path"])
    if not raw_path.exists():
        return []
    soup = BeautifulSoup(raw_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    include_terms = [term.lower() for term in spec.get("link_include_terms", ["cuet", "business", "studies", "pdf"])]
    required_any = [term.lower() for term in spec.get("link_required_any_terms", [])]
    max_links = int(load_sources().get("max_links_per_page", 80))
    rows: list[dict[str, Any]] = []
    base_url = page_row["source_url"]
    seen_hrefs = {row.get("source_url", "") for row in read_jsonl(MANIFEST)}
    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"])
        href = href.split("#", 1)[0]
        if href in seen_hrefs:
            continue
        label = " ".join(anchor.get_text(" ", strip=True).split())
        haystack = f"{href} {label}".lower()
        if not any(term in haystack for term in include_terms):
            continue
        if required_any and not any(term in haystack for term in required_any):
            continue
        if not (_looks_like_document(href) or "question" in haystack or "answer" in haystack):
            continue
        linked_spec = {
            "name": f"{spec.get('name', 'source')} linked file",
            "kind": "auto",
            "url": href,
            "priority": spec.get("priority", 3),
            "parent_url": base_url,
            "link_text": label,
        }
        rows.append(_download_url(linked_spec, session, robots, delay, timeout, logger))
        seen_hrefs.add(href)
        if len(rows) >= max_links:
            break
    return rows


def _download_url(
    spec: dict[str, Any],
    session: requests.Session,
    robots: RobotsCache,
    delay: float,
    timeout: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    url = spec["url"]
    base = {
        "source_name": spec.get("name", ""),
        "source_url": url,
        "parent_url": spec.get("parent_url", ""),
        "priority": spec.get("priority", ""),
        "downloaded_at": utc_now(),
    }
    if not robots.can_fetch(url):
        logger.warning("Robots disallow fetch: %s", url)
        return {**base, "status": "skipped_robots", "raw_file_path": "", "error": "robots.txt disallows fetch"}
    robots.wait(url, delay)
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Download failed: %s (%s)", url, exc)
        return {**base, "status": "failed", "raw_file_path": "", "error": str(exc)}

    content_type = response.headers.get("content-type", "")
    ext = url_extension(response.url, content_type)
    host = urlparse(response.url).netloc.replace("www.", "")
    name = f"{safe_filename(host)}_{stable_hash(response.url)}{ext}"
    raw_path = RAW_DIR / name
    mode = "wb"
    data = response.content
    with raw_path.open(mode) as handle:
        handle.write(data)
    logger.info("Downloaded %s -> %s", url, raw_path)
    return {
        **base,
        "status": "downloaded",
        "final_url": response.url,
        "content_type": content_type,
        "raw_file_path": str(raw_path),
        "bytes": len(data),
        "sha256": stable_hash(data.decode("latin-1", errors="ignore"), 64),
        "error": "",
    }


def _looks_like_document(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".pdf", ".html", ".htm", ".csv", ".json", ".txt"))


def _dedupe_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("source_url") or row.get("raw_file_path") or stable_hash(str(row))
        if key not in by_key or by_key[key].get("status") != "downloaded":
            by_key[key] = row
    return list(by_key.values())
