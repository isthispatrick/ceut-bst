from __future__ import annotations

import argparse
from pathlib import Path

from cuet_bst.downloader import collect_data, load_sources
from cuet_bst.settings import MANUAL_IMPORT_DIR, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public CUET UG Business Studies PYQ source files.")
    parser.add_argument("--extra-url", action="append", default=[], help="Additional public URL to download.")
    parser.add_argument("--official-only", action="store_true", help="Skip secondary public sources.")
    parser.add_argument("--no-manual-discovered", action="store_true", help="Do not read data/manual_imports/discovered_urls.txt.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_logging("collect_data", args.verbose)
    _write_discovery_queries()
    rows = collect_data(
        extra_urls=args.extra_url,
        include_secondary=not args.official_only,
        include_manual_discovered=not args.no_manual_discovered,
        logger=logger,
    )
    downloaded = sum(1 for row in rows if row.get("status") == "downloaded")
    skipped = sum(1 for row in rows if str(row.get("status", "")).startswith("skipped"))
    logger.info("Collection complete. Downloaded=%s skipped=%s manifest_rows=%s", downloaded, skipped, len(rows))


def _write_discovery_queries() -> None:
    config = load_sources()
    path = MANUAL_IMPORT_DIR / "search_queries.md"
    lines = [
        "# Manual Discovery Queries",
        "",
        "Use these in a normal browser/search engine, then paste public result URLs into `data/manual_imports/discovered_urls.txt`.",
        "The automated collector does not scrape search engines.",
        "",
    ]
    for query in config.get("discovery_queries", []):
        lines.append(f"- {query}")
    lines.append("")
    lines.append("## Reddit Queries")
    for query in config.get("reddit_queries", []):
        lines.append(f"- {query}")
    path.write_text("\n".join(lines), encoding="utf-8")
    (MANUAL_IMPORT_DIR / "discovered_urls.txt").touch(exist_ok=True)


if __name__ == "__main__":
    main()
