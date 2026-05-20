from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cuet_bst.downloader import MANIFEST
from cuet_bst.settings import MANUAL_IMPORT_DIR, RAW_DIR, setup_logging
from cuet_bst.util import read_jsonl, stable_hash, utc_now, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Import manually downloaded CUET PDFs, HTML, text, screenshots, or CSV files.")
    parser.add_argument("paths", nargs="+", help="Files to copy into data/manual_imports and register for processing.")
    parser.add_argument("--source-url", default="", help="Original public URL, if known.")
    parser.add_argument("--source-name", default="manual-import", help="Source label.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("import_manual", args.verbose)

    manifest = read_jsonl(MANIFEST)
    for value in args.paths:
        src = Path(value).expanduser().resolve()
        if not src.exists():
            logger.warning("Missing file: %s", src)
            continue
        destination_dir = MANUAL_IMPORT_DIR if src.suffix.lower() == ".csv" else RAW_DIR
        destination = destination_dir / f"manual_{stable_hash(str(src))}_{src.name}"
        shutil.copy2(src, destination)
        if destination_dir == RAW_DIR:
            manifest.append(
                {
                    "source_name": args.source_name,
                    "source_url": args.source_url or f"manual://{src.name}",
                    "parent_url": "",
                    "priority": "manual",
                    "downloaded_at": utc_now(),
                    "status": "downloaded",
                    "final_url": args.source_url,
                    "content_type": "",
                    "raw_file_path": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": stable_hash(destination.read_bytes().decode("latin-1", errors="ignore"), 64),
                    "error": "",
                }
            )
        logger.info("Imported %s -> %s", src, destination)
    write_jsonl(MANIFEST, manifest)


if __name__ == "__main__":
    main()
