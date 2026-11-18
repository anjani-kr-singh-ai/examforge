"""Download the ICAI source PDFs listed in data/metadata/sources.json into data/raw/.

ICAI documents are copyrighted and are NOT redistributed in this repository. This script fetches them from
the official ICAI URLs recorded in the manifest, checks they are real PDFs, and verifies the SHA-256 recorded
at ingestion time (a mismatch means ICAI changed the file - re-ingest and review the metadata).

    python scripts/fetch_sources.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "metadata" / "sources.json"
RAW = ROOT / "data" / "raw"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True
    for d in json.loads(MANIFEST.read_text(encoding="utf-8")):
        dest = RAW / d["filename"]
        if not dest.exists():
            print(f"downloading {d['document_id']} <- {d['source_url']}")
            r = requests.get(d["source_url"], timeout=120, headers={"User-Agent": "examforge-ai-dataset-fetch"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                print(f"  !! not a PDF (got {r.headers.get('content-type')}); skipped")
                ok = False
                continue
            dest.write_bytes(r.content)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if d.get("sha256") and d["sha256"] != digest:
            print(f"  !! {d['filename']}: sha256 differs from manifest - document changed upstream, re-check metadata")
            ok = False
        else:
            print(f"  ok {d['filename']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
